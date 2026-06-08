#!/usr/bin/env python3
"""
劳动法 PDF 离线入库脚本
===========================
功能：读取 data/ 目录下的所有 PDF，用 pymupdf 加载 → 按「第X条」切块 →
     调用 DashScope Embeddings 向量化 → 批量写入 Pinecone 云端索引。

运行前请配置 .env 中的 PINECONE_API_KEY / DASHSCOPE_API_KEY。
本脚本仅在本地运行一次（或数据更新后重新运行）。
"""

import os
import re
import sys
import time
import traceback
from typing import List, Dict
from pathlib import Path

# ---- 环境变量 ----
from dotenv import load_dotenv
load_dotenv(override=True)

# ---- 配置 ----
DASHSCOPE_API_KEY     = os.getenv("DASHSCOPE_API_KEY", "")
DASHSCOPE_BASE_URL    = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
DASHSCOPE_EMBED_MODEL = os.getenv("DASHSCOPE_EMBED_MODEL", "text-embedding-v2")
EMBED_DIMENSION       = int(os.getenv("EMBED_DIMENSION", "1536"))

PINECONE_API_KEY      = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME   = os.getenv("PINECONE_INDEX_NAME", "labor-law")
PINECONE_CLOUD        = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION       = os.getenv("PINECONE_REGION", "us-east-1")

DATA_DIR              = os.getenv("DATA_DIR", "./data/")
VERIFY_SSL            = os.getenv("VERIFY_SSL", "false").lower() == "true"

# 每批上传条数（DashScope API 单次最多 25 条，保守用 10）
BATCH_SIZE            = int(os.getenv("UPLOAD_BATCH_SIZE", "10"))
# 批次间延时（秒），避免触发 API 速率限制
BATCH_DELAY           = float(os.getenv("UPLOAD_BATCH_DELAY", "0.5"))

# ---- 校验密钥 ----
def _mask(s):
    return s[:8] + "..." + s[-4:] if len(s) > 12 else "(KEY too short)"

if not DASHSCOPE_API_KEY:
    print("❌ 缺少 DASHSCOPE_API_KEY，请在 .env 中配置。")
    sys.exit(1)
if not PINECONE_API_KEY:
    print("❌ 缺少 PINECONE_API_KEY，请在 .env 中配置。")
    sys.exit(1)

print(f"📌 DashScope  Key : {_mask(DASHSCOPE_API_KEY)}")
print(f"📌 DashScope  URL : {DASHSCOPE_BASE_URL}")
print(f"📌 Pinecone   Key : {_mask(PINECONE_API_KEY)}")
print(f"📌 Pinecone Index : {PINECONE_INDEX_NAME}")
print(f"📌 数据目录       : {os.path.abspath(DATA_DIR)}")
print(f"📌 批次大小       : {BATCH_SIZE} 条/批")
print("=" * 60)

# ---- 提前导入重依赖（pymupdf / pinecone） ----
try:
    from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
    from langchain_core.documents import Document as LangchainDocument
except ImportError as e:
    print(f"❌ 缺少依赖: {e}")
    print("   请执行: pip install langchain-community pymupdf langchain-core")
    sys.exit(1)

try:
    from pinecone import Pinecone as PineconeClient, ServerlessSpec
    from langchain_pinecone import PineconeVectorStore
except ImportError as e:
    print(f"❌ 缺少依赖: {e}")
    print("   请执行: pip install pinecone-client langchain-pinecone")
    sys.exit(1)


# ==========================================
# 1. DashScope Embeddings（用原生 SDK 批量调用）
# ==========================================
class DashScopeEmbeddings:
    """DashScope 文本向量化，兼容 LangChain 的 embed_documents/embed_query 接口。
    使用 DashScope 原生 SDK 并支持批量输入（单次最多 25 条）。"""

    MAX_BATCH_SIZE = 25   # DashScope API 单次最大输入条数

    def __init__(self, api_key: str, model: str = "text-embedding-v2"):
        self.api_key = api_key
        self.model = model
        import dashscope
        dashscope.api_key = api_key

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        from dashscope import TextEmbedding
        all_vectors = []

        for i in range(0, len(texts), self.MAX_BATCH_SIZE):
            batch = texts[i:i + self.MAX_BATCH_SIZE]
            try:
                # 批量输入：input 可以是字符串列表或单个字符串
                resp = TextEmbedding.call(
                    model=self.model,
                    input=batch if len(batch) > 1 else batch[0],
                )
                if resp.status_code == 200 and resp.output and resp.output.get("embeddings"):
                    for emb in resp.output["embeddings"]:
                        all_vectors.append(emb["embedding"])
                else:
                    err_msg = resp.message if resp.message else f"HTTP {resp.status_code}"
                    raise RuntimeError(f"DashScope API 错误: {err_msg}")
            except Exception as e:
                print(f"  ❌ Embedding 批次 [{i}:{i+len(batch)}] 失败: {e}")
                raise  # 入库脚本不容忍静默失败

        return all_vectors

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


# ==========================================
# 2. 法条切分器（与 api.py 保持一致）
# ==========================================
class LegalRegexSplitter:
    """按「第X条」切分中国法律文档，支持中文/阿拉伯数字。无条款文档回退到段落切分。"""

    # 匹配「第X条」— 中文字号 + 阿拉伯数字
    ARTICLE_PATTERN = re.compile(
        r"(?:第[一二三四五六七八九十百千万零]+条)"   # 第一条、第十二条...
        r"|(?:第\d+条)",                            # 第1条、第12条...
    )

    MIN_CHUNK_CHARS = 80       # 段落切分时的最小 chunk 长度
    MAX_CHUNK_CHARS = 1200     # 段落切分时的最大 chunk 长度（用于拆分过长段落）

    def split_documents(self, documents: list) -> list:
        source_text_map: Dict[str, str] = {}
        for doc in documents:
            src = doc.metadata.get("source", "未知法律文件")
            source_text_map[src] = source_text_map.get(src, "") + doc.page_content + "\n"

        final_chunks: list = []
        total_sources = len(source_text_map)
        for idx, (source, text) in enumerate(source_text_map.items(), 1):
            law_name = os.path.basename(source).replace(".pdf", "")

            # 清理硬回车断行
            text = re.sub(r'(?<=[^。；：？！\n])\n(?=[^\n])', '', text)

            # 按「第X条」切分（不再要求条后有特定字符）
            raw_chunks = re.split(f"(?={self.ARTICLE_PATTERN.pattern})", text)

            chunk_count = 0
            for chunk in raw_chunks:
                chunk = chunk.strip()
                if len(chunk) >= 15 and chunk[0] == "第":
                    enhanced = f"《{law_name}》 {chunk}"
                    final_chunks.append(LangchainDocument(
                        page_content=enhanced,
                        metadata={"source": source, "law_name": law_name}
                    ))
                    chunk_count += 1

            # 兜底：无条款的文档（通知、办法、意见等）→ 段落切分
            if chunk_count == 0 and len(text.strip()) > 50:
                fallback_count = self._split_by_paragraph(
                    text, law_name, source, final_chunks
                )
                print(f"  [{idx}/{total_sources}] {law_name}: {chunk_count} 条 (段落切分: {fallback_count} 段)")
            else:
                print(f"  [{idx}/{total_sources}] {law_name}: {chunk_count} 条")

        return final_chunks

    def _split_by_paragraph(self, text: str, law_name: str, source: str, output: list) -> int:
        """将无条款文档按段落（连续双换行 / PDF 空行）切分成 chunk"""
        # 按空行或分页标记分段
        paragraphs = re.split(r'\n\s*\n|(?:\f)', text.strip())
        count = 0
        buffer = ""
        for para in paragraphs:
            para = para.strip()
            if len(para) < 20:           # 跳过太短的片段（页码/空行）
                continue
            if len(buffer) + len(para) < self.MAX_CHUNK_CHARS:
                buffer += para + "\n"
            else:
                if len(buffer) >= self.MIN_CHUNK_CHARS:
                    output.append(LangchainDocument(
                        page_content=f"《{law_name}》\n{buffer.strip()}",
                        metadata={"source": source, "law_name": law_name}
                    ))
                    count += 1
                buffer = para + "\n"
            # 如果当前段落本身就达到最小长度且 buffer 为空，直接存
            if not buffer or len(buffer) >= self.MIN_CHUNK_CHARS:
                pass  # 继续累积

        # 处理最后一段
        if len(buffer.strip()) >= self.MIN_CHUNK_CHARS:
            output.append(LangchainDocument(
                page_content=f"《{law_name}》\n{buffer.strip()}",
                metadata={"source": source, "law_name": law_name}
            ))
            count += 1

        return count


# ==========================================
# 3. 主流程
# ==========================================
def main():
    # ---- 3a. 连接 Pinecone，准备/创建索引 ----
    print("\n🔌 连接 Pinecone ...")
    pc = PineconeClient(api_key=PINECONE_API_KEY)

    existing_raw = pc.list_indexes()
    try:
        existing_names = [idx.name for idx in existing_raw]
    except (AttributeError, TypeError):
        existing_names = [idx["name"] for idx in existing_raw]

    if PINECONE_INDEX_NAME not in existing_names:
        print(f"📦 创建 Pinecone 索引 '{PINECONE_INDEX_NAME}' (dim={EMBED_DIMENSION}) ...")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBED_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )
        print("✅ 索引已创建（等待就绪...）")
        time.sleep(10)  # Serverless 索引就绪需要几秒
    else:
        # 检查已有向量数量
        try:
            idx_stats = pc.Index(PINECONE_INDEX_NAME).describe_index_stats()
            existing_count = idx_stats.get("total_vector_count", 0)
            if existing_count > 0:
                print(f"⚠️  索引 '{PINECONE_INDEX_NAME}' 已有 {existing_count} 条向量。")
                ans = input("是否清空后重新导入？(y/N): ").strip().lower()
                if ans == "y":
                    print("🗑️  清空已有向量...")
                    pc.Index(PINECONE_INDEX_NAME).delete(delete_all=True)
                    time.sleep(3)
                    print("✅ 已清空")
                else:
                    print("❌ 已取消操作")
                    return
            else:
                print(f"✅ 索引 '{PINECONE_INDEX_NAME}' 就绪（当前 0 条向量）")
        except Exception as e:
            print(f"⚠️  无法获取索引状态: {e}，继续尝试导入")

    # ---- 3b. 加载 PDF ----
    print(f"\n📂 扫描 PDF 目录: {os.path.abspath(DATA_DIR)}")
    if not os.path.isdir(DATA_DIR):
        print(f"❌ 目录不存在: {DATA_DIR}")
        sys.exit(1)

    loader = DirectoryLoader(DATA_DIR, glob="**/*.pdf", loader_cls=PyMuPDFLoader)
    documents = loader.load()
    print(f"📄 加载了 {len(documents)} 页 PDF 内容")

    if not documents:
        print("❌ 未找到任何 PDF 文件")
        sys.exit(1)

    # ---- 3c. 法条切分 ----
    print("\n✂️  正在按「第X条」切分法律条款 ...")
    splitter = LegalRegexSplitter()
    chunks = splitter.split_documents(documents)
    print(f"✅ 切分出 {len(chunks)} 条法律条款")

    if not chunks:
        print("❌ 切分结果为空，请检查 PDF 内容")
        sys.exit(1)

    # ---- 3d. 初始化 Embeddings ----
    print("\n🧠 初始化 DashScope Embeddings ...")
    embeddings = DashScopeEmbeddings(
        api_key=DASHSCOPE_API_KEY,
        model=DASHSCOPE_EMBED_MODEL,
    )

    # ---- 3e. 批量上传到 Pinecone ----
    print(f"\n📤 开始批量上传到 Pinecone（共 {len(chunks)} 条，每批 {BATCH_SIZE} 条）...")
    print("-" * 60)

    total_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE
    uploaded = 0
    failed = 0
    start_time = time.time()

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1

        # 提取文本预览
        previews = [c.page_content[:40].replace("\n", " ") for c in batch]
        print(f"\n  批次 [{batch_num}/{total_batches}]  {len(batch)} 条")
        print(f"    示例: {previews[0]}...")
        if len(previews) > 1:
            print(f"          {previews[-1]}...")

        try:
            # 使用 langchain_pinecone 的 from_documents 会自动处理 embedding + upsert
            PineconeVectorStore.from_documents(
                documents=batch,
                embedding=embeddings,
                index_name=PINECONE_INDEX_NAME,
                pinecone_api_key=PINECONE_API_KEY,
            )
            uploaded += len(batch)
            elapsed = time.time() - start_time
            pct = (uploaded / len(chunks)) * 100
            eta = (elapsed / uploaded) * (len(chunks) - uploaded) if uploaded > 0 else 0
            print(f"    ✅ 上传成功 ({uploaded}/{len(chunks)} = {pct:.1f}%)")
            print(f"       已用: {elapsed:.0f}s | 预计剩余: {eta:.0f}s")
        except Exception as e:
            failed += len(batch)
            print(f"    ❌ 批次失败: {type(e).__name__}: {e}")
            traceback.print_exc()

            # 失败后等待更长时间再重试下一批
            print("    ⏳ 等待 5 秒后继续...")
            time.sleep(5)

        # 批次间延时
        if i + BATCH_SIZE < len(chunks) and BATCH_DELAY > 0:
            time.sleep(BATCH_DELAY)

    # ---- 3f. 验证 ----
    print("\n" + "=" * 60)
    total_elapsed = time.time() - start_time
    print(f"🎉 导入完成！")
    print(f"   成功: {uploaded} 条")
    if failed > 0:
        print(f"   失败: {failed} 条")
    print(f"   耗时: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")

    # 验证
    try:
        index = pc.Index(PINECONE_INDEX_NAME)
        stats = index.describe_index_stats()
        count = stats.get("total_vector_count", 0)
        print(f"   索引中向量总数: {count}")
    except Exception as e:
        print(f"   ⚠️ 无法验证索引: {e}")

    print("=" * 60)
    print("✅ 离线入库脚本执行完毕。现在可以启动 api.py 使用新的向量库。")
    print(f"   uvicorn api:fastapi_app --host 0.0.0.0 --port 8000")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断")
    except Exception:
        print("\n❌ 未捕获异常:")
        traceback.print_exc()
        sys.exit(1)
