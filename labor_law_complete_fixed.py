#!/usr/bin/env python3
"""
劳动法律师智能助理 - 终极工业版 (Copilot 旗舰版终极核心)
特色：无损记忆压缩防失忆、前台 AI 动态读取右侧卷宗防重复提问、思维链推理
"""

import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import sys
import json
import streamlit as st
from dotenv import load_dotenv
from typing import TypedDict, List, Dict, Any, Annotated
from langchain_openai import ChatOpenAI
from langchain.embeddings.base import Embeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, RemoveMessage
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field

# 加载环境变量（优先从 Streamlit secrets 读取，其次从 .env 读取）
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))
api_key = st.secrets.get("DASHSCOPE_API_KEY") or os.getenv("DASHSCOPE_API_KEY", "")
base_url = st.secrets.get("DASHSCOPE_BASE_URL") or os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")

# ==========================================
# 1. 自定义兼容 DashScope 的 Embeddings
# ==========================================
class DashScopeEmbeddings(Embeddings):
    def __init__(self, api_key: str, model: str = "text-embedding-v2"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        import requests
        embeddings = []
        for text in texts:
            response = requests.post(
                f"{self.base_url}/embeddings",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "input": text,
                    "encoding_format": "float"
                },
                verify=False
            )
            if response.status_code == 200:
                result = response.json()
                if "data" in result and len(result["data"]) > 0:
                    embeddings.append(result["data"][0]["embedding"])
                else:
                    embeddings.append([0.0] * 1536)
            else:
                embeddings.append([0.0] * 1536)
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

# ==========================================
# 2. 初始化 LLM 和 Embeddings
# ==========================================
print(">>> 正在初始化大模型和知识库...")

embeddings = DashScopeEmbeddings(api_key=api_key, model="text-embedding-v2")
llm = ChatOpenAI(
    model="qwen-plus",
    api_key=st.secrets["DASHSCOPE_API_KEY"],
    base_url=st.secrets.get("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    temperature=0.3,
    max_tokens=4000
)

# 轻量级 LLM：用于 triage、query rewrite、质检等场景，保留足够输出长度
llm_fast = ChatOpenAI(
    model="qwen-plus",
    api_key=st.secrets["DASHSCOPE_API_KEY"],
    base_url=st.secrets.get("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    temperature=0.1,
    max_tokens=3000
)

# ==========================================
# 3. 初始化 RAG 知识库 (高级 Semantic Chunking 法条级切分)
# ==========================================
persist_dir = "./chroma_db"

class LegalRegexSplitter:
    """专为中国法律文档定制的正则切分器"""
    def split_documents(self, documents):
        import re
        import os
        from langchain_core.documents import Document
        
        # 1. 跨页缝合：把碎片的 PDF 拼成整篇法律
        source_text_map = {}
        for doc in documents:
            source = doc.metadata.get("source", "未知法律文件")
            if source not in source_text_map:
                source_text_map[source] = ""
            source_text_map[source] += doc.page_content + "\n"

        final_chunks = []
        for source, text in source_text_map.items():
            # 2. 清理硬回车断行
            text = re.sub(r'(?<=[^。；：？！\n])\n(?=[^\n])', '', text)
            
            # 3. 核心正则：找到"第X条"切一刀
            pattern = r"(?=第[一二三四五六七八九十百千万]+条[\s、，。])"
            raw_chunks = re.split(pattern, text)
            
            # 4. 提取文件名作为法律名称，强行打上 Metadata 标签
            law_name = os.path.basename(source).replace(".pdf", "")
            
            for chunk in raw_chunks:
                chunk = chunk.strip()
                if len(chunk) > 15 and chunk.startswith("第"):
                    # 把法律名称强行缝合进文本开头，防止 AI 瞎猜
                    enhanced_content = f"《{law_name}》 {chunk}"
                    final_chunks.append(Document(
                        page_content=enhanced_content, 
                        metadata={"source": source, "law_name": law_name}
                    ))
        return final_chunks

if os.path.exists(persist_dir):
    print(">>> 发现已存在的高级语义向量库，正在加载...")
    vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
else:
    print(">>> 启动高级法条切分引擎，正在重构知识库...")
    data_dir = './data/'
    os.makedirs(data_dir, exist_ok=True)
    loader = DirectoryLoader(data_dir, glob="**/*.pdf", loader_cls=PyMuPDFLoader)
    documents = loader.load()
    
    if documents:
        splitter = LegalRegexSplitter()
        splits = splitter.split_documents(documents)
        print(f"[成功] 完美切分出 {len(splits)} 条独立的法律条款！正在入库...")
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory=persist_dir)
    else:
        print("[警告] ./data/ 目录下没有 PDF，请放入法律文件后重启。")
        # 创建空向量库以避免未绑定错误
        from langchain_core.documents import Document
        vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)

# 检索器配置：因为切得很准，我们可以放心捞 Top 5
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# ==========================================
# 4. 定义全局状态 (新增并行与自我纠错字段)
# ==========================================
class LaborLawState(TypedDict):
    messages: Annotated[list, add_messages] 
    chat_summary: str                       
    
    triage_result: dict                     
    form_data: dict                         
    legal_facts_summary: str
    
    # 🌟 优化二新增：并行子图的各自产出
    relevant_laws: str
    similar_cases: str
    
    # 🌟 优化一新增：审查与纠错机制字段
    final_review: str
    reviewer_feedback: str # 质检员的打回修改意见
    retry_count: int       # 重试计数器，防止死循环
    is_pass_flag: bool     # 是否通过质检

# ==========================================
# 5. 定义智能体节点函数
# ==========================================

from typing import Literal

class TriageOutput(BaseModel):
    """分诊台结构化输出模型"""
    action: Literal["chat", "form"] = Field(description="动作类型：只能是 'chat' 或者是 'form'")
    category: str = Field(description="意图分类")
    reply: str = Field(description="回复给用户的话术")

class SearchQueries(BaseModel):
    """查询重写结构化输出模型"""
    queries: List[str] = Field(description="包含3个核心法律检索短语的数组")

# 🌟 新增：质检员结构化输出
class QualityOutput(BaseModel):
    """质检员输出模型"""
    is_pass: bool = Field(description="审查是否合格。若有明显漏洞则为False")
    feedback: str = Field(description="如果不合格，指出具体修改意见。如果合格，填'无'")

def summarize_conversation_node(state: LaborLawState):
    """节点0：Context 工程记忆压缩机（每10轮对话压缩一次）"""
    messages = state.get("messages", [])
    old_summary = state.get("chat_summary", "")
    
    if len(messages) <= 10:
        return {}
        
    print(f"\n[CONTEXT] 检测到对话达到 {len(messages)} 条，触发第 {len(messages)//10} 次 Context 压缩...")
    # 保留最近4条（2轮完整对话），压缩更早的内容
    messages_to_summarize = messages[:-4]
    
    # 获取当前卷宗状态，指导压缩时重点保留
    form_data = state.get("form_data", {})
    if not form_data:
        form_data = {"案件发生地": "", "单位名称": "", "平均月薪": "", "时间节点": "", "核心诉求": "", "详细经过": ""}
    
    missing_fields = [k for k, v in form_data.items() if not v]
    filled_fields = [f"{k}: {v}" for k, v in form_data.items() if v]
    missing_hint = f"目前仍缺失的字段：{missing_fields}" if missing_fields else "所有字段已收集完毕"
    filled_hint = f"已收集到的字段：{'; '.join(filled_fields)}" if filled_fields else "暂无已收集字段"
    
    prompt = f"""你是一名极其严谨的法庭书记员，正在进行 Context 工程压缩。

⚠️ 压缩核心目标：在大幅缩减对话长度的同时，绝对保留所有对填写右侧卷宗有用的信息！

【当前右侧卷宗状态】：
{filled_hint}
{missing_hint}

⚠️ 绝对禁止省略以下关键信息（如果出现过）：
1. 具体的数字（工资数额、索赔金额、工作年限、赔偿金额等）
2. 具体的时间节点（哪年哪月入职/离职/发生争议，几号）
3. 具体的地点（城市、省份）
4. 公司/单位的具体名称和行为
5. 用户的核心诉求和情绪
6. 任何与缺失字段「{missing_fields}」相关的线索

⚠️ 可以安全省略的内容：
- 问候、寒暄、客套话
- AI 的重复解释和安抚
- 已在卷宗中明确记录的信息（但保留细节补充）

【核心限制】：请用精准的要点（Bullet Points）罗列关键事实。合并后的总字数必须严格控制在 600 字以内！

【已有案件档案】：
{old_summary if old_summary else "暂无"}

【待压缩的早期对话】：
{messages_to_summarize}

请直接输出更新后的完整档案文本。"""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    new_summary = response.content
    delete_messages = [RemoveMessage(id=m.id) for m in messages_to_summarize]
    
    return {
        "chat_summary": new_summary,
        "messages": delete_messages 
    }

def triage_node(state: LaborLawState) -> LaborLawState:
    """节点1：前台分诊 + Context 工程信息完善度审视"""
    print("\n[AI] [分诊台 AI] 正在结合右侧卷宗与聊天记录分析意图...")
    chat_history = state.get("messages", [])
    summary = state.get("chat_summary", "")
    summary_text = f"\n【⚠️ 长期背景案件档案（压缩后的历史记忆）】：\n{summary}" if summary else ""
    
    form_data = state.get("form_data", {})
    if not form_data: 
        form_data = {"案件发生地": "", "单位名称": "", "平均月薪": "", "时间节点": "", "核心诉求": "", "详细经过": ""}
        
    # 信息完善度分析
    filled = {k: v for k, v in form_data.items() if v}
    missing = [k for k, v in form_data.items() if not v]
    total_fields = len(form_data)
    filled_count = len(filled)
    completeness_pct = int(filled_count / total_fields * 100) if total_fields > 0 else 0
    
    form_status = "\n".join([
        f"- {k}: ✅ {v}" if v else f"- {k}: ❌ [缺失，待补充]"
        for k, v in form_data.items()
    ])
    
    # 判断是否刚经历过压缩（summary 非空且对话很短说明刚压缩完）
    is_post_compression = bool(summary) and len(chat_history) <= 6
    
    compression_note = ""
    if is_post_compression:
        compression_note = f"""
【🔔 Context 压缩后审视指令（重要！）】：
刚刚完成了对话记忆压缩。请你现在做两件事：
1. 仔细审视右侧卷宗中每个字段的状态，检查是否有信息遗漏或矛盾
2. 根据缺失字段，规划下一个最关键的问题来完善信息
3. 如果核心信息（至少有：案件发生地、核心诉求、详细经过）已齐全，请立即转交表单
"""
    
    # 核心信息判断：至少3个关键字段已填写
    core_fields = ["核心诉求", "详细经过"]
    core_filled = sum(1 for f in core_fields if form_data.get(f, ""))
    has_basic_info = filled_count >= 3 and core_filled >= 1
    
    completeness_note = f"""
【📊 当前信息完善度】：{completeness_pct}%（{filled_count}/{total_fields} 项已填写）
已收集：{list(filled.keys()) if filled else '无'}
待补充：{missing if missing else '无'}
{'✅ 核心信息基本充足，如用户不再补充细节，可考虑转交分析。' if has_basic_info else '⚠️ 核心信息仍然不足，需要继续追问。'}
"""
    
    prompt = f"""你是一个专业的劳动法律师前台分诊智能体，同时负责 Context 工程中的信息完善度管理。
    在开口前，请务必仔细核对【当前右侧卷宗状态】和【信息完善度】。
    
    【当前右侧卷宗状态】：
    {form_status}
    {summary_text}
    
    {completeness_note}
    {compression_note}
    
    【Copilot 极严格提问与引导策略】：
    1. 审视卷宗和信息完善度。如果某个字段已有具体内容，**绝对禁止**再次询问！
    2. 只有发现"❌ [缺失，待补充]"的字段时，才可以提问。
    3. 每次**只挑选 1 个最关键的缺失字段**进行自然追问。优先级：核心诉求 > 详细经过 > 案件发生地 > 时间节点 > 单位名称 > 平均月薪。
    4. 追问时自然告诉用户："我已经将您的XX信息记录在右侧表格中了，请问您的YY是什么？"
    5. **关键判断**：如果信息完善度 ≥ 60%，且核心诉求和详细经过至少有1个已填写，且用户没有主动补充更多细节的意愿，请将 action 设为 "form"，表示信息收集完毕。
    6. 如果用户明确说"够了""出报告""开始分析"等，立即设 action 为 "form"。
    7. 如果用户在提供新信息，即使完善度较高，也应继续 chat 以收集更多细节，除非已超10轮对话。
    """
    messages_for_llm = [SystemMessage(content=prompt)] + chat_history
    
    try:
        triage_output = llm.with_structured_output(TriageOutput).invoke(messages_for_llm)
        triage_result = {
            "action": triage_output.action,
            "category": triage_output.category,
            "reply": triage_output.reply
        }
    except Exception as e:
        print(f"[ERROR] 分诊台结构化输出失败: {e}")
        triage_result = {"action": "chat", "category": "系统降级", "reply": "抱歉，系统刚刚开小差了，您可以再详细描述一下您的诉求吗？"}
    
    print(f"[TARGET] [意图识别结果] 动作: {triage_result['action']}, 分类: {triage_result['category']}, 完善度: {completeness_pct}%")
    ai_reply_message = AIMessage(content=triage_result["reply"])
    return {"triage_result": triage_result, "messages": [ai_reply_message]}

def extract_output(text: str) -> str:
    import re
    match = re.search(r'<output>(.*?)</output>', text, re.DOTALL)
    if match: return match.group(1).strip()
    match = re.search(r'</thinking>(.*)', text, re.DOTALL)
    if match: return match.group(1).strip()
    return text.strip()

def fact_summarizer_node(state: LaborLawState) -> LaborLawState:
    """节点2：事实梳理员（使用轻量 LLM 提速，保留思考链）"""
    form_data = state.get("form_data", {})
    form_text = "\n".join([f"- {k}: {v}" for k, v in form_data.items()])
    prompt = f"""请根据以下案件信息梳理关键事实：\n【用户表单提交信息】：\n{form_text}
    请你必须先在 <thinking> 标签内进行沙盘推演和逻辑自洽检查。
    思考完成后，再在 <output> 标签内梳理：1. 争议焦点 2. 关键时间节点 3. 证据情况分析 4. 法律适用预判"""
    messages = [SystemMessage(content="你是劳动法律师助理"), HumanMessage(content=prompt)]
    return {"legal_facts_summary": extract_output(llm_fast.invoke(messages).content)}

def legal_researcher_node(state: LaborLawState) -> LaborLawState:
    """节点3：法条检索 + 案例参考（合并原并行双节点，减少1次LLM调用）"""
    print("\n[法条+案例专员] 正在查询知识库并检索参考案例...")
    summary = state.get("legal_facts_summary", "")
    if not summary: return {"relevant_laws": "暂无相关事实，无法检索法条", "similar_cases": ""}

    # 用轻量 LLM 快速提取检索词（替代 structured_output，更快）
    rewrite_prompt = f"根据以下案件事实，提取3个用于法律检索的中文关键词，用逗号分隔。不要输出其他内容。\n事实：{summary[:300]}"
    try:
        queries_text = llm_fast.invoke([HumanMessage(content=rewrite_prompt)]).content
        queries = [q.strip() for q in queries_text.split('，') if q.strip()][:3]
        if not queries: queries = [summary[:50]]
    except Exception:
        queries = [summary[:50]]

    # 向量检索
    all_docs = []
    seen_content = set()
    try:
        batch_results = retriever.batch(queries)
        for docs in batch_results:
            for doc in docs:
                if doc.page_content not in seen_content:
                    seen_content.add(doc.page_content)
                    all_docs.append(doc)
    except Exception:
        for q in queries:
            for doc in retriever.invoke(q):
                if doc.page_content not in seen_content:
                    seen_content.add(doc.page_content)
                    all_docs.append(doc)
    
    all_docs = all_docs[:6]
    relevant_docs = "\n\n".join([doc.page_content for doc in all_docs]) if all_docs else "暂无法条"
    
    prompt = f"""你是一名精通中国劳动法的资深裁判者。
    【案件事实】：{summary}
    【相关法条】：{relevant_docs}
    
    请在 <thinking> 标签中排查法条冲突（注意特别法优于一般法，上位法优于下位法）。
    思考后在 <output> 标签输出：
    ## 法条适用分析
    1. 适用具体法律条款 2. 适用说明 3. 赔偿计算依据 4. 程序建议

    ## 参考案例要旨
    简述1个国内劳动争议领域的典型相似判例要旨及对本案参考价值。"""
    messages = [SystemMessage(content="你是精通法理的资深专家"), HumanMessage(content=prompt)]
    result = extract_output(llm.invoke(messages).content)
    return {"relevant_laws": result, "similar_cases": ""}

def compliance_reviewer_node(state: LaborLawState) -> LaborLawState:
    """节点4 (汇合点)：合规审核员"""
    print("\n[合规审核员] 正在汇编最终报告...")
    facts = state.get("legal_facts_summary", "")
    laws = state.get("relevant_laws", "")
    feedback = state.get("reviewer_feedback", "")
    
    form_text = "\n".join([f"- {k}: {v}" for k, v in state.get("form_data", {}).items()])
    prompt = f"原始信息：\n{form_text}\n事实：\n{facts}\n法条分析：\n{laws}"
    
    if feedback:
        print(f"⚠️ 收到主编修改意见，开始重写：{feedback}")
        prompt += f"\n\n【主编打回修改意见】：\n{feedback}\n请务必严格修正上述漏洞！"
        
    prompt += """
请你在 <thinking> 标签内审视前置分析有无漏洞。思考后在 <output> 标签输出一份精炼的合规法律建议报告，必须包含以下三部分：

## 适用法条
列出本案最核心的2-3条法律条款及其适用要点，简明扼要。

## 关键证据与注意事项
指出用户需要准备的核心证据及取证注意事项。

## 操作建议
给出具体可执行的操作步骤和风险提示。

⚠️ 严格字数限制：输出正文必须在2000字以内，超出部分直接截断。语言精炼，直击要害，不要赘述。"""
    messages = [SystemMessage(content="你是资深劳动法律师"), HumanMessage(content=prompt)]
    return {"final_review": extract_output(llm.invoke(messages).content)}

# 🌟 新增：质检员把关
def quality_inspector_node(state: LaborLawState) -> LaborLawState:
    """节点5：主编质检员（打回重写一次后直接通过，不再二次审查）"""
    print("\n[主编质检员] 正在快速审核...")
    review = state.get("final_review", "")
    retry_count = state.get("retry_count", 0)
    
    # 已经重写过了，直接通过，省掉一次 LLM 调用
    if retry_count >= 1:
        print(f"[QA] ⏭️ 已重写过一次，直接通过 | 重试: {retry_count}")
        return {
            "reviewer_feedback": "",
            "retry_count": retry_count + 1,
            "is_pass_flag": True
        }
    
    form_text = "\n".join([f"- {k}: {v}" for k, v in state.get("form_data", {}).items()])
    
    prompt = f"""审查这份报告是否正面回答了用户诉求？有无明显逻辑漏洞或法条引用不当？只需回答 PASS 或 FAIL + 简要意见。
    【用户诉求】：\n{form_text}
    【报告】：\n{review[:1500]}"""
    
    try:
        qa_out = llm_fast.with_structured_output(QualityOutput).invoke([HumanMessage(content=prompt)])
        is_pass, feedback = qa_out.is_pass, qa_out.feedback
    except Exception as e:
        print(f"[QA ERROR] 质检解析失败，强行放行: {e}")
        is_pass, feedback = True, "无"
        
    print(f"[QA] {'✅ 通过' if is_pass else '❌ 打回'} | 重试: {retry_count} | 意见: {feedback}")

    return {
        "reviewer_feedback": feedback if not is_pass else "",
        "retry_count": retry_count + 1,
        "is_pass_flag": is_pass
    }

# ==========================================
# 6. 构建 LangGraph 工作流
# ==========================================
print(">>> 正在构建精简高速多智能体架构流...")
workflow = StateGraph(LaborLawState)

workflow.add_node("summarizer", summarize_conversation_node)
workflow.add_node("triage", triage_node)
workflow.add_node("fact_summarizer", fact_summarizer_node)
workflow.add_node("legal_researcher", legal_researcher_node)
workflow.add_node("compliance_reviewer", compliance_reviewer_node)
workflow.add_node("quality_inspector", quality_inspector_node)

workflow.set_entry_point("summarizer")
workflow.add_edge("summarizer", "triage")

# 路由 1：分诊
def route_after_triage(state: LaborLawState):
    return "process_case" if state.get("triage_result", {}).get("action") == "form" else "end"

workflow.add_conditional_edges("triage", route_after_triage, {"process_case": "fact_summarizer", "end": END})

# 串行流水线（合并原并行双节点，减少1次LLM调用）
workflow.add_edge("fact_summarizer", "legal_researcher")
workflow.add_edge("legal_researcher", "compliance_reviewer")
workflow.add_edge("compliance_reviewer", "quality_inspector")

# 质检纠错环
def route_after_qa(state: LaborLawState):
    if state.get("is_pass_flag", True) or state.get("retry_count", 0) >= 2:
        return "end"
    return "retry"

workflow.add_conditional_edges("quality_inspector", route_after_qa, {"end": END, "retry": "compliance_reviewer"})

memory = MemorySaver()
app = workflow.compile(checkpointer=memory, interrupt_before=["fact_summarizer"])
print("=" * 50)

# ==========================================
# 7. 真实交互模拟测试
# ==========================================
def test_workflow():
    print(">>> [MOVIE] 开始进行全链路测试...\n")
    config = {"configurable": {"thread_id": "多轮测试-001"}}

    # --- 阶段1：普法闲聊 ---
    print("[USER] 用户: 劳动法规定辞退要赔钱吗？")
    state_1 = {"messages": [HumanMessage(content="劳动法规定辞退要赔钱吗？")]}
    res_1 = app.invoke(state_1, config)
    print(f"[AI] AI: {res_1['triage_result']['reply']}\n" + "-"*30)

    # --- 阶段2：不完整案情，触发单步追问 ---
    print("\n[USER] 用户: 老板今天把我开了。")
    state_2 = {"messages": [HumanMessage(content="老板今天把我开了。")]}
    res_2 = app.invoke(state_2, config)
    print(f"[AI] AI: {res_2['triage_result']['reply']}")
    print("[STATUS] 当前 action:", res_2['triage_result']['action'])
    
    # --- 阶段3：补充案情，触发表单就绪 ---
    print("\n[USER] 用户: 在这个公司干了3年，平时月薪8000。我想要经济补偿。")
    state_3 = {"messages": [HumanMessage(content="在这个公司干了3年，平时月薪8000。我想要经济补偿。")]}
    res_3 = app.invoke(state_3, config)
    print(f"[AI] AI: {res_3['triage_result']['reply']}")
    print("[PAUSE]  (系统已侦测到核心要素，流转至梳理专员前已自动挂起...)")
    
    # --- 阶段4：模拟 Copilot 面板提交 ---
    mock_form = {
        "单位名称": "未知",
        "平均月薪": "8000",
        "时间节点": "干了3年，今天被开除",
        "核心诉求": "要经济补偿",
        "详细经过": "被老板突然开除"
    }
    
    print("\n[POINT] 模拟用户在右侧面板点击确认...")
    app.update_state(config, {"form_data": mock_form})
    
    print("[FLASH] 系统读档唤醒，开始带有<thinking>标签的深度推理...")
    final_res = app.invoke(None, config)
    
    print("\n[SUCCESS] 最终审查报告：\n")
    print(final_res.get('final_review', '生成失败'))

if __name__ == "__main__":
    test_workflow()