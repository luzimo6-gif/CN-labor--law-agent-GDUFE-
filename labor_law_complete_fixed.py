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
    max_tokens=2000
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
    """节点0：无损记忆压缩机"""
    messages = state.get("messages", [])
    old_summary = state.get("chat_summary", "")
    
    if len(messages) <= 12:
        return {}
        
    print("\n[CLEAN] 检测到对话超过12条，触发无损记忆压缩...")
    messages_to_summarize = messages[:-6]
    
    prompt = f"""你是一名极其严谨的法庭书记员。请将下面的【早期对话】与【已有案件档案】合并。
    
    ⚠️ 绝对禁止省略以下关键信息（如果出现过）：
    1. 具体的数字（工资数额、索赔金额、工作年限等）
    2. 具体的时间节点（哪年哪月、几号发生的事情）
    3. 用户的具体情绪或已经做出的动作
    4. 公司行为的具体描述
    
    【核心限制】：请用精准的要点（Bullet Points）罗列关键事实，去除所有客套话。合并后的总字数必须严格控制在 500 字以内！
    
    【已有案件档案】：\n{old_summary if old_summary else "暂无"}
    
    【待合并的早期对话】：\n{messages_to_summarize}
    
    请直接输出更新后的完整档案文本。"""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    new_summary = response.content
    delete_messages = [RemoveMessage(id=m.id) for m in messages_to_summarize]
    
    return {
        "chat_summary": new_summary,
        "messages": delete_messages 
    }

def triage_node(state: LaborLawState) -> LaborLawState:
    """节点1：前台分诊"""
    print("\n[AI] [分诊台 AI] 正在结合右侧卷宗与聊天记录分析意图...")
    chat_history = state.get("messages", [])
    summary = state.get("chat_summary", "")
    summary_text = f"\n【⚠️ 长期背景案件档案】：\n{summary}" if summary else ""
    
    form_data = state.get("form_data", {})
    if not form_data: 
        form_data = {"案件发生地": "", "单位名称": "", "平均月薪": "", "时间节点": "", "核心诉求": "", "详细经过": ""}
        
    form_status = "\n".join([
        f"- {k}: {v if v else '❌ [缺失，待补充]'}" 
        for k, v in form_data.items()
    ])
    
    prompt = f"""你是一个专业的劳动法律师前台分诊智能体。
    在开口前，请务必仔细核对【当前右侧卷宗状态】。
    
    【当前右侧卷宗状态】：
    {form_status}
    {summary_text}
    
    【Copilot 极严格提问与引导策略】：
    1. 审视上面的卷宗。如果某个字段已经有具体内容，**绝对禁止**再次询问相关信息！
    2. 只有当发现带有"❌ [缺失，待补充]"的字段时，才可以提问。
    3. 每次**只挑选 1 个最关键的缺失字段**进行自然追问。在追问时，你可以非常自然地告诉用户："我已经将您的XX信息自动记录在右侧表格中了，请问您的YY是什么？"
    4. 如果所有核心信息已经基本收齐，或者用户明确要求"出报告/开始分析"，请立即转交表单，停止任何追问。
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
    
    print(f"[TARGET] [意图识别结果] 动作: {triage_result['action']}, 分类: {triage_result['category']}")
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
    """节点2：事实梳理员"""
    form_data = state.get("form_data", {})
    form_text = "\n".join([f"- {k}: {v}" for k, v in form_data.items()])
    prompt = f"""请根据以下案件信息梳理关键事实：\n【用户表单提交信息】：\n{form_text}
    请你必须先在 <thinking> 标签内进行沙盘推演和逻辑自洽检查。
    思考完成后，再在 <output> 标签内梳理：1. 争议焦点 2. 关键时间节点 3. 证据情况分析 4. 法律适用预判"""
    messages = [SystemMessage(content="你是劳动法律师助理"), HumanMessage(content=prompt)]
    return {"legal_facts_summary": extract_output(llm.invoke(messages).content)}

def legal_researcher_node(state: LaborLawState) -> LaborLawState:
    """节点3 (并行分支A)：法条检索专员"""
    print("\n[并发-分支A] [法条专员] 正在查询知识库...")
    summary = state.get("legal_facts_summary", "")
    if not summary: return {"relevant_laws": "暂无相关事实，无法检索法条"}

    rewrite_prompt = f"你是一个专业的劳动法检索专家。请根据以下事实，提炼出 3 个用于检索的核心【法律短语】。\n要求：必须是纯正的法律术语，不要带人名公司名。\n事实：{summary}"
    try:
        rewrite_output = llm.with_structured_output(SearchQueries).invoke([
            SystemMessage(content="你是专业的法律检索专家，请提取查询词。"), 
            HumanMessage(content=rewrite_prompt)
        ])
        queries = rewrite_output.queries
        if not queries: queries = [summary]
    except Exception:
        queries = [summary]

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
    
    all_docs = all_docs[:8]
    relevant_docs = "\n\n".join([doc.page_content for doc in all_docs]) if all_docs else "暂无法条"

    region = state.get("form_data", {}).get("案件发生地", "未明确")
    if not region: region = "未明确"
    
    prompt = f"""你是一名精通中国法律适用规则的资深裁判者。
    【案件事实】：{summary}
    【多路召回的相关法条】：{relevant_docs}
    
    【⚖️ 绝对指令】：请在 <thinking> 中排查冲突（注意特别法优于一般法，上位法优于下位法）。
    思考后在 <output> 标签输出：1. 适用具体法律条款 2. 适用说明 3. 赔偿计算依据 4. 程序建议"""
    messages = [SystemMessage(content="你是精通法理的资深专家"), HumanMessage(content=prompt)]
    return {"relevant_laws": extract_output(llm.invoke(messages).content)}

# 🌟 新增：并行分支B - 案例专员
def case_researcher_node(state: LaborLawState) -> LaborLawState:
    """节点4 (并行分支B)：相似案例检索员"""
    print("\n[并发-分支B] [案例专员] 正在寻找典型相似判例...")
    summary = state.get("legal_facts_summary", "")
    if not summary: return {"similar_cases": "暂无相似案例"}
    
    prompt = f"""你是高级判例检索专家。请基于以下事实，简述1个国内劳动争议领域的典型相似判例（可直接生成典型法理要旨）。
    案件事实：{summary}
    请在 <thinking> 中推演相似度。在 <output> 标签中输出：1. 案例核心要旨 2. 对本案的参考价值。"""
    
    messages = [SystemMessage(content="你是高级判例检索专家"), HumanMessage(content=prompt)]
    return {"similar_cases": extract_output(llm.invoke(messages).content)}

def compliance_reviewer_node(state: LaborLawState) -> LaborLawState:
    """节点5 (汇合点)：合规审核员 (支持接受质检意见重写)"""
    print("\n[AI] [合规审核员] 正在汇编最终报告...")
    facts = state.get("legal_facts_summary", "")
    laws = state.get("relevant_laws", "")
    cases = state.get("similar_cases", "") # 获取并行节点的案例
    feedback = state.get("reviewer_feedback", "") # 获取质检员意见
    
    form_text = "\n".join([f"- {k}: {v}" for k, v in state.get("form_data", {}).items()])
    prompt = f"原始信息：\n{form_text}\n事实：\n{facts}\n法条：\n{laws}\n参考案例：\n{cases}"
    
    # 🌟 纠错触发：如果主编有意见，强令重写
    if feedback:
        print(f"⚠️ 收到主编修改意见，开始重写：{feedback}")
        prompt += f"\n\n【⚠️ 主编打回修改意见】：\n{feedback}\n请务必严格修正上述漏洞，重新出具最终法律建议！"
        
    prompt += "\n请在 <thinking> 标签内审视前置分析有无漏洞。思考后在 <output> 标签提供：1. 最终法律建议 2. 操作步骤 3. 风险提示 4. 沟通策略 5. 证据建议"
    messages = [SystemMessage(content="你是资深专家"), HumanMessage(content=prompt)]
    return {"final_review": extract_output(llm.invoke(messages).content)}

# 🌟 新增：质检员把关
def quality_inspector_node(state: LaborLawState) -> LaborLawState:
    """节点6：主编质检员"""
    print("\n[AI] [主编质检员] 正在对初稿进行出厂审核...")
    review = state.get("final_review", "")
    retry_count = state.get("retry_count", 0)
    form_text = "\n".join([f"- {k}: {v}" for k, v in state.get("form_data", {}).items()])
    
    prompt = f"""你是律所极其严苛的高级合伙人。请审查这份报告初稿：
    【用户原始诉求】：\n{form_text}
    【待审初稿】：\n{review}
    
    请严苛判断初稿是否正面回答了用户诉求？是否存在逻辑漏洞、计算缺失或法条引用不当？
    """
    try:
        qa_out = llm.with_structured_output(QualityOutput).invoke([HumanMessage(content=prompt)])
        is_pass, feedback = qa_out.is_pass, qa_out.feedback
    except Exception as e:
        print(f"[QA ERROR] 质检解析失败，为防卡死强行放行: {e}")
        is_pass, feedback = True, "无"
        
    print(f"====== QA 质检结果 ======\n状态: {'✅ 通过' if is_pass else '❌ 打回'} \n重试次数: {retry_count}\n意见: {feedback}\n=========================")

    return {
        "reviewer_feedback": feedback if not is_pass else "",
        "retry_count": retry_count + 1,
        "is_pass_flag": is_pass
    }

# ==========================================
# 6. 构建 LangGraph 工作流
# ==========================================
print(">>> 正在构建多智能体并发与自纠错架构流...")
workflow = StateGraph(LaborLawState)

workflow.add_node("summarizer", summarize_conversation_node)
workflow.add_node("triage", triage_node)  
workflow.add_node("fact_summarizer", fact_summarizer_node)

# 并行双节点
workflow.add_node("legal_researcher", legal_researcher_node)
workflow.add_node("case_researcher", case_researcher_node)

workflow.add_node("compliance_reviewer", compliance_reviewer_node)
workflow.add_node("quality_inspector", quality_inspector_node)

workflow.set_entry_point("summarizer")
workflow.add_edge("summarizer", "triage")

# 路由 1：分诊
def route_after_triage(state: LaborLawState):
    return "process_case" if state.get("triage_result", {}).get("action") == "form" else "end"

workflow.add_conditional_edges("triage", route_after_triage, {"process_case": "fact_summarizer", "end": END})

# 🌟 路由 2：并发扇出 (Fan-out)
workflow.add_edge("fact_summarizer", "legal_researcher")
workflow.add_edge("fact_summarizer", "case_researcher")

# 🌟 路由 3：并发汇合 (Fan-in)
workflow.add_edge(["legal_researcher", "case_researcher"], "compliance_reviewer")

# 路由 4：送交质检
workflow.add_edge("compliance_reviewer", "quality_inspector")

# 🌟 路由 5：质检纠错环
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