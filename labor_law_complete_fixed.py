#!/usr/bin/env python3
"""
劳动法律师智能助理 - 终极工业版 (Copilot 旗舰版终极核心)
特色：无损记忆压缩防失忆、前台 AI 动态读取右侧卷宗防重复提问、思维链推理
"""

import os
import sys
import json
import streamlit as st
from dotenv import load_dotenv
from typing import TypedDict, List, Dict, Any, Annotated
from langchain_openai import ChatOpenAI
from langchain.embeddings.base import Embeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, RemoveMessage
from langgraph.checkpoint.memory import MemorySaver

# 加载环境变量（优先从 Streamlit secrets 读取，其次从 .env 读取）
load_dotenv()
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
# 3. 初始化 RAG 知识库
# ==========================================
persist_dir = "./chroma_db"
if os.path.exists(persist_dir):
    vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
else:
    data_dir = './data/'
    os.makedirs(data_dir, exist_ok=True)
    loader = DirectoryLoader(data_dir, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    splits = text_splitter.split_documents(documents)
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory=persist_dir)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) 

# ==========================================
# 4. 定义全局状态 (带压缩记忆版)
# ==========================================
class LaborLawState(TypedDict):
    messages: Annotated[list, add_messages] 
    chat_summary: str                       
    
    triage_result: dict                     
    form_data: dict                         
    legal_facts_summary: str
    relevant_laws: str
    final_review: str

# ==========================================
# 5. 定义智能体节点函数
# ==========================================

def summarize_conversation_node(state: LaborLawState):
    """节点0：无损记忆压缩机 (修复金鱼记忆)"""
    messages = state.get("messages", [])
    old_summary = state.get("chat_summary", "")
    
    # 放宽触发条件，积攒到 12 条消息才压缩
    if len(messages) <= 12:
        return {}
        
    print("\n[CLEAN] 检测到对话超过12条，触发无损记忆压缩...")
    
    # 强制保留最近的 6 条作为鲜活记忆，绝不压缩！
    messages_to_summarize = messages[:-6]
    
    prompt = f"""你是一名极其严谨的法庭书记员。请将下面的【早期对话】与【已有案件档案】合并。
    
    ⚠️ 绝对禁止省略以下关键信息（如果出现过）：
    1. 具体的数字（工资数额、索赔金额、工作年限等）
    2. 具体的时间节点（哪年哪月、几号发生的事情）
    3. 用户的具体情绪或已经做出的动作
    4. 公司行为的具体描述
    
    请用精准的要点（Bullet Points）罗列关键事实档案。
    
    【已有案件档案】：\n{old_summary if old_summary else "暂无"}
    
    【待合并的早期对话】：\n{messages_to_summarize}
    
    请直接输出更新后的完整档案文本。"""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    new_summary = response.content
    
    delete_messages = [RemoveMessage(id=m.id) for m in messages_to_summarize]
    print(f"[PACKAGE] 案件档案已更新：\n{new_summary}")
    
    return {
        "chat_summary": new_summary,
        "messages": delete_messages 
    }

def triage_node(state: LaborLawState) -> LaborLawState:
    """节点1：前台分诊 (强化：动态读取表单状态防重复提问)"""
    print("\n[AI] [分诊台 AI] 正在结合右侧卷宗与聊天记录分析意图...")
    
    chat_history = state.get("messages", [])
    summary = state.get("chat_summary", "")
    summary_text = f"\n【⚠️ 长期背景案件档案】：\n{summary}" if summary else ""
    
    # 🌟 提取右侧表格状态，并打上明确的“缺失/已填”标记
    form_data = state.get("form_data", {})
    if not form_data: # 兜底默认结构
        form_data = {"单位名称": "", "平均月薪": "", "时间节点": "", "核心诉求": "", "详细经过": ""}
        
    form_status = "\n".join([
        f"- {k}: {v if v else '❌ [缺失，待补充]'}" 
        for k, v in form_data.items()
    ])
    
    prompt = f"""你是一个专业的劳动法前台分诊智能体。
    在开口前，请务必仔细核对【当前右侧卷宗状态】。
    
    【当前右侧卷宗状态】：
    {form_status}
    {summary_text}
    
    【Copilot 极严格提问策略】：
    1. 审视上面的卷宗。如果某个字段已经有具体内容，**绝对禁止**再次询问相关信息！
    2. 只有当发现带有“❌ [缺失，待补充]”的字段时，才可以提问。
    3. 每次**只挑选 1 个最关键的缺失字段**进行自然追问（比如先问诉求，再问薪资）。不要像机器人一样机械报菜名。
    4. 如果没有明显的缺失，或者用户明确要求“出报告/开始分析”，请立即转交表单。

    请严格按照以下 JSON 格式返回：
    1. 普法或信息收集阶段：
    返回 {{"action": "chat", "category": "普法/收集", "reply": "解答内容或基于缺失字段的单步追问话术"}}
    2. 核心信息已基本收齐（没有重要缺失）：
    返回 {{"action": "form", "category": "案件类型", "reply": "您的案卷信息我已经记录完毕。请核对右侧面板，确认无误后点击生成报告。"}}
    """

    messages_for_llm = [SystemMessage(content=prompt)] + chat_history
    response = llm.invoke(messages_for_llm)
    
    try:
        clean_content = response.content.replace('```json', '').replace('```', '').strip()
        triage_result = json.loads(clean_content)
    except Exception:
        triage_result = {"action": "chat", "category": "未知", "reply": "抱歉，请您再说详细一点可以吗？"}
        
    print(f"[TARGET] [意图识别结果] 动作: {triage_result['action']}, 分类: {triage_result['category']}")
    
    ai_reply_message = AIMessage(content=triage_result["reply"])
    return {"triage_result": triage_result, "messages": [ai_reply_message]}

def fact_summarizer_node(state: LaborLawState) -> LaborLawState:
    form_data = state.get("form_data", {})
    form_text = "\n".join([f"- {k}: {v}" for k, v in form_data.items()])
    prompt = f"""请根据以下案件信息梳理关键事实：\n【用户表单提交信息】：\n{form_text}
    请你必须先在 <thinking> 标签内进行沙盘推演和逻辑自洽检查。
    思考完成后，再在 <output> 标签内梳理：1. 争议焦点 2. 关键时间节点 3. 证据情况分析 4. 法律适用预判"""
    messages = [SystemMessage(content="你是劳动法律师助理"), HumanMessage(content=prompt)]
    return {"legal_facts_summary": llm.invoke(messages).content}

def legal_researcher_node(state: LaborLawState) -> LaborLawState:
    summary = state.get("legal_facts_summary", "")
    relevant_docs = "\n\n".join([doc.page_content for doc in retriever.invoke(summary)]) if summary else "暂无法条"
    prompt = f"""案件事实：\n{summary}\n相关法条：\n{relevant_docs}
    请在 <thinking> 标签内思考：法条是否覆盖诉求？有没有冲突？
    思考后在 <output> 标签输出：1. 适用条款 2. 适用说明 3. 赔偿计算依据 4. 程序性建议"""
    messages = [SystemMessage(content="你是专业律师"), HumanMessage(content=prompt)]
    return {"relevant_laws": llm.invoke(messages).content}

def compliance_reviewer_node(state: LaborLawState) -> LaborLawState:
    facts = state.get("legal_facts_summary", "")
    laws = state.get("relevant_laws", "")
    form_text = "\n".join([f"- {k}: {v}" for k, v in state.get("form_data", {}).items()])
    prompt = f"""原始信息：\n{form_text}\n事实：\n{facts}\n法条：\n{laws}
    请在 <thinking> 标签内审视前置分析有无漏洞。
    思考后在 <output> 标签提供：1. 最终法律建议 2. 操作步骤 3. 风险提示 4. 沟通策略 5. 证据建议"""
    messages = [SystemMessage(content="你是资深专家"), HumanMessage(content=prompt)]
    return {"final_review": llm.invoke(messages).content}

# ==========================================
# 6. 构建 LangGraph 工作流
# ==========================================
print(">>> 正在构建最终完美闭环的 LangGraph 工作流...")
workflow = StateGraph(LaborLawState)

workflow.add_node("summarizer", summarize_conversation_node)
workflow.add_node("triage", triage_node)  
workflow.add_node("fact_summarizer", fact_summarizer_node)
workflow.add_node("legal_researcher", legal_researcher_node)
workflow.add_node("compliance_reviewer", compliance_reviewer_node)

workflow.set_entry_point("summarizer")
workflow.add_edge("summarizer", "triage")

def route_after_triage(state: LaborLawState):
    return "process_case" if state.get("triage_result", {}).get("action") == "form" else "end"

workflow.add_conditional_edges("triage", route_after_triage, {"process_case": "fact_summarizer", "end": END})
workflow.add_edge("fact_summarizer", "legal_researcher")
workflow.add_edge("legal_researcher", "compliance_reviewer")
workflow.add_edge("compliance_reviewer", END)

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