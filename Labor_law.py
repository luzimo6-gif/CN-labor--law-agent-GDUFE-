import os
from typing import TypedDict
from dotenv import load_dotenv

# --- LangGraph 核心组件 ---
from langgraph.graph import StateGraph, START, END

# --- LangChain 核心与模型组件 ---
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# --- RAG 相关组件 (使用 PDF) ---
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# ==========================================
# 1. 基础配置与大模型初始化
# ==========================================
_ = load_dotenv()
api_key = os.getenv("DASHSCOPE_API_KEY") or ""
base_url = "https://dashscope.console.aliyun.com/compatible-mode/v1"

# 实例化 Qwen 大语言模型 (用于文字分析)
llm = ChatOpenAI(
    api_key=api_key,
    base_url=base_url,
    model_name="qwen-plus", 
    temperature=0.1 
)

# 实例化 Embeddings 模型 (用于将文本转为向量)
embeddings = OpenAIEmbeddings(
    model="text-embedding-v2",
    api_key=api_key,
    base_url=base_url
)

# ==========================================
# 2. 初始化 RAG 知识库 (带本地持久化缓存)
# ==========================================
persist_dir = "./chroma_db_labor_law" # 定义向量库本地保存的文件夹

if os.path.exists(persist_dir):
    print(">>> 发现已存在的本地向量库，正在直接加载 (秒开)...")
    vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
else:
    print(">>> 未发现本地库，正在扫描并读取 PDF 法律文档...")
    # 注意：请确保当前目录下有 data 文件夹，并且里面放了 pdf 格式的法条文档
    loader = DirectoryLoader('./data/', glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    print(f"[成功] 成功加载了 {len(documents)} 份文档，正在切分...")
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    splits = text_splitter.split_documents(documents)
    
    print(">>> 正在调用大模型构建并保存向量数据库 (仅需执行一次)...")
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=embeddings, 
        persist_directory=persist_dir # 写入本地磁盘
    )

retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) 
print("[成功] 法律知识库准备就绪！\n")

# ==========================================
# 3. 定义全局状态 (State) - 智能体们的“共享案卷”
# ==========================================
class LaborLawForm(TypedDict):
    company_name: str
    has_contract: str
    hire_date: str
    fire_date: str
    monthly_salary: float
    dispute_type: str
    user_demand: str
    extra_details: str

class LaborLawState(TypedDict):
    form_data: LaborLawForm              # 前端传来的表单原始数据
    legal_facts_summary: str             # 节点1生成的：标准案情摘要
    relevant_laws: str                   # 节点2生成的：检索到的法条与分析
    final_review: str                    # 节点3生成的：最终审查意见

# ==========================================
# 4. 编写智能体节点 (Nodes)
# ==========================================

# 【智能体 1：事实梳理专员】
def form_processor_node(state: LaborLawState):
    form = state["form_data"]
    
    # 纯代码逻辑处理，将表单拼成规范文本
    summary = f"""【劳动争议案件基本事实】
1. 用人单位：{form.get('company_name', '未提供')}
2. 劳动合同：{form.get('has_contract', '未知')}
3. 入职时间：{form.get('hire_date', '未知')}
4. 争议时间：{form.get('fire_date', '未知')}
5. 平均月薪：{form.get('monthly_salary', '未知')} 元
6. 争议类型：{form.get('dispute_type', '未知')}
7. 核心诉求：{form.get('user_demand', '未知')}
8. 补充说明：{form.get('extra_details', '无')}
"""
    print("--- [节点1] 事实梳理完毕，已生成标准案卷 ---")
    return {"legal_facts_summary": summary}


# 【智能体 2：法条检索与分析专员】
def legal_researcher_node(state: LaborLawState):
    summary = state["legal_facts_summary"]
    print(">>> [节点2] 正在从本地知识库检索相关法条...")
    
    # 步骤 A：利用 RAG 检索器，根据案情找法条
    docs = retriever.invoke(summary)
    retrieved_context = "\n\n".join(doc.page_content for doc in docs)
    
    # 步骤 B：构造提示词，强制大模型进行“闭卷考试”
    system_prompt = """你是一位专业的劳动法案件研究员。
你的任务是根据给定的【案件事实】和【参考法律资料】，出具一份法条适用分析。
请严格遵守以下要求：
1. 只能使用【参考法律资料】中提供的法条，绝对不能自己捏造法律条文。
2. 列出适用的具体法律名称及条款序号。
3. 结合案件事实，给出初步的法律分析意见。"""

    user_prompt = f"【案件事实】:\n{summary}\n\n【参考法律资料】:\n{retrieved_context}\n\n请进行法条适用分析："
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    # 步骤 C：调用大模型
    response = llm.invoke(messages)
    print("--- [节点2] 法条检索与分析完成 ---")
    
    return {"relevant_laws": response.content}


# 【智能体 3：合规与风险审查高级专家】
def reviewer_node(state: LaborLawState):
    summary = state["legal_facts_summary"]
    analysis = state["relevant_laws"]
    print(">>> [节点3] 资深合规专家正在进行最终审查与出具实操建议...")
    
    # 构造提示词，让大模型扮演高级合伙人
    system_prompt = """你是一位资深的劳动法合规专家（律所高级合伙人）。
你的任务是审查“助理律师”出具的法条适用分析，并结合案件事实，给出最终的法律建议。
请严格按照以下结构输出报告：
1. 【分析审查】：助理的法条适用是否准确？有无遗漏？
2. 【维权策略】：针对当事人的核心诉求，第一步该做什么，第二步该做什么？（如：发送被迫解除通知书、收集打卡记录等）
3. 【风险提示】：当事人在实操中可能面临哪些不利因素或败诉风险？"""

    user_prompt = f"【案件事实】:\n{summary}\n\n【助理律师的法条分析】:\n{analysis}\n\n请出具最终的审查意见与实操建议："
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    # 调用大模型
    response = llm.invoke(messages)
    print("--- [节点3] 最终审查与建议生成完成 ---")
    
    return {"final_review": response.content}

# ==========================================
# 5. 构建与编译多智能体工作流 (LangGraph)
# ==========================================
workflow = StateGraph(LaborLawState)

# 注册我们的三个智能体
workflow.add_node("form_processor", form_processor_node)
workflow.add_node("legal_researcher", legal_researcher_node)
workflow.add_node("reviewer", reviewer_node)

# 绘制流程图连线 (流水线模式)
workflow.add_edge(START, "form_processor")              # 1. 启动 -> 事实梳理
workflow.add_edge("form_processor", "legal_researcher") # 2. 事实梳理 -> 法条检索
workflow.add_edge("legal_researcher", "reviewer")       # 3. 法条检索 -> 专家审查
workflow.add_edge("reviewer", END)                      # 4. 专家审查 -> 结束

# 编译成可执行的应用
app = workflow.compile()

# ==========================================
# 6. 本地模拟测试运行
# ==========================================
if __name__ == "__main__":
    # 假设这是前端用户填写的表格
    mock_form_data = {
        "company_name": "某信息技术有限公司",
        "has_contract": "签了，但是公司没给我一份",
        "hire_date": "2021年5月",
        "fire_date": "2024年3月15日",
        "monthly_salary": 12000.0,
        "dispute_type": "违法解除劳动合同",
        "user_demand": "要求支付2N赔偿金",
        "extra_details": "HR口头通知我被优化了，没有任何书面辞退通知。"
    }
    
    print("\n================ 开始立案处理 ================\n")
    
    # 初始化状态 (精简版)
    initial_state = {
        "form_data": mock_form_data,
        "legal_facts_summary": "",
        "relevant_laws": "",
        "final_review": ""
    }
    
    # 启动工作流
    result = app.invoke(initial_state)
    
    # 打印最终结果
    print("\n================ 最终输出结果 ================\n")
    print("【第一步：事实梳理摘要】\n")
    print(result["legal_facts_summary"])
    print("-" * 50)
    print("【第二步：法条检索与初步分析】\n")
    print(result["relevant_laws"])
    print("-" * 50)
    print("【第三步：专家审查与实操建议】\n")
    print(result["final_review"])