import os
import re
import transformers
import streamlit as st
transformers.logging.set_verbosity_error()
transformers.logging.disable_progress_bar()
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from collections import defaultdict

# ============================================================
# 文本清洗
# ============================================================
def clean_text(text):
    text = text.replace("$", "")
    text = text.replace("\\(", "")
    text = text.replace("\\)", "")
    text = text.replace("\\mathrm", "")
    text = text.replace("\\mathsf", "")
    text = text.replace("\\mathbf", "")
    text = text.replace("\\mathit", "")
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def force_markdown_table(text):
    if "|" not in text:
        return text
    parts = text.split("|")
    cleaned = [p.strip() for p in parts if p.strip()]
    if len(cleaned) < 4:
        return text
    header = f"| {cleaned[0]} | {cleaned[1]} |"
    separator = "|---|---|"
    rows = []
    for i in range(2, len(cleaned), 2):
        if i + 1 < len(cleaned):
            rows.append(f"| {cleaned[i]} | {cleaned[i+1]} |")
    table = "\n".join([header, separator] + rows)
    return table

def highlight_keywords(text, keywords):
    if not keywords:
        return text
    pattern = "|".join(map(re.escape, keywords))
    return re.sub(
        pattern,
        lambda m: f"<span style='background-color: yellow; font-weight: bold;'>{m.group(0)}</span>",
        text,
        flags=re.IGNORECASE
    )

# ============================================================
# 路径
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMB_DIR = os.path.join(BASE_DIR, "embeddings")

# ============================================================
# Streamlit UI
# ============================================================
st.set_page_config(page_title="本地 RAG Chat（Ollama + LangChain）", layout="wide")
st.title("💬 本地 RAG Chat（Ollama + LangChain）")

if "messages" not in st.session_state:
    st.session_state.messages = []

# ============================================================
# 加载所有向量库（自动遍历子目录）
# ============================================================
@st.cache_resource
def load_all_vectorstores():
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-zh")
    vectorstores = {}

    for folder in os.listdir(EMB_DIR):
        path = os.path.join(EMB_DIR, folder)
        if os.path.isdir(path):
            try:
                vs = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
                vectorstores[folder] = vs
                print(f"📚 已加载向量库：{folder}")
            except Exception as e:
                print(f"❌ 加载失败：{folder} - {e}")

    return vectorstores

db_all = load_all_vectorstores()

# ============================================================
# 加载本地模型（Ollama）
# ============================================================
llm = ChatOllama(
    model="qwen2.5",
    temperature=0.2
)

# ============================================================
# 渲染历史消息
# ============================================================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)

    if msg["role"] == "assistant" and msg.get("sources"):
        question = msg.get("question", "")
        keywords = question.split()

        with st.expander("📚 引用来源（对应本条回答）"):
            for i, src in enumerate(msg["sources"]):
                cleaned = clean_text(src[:500])
                highlighted = highlight_keywords(cleaned, keywords)
                st.markdown(f"**[{i+1}]**<br>{highlighted}", unsafe_allow_html=True)

# ============================================================
# 输入框
# ============================================================
user_input = st.chat_input("请输入你的问题...")

if user_input:
    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
        "question": user_input,
        "sources": []
    })

    with st.chat_message("user"):
        st.write(user_input)

    # ============================================================
    # 多向量库 MMR 检索（公平检索版）
    # ============================================================
    with st.spinner("正在检索知识库..."):
        raw_docs = []

        for name, db in db_all.items():
            try:
                r = db.max_marginal_relevance_search(user_input, k=3, fetch_k=20)
                print(f"🔍 来自向量库：{name}，命中 {len(r)} 条")
                for d in r:
                    d.metadata["source_db"] = name
                raw_docs.extend(r)
            except Exception as e:
                print(f"❌ 检索失败：{name} - {e}")

        # 按库分组
        grouped = defaultdict(list)
        for d in raw_docs:
            grouped[d.metadata["source_db"]].append(d)

        # 每个库至少保留 1 条
        final_docs = []
        for name, items in grouped.items():
            final_docs.append(items[0])

        # 如果不足 3 条，从剩余结果中补齐
        if len(final_docs) < 3:
            others = [d for d in raw_docs if d not in final_docs]
            final_docs.extend(others[:3 - len(final_docs)])

        docs = final_docs
        sources_raw = [d.page_content for d in docs]

    # ============================================================
    # Prompt
    # ============================================================
    formatted_context = ""
    for i, d in enumerate(docs):
        formatted_context += f"[{i + 1}] {d.page_content}\n\n"

    prompt = f"""
    你是一个严格的 RAG 检索助手。

    【回答规则】：
    1. 回答必须完全基于知识库内容。
    2. 必须输出干净的中文，不允许出现 LaTeX、HTML、公式符号、特殊字符。
    3. 禁止输出 $、\\(、\\mathrm、<table> 等格式符号。
    4. 不允许扩写、不允许推测、不允许加入额外解释。
    5. 如果知识库中没有答案，回答：“知识库未包含相关内容。”

    【引用规则】：
    你必须在回答最后输出：
    【引用】1,3

    【知识库内容（带编号）】：
    {formatted_context}

    【用户问题】：
    {user_input}

    请输出：
    1. 回答
    2. 【引用】chunk编号列表
    """

    # ============================================================
    # 调用模型
    # ============================================================
    with st.spinner("正在生成回答..."):
        answer_obj = llm.invoke(prompt)
        answer = clean_text(answer_obj.content)
        answer = force_markdown_table(answer)

    # 保存消息
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "question": user_input,
        "sources": sources_raw
    })

    # 显示回答
    with st.chat_message("assistant"):
        st.markdown(answer, unsafe_allow_html=True)

    # 显示引用来源
    keywords = user_input.split()
    with st.expander("📚 引用来源（对应本条回答）"):
        for i, src in enumerate(sources_raw):
            cleaned = clean_text(src[:500])
            highlighted = highlight_keywords(cleaned, keywords)
            st.markdown(f"**[{i+1}]**<br>{highlighted}", unsafe_allow_html=True)
