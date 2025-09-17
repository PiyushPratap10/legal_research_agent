import streamlit as st
import asyncio
from agent import legal_agent

st.set_page_config(
    page_title="⚖️ Legal Research AI",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
        body {
            background: linear-gradient(120deg, #0f2027, #203a43, #2c5364);
            color: white;
        }
        .stTextInput > div > div > input {
            border-radius: 12px;
            padding: 12px;
            font-size: 16px;
        }
        .stButton > button {
            border-radius: 12px;
            background: #1f4068;
            color: white;
            font-weight: bold;
            transition: 0.3s;
        }
        .stButton > button:hover {
            background: #e43f5a;
            transform: scale(1.05);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title
st.title("⚖️ Futuristic Legal Research Agent")
st.subheader("Ask complex legal queries and get multi-hop refined answers.")

# Input
query = st.text_area("💬 Enter your legal query:", placeholder="e.g. Recent amendments to laws regarding rape in India")

if st.button("🔍 Search"):
    if query.strip():
        with st.spinner("⚡ Processing your query with multi-hop RAG..."):
            res = asyncio.run(legal_agent(query))
            st.markdown("### 📑 Refined Legal Answer")
            st.write(res)
    else:
        st.warning("Please enter a query.")

# Sidebar
st.sidebar.title("⚙️ Settings")
st.sidebar.info(
    """
    This AI agent uses:
    - LlamaIndex FunctionAgent
    - Gemini 2.5 Flash (Google GenAI)
    - Chroma + Keyword & Summary Tools  
    """
)
st.sidebar.markdown("🚀 Designed for **dynamic legal research**")