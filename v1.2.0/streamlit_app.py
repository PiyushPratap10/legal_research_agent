import streamlit as st
import asyncio
from agent import legal_agent
st.set_page_config(
    page_title="⚖️ Futuristic Legal Research Agent",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    body {background-color: #0e1117; color: #f0f0f0;}
    .stChatMessage {border-radius: 12px; padding: 12px;}
    .user {background-color: #1e293b; color: #f1f5f9;}
    .assistant {background-color: #334155; color: #e2e8f0;}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("⚖️ Futuristic Legal Research Agent")
st.caption("🔍 Research Indian Constitution, Criminal Law, Civil Law, or search the web for legal updates.")

# Session state for chat
if "history" not in st.session_state:
    st.session_state["history"] = []

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    st.markdown("Choose your query domain:")
    st.info("Agent will auto-select the right tool based on your query.")

# Chat input
user_query = st.chat_input("Ask your legal research question...")

if user_query:
    st.session_state.history.append({"role": "user", "content": user_query})

    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("⚡ Analyzing your query..."):
            response = asyncio.run(legal_agent(user_query))
            st.markdown(response)
            st.session_state.history.append({"role": "assistant", "content": response})

# Display chat history
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])