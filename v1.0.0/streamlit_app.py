import streamlit as st
from query_engine import get_search_engine,get_summarize_engine

st.set_page_config(page_title="Legal RAG Assistant", layout="wide")

st.title("⚖️ Legal Research Assistant")
st.markdown("Ask questions about **Indian Constitution, Criminal Law, or Civil Law**.")

# Sidebar options
engine_choice = st.sidebar.radio(
    "Select Query Engine",
    ["Summarize Engine", "Search Engine"],
    help="Choose whether you want summarized responses or direct keyword search."
)

# Input box
query = st.text_area("Enter your legal question:", placeholder="e.g. Explain Article 25 in Indian Constitution")

if st.button("🔍 Get Answer"):
    if not query.strip():
        st.warning("⚠️ Please enter a question.")
    else:
        with st.spinner("Processing your query..."):
            if engine_choice == "Summarize Engine":
                engine = get_summarize_engine()
            else:
                engine = get_search_engine()

            res = engine.query(query)

        # Display response
        st.subheader("📌 Answer")
        st.write(res.response)
