import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="IPL Analytics Assistant", layout="wide")
st.title("🏏 IPL Analytics Assistant")

query = st.text_input("Ask a question:")

col1, col2 = st.columns(2)

if col1.button("Ask"):
    if query.strip():
        with st.spinner("Thinking..."):
            response = requests.post(
                f"{API_URL}/chat",
                json={"query": query},
                timeout=120
            )
            res = response.json()

        st.subheader("Answer")
        st.write(res.get("answer", "No answer returned."))

        st.markdown("---")
        st.write(f"**Route:** {res.get('route')} | **Intent:** {res.get('intent')}")
        st.write(
            f"**Structured:** {res.get('structured_count', 0)} | "
            f"**Insights:** {res.get('insight_count', 0)}"
        )

if col2.button("Debug"):
    if query.strip():
        with st.spinner("Retrieving..."):
            response = requests.post(
                f"{API_URL}/debug/retrieve",
                json={"query": query},
                timeout=120
            )
            res = response.json()

        st.subheader("Debug Info")
        st.write("Route:", res.get("route"))
        st.write("Intent:", res.get("intent"))
        st.write("Entities:", res.get("entities"))
        st.write("Structured:", res.get("structured_count"))
        st.write("Insights:", res.get("insight_count"))

        st.markdown("### Context")
        st.write(res.get("context"))