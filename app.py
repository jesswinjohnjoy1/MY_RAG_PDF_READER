
import os
import streamlit as st
from utils import load_pdf_bytes, simple_text_split, embed_texts, build_faiss_index, retrieve_top_k
import json
from groq import Groq

# Full-width config
st.set_page_config(page_title="RAG PDF Reader", layout="wide")

st.markdown("<h1 style='text-align:center; color: #4B8BBE;'>📚 RAG PDF Reader with LLaMA 3</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center;'>Upload PDFs → Build Index → Chat with Document</h4>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar for settings
st.sidebar.header("⚙️ Indexing Settings")
chunk_size = st.sidebar.number_input("Chunk size (chars)", min_value=200, max_value=3000, value=1000, step=100)
overlap = st.sidebar.number_input("Chunk overlap (chars)", min_value=0, max_value=1000, value=200, step=50)

uploaded_files = st.sidebar.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

# Sidebar Utilities
st.sidebar.header("📂 Utilities")
if st.sidebar.button("Clear Index"):
    st.session_state.clear()
    st.success("Index and history cleared.")

if "docs" not in st.session_state:
    st.session_state.docs = []
if "index_built" not in st.session_state:
    st.session_state.index_built = False
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None
if "history" not in st.session_state:
    st.session_state.history = []

# Build Index
if uploaded_files and st.sidebar.button("Process & Build Index"):
    all_texts = []
    file_sources = []
    for f in uploaded_files:
        file_bytes = f.read()
        text = load_pdf_bytes(file_bytes)
        chunks = simple_text_split(text, chunk_size=chunk_size, overlap=overlap)
        for i, c in enumerate(chunks):
            all_texts.append(c)
            file_sources.append({"file_name": f.name, "chunk_id": i})

    if not all_texts:
        st.error("❌ No textual content found in PDFs.")
    else:
        with st.spinner("🛠 Building embeddings and index..."):
            embeddings = embed_texts(all_texts)
            index, dim = build_faiss_index(embeddings)
            st.session_state.docs = all_texts
            st.session_state.file_sources = file_sources
            st.session_state.faiss_index = index
            st.session_state.index_built = True
        st.success(f"✅ Index built with {len(all_texts)} chunks (Embedding dim = {dim})")

st.markdown("---")

if st.session_state.index_built:
    col1, col2 = st.columns([2, 3])

    with col1:
        st.header("💬 Chat with Document")
        query = st.text_input("Enter your question here:")

        if st.button("Ask") and query.strip():
            top_k = 5
            results = retrieve_top_k(query, st.session_state.docs, st.session_state.faiss_index, top_k=top_k)

            context = "\n\n".join([r["chunk"] for r in results])
            groq_key = os.environ.get("GROQ_API_KEY", "YOUR_GROQ_KEY_HERE")
            if not groq_key:
               groq_key=st.secrets["GROQ_API_KEY"]

            if groq_key:
                client = Groq(api_key=groq_key)
                messages = [{"role": "system", "content": "You are a helpful assistant that answers based only on provided PDF context."}]
                for h in st.session_state.history:
                    messages.append({"role": "user", "content": h["question"]})
                    messages.append({"role": "assistant", "content": h["answer"]})
                messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{query}\n\nAnswer concisely:"})

                with st.spinner("🤖 Generating answer with LLaMA 3..."):
                    try:
                        resp = client.chat.completions.create(
                            model="llama-3.1-8b-instant",
                            messages=messages,
                            max_tokens=400,
                            temperature=0.2,
                        )
                        answer = resp.choices[0].message.content.strip()
                        st.subheader("📝 Final Answer")
                        st.success(answer)
                        st.session_state.history.append({"question": query, "answer": answer})
                    except Exception as e:
                        st.error(f"Groq request failed: {e}")

    with col2:
        st.header("📄 Retrieved Chunks")
        if st.session_state.history:
            for i, h in enumerate(st.session_state.history[::-1], 1):
                st.markdown(f"**Q{i}:** {h['question']}")
                st.markdown(f"**A{i}:** {h['answer']}")
                st.markdown("---")

        st.subheader("Export Chat History")
        chat_history_json = json.dumps(st.session_state.history, ensure_ascii=False, indent=2)
        st.download_button("Download Chat History (JSON)", data=chat_history_json, file_name="chat_history.json", mime="application/json")

else:
    st.info("📌 Upload PDFs and click 'Process & Build Index' to start chatting.")
