import streamlit as st
import requests, os, tempfile, uuid
from PyPDF2 import PdfReader
from bs4 import BeautifulSoup
from newspaper import Article
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from datetime import datetime

# === CONFIG ===
GROQ_API_KEY = "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"
TAVILY_API_KEY = "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"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "moonshotai/kimi-k2-instruct"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

embedder = SentenceTransformer(EMBED_MODEL_NAME, device='cpu')
faiss_index = faiss.IndexFlatL2(384)
document_chunks = []

# === GROQ CALL ===
def call_groq_llm(prompt, history=None):
    messages = [{"role": "system", "content": "You are a market analyst providing actionable insights for business strategy."}]
    if history:
        messages += history
    messages.append({"role": "user", "content": prompt})

    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    body = {
        "model": GROQ_MODEL,
        "messages": messages,
        "temperature": 0.6
    }

    response = requests.post(GROQ_API_URL, json=body, headers=headers)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

# === TAVILY FETCH ===
def fetch_trend_data_tavily(query: str, max_results: int = 3) -> str:
    if not TAVILY_API_KEY:
        return "❌ Tavily API Key not configured."

    url = "https://api.tavily.com/search"
    headers = {"Content-Type": "application/json"}
    payload = {
        "api_key": TAVILY_API_KEY,
        "query": f"latest trends {query}",
        "search_depth": "advanced",
        "max_results": max_results,
        "include_answer": False,
        "include_raw_content": True
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        results = data.get("results", [])

        content = ""
        for res in results:
            title = res.get("title", "No Title")
            body = res.get("content", "")[:2000]
            source = res.get("url", "")
            content += f"\n### {title}\n{body}\n🔗 [Source]({source})\n"

        return content.strip() or "No relevant Tavily results found."

    except Exception as e:
        return f"❌ Error calling Tavily: {e}"

# === PDF + FAISS ===
def process_pdf(file):
    reader = PdfReader(file)
    return "\n".join(page.extract_text() or "" for page in reader.pages)

def chunk_text(text, max_tokens=200):
    sentences = text.split(". ")
    chunks, chunk = [], ""
    for sent in sentences:
        if len(chunk) + len(sent) < max_tokens:
            chunk += sent + ". "
        else:
            chunks.append(chunk.strip())
            chunk = sent + ". "
    if chunk:
        chunks.append(chunk.strip())
    return chunks

def add_to_faiss(chunks):
    global document_chunks
    embeddings = embedder.encode(chunks)
    faiss_index.add(np.array(embeddings).astype("float32"))
    document_chunks.extend(chunks)

def search_faiss(query, top_k=3):
    embedding = embedder.encode([query])
    D, I = faiss_index.search(np.array(embedding).astype("float32"), top_k)
    return [document_chunks[i] for i in I[0] if i < len(document_chunks)]

# === STREAMLIT UI ===
st.set_page_config("Trend Analyzer: Tavily + Groq", layout="centered")
st.title("📈 AI Trend Analyzer & competitive Insights Generator ")
st.markdown("Get strategic insights from internal PDFs + Tavily web results via **Groq GenAI**.")

domain = st.selectbox("🌐 Select Business Domain", ["Furniture", "Retail", "Healthcare", "Energy", "Consumer Electronics", "Fashion", "Other"])

uploaded_files = st.file_uploader("📎 Upload Internal Reports or Catalog PDFs", type="pdf", accept_multiple_files=True)
if uploaded_files:
    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(file.read())
            pdf_text = process_pdf(tmp_file.name)
            chunks = chunk_text(pdf_text)
            add_to_faiss(chunks)
    st.success("✅ Uploaded PDF(s) indexed.")

query = st.text_input("🔍 Enter a trend to explore", placeholder="e.g. smart ceiling fans, solar appliances")

if st.button("🔎 Analyze Now"):
    if not GROQ_API_KEY:
        st.error("❌ GROQ API Key not set.")
    elif not query.strip():
        st.warning("⚠️ Please enter a valid trend.")
    else:
        with st.spinner("🔬 Fetching Tavily data and analyzing..."):
            try:
                sanitized = query.replace("client", "[REDACTED]").strip()
                web_text = fetch_trend_data_tavily(sanitized)
                rag_context = "\n".join(search_faiss(sanitized)) if len(document_chunks) else ""

                # === Show raw Tavily data
                st.markdown("### 🌐 Tavily Web Data")
                st.text_area("Raw Web Data", value=web_text, height=300)

                if rag_context:
                    st.markdown("### 📎 Internal PDF Insights (RAG Context)")
                    st.text_area("Internal Data", value=rag_context, height=200)

                # === LLM Prompt
                prompt = (
                    f"Analyze the trend: '{sanitized}' in the domain: '{domain}'.\n"
                    f"Use Internal Catalog:\n{rag_context}\n\n"
                    f"Use Web Data:\n{web_text}\n\n"
                    f"Generate:\n"
                    f"- Key market trends\n"
                    f"- Competitive landscape\n"
                    f"- Estimated impact score (0-10)\n"
                    f"- Strategic business recommendations"
                )

                result = call_groq_llm(prompt)
                st.markdown("### 🤖 Groq GenAI Summary")
                st.markdown(result)

                session_id = uuid.uuid4().hex
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
                st.session_state["last_summary"] = result
                st.session_state["chat_history"] = [{"role": "user", "content": query}, {"role": "assistant", "content": result}]

                st.download_button(
                    label="📥 Download Summary",
                    data=result,
                    file_name=f"trend_summary_{session_id}_{timestamp}.txt",
                    mime="text/plain"
                )

            except Exception as e:
                st.error(f"❌ Error: {e}")

# === Optional Follow-Up Q&A ===
if "last_summary" in st.session_state:
    followup = st.text_input("💬 Ask a follow-up question (optional)", placeholder="e.g. What are the pricing implications?")
    if st.button("💡 Get Follow-up Insight"):
        if followup:
            st.session_state["chat_history"].append({"role": "user", "content": followup})
            with st.spinner("🧠 Thinking..."):
                followup_result = call_groq_llm(followup, history=st.session_state["chat_history"])
                st.session_state["chat_history"].append({"role": "assistant", "content": followup_result})
                st.markdown(followup_result)
        else:
            st.warning("⚠️ Please enter a follow-up question.")
