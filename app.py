import streamlit as st
import requests
import os
import tempfile
import uuid
from datetime import datetime
import os
from dotenv import load_dotenv
from watsonxclient import WatsonXClient as wxc

# Initialize WatsonXClient
wxc_client = wxc()

# Enhanced version without ML dependencies for immediate functionality
try:
    from PyPDF2 import PdfReader
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

try:
    from newspaper import Article
    NEWSPAPER_AVAILABLE = True
except ImportError:
    NEWSPAPER_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# === CONFIG ===
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GROQ_API_URL = os.getenv("GROQ_API_URL")
GROQ_MODEL = os.getenv("GROQ_MODEL")

# Initialize session state for document storage
if "document_chunks" not in st.session_state:
    st.session_state.document_chunks = []

# Load custom CSS
def load_css():
    try:
        with open("styles.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass  # Gracefully handle missing CSS file

# === GROQ CALL ===
def call_groq_llm(prompt, history=None):
    messages = [{"role": "system", "content": "You are a senior management consultant providing strategic insights for business leaders. Structure your responses with clear headings, bullet points, and actionable recommendations."}]
    if history:
        messages += history
    messages.append({"role": "user", "content": prompt})

    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    body = {
        "model": GROQ_MODEL,
        "messages": messages,
        "temperature": 0.6
    }

    try:
        response = requests.post(GROQ_API_URL, json=body, headers=headers)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error calling AI service: {e}"
    
def call_watsonx_llm(wxc_client, prompt, history=None):
    """
    Calls WatsonXClient's generate_completion method and returns the result.
    """
    system_prompt = "You are a market analyst providing actionable insights for business strategy."
    if history:
        system_prompt += "\n" + "\n".join(
            f"{msg['role'].capitalize()}: {msg['content']}" for msg in history
        )
    result = wxc_client.generate_completion(
        prompt=prompt,
        system_prompt=system_prompt,
        max_tokens=500,
        temperature=0.2
    )
    return result  # <-- Just return the string result

# === TAVILY FETCH ===
def fetch_trend_data_tavily(query: str, max_results: int = 3) -> dict:
    if not TAVILY_API_KEY:
        return {"error": "Tavily API Key not configured", "results": [], "content": ""}

    url = "https://api.tavily.com/search"
    headers = {"Content-Type": "application/json"}
    payload = {
        "api_key": TAVILY_API_KEY,
        "query": f"latest trends market analysis {query}",
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

        processed_results = []
        content = ""
        for res in results:
            title = res.get("title", "No Title")
            body = res.get("content", "")[:2000]
            source = res.get("url", "")
            processed_results.append({
                "title": title,
                "content": body,
                "url": source
            })
            content += f"\n### {title}\n{body}\n🔗 [Source]({source})\n"

        return {
            "content": content.strip() or "No relevant results found.",
            "results": processed_results,
            "error": None
        }

    except Exception as e:
        return {"error": f"Error calling Tavily: {e}", "results": [], "content": ""}

# === PDF PROCESSING ===
def process_pdf(file):
    if not PDF_AVAILABLE:
        return "PDF processing unavailable - PyPDF2 not installed"
    
    try:
        reader = PdfReader(file)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as e:
        return f"Error processing PDF: {e}"

def chunk_text(text, max_tokens=500):
    # Simple text chunking without ML dependencies
    sentences = text.split(". ")
    chunks, chunk = [], ""
    for sent in sentences:
        if len(chunk) + len(sent) < max_tokens:
            chunk += sent + ". "
        else:
            if chunk.strip():
                chunks.append(chunk.strip())
            chunk = sent + ". "
    if chunk.strip():
        chunks.append(chunk.strip())
    return chunks

def add_to_knowledge_base(chunks):
    # Simple storage without vector search
    st.session_state.document_chunks.extend(chunks)

def search_knowledge_base(query):
    # Simple keyword-based search
    query_words = query.lower().split()
    relevant_chunks = []
    
    for chunk in st.session_state.document_chunks:
        chunk_lower = chunk.lower()
        score = sum(1 for word in query_words if word in chunk_lower)
        if score > 0:
            relevant_chunks.append((chunk, score))
    
    # Sort by relevance and return top 3
    relevant_chunks.sort(key=lambda x: x[1], reverse=True)
    return [chunk for chunk, score in relevant_chunks[:3]]

def create_impact_chart(impact_score):
    if not PLOTLY_AVAILABLE:
        return None
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = impact_score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Market Impact Score"},
        gauge = {
            'axis': {'range': [None, 10]},
            'bar': {'color': "#1E3A8A"},
            'steps': [
                {'range': [0, 4], 'color': "#FEF3C7"},
                {'range': [4, 7], 'color': "#FDE68A"},
                {'range': [7, 10], 'color': "#059669"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 9
            }
        }
    ))
    fig.update_layout(height=300, font={'color': "#1E293B", 'family': "Inter"})
    return fig

def parse_impact_score(response_text):
    # Extract impact score from response
    lines = response_text.split('\n')
    impact_score = 7.5  # Default
    
    for line in lines:
        if 'impact score' in line.lower() or 'score' in line.lower():
            import re
            numbers = re.findall(r'\d+\.?\d*', line)
            if numbers:
                try:
                    impact_score = min(float(numbers[0]), 10)
                    break
                except:
                    pass
    
    return impact_score

# === STREAMLIT UI ===
st.set_page_config(
    page_title="AI Trend Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
load_css()

# Header
st.markdown("""
<div class="header">
    <h1>📈 AI Trend Analyzer & Competitive Insights</h1>
    <p>Strategic market intelligence powered by advanced AI analytics</p>
</div>
""", unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.markdown("### 🎯 Analysis Configuration")
    
    domain = st.selectbox(
        "Business Domain", 
        ["Consumer Electronics", "Furniture", "Retail", "Healthcare", "Energy", "Fashion", "Automotive", "Fintech", "Other"],
        help="Select your primary business domain for contextual analysis"
    )
    
    st.markdown("### 📄 Internal Documents")
    
    if PDF_AVAILABLE:
        uploaded_files = st.file_uploader(
            "Upload Reports/Catalogs", 
            type="pdf", 
            accept_multiple_files=True,
            help="Upload internal reports, product catalogs, or market research documents"
        )
        
        if uploaded_files:
            with st.spinner("Processing documents..."):
                for file in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                        tmp_file.write(file.read())
                        pdf_text = process_pdf(tmp_file.name)
                        chunks = chunk_text(pdf_text)
                        add_to_knowledge_base(chunks)
            st.success(f"✅ {len(uploaded_files)} document(s) processed and indexed")
            st.info(f"📊 Total chunks in knowledge base: {len(st.session_state.document_chunks)}")
    else:
        st.info("📋 PDF upload will be available after installing PyPDF2")
    
    # Manual text input as alternative
    st.markdown("### ✍️ Manual Knowledge Input")
    manual_text = st.text_area(
        "Add internal knowledge manually",
        placeholder="Paste company reports, product information, or market research here...",
        height=100,
        help="Add text-based knowledge to enhance analysis"
    )
    
    if manual_text and st.button("➕ Add to Knowledge Base"):
        chunks = chunk_text(manual_text)
        add_to_knowledge_base(chunks)
        st.success(f"✅ Added {len(chunks)} text chunks to knowledge base")
        st.rerun()

# Main Content Area
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 🔍 Trend Analysis Query")
    query = st.text_input(
        "Enter trend or market topic to analyze",
        placeholder="e.g., smart home devices, sustainable packaging, AI in healthcare",
        help="Describe the market trend, technology, or business area you want to analyze"
    )

with col2:
    st.markdown("### 🚀 Actions")
    analyze_button = st.button("🔬 Analyze Market Trend", type="primary", use_container_width=True)

if analyze_button:
    if not wxc_client.API_KEY:
        st.error("❌ WatsonX API Key not configured. Please check your environment variables.")
    # elif not GROQ_API_KEY:
    #     st.error("❌ GROQ API Key not configured. Please check your environment variables.")
    elif not query.strip():
        st.warning("⚠️ Please enter a valid trend or topic to analyze.")
    else:
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Sanitize query and fetch web data
            status_text.text("🌐 Fetching market intelligence from web sources...")
            progress_bar.progress(20)
            
            sanitized_query = query.replace("client", "[REDACTED]").strip()
            web_data = fetch_trend_data_tavily(sanitized_query)
            
            # Step 2: Search internal documents
            status_text.text("📎 Searching internal documents...")
            progress_bar.progress(40)
            
            rag_context = ""
            if len(st.session_state.document_chunks) > 0:
                relevant_chunks = search_knowledge_base(sanitized_query)
                rag_context = "\n".join(relevant_chunks)
            
            # Step 3: Generate analysis
            status_text.text("🤖 Generating strategic insights...")
            progress_bar.progress(60)
            
            # Enhanced LLM prompt for structured output
            prompt = (
                f"As a senior management consultant, analyze the market trend: '{sanitized_query}' in the {domain} domain.\n\n"
                f"INTERNAL KNOWLEDGE BASE:\n{rag_context}\n\n"
                f"MARKET INTELLIGENCE:\n{web_data.get('content', '')}\n\n"
                f"Provide a comprehensive strategic analysis with the following structure:\n\n"
                f"## EXECUTIVE SUMMARY\n"
                f"- Brief overview of the trend's significance\n"
                f"- Key market implications\n\n"
                f"## MARKET TRENDS & DRIVERS\n"
                f"- List 5 key trends driving this market\n"
                f"- Underlying technological/social factors\n\n"
                f"## COMPETITIVE LANDSCAPE\n"
                f"- Major players and market dynamics\n"
                f"- Competitive advantages and threats\n\n"
                f"## IMPACT ASSESSMENT\n"
                f"- Provide a market impact score from 0-10 (where 10 is transformational)\n"
                f"- Justify the score with specific factors\n\n"
                f"## STRATEGIC RECOMMENDATIONS\n"
                f"- 3-5 actionable business recommendations\n"
                f"- Implementation priorities\n\n"
                f"## RISK FACTORS\n"
                f"- Key risks and mitigation strategies\n\n"
                f"Format your response with clear headings and bullet points for executive readability."
            )
            
            # llm_response = call_groq_llm(prompt)  # Original call with groq
            llm_response = call_watsonx_llm(wxc_client, prompt)
            
            progress_bar.progress(100)
            status_text.text("✅ Analysis complete!")
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            # === RESULTS DISPLAY ===
            
            # Executive Summary Cards
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Analysis Status</h3>
                    <div class="metric-value">✅</div>
                    <p>Complete</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Domain</h3>
                    <div class="metric-value">{domain}</div>
                    <p>Business Sector</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Data Sources</h3>
                    <div class="metric-value">{len(web_data.get('results', []))}</div>
                    <p>Web Intelligence</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Main Analysis with visualization
            st.markdown("---")
            
            if PLOTLY_AVAILABLE:
                col_analysis, col_viz = st.columns([2, 1])
                
                with col_analysis:
                    st.markdown("## 📊 Strategic Analysis")
                    st.markdown(llm_response)
                
                with col_viz:
                    st.markdown("### 📈 Impact Visualization")
                    impact_score = parse_impact_score(llm_response)
                    impact_chart = create_impact_chart(impact_score)
                    if impact_chart:
                        st.plotly_chart(impact_chart, use_container_width=True)
                    
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Impact Score</h3>
                        <div class="metric-value">{impact_score}/10</div>
                        <p>Market Significance</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("## 📊 Strategic Analysis")
                st.markdown(llm_response)
                
                # Show impact score without chart
                impact_score = parse_impact_score(llm_response)
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Market Impact Score</h3>
                    <div class="metric-value">{impact_score}/10</div>
                    <p>Strategic Significance Assessment</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Market Intelligence Sources
            if web_data.get('results'):
                st.markdown("---")
                st.markdown("## 🌐 Market Intelligence Sources")
                
                for i, result in enumerate(web_data['results'][:3]):
                    with st.expander(f"📰 {result['title']}", expanded=False):
                        st.write(result['content'])
                        st.markdown(f"**Source:** [{result['url']}]({result['url']})")
            
            # Internal Knowledge Context
            if rag_context:
                st.markdown("---")
                st.markdown("## 📎 Internal Knowledge Context")
                with st.expander("View relevant internal document excerpts", expanded=False):
                    st.text_area("Internal Context", value=rag_context, height=200)
            
            # Export Options
            st.markdown("---")
            session_id = uuid.uuid4().hex[:8]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            
            # Prepare comprehensive report
            full_report = f"""
TREND ANALYSIS REPORT
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Domain: {domain}
Query: {sanitized_query}
Session ID: {session_id}
Impact Score: {parse_impact_score(llm_response)}/10

=== STRATEGIC ANALYSIS ===
{llm_response}

=== MARKET INTELLIGENCE SOURCES ===
{web_data.get('content', '')}

=== INTERNAL KNOWLEDGE CONTEXT ===
{rag_context if rag_context else "No internal documents processed"}
"""
            
            st.download_button(
                label="📥 Download Complete Report",
                data=full_report,
                file_name=f"trend_analysis_{session_id}_{timestamp}.txt",
                mime="text/plain",
                use_container_width=True
            )
            
            # Store in session state for follow-up
            st.session_state["last_summary"] = llm_response
            st.session_state["chat_history"] = [
                {"role": "user", "content": query}, 
                {"role": "assistant", "content": llm_response}
            ]

        except Exception as e:
            st.error(f"❌ Error during analysis: {e}")
            st.info("Please check your API keys and network connection.")

# === FOLLOW-UP Q&A ===
if "last_summary" in st.session_state:
    st.markdown("---")
    st.markdown("## 💬 Follow-up Analysis")
    
    col_followup1, col_followup2 = st.columns([3, 1])
    
    with col_followup1:
        followup = st.text_input(
            "Ask a follow-up question", 
            placeholder="e.g., What are the pricing implications? How does this affect our product roadmap?",
            help="Ask specific questions about the analysis to get deeper insights"
        )
    
    with col_followup2:
        followup_button = st.button("💡 Get Follow-up Insight", use_container_width=True)
    
    if followup_button and followup:
        with st.spinner("🧠 Generating follow-up insights..."):
            try:
                st.session_state["chat_history"].append({"role": "user", "content": followup})
                # followup_result = call_groq_llm(followup, history=st.session_state["chat_history"])
                followup_result = call_watsonx_llm(wxc_client, followup, history=st.session_state["chat_history"])
                st.session_state["chat_history"].append({"role": "assistant", "content": followup_result})
                
                st.markdown("### 🎯 Follow-up Insights")
                st.markdown(followup_result)
                
            except Exception as e:
                st.error(f"❌ Error generating follow-up: {e}")
    elif followup_button and not followup:
        st.warning("⚠️ Please enter a follow-up question.")

# Footer
st.markdown("---")
st.markdown("""
<div class="footer">
    <p>Powered by watsonx AI • Market Intelligence by Tavily • Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)

# Status indicator
with st.sidebar:
    st.markdown("---")
    st.markdown("### 🔧 System Status")
    
    status_items = [
        ("📊 Streamlit", "✅ Active"),
        # ("🤖 Groq AI", "✅ Connected" if GROQ_API_KEY else "❌ Missing Key"),
        ("🤖 WatsonX AI", "✅ Connected" if wxc_client.API_KEY else "❌ Missing Key"),
        ("🌐 Tavily Search", "✅ Connected" if TAVILY_API_KEY else "❌ Missing Key"),
        ("📄 PDF Processing", "✅ Available" if PDF_AVAILABLE else "⚠️ Limited"),
        ("📈 Visualizations", "✅ Available" if PLOTLY_AVAILABLE else "⚠️ Limited"),
    ]
    
    for item, status in status_items:
        st.markdown(f"**{item}:** {status}")
