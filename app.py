import streamlit as st
import time
import os
import concurrent.futures
from fpdf import FPDF
from dotenv import load_dotenv

# Import pipeline wrappers
from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question

load_dotenv()

st.set_page_config(
    page_title="AI Video Assistant Pro",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom High-Contrast Dark Interface Styling
st.markdown("""
    <style>
        .stApp {
            background-color: #0f172a;
            color: #f8fafc;
            font-family: 'Inter', -apple-system, sans-serif;
        }
        .section-header {
            color: #38bdf8;
            font-weight: 600;
            font-size: 1.25rem;
            margin-bottom: 16px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        div.stTextInput > div > div > input {
            background-color: #1e293b !important;
            color: #ffffff !important;
            border: 1px solid #475569 !important;
        }
        div.stSelectbox > div > div > div {
            background-color: #1e293b !important;
            color: #ffffff !important;
        }
        div.stButton > button {
            transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
            border-radius: 8px !important;
            background-color: #0284c7 !important;
            color: #ffffff !important;
            border: none !important;
            font-weight: 600 !important;
            height: 42px !important;
        }
        div.stButton > button:hover {
            transform: scale(1.01);
            background-color: #0ea5e9 !important;
            box-shadow: 0 0 15px rgba(56, 189, 248, 0.4);
        }
        p, li {
            color: #e2e8f0 !important;
        }
        .stTabs [data-baseweb="tab"] {
            color: #94a3b8 !important;
            font-weight: 500;
        }
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            color: #38bdf8 !important;
            font-weight: 700;
        }
        /* Custom adjustment to make drag and drop area look clean */
        .stFileUploader section {
            background-color: #1e293b !important;
            border: 1px dashed #475569 !important;
            border-radius: 8px !important;
            padding: 10px !important;
        }
    </style>
""", unsafe_allow_html=True)

def purge_temporary_chunks(chunk_paths: list):
    for path in chunk_paths:
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception as e:
            print(f"Error purging file path {path}: {e}")

def generate_pdf(results, chat_history):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 10, "AI Video Intelligence Workspace Report", ln=True, align="C")
    pdf.ln(5)
    
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 10, f"Asset Title: {results['title']}", ln=True)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)
    
    sections = [
        ("Executive Summary", results['summary']),
        ("Action Items", results['action_items']),
        ("Key Decisions", results['key_decisions']),
        ("Open Questions", results['open_questions'])
    ]
    
    for title, content in sections:
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, title, ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 6, content.encode('latin-1', 'replace').decode('latin-1'))
        pdf.ln(5)
        
    if chat_history:
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(5)
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, "Interactive Copilot Conversation Logs", ln=True)
        pdf.ln(2)
        
        for msg in chat_history:
            role = "User" if msg["role"] == "user" else "Assistant"
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(0, 5, f"{role}:", ln=True)
            pdf.set_font("Helvetica", "", 10)
            pdf.multi_cell(0, 5, msg["content"].encode('latin-1', 'replace').decode('latin-1'))
            pdf.ln(3)
            
    return bytes(pdf.output())

# Session Setup tracking
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "pipeline_results" not in st.session_state:
    st.session_state.pipeline_results = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.markdown("<h1 style='color: #ffffff; font-weight: 800; letter-spacing: -0.5px; margin-bottom: 0px;'>🎥 AI Video Intelligence Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #94a3b8; font-size: 1.05rem; margin-top: 4px;'>Analyze media assets, extract deep insights, and converse with your files natively.</p>", unsafe_allow_html=True)
st.divider()

# --- NEW INTEGRATED TABBED INPUT CARDS WORKSPACE ---
with st.container(border=True):
    st.markdown("<p style='font-weight: 600; font-size: 0.95rem; color: #94a3b8; margin-bottom: -5px;'>Select Media Input Source</p>", unsafe_allow_html=True)
    input_tab, upload_tab = st.tabs(["🔗 Stream from Remote URL / Path", "📁 Upload Local File Asset"])
    
    with input_tab:
        source_text = st.text_input(
            "Video Source URL", 
            placeholder="Paste a YouTube link or a valid absolute local file path...", 
            label_visibility="collapsed"
        ).strip()
    
    with upload_tab:
        uploaded_file = st.file_uploader(
            "Drag and drop your file here", 
            type=["mp3", "wav", "mp4", "m4a"], 
            label_visibility="collapsed"
        )

# Action Control Strip (Language + Process Button sits cleanly unified beneath options)
st.markdown("<div style='margin-top: -10px;'></div>", unsafe_allow_html=True)
control_col1, control_col2 = st.columns([3, 1])

with control_col1:
    language = st.selectbox("Language Preference", options=["english", "hinglish"], index=0, label_visibility="collapsed")
with control_col2:
    process_btn = st.button("🚀 Process Asset", type="primary", use_container_width=True)

# Determine asset routing source path securely
source = source_text
if uploaded_file is not None:
    source = os.path.join("Downloads", uploaded_file.name)
    with open(source, "wb") as f:
        f.write(uploaded_file.getbuffer())

st.markdown("<br>", unsafe_allow_html=True)

# Parallel Pipeline logic engine
if process_btn and source:
    st.session_state.pipeline_results = None
    st.session_state.chat_history = []
    
    progress_container = st.container(border=True)
    with progress_container:
        st.markdown("### ⚡ Execution Pipeline Status")
        status_text = st.empty()
        progress_bar = st.progress(0)
        
        try:
            status_text.markdown("⏳ **Step 1/3:** Isolating audio stream and prepping chunks...")
            chunks = process_input(source)
            progress_bar.progress(30)
            
            status_text.markdown("⏳ **Step 2/3:** Transcribing audio files with localized context...")
            transcript = transcribe_all(chunks, language)
            progress_bar.progress(60)
            
            purge_temporary_chunks(chunks)
            
            status_text.markdown("⏳ **Step 3/3:** Running Parallel Extraction Engines & Hybrid Search Vector Indexing...")
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_title = executor.submit(generate_title, transcript)
                future_summary = executor.submit(summarize, transcript)
                future_actions = executor.submit(extract_action_items, transcript)
                future_decisions = executor.submit(extract_key_decisions, transcript)
                future_questions = executor.submit(extract_questions, transcript)
                future_rag = executor.submit(build_rag_chain, transcript)
                
                title = future_title.result()
                summary = future_summary.result()
                action_items = future_actions.result()
                decisions = future_decisions.result()
                questions = future_questions.result()
                rag_chain = future_rag.result()
                
            progress_bar.progress(100)
            
            st.session_state.pipeline_results = {
                "title": title,
                "transcript": transcript,
                "summary": summary,
                "action_items": action_items,
                "key_decisions": decisions,
                "open_questions": questions
            }
            st.session_state.rag_chain = rag_chain
            
            status_text.markdown("✅ **Processing Complete!** Workspace generated successfully below.")
            time.sleep(1)
            progress_container.empty()
            
        except Exception as e:
            status_text.markdown("❌ **Critical Execution Error**")
            st.error(f"Pipeline failure details: {str(e)}")

# Active Report Dashboard Rendering Layer
if st.session_state.pipeline_results:
    res = st.session_state.pipeline_results
    
    header_col, download_col = st.columns([3, 1])
    with header_col:
        st.markdown(f"<h3 style='color: #ffffff; margin-bottom: 0px;'>📌 Active Asset: <span style='color: #38bdf8;'>{res['title']}</span></h3>", unsafe_allow_html=True)
    with download_col:
        pdf_bytes = generate_pdf(res, st.session_state.chat_history)
        st.download_button(
            label="📥 Export Workspace to PDF",
            data=pdf_bytes,
            file_name="AI_Assistant_Workspace_Report.pdf",
            mime="application/pdf",
            use_container_width=True
        )
    
    st.markdown("<br>", unsafe_allow_html=True)
    col_left, col_right = st.columns([1.4, 1.0], gap="medium")
    
    with col_left:
        with st.container(border=True):
            st.markdown("<div class='section-header'>📝 Executive Summary</div>", unsafe_allow_html=True)
            st.markdown(res['summary'])
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        with st.container(border=True):
            st.markdown("<div class='section-header'>⚡ Extracted Artifacts</div>", unsafe_allow_html=True)
            tab1, tab2, tab3 = st.tabs(["🎯 Action Items", "🔑 Key Decisions", "❓ Open Questions"])
            with tab1:
                st.markdown(res['action_items'])
            with tab2:
                st.markdown(res['key_decisions'])
            with tab3:
                st.markdown(res['open_questions'])
        
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("🔍 View Raw Transcript File"):
            st.text_area("Transcript Raw Content", res["transcript"], height=200, disabled=True, label_visibility="collapsed")
            
    with col_right:
        with st.container(border=True):
            st.markdown("<div class='section-header'>💬 Interactive RAG Copilot</div>", unsafe_allow_html=True)
            
            if not st.session_state.chat_history:
                st.markdown("<p style='color: #64748b !important; font-style: italic; font-size: 0.9rem;'>No active conversation yet. Ask a question below to begin.</p>", unsafe_allow_html=True)
            else:
                for message in st.session_state.chat_history:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])
                        
            if user_query := st.chat_input("Ask a question about the video contents..."):
                with st.chat_message("user"):
                    st.markdown(user_query)
                st.session_state.chat_history.append({"role": "user", "content": user_query})
                
                with st.chat_message("assistant"):
                    with st.spinner("Analyzing document nodes..."):
                        try:
                            stream_generator = ask_question(
                                st.session_state.rag_chain, 
                                user_query, 
                                st.session_state.chat_history[:-1]
                            )
                            full_answer = st.write_stream(stream_generator)
                            st.session_state.chat_history.append({"role": "assistant", "content": full_answer})
                            st.rerun()
                        except Exception as e:
                            st.error(f"RAG query timeout: {str(e)}")
else:
    st.info("💡 **Ready for Input:** Provide an asset source link or upload a meeting recording above to activate workspace insights.")