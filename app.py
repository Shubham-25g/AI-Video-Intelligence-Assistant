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
    page_title="AI Video Assistant",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────────────────────
# PREMIUM STYLES
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Newsreader:opsz,wght@6..72,400;6..72,500&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    /* Deep graphite palette — Professional and easy on the eyes */
    --bg: #0b0c10;
    --surface: #14161a;
    --border: #282b33;
    --border-hover: #3d424e;
    
    --text-primary: #f0f2f5;
    --text-secondary: #9ba1a6;
    --text-tertiary: #686e75;
    
    --accent: #5d9cec;
    --positive: #51cf66;
    --negative: #ff6b6b;
}

*, *::before, *::after { box-sizing: border-box; }

html, body, .stApp {
    background: var(--bg) !important;
    color: var(--text-primary);
    font-family: 'Inter', system-ui, sans-serif;
}

/* Hide default streamlit items */
footer, [data-testid="stDecoration"], [data-testid="stToolbar"] { display: none !important; }
.block-container { max-width: 1100px !important; padding: 4rem 2rem 6rem !important; }

/* ── Page header ── */
.page-hdr {
    display: flex; align-items: flex-end; justify-content: space-between;
    padding-bottom: 24px; margin-bottom: 32px;
    border-bottom: 1px solid var(--border);
    flex-wrap: wrap; gap: 20px;
}
.ph-title { font-family: 'Newsreader', serif; font-size: 36px; font-weight: 500; letter-spacing: -0.02em; color: var(--text-primary); display: flex; align-items: baseline; line-height: 1; }
.ph-title em { font-style: normal; color: var(--text-secondary); font-family: 'Inter', sans-serif; font-size: 20px; font-weight: 500; margin-left: 6px; letter-spacing: 0; }
.ph-sub { font-size: 14px; color: var(--text-secondary); margin-top: 10px; }
.ph-badges { display: flex; gap: 8px; flex-wrap: wrap; }
.ph-badge { font-family: 'JetBrains Mono', monospace; font-size: 10px; color: var(--text-secondary); 
            border: 1px solid var(--border); background: var(--surface); border-radius: 4px; padding: 4px 10px; text-transform: uppercase; letter-spacing: 0.04em; }

.section-header {
    font-family: 'Inter', sans-serif;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--text-secondary);
    margin-bottom: 16px;
    border-bottom: 1px solid var(--border);
    padding-bottom: 8px;
}

/* ── Streamlit Container Overrides (Cards) ── */
div[data-testid="stVerticalBlockBorderWrapper"] > div {
    background-color: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 16px !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.15) !important;
}

/* ── Tabs ── */
button[data-baseweb="tab"] {
    background-color: transparent !important;
    color: var(--text-tertiary) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
    font-weight: 500 !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: var(--text-primary) !important;
    border-bottom-color: var(--text-primary) !important;
}
button[data-baseweb="tab"]:hover {
    color: var(--text-secondary) !important;
}

/* ── Inputs ── */
.stTextInput>div>div>input {
    background: var(--bg) !important; border: 1px solid var(--border) !important;
    border-radius: 6px !important; color: var(--text-primary) !important;
    font-size: 14px !important; padding: 0 16px !important; height: 42px !important;
    font-family: 'Inter', sans-serif !important; transition: border-color .2s ease;
}
.stTextInput>div>div>input::placeholder { color: var(--text-tertiary) !important; }
.stTextInput>div>div>input:focus { border-color: var(--text-primary) !important; box-shadow: none !important; }

/* ── Selectbox ── */
.stSelectbox>div>div>div {
    background: var(--bg) !important; border: 1px solid var(--border) !important;
    border-radius: 6px !important; color: var(--text-primary) !important;
    height: 42px !important; min-height: 42px !important;
}

/* ── File Uploader ── */
[data-testid="stFileUploader"] {
    background: var(--bg);
    border: 1px dashed var(--border);
    border-radius: 6px;
    padding: 16px;
}
[data-testid="stFileUploader"] section {
    background: transparent !important; border: none !important;
}

/* ── Buttons ── */
.stButton>button {
    background: var(--surface) !important; color: var(--text-primary) !important;
    border: 1px solid var(--border) !important; border-radius: 6px !important;
    font-weight: 500 !important; font-size: 14px !important;
    padding: 0 24px !important; height: 42px !important; letter-spacing: 0 !important;
    font-family: 'Inter', sans-serif !important; transition: all .2s ease !important;
    display: flex !important; align-items: center !important; justify-content: center !important; line-height: 1 !important;
}
/* Ensure the text inside the button inherits the correct color, overriding global p tags */
.stButton>button p { color: inherit !important; margin: 0 !important; }
.stButton>button:hover {
    border-color: var(--text-secondary) !important; background: var(--border) !important;
}

/* Primary Button */
.stButton>button[kind="primary"] {
    background: var(--text-primary) !important; color: #0b0c10 !important;
    border: none !important;
}
.stButton>button[kind="primary"] p { color: #0b0c10 !important; font-weight: 600 !important; }
.stButton>button[kind="primary"]:hover { 
    background: #ffffff !important; box-shadow: 0 4px 12px rgba(255,255,255,0.1) !important; transform: translateY(-1px); 
}

/* Download Button */
.stDownloadButton>button {
    background: var(--surface) !important; color: var(--text-primary) !important;
    border: 1px solid var(--border) !important; border-radius: 6px !important;
    font-size: 13px !important; font-weight: 500 !important; height: 42px !important;
    transition: all .2s ease !important;
}
.stDownloadButton>button p { color: inherit !important; margin: 0 !important; }
.stDownloadButton>button:hover { border-color: var(--text-secondary) !important; background: var(--border) !important; }

/* ── Progress Bar ── */
.stProgress>div>div>div>div { background: var(--text-primary) !important; border-radius: 99px !important; }

/* ── Expander ── */
[data-testid="stExpander"] {
    background: var(--surface) !important; border: 1px solid var(--border) !important;
    border-radius: 8px !important; overflow: hidden;
}
[data-testid="stExpander"]>details>summary {
    font-size: 13px !important; color: var(--text-primary) !important;
    font-weight: 500 !important; padding: 12px 16px !important;
    font-family: 'Inter', sans-serif !important; background: transparent !important;
}
[data-testid="stExpander"]>details>summary:hover { color: var(--text-secondary) !important; }

/* ── Chat ── */
[data-testid="stChatMessage"] { background-color: transparent !important; padding: 1rem 0 !important; }
[data-testid="stChatMessage"] p { font-size: 14px !important; line-height: 1.6 !important; color: var(--text-primary) !important; }
[data-testid="chatAvatarIcon-user"] { background-color: var(--text-secondary) !important; }
[data-testid="chatAvatarIcon-assistant"] { background-color: var(--accent) !important; }
.stChatInput>div { background: var(--surface) !important; border: 1px solid var(--border) !important; border-radius: 8px !important; }
.stChatInput>div:focus-within { border-color: var(--text-primary) !important; }

/* Typography */
p, li { color: var(--text-secondary) !important; font-size: 14px; line-height: 1.6; }
h1, h2, h3 { color: var(--text-primary) !important; }
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
    pdf.cell(0, 10, "AI Video Assistant Workspace Report", ln=True, align="C")
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

# ─────────────────────────────────────────────────────────────
# PAGE HEADER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-hdr">
    <div>
        <div class="ph-title">AI Video <em>Assistant</em></div>
        <div class="ph-sub">Analyze media, extract insights, and converse with assets natively.</div>
    </div>
    <div class="ph-badges">
        <span class="ph-badge">RAG Copilot</span>
        <span class="ph-badge">Audio Processing</span>
        <span class="ph-badge">Vector Indexing</span>
    </div>
</div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# WORKSPACE INPUT
# ─────────────────────────────────────────────────────────────
with st.container(border=True):
    st.markdown("<div class='section-header' style='border:none; margin-bottom:0;'>Select Media Source</div>", unsafe_allow_html=True)
    input_tab, upload_tab = st.tabs(["Remote URL / Path", "Upload Local File"])
    
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

# Action Control Strip
st.markdown("<div style='margin-top: 16px;'></div>", unsafe_allow_html=True)
control_col1, control_col2 = st.columns([3, 1])

with control_col1:
    language = st.selectbox("Language Context", options=["english", "hinglish"], index=0, label_visibility="collapsed")
with control_col2:
    process_btn = st.button("Process Asset", type="primary", use_container_width=True)

# Determine asset routing source path securely
source = source_text
if uploaded_file is not None:
    source = os.path.join("Downloads", uploaded_file.name)
    with open(source, "wb") as f:
        f.write(uploaded_file.getbuffer())

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# EXECUTION PIPELINE
# ─────────────────────────────────────────────────────────────
if process_btn and source:
    st.session_state.pipeline_results = None
    st.session_state.chat_history = []
    
    progress_container = st.container(border=True)
    with progress_container:
        st.markdown("<div class='section-header'>Execution Pipeline Status</div>", unsafe_allow_html=True)
        status_text = st.empty()
        progress_bar = st.progress(0)
        
        try:
            status_text.markdown("⬡ **Step 1/3:** Isolating audio stream and prepping chunks...")
            chunks = process_input(source)
            progress_bar.progress(30)
            
            status_text.markdown("⬡ **Step 2/3:** Transcribing audio files with localized context...")
            transcript = transcribe_all(chunks, language)
            progress_bar.progress(60)
            
            purge_temporary_chunks(chunks)
            
            status_text.markdown("⬡ **Step 3/3:** Running Parallel Extraction Engines & Hybrid Search Vector Indexing...")
            
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
            
            status_text.markdown("<span style='color: var(--positive);'>✓ Processing Complete. Workspace generated below.</span>", unsafe_allow_html=True)
            time.sleep(1)
            progress_container.empty()
            
        except Exception as e:
            status_text.markdown("<span style='color: var(--negative);'>✕ Critical Execution Error</span>", unsafe_allow_html=True)
            st.error(f"Pipeline failure details: {str(e)}")

# ─────────────────────────────────────────────────────────────
# RESULTS DASHBOARD
# ─────────────────────────────────────────────────────────────
if st.session_state.pipeline_results:
    res = st.session_state.pipeline_results
    
    header_col, download_col = st.columns([3, 1])
    with header_col:
        st.markdown(f"<div class='ph-title' style='font-size: 28px;'>Asset: <em>{res['title']}</em></div>", unsafe_allow_html=True)
    with download_col:
        pdf_bytes = generate_pdf(res, st.session_state.chat_history)
        st.download_button(
            label="Export to PDF",
            data=pdf_bytes,
            file_name="AI_Assistant_Workspace_Report.pdf",
            mime="application/pdf",
            use_container_width=True
        )
    
    st.markdown("<br>", unsafe_allow_html=True)
    col_left, col_right = st.columns([1.4, 1.0], gap="large")
    
    with col_left:
        with st.container(border=True):
            st.markdown("<div class='section-header'>Executive Summary</div>", unsafe_allow_html=True)
            st.markdown(res['summary'])
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        with st.container(border=True):
            st.markdown("<div class='section-header'>Extracted Artifacts</div>", unsafe_allow_html=True)
            tab1, tab2, tab3 = st.tabs(["Action Items", "Key Decisions", "Open Questions"])
            with tab1:
                st.markdown(res['action_items'])
            with tab2:
                st.markdown(res['key_decisions'])
            with tab3:
                st.markdown(res['open_questions'])
        
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("View Raw Transcript File"):
            st.text_area("Transcript Raw Content", res["transcript"], height=250, disabled=True, label_visibility="collapsed")
            
    with col_right:
        with st.container(border=True):
            st.markdown("<div class='section-header'>Interactive RAG Copilot</div>", unsafe_allow_html=True)
            
            if not st.session_state.chat_history:
                st.markdown("<p style='color: var(--text-tertiary) !important; font-style: italic; font-size: 13px;'>No active conversation yet. Ask a question below to begin.</p>", unsafe_allow_html=True)
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
    # Resting State
    pass