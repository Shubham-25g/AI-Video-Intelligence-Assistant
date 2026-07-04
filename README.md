# 🎥 AI Video Intelligence Assistant Pro

An enterprise-grade **AI Video Intelligence Assistant** that automates the transcription, analysis, extraction, and interactive exploration of media content. Built using **Streamlit**, **LangChain**, **Mistral AI**, **OpenAI Whisper**, **Sarvam AI**, **Chroma DB**, and **Rank-BM25**, the system enables users to provide a multimedia asset (via a remote URL or a drag-and-drop file upload), extract deeply structured insights instantly via a multi-threaded parallel extraction engine, and have continuous, context-aware conversations with the content.

---

## 🌐 Live Demo

🚀 **Application:** https://ai-video-intelligence-assistant.streamlit.app/

📂 **GitHub Repository:** https://github.com/Shubham-25g/AI-Video-Intelligence-Assistant

---

# ✨ Features

### 📦 Hybrid Media Ingestion

* Supports streaming online video assets via direct YouTube URLs.
* Handles physical local file uploads (`.mp3`, `.wav`, `.mp4`, `.m4a`) using an integrated drag-and-drop tab system.
* Standardizes diverse audio codecs by converting streams to single-channel 16kHz WAV formats via `pydub`.

### 🗣️ Context-Driven Localization

* Features dual-engine transcription matching localized video dialects.
* Employs a local **OpenAI Whisper** model for rapid English audio conversions.
* Integrates the **Sarvam AI API** for multi-piece translation and processing of Hinglish media assets into English text.

### ⚡ Parallel Analytical Engines

* Bypasses sequential execution lag by utilizing `concurrent.futures` multi-threading to evaluate tasks simultaneously.
* Kicks off independent language model requests in parallel to instantly generate:
* Contextual meeting/video titles (max 8 words).
* Bulleted Executive Summaries using map-reduce grouping strategies.
* Action Items coupled with descriptions, task owners, and deadline metrics.
* Key Decisions finalized across the conversational track.
* Open Questions and unresolved topics requiring future attention.



### 🔍 Advanced Hybrid Search Core

* Upgrades standard document indexing into a resilient **Hybrid Search** configuration.
* Pairs exact keyword token extraction via a `Rank-BM25` lexical retriever with deep semantic similarity lookups using `Chroma DB` and `all-MiniLM-L6-v2` embeddings.
* Drastically mitigates RAG omission and context errors by maintaining large, multi-token text chunk sizes.

### 💬 Memory-Infused Streaming Chat

* Employs a LangChain LCEL pipeline that converts sequential dialog history lists into explicit system prompts.
* Allows the model to recognize context and handle conversational dependencies or follow-up questions gracefully.
* Features live token streaming using Mistral's `.stream()` generator engine to stream text onto the UI in real time.

### 🗂️ Workspace Exporter & Optimizer

* Integrates a one-click PDF generation engine using `FPDF2` to export structured results and active chat conversation history arrays.
* Features automatic resource cleanup utilities that immediately sweep and purge temporary audio chunks from local storage post-transcription to preserve system memory.

---

# 🏗️ System Workflow

```text
                     Video URL or Local File
                                │
                                ▼
               📦 Audio Processor (yt-dlp / pydub)
                                │
                                ▼
              🗣️ Transcription Core (Whisper / Sarvam)
                                │
            ┌───────────────────┴───────────────────┐
            ▼                                       ▼
    📚 Hybrid Search Indexing               ⚙️ Parallel Extraction
 (Chroma Vector + Rank-BM25)             (ThreadPoolExecutor)
            │                                       │
            │   ┌───────────────────────────────────┼───────────────────────────────────┐
            │   ▼                                   ▼                                   ▼
            │ 📝 Title Generator         📋 Executive Summarizer          ⚡ Artifact Extractors
            │   │                                   │                                   │
            └───┼───────────────────────────────────┼───────────────────────────────────┘
                ▼                                   ▼
         💬 Context-Aware RAG Chat           🗂️ Interactive Workspace Dashboard
         (Live Token Streaming)               (Summary, Analytics Tabs, PDF Export)

```

---

# 🛠️ Tech Stack

| Category | Technologies |
| --- | --- |
| Programming Language | Python |
| Frontend Framework | Streamlit (Custom Slate Dark Theme Layout) |
| AI Framework | LangChain (LCEL Orchestration) |
| Large Language Model | Mistral AI (`mistral-small-latest`) |
| Local Transcription | OpenAI Whisper (Local Model Node) |
| Localization API | Sarvam AI API (Hinglish Context Translation) |
| Vector Store | Chroma DB |
| Keyword Retrieval | Rank-BM25 Lexical Engine |
| Vector Embeddings | HuggingFace sentence-transformers (`all-MiniLM-L6-v2`) |
| Audio Engineering | Pydub (AudioSegment format manipulations) |
| Media Downloader | yt-dlp |
| Report Generation | FPDF2 |
| Version Control | Git & GitHub |

---

# 📂 Project Structure

```text
AI-Video-Intelligence-Assistant/
│
├── app.py                  # Master Streamlit layout dashboard & thread coordination
├── main.py                 # CLI execution entrypoint pipeline
├── requirements.txt        # Active python environment tracking configurations
├── .gitignore              # Staging filter rules tracking system exceptions
│
├── core/
│   ├── transcriber.py      # Whisper resource caching & Sarvam API chunk segmenting
│   ├── summarizer.py       # Map-Reduce text summarizations & metadata title handlers
│   ├── extractor.py        # Independent language chain definitions for meeting outputs
│   ├── vector_store.py     # Hybrid search core blending dense Chroma indexing and BM25
│   └── rag_engine.py       # Memory- infused RAG execution pipelines and text streaming
│
├── utils/
│   └── audio_processor.py  # yt-dlp media extractions and AudioSegment chunk subdivisions
│
└── Downloads/              # Directory mapping transient media processing streams

```

---

# 🚀 Future Improvements

* Interactive time-stamped citations inside the RAG chat answers.
* Advanced speaker diarization to explicitly distinguish meeting participants.
* Multi-video batch processing queues for massive asynchronous content research.
* Integration with advanced cross-encoder re-ranking layers (e.g., Cohere) to maximize search relevance.
* Dynamic analytical charts and visual participant engagement timelines.

---

# 🎯 Key Learning Outcomes

This project demonstrates practical experience with:

* Asynchronous multi-threaded API architectures (`ThreadPoolExecutor`).
* Designing hybrid search logic boundaries that balance lexical precision with vector context.
* Optimizing conversational systems by managing conversation history states across stateless REST models.
* Resource constraints, memory performance, and local machine learning model lifecycle management.
* Custom application frontend container styling using low-level CSS injections in Streamlit.
* Building immutable report generation structures for corporate record-keeping.

---

# 👨‍💻 Author

**Shubham Gupta**

* GitHub: [https://github.com/Shubham-25g](https://github.com/Shubham-25g)
* LinkedIn: [www.linkedin.com/in/shubhamgupta2510](https://www.google.com/search?q=https://www.linkedin.com/in/shubhamgupta2510)
* Live Demo: https://ai-video-intelligence-assistant.streamlit.app/
