---
title: PDF Q&A Chatbot (RAG)
emoji: 📄
colorFrom: red
colorTo: blue
sdk: gradio
sdk_version: 5.23.0
app_file: app.py
pinned: false
license: mit
short_description: RAG-based PDF chatbot using NVIDIA API and DeepSeek-V4-Pro
---

<div align="center">

# 📄 PDF Q&A Chatbot — RAG (DeepSeek-V4-Pro Edition)

### Retrieval-Augmented Generation · Ask anything from any PDF

[![HuggingFace Space](https://img.shields.io/badge/🤗%20HuggingFace-Space-blue?style=for-the-badge)](https://huggingface.co/spaces/LovnishVerma/rag)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](https://opensource.org/licenses/MIT)

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Gradio](https://img.shields.io/badge/Gradio-5.x-FF7C00?style=flat-square&logo=gradio&logoColor=white)](https://gradio.app)
[![LangChain](https://img.shields.io/badge/LangChain-0.3-1C3C3C?style=flat-square&logo=chainlink&logoColor=white)](https://langchain.com)
[![FAISS](https://img.shields.io/badge/FAISS-Local%20Vector%20DB-0467DF?style=flat-square&logo=meta&logoColor=white)](https://faiss.ai)
[![DeepSeek](https://img.shields.io/badge/LLM-DeepSeek--V4--Pro-7B68EE?style=flat-square)](https://huggingface.co/deepseek-ai)
[![NVIDIA NIM](https://img.shields.io/badge/API-NVIDIA%20NIM-76B900?style=flat-square&logo=nvidia&logoColor=white)](https://build.nvidia.com)

> 🎓 **RAG Project** · AI with ML 6-Month Course · NIELIT Ropar  
> **Level:** Intermediate · **Architecture:** Hybrid (Local Embeddings + Cloud LLM)
>
> GitHub Repo: https://github.com/lovnishverma/pdf-rag-chatbot

</div>

---

## 📸 Preview

> Upload a PDF → Ask questions → Get grounded answers with page citations

```text
┌─────────────────────────────────────────────────────────┐
│  📁 Upload PDF          │  💬 Ask Questions             │
│  ─────────────────────  │  ───────────────────────────  │
│  [ Drop PDF here ]      │  You: What is the summary?    │
│                         │                               │
│  [ ⚡ Process PDF ]     │  Bot: The document covers...  │
│                         │  📌 Sources: Page(s) 2, 5     │
│  ✅ 12 pages, 48 chunks │                               │
└─────────────────────────────────────────────────────────┘

```

---

## 🏗️ Architecture

```text
┌──────────────┐
│  PDF Upload  │  (PyPDFLoader)
└──────┬───────┘
       ↓
┌──────────────────┐
│  Text Chunking   │  RecursiveCharacterTextSplitter
│  chunk=800       │  overlap=100
└──────┬───────────┘
       ↓
┌────────────────────────┐
│  Embed Chunks          │  BAAI/bge-small-en-v1.5 (Local)
│  normalize_embeddings  │  CPU / CUDA auto-detect
└──────┬─────────────────┘
       ↓
┌──────────────────┐
│  FAISS Index     │  Local vector store
└──────┬───────────┘
       ↓  ← User asks question
┌──────────────────────┐
│  MMR Retrieval       │  Top-3 chunks (fetch_k=9)
│  (diversity-aware)   │
└──────┬───────────────┘
       ↓
┌─────────────────────────────────────────┐
│  DeepSeek-V4-Pro (via NVIDIA API)       │  Strict grounding prompt
│  openai client stream=True              │  max_tokens=2048
└──────┬──────────────────────────────────┘
       ↓
  Grounded answer + page citations

```

---

## 🛠️ Tech Stack

| Layer | Component | Tool |
| --- | --- | --- |
| 🖥️ Frontend | UI Framework | Gradio 5 |
| 📄 Ingestion | PDF Loader | LangChain `PyPDFLoader` |
| ✂️ Ingestion | Text Splitter | `RecursiveCharacterTextSplitter` |
| 🧠 Embedding | Encoder Model | `BAAI/bge-small-en-v1.5` (Local) |
| 🗄️ Retrieval | Vector Store | FAISS (local) + MMR |
| 🤖 Generation | LLM | `deepseek-ai/deepseek-v4-pro` (via NVIDIA NIM) |

---

## ✨ Features

* 📤 **Upload any PDF** — research papers, textbooks, notes, reports
* ⚡ **Fast processing** — chunking + embedding in seconds using lightweight BGE models
* 🔍 **MMR retrieval** — diverse, non-redundant context chunks
* 🧠 **High-Tier Reasoning** — utilizes DeepSeek-V4-Pro for highly accurate synthesis
* 📌 **Page citations** — every answer shows source page numbers
* 💬 **Conversational UI** — chat-style interface with real-time streaming text

---

## 🚀 Quick Start

### Run Locally

```bash
# 1. Clone the repo
git clone [https://github.com/lovnishverma/pdf-rag-chatbot
cd pdf-rag-chatbot

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch
python app.py
# Open http://localhost:7860

```

*(Note: The current `app.py` uses a hardcoded NVIDIA API key for demonstration. For production, please secure your API keys using environment variables.)*

---

## 💡 How to Use

| Step | Action |
| --- | --- |
| 1️⃣ | Upload any PDF using the file picker |
| 2️⃣ | Click **⚡ Process PDF** and wait for chunking & indexing |
| 3️⃣ | Type your question in the chat box |
| 4️⃣ | Hit **Send** or press `Enter` |
| 5️⃣ | Read the grounded answer with source page numbers streaming in |

---

## 🔧 Customization

### Swap the LLM (via API)

You can easily swap the LLM to any other model supported by the NVIDIA integration or standard OpenAI API by changing the variables in `app.py`:

```python
# In app.py — change LLM_MODEL:
LLM_MODEL = "meta/llama-3.1-70b-instruct"      # Example: Llama 3.1
LLM_MODEL = "mistralai/mixtral-8x7b-instruct-v0.1" # Example: Mixtral

```

### Tune chunking & retrieval

```python
CHUNK_SIZE    = 800   # ↑ larger = more context per chunk
CHUNK_OVERLAP = 100   # overlap to avoid splitting sentences
TOP_K         = 3     # number of chunks retrieved per query

```

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

1. Fork the repo
2. Create your branch: `git checkout -b feature/my-feature`
3. Commit: `git commit -m "Add my feature"`
4. Push: `git push origin feature/my-feature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the [MIT License](https://www.google.com/search?q=LICENSE).

---

## 👨‍💻 Author

**Lovnish Verma** · Project Engineer, NIELIT Ropar

---

*Built with ❤️ for the AI with ML course at NIELIT Ropar*