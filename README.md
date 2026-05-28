---
title: PDF Q&A Chatbot (RAG)
emoji: 📄
colorFrom: yellow
colorTo: green
sdk: gradio
sdk_version: 5.23.0
app_file: app.py
pinned: false
license: mit
short_description: RAG-based PDF chatbot using Groq + Llama 3.3
---

<div align="center">

# 📄 Multi-Document RAG System (Groq + Llama 3.3 Edition)

### Retrieval-Augmented Generation · Query multiple PDFs instantly

[![HuggingFace Space](https://img.shields.io/badge/🤗%20HuggingFace-Space-blue?style=for-the-badge)](https://huggingface.co/spaces/LovnishVerma/rag)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Gradio](https://img.shields.io/badge/Gradio-Mobile%20Responsive-FF7C00?style=flat-square&logo=gradio&logoColor=white)](https://gradio.app)
[![LangChain](https://img.shields.io/badge/LangChain-0.3-1C3C3C?style=flat-square&logo=chainlink&logoColor=white)](https://langchain.com)
[![FAISS](https://img.shields.io/badge/FAISS-Local%20Vector%20DB-0467DF?style=flat-square&logo=meta&logoColor=white)](https://faiss.ai)
[![Llama](https://img.shields.io/badge/LLM-Llama--3.3--70B-0467DF?style=flat-square)](https://ai.meta.com/llama/)
[![Groq](https://img.shields.io/badge/API-Groq%20Cloud-F55036?style=flat-square)](https://groq.com)

> 🎓 **RAG Project** · AI with ML Course · NIELIT Ropar  
> **Level:** Advanced · **Architecture:** Hybrid (Local Embeddings + Ultra-Fast Cloud LLM)
>
> GitHub Repo: https://github.com/lovnishverma/pdf-rag-chatbot

</div>

---

<img width="1768" height="976" alt="image" src="https://github.com/user-attachments/assets/6f084f39-5199-4ac4-9d5d-f30773495b20" />


## 📸 Preview

> Upload multiple PDFs → Ask questions across all documents → Get grounded answers with file/page citations streamed instantly.

```text
┌─────────────────────────────────────────────────────────┐
│  📁 Upload Documents    │  💬 Query Interface           │
│  ─────────────────────  │  ───────────────────────────  │
│  [ Drop PDF(s) here ]   │  You: Compare the findings... │
│                         │                               │
│  [ ⚡ Build Vector Space]│ Bot: The reports differ...   │
│                         │  📌 Sources: Doc1 (Pg 2),     │
│  ✅ 3 Docs, 48 chunks   │               Doc2 (Pg 5)     │
└─────────────────────────────────────────────────────────┘

```

---

## 🏗️ Architecture

```text
┌─────────────────┐
│  Multiple PDFs  │  (PyPDFLoader) Batch Ingestion
└──────┬──────────┘
       ↓
┌──────────────────┐
│  Text Chunking   │  RecursiveCharacterTextSplitter
│  chunk=800       │  overlap=100 + Metadata Tagging
└──────┬───────────┘
       ↓
┌────────────────────────┐
│  Embed Chunks          │  BAAI/bge-small-en-v1.5 (Local)
│  normalize_embeddings  │  CPU / CUDA auto-detect
└──────┬─────────────────┘
       ↓
┌──────────────────┐
│  FAISS Index     │  Combined Local Vector Store
└──────┬───────────┘
       ↓  ← User asks question
┌──────────────────────┐
│  MMR Retrieval       │  Top-4 chunks across all docs
│  (diversity-aware)   │  fetch_k=12
└──────┬───────────────┘
       ↓
┌─────────────────────────────────────────┐
│  Llama-3.3-70B-Versatile (via Groq API) │  Strict grounding prompt
│  ChatGroq (LangChain) stream=True       │  temperature=0.5
└──────┬──────────────────────────────────┘
       ↓
  Grounded answer + Cross-document citations

```

---

## 🛠️ Tech Stack

| Layer | Component | Tool |
| --- | --- | --- |
| 🖥️ Frontend | UI Framework | Gradio 5 (Mobile Responsive Flexbox) |
| 📄 Ingestion | PDF Loader | LangChain `PyPDFLoader` (Batch Processing) |
| ✂️ Ingestion | Text Splitter | `RecursiveCharacterTextSplitter` |
| 🧠 Embedding | Encoder Model | `BAAI/bge-small-en-v1.5` (Local) |
| 🗄️ Retrieval | Vector Store | FAISS (local) + MMR Search |
| 🤖 Generation | LLM API Client | `langchain-groq` (`ChatGroq`) |
| 🧠 Generation | Base Model | `llama-3.3-70b-versatile` (via Groq Cloud) |

---

## ✨ Features

* 📚 **Multi-Document Batch Processing** — Upload and digest multiple PDFs simultaneously to cross-reference data.
* ⚡ **Lightning-Fast Inference** — Swapped to Groq API (LPUs) to eliminate gateway timeouts and deliver near-instantaneous token streaming.
* 📱 **Mobile-First UI** — Completely refactored Gradio interface utilizing dynamic scaling and flexbox wraps for seamless use on phones and tablets.
* 🔍 **Cross-Document Citations** — Generated answers strictly cite both the *Source File Name* and the *Page Number*.
* 🧠 **High-Tier Reasoning** — Utilizes Meta's flagship `Llama-3.3-70B` for deep comprehension and synthesis.
* 🔒 **Secure API Key Handling** — Safely reads your Groq API key from environment variables/secrets.

---

## 🔑 Prerequisites: Getting Your Groq API Key

This project relies on **Groq Cloud** for ultra-low latency LLM inference.

1. Go to [console.groq.com](https://console.groq.com/).
2. Log in or create a free developer account.
3. Navigate to **API Keys** in the left sidebar.
4. Click **Create API Key**, name it, and copy the string (e.g., `gsk_...`).
5. **Keep this key secret!** Never commit it to GitHub.

---

## 🚀 Quick Start & Deployment

### Option 1: Run Locally

**1. Clone the repository:**

```bash
git clone [https://github.com/lovnishverma/pdf-rag-chatbot.git](https://github.com/lovnishverma/pdf-rag-chatbot.git)
cd pdf-rag-chatbot

```

**2. Install dependencies:**

```bash
pip install -r requirements.txt

```

**3. Set your API Key Secret:** Before running the app, expose your Groq API key as an environment variable.

* **On Windows (Command Prompt):**

```cmd
set GROQ=your-api-key-here

```

* **On Windows (PowerShell):**

```powershell
$env:GROQ="your-api-key-here"

```

* **On Mac/Linux:**

```bash
export GROQ="your-api-key-here"

```

**4. Launch the App:**

```bash
python app.py

```

Open `http://localhost:7860` in your web browser.

### Option 2: Deploy to HuggingFace Spaces

If you are hosting this project on HuggingFace, you must add your API key to the Space's Secrets.

1. Go to your HuggingFace Space.
2. Click on **Settings** (the gear icon).
3. Scroll down to the **Variables and secrets** section.
4. Click **New secret**.
5. Set the **Name** to: `GROQ`
6. Set the **Value** to your actual key (e.g., `gsk_...`).
7. Click **Save** and **Restart** your Space.

---

## 💡 How to Use

| Step | Action |
| --- | --- |
| 1️⃣ | **Upload:** Drag and drop one or multiple PDFs into the upload box. |
| 2️⃣ | **Process:** Click **⚡ Build Vector Space** and wait for the batch ingestion to complete. |
| 3️⃣ | **Ask:** Type your query in the chat input (e.g., "Summarize the differences between Doc A and Doc B"). |
| 4️⃣ | **Send:** Hit **Send** or press `Enter`. |
| 5️⃣ | **Read:** Watch the fast-streaming answer appear with precise file/page citations. |

---

## 🔧 Customization

### Tune Retrieval Parameters

Adjust these global variables at the top of `app.py` to change how the documents are digested, particularly if you are processing massive batches:

```python
CHUNK_SIZE    = 800   # ↑ larger = more context per chunk
CHUNK_OVERLAP = 100   # overlap to avoid splitting sentences in half
TOP_K         = 4     # Number of chunks retrieved. Increase if querying many docs at once.

```

---

## 🤝 Contributing

Contributions, issues, and feature requests are always welcome!

1. Fork the repo
2. Create your branch: `git checkout -b feature/my-feature`
3. Commit: `git commit -m "Add my feature"`
4. Push: `git push origin feature/my-feature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

---

## 👨‍💻 Author

**Lovnish Verma** · Project Engineer, AI/ML Engineer · NIELIT Ropar

---

*Built with ❤️ for the AI with ML course at NIELIT Ropar*
