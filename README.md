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
* 🔒 **Secure API Key Handling** — securely reads API keys from environment variables/secrets

---

## 🔑 Prerequisites: Getting Your NVIDIA API Key

This project uses the **NVIDIA NIM API** to run the DeepSeek-V4-Pro model for free. Follow these steps to generate your API key:

1. Go to [build.nvidia.com](https://build.nvidia.com).
2. Click **Sign In** (or create a free NVIDIA Developer account if you don't have one).
3. Once logged in, browse the models and click on **DeepSeek-V4-Pro** (or any other text-generation model).
4. On the model's playground page, look for the **"Get API Key"** or **"Generate Key"** button (usually located in the code snippet section).
5. Generate the key and copy it. It will look something like this: `nvapi-LU2...`
6. **Keep this key secret!** Never commit it to GitHub.

---

## 🚀 Quick Start & Deployment

### Option 1: Run Locally

```bash
# 1. Clone the repo
git clone [https://github.com/lovnishverma/pdf-rag-chatbot.git](https://github.com/lovnishverma/pdf-rag-chatbot.git)
cd pdf-rag-chatbot

# 2. Install dependencies
pip install -r requirements.txt

```

**3. Set your API Key Secret:** Before running the app, you need to expose your NVIDIA API key as an environment variable so `app.py` can read it securely.

* **On Windows (Command Prompt):**
```cmd
set NVIDIA_API_KEY=your-api-key-here

```


* **On Windows (PowerShell):**
```powershell
$env:NVIDIA_API_KEY="your-api-key-here"

```


* **On Mac/Linux:**
```bash
export NVIDIA_API_KEY="your-api-key-here"

```



**4. Launch the App:**

```bash
python app.py
# Open http://localhost:7860 in your browser

```

### Option 2: Deploy to HuggingFace Spaces

If you are hosting this project on HuggingFace, you must add your API key to the Space's Secrets.

1. Go to your HuggingFace Space.
2. Click on **Settings** (the gear icon).
3. Scroll down to the **Variables and secrets** section.
4. Click **New secret**.
5. Set the **Name** to: `NVIDIA_API_KEY`
6. Set the **Value** to your actual key (e.g., `nvapi-LU2...`).
7. Click **Save**.
8. Restart your Space. The app will now securely fetch the key on startup!

---

## 💡 How to Use

| Step | Action |
| --- | --- |
| 1️⃣ | Upload any PDF using the file picker on the left |
| 2️⃣ | Click **⚡ Process PDF** and wait for the status to turn green |
| 3️⃣ | Type your question in the chat box on the right |
| 4️⃣ | Hit **Send ➤** or press `Enter` |
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

**Lovnish Verma** · Project Engineer, AI/ML Engineer · NIELIT Ropar

---

*Built with ❤️ for the AI with ML course at NIELIT Ropar*