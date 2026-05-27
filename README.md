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
---

# 📄 PDF Q&A Chatbot — RAG (Retrieval-Augmented Generation)

> **RAG Project** · AI with ML 6 Months Course
> **Level:** Beginner · **Build time:** 1–2 hours · **Deploy:** HuggingFace Spaces (Free)

---

## 🏗️ Architecture

```
PDF Upload (PyPDF)
       ↓
Text Chunking (RecursiveCharacterTextSplitter)
       ↓
Embed Chunks (BAAI/bge-small-en-v1.5)
       ↓
Store in FAISS (local, free)
       ↓
User asks question
       ↓
Retrieve Top-3 chunks via MMR
       ↓
LLM generates grounded answer (Qwen/Qwen3-0.6B — non-thinking mode)
```

---

## 🛠️ Tech Stack

| Component    | Tool                                          |
|--------------|-----------------------------------------------|
| Frontend     | Gradio 5                                      |
| PDF Loading  | LangChain `PyPDFLoader`                       |
| Chunking     | `RecursiveCharacterTextSplitter`              |
| Embeddings   | `BAAI/bge-small-en-v1.5` (free)               |
| Vector DB    | FAISS (local, free) + MMR retrieval           |
| LLM          | `Qwen/Qwen3-0.6B` (free, CPU-friendly)        |
| Deploy       | HuggingFace Spaces (free tier)                |

---

## 🚀 Deploy on HuggingFace Spaces (Step-by-Step)

### Option A: Upload via Web UI

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces)
2. Click **Create new Space**
3. Choose:
   - **SDK:** Gradio
   - **Hardware:** CPU Basic (free)
4. Upload `app.py` and `requirements.txt`
5. Wait ~3 minutes for build → Your app is live!

### Option B: Git Push

```bash
# Install HF CLI
pip install huggingface_hub

# Login
huggingface-cli login

# Create space & push
git clone https://huggingface.co/spaces/YOUR_USERNAME/pdf-rag-chatbot
cp app.py requirements.txt pdf-rag-chatbot/
cd pdf-rag-chatbot
git add . && git commit -m "Initial RAG chatbot" && git push
```

---

## 📁 Project Structure

```
pdf-rag-chatbot/
├── app.py           # Main Gradio application
├── requirements.txt # Dependencies
└── README.md        # This file
```

---

## 💡 How to Use

1. **Upload** any PDF document (research paper, textbook, notes)
2. Click **⚡ Process PDF** — wait for chunking & embedding
3. **Ask questions** in natural language
4. Get grounded answers with **source page numbers**

---

## 🔧 Customization

### Switch to a different LLM (still free):

```python
# In app.py, change LLM_MODEL to:
LLM_MODEL = "Qwen/Qwen3-1.7B"                          # Larger Qwen3, better quality
LLM_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"       # Lightweight chat model
LLM_MODEL = "microsoft/Phi-3-mini-4k-instruct"          # Strong small model
```

### Tune retrieval:

```python
# In app.py, adjust these constants:
CHUNK_SIZE    = 800   # Larger = more context per chunk
CHUNK_OVERLAP = 100   # Overlap to avoid cutting sentences
TOP_K         = 3     # Number of chunks retrieved per query
```

### For GPU Spaces (still free tier available):

```python
# app.py already auto-detects CUDA:
device=0 if torch.cuda.is_available() else -1
```

---

## 🐛 Known Fixes Applied

| Issue | Fix |
|-------|-----|
| `TypeError: multiple values for keyword argument 'generation_config'` | Removed `GenerationConfig` object; pass all generation params directly into `pipeline()` |
| `presence_penalty` not supported | Replaced with `repetition_penalty=1.3` (native HuggingFace param) |
| `dtype` deprecation warning | Changed `dtype=` to `torch_dtype=` in `AutoModelForCausalLM.from_pretrained()` |

---

## 📚 Learning Concepts Covered

- **RAG (Retrieval-Augmented Generation)** pipeline
- **Text chunking** strategies
- **Dense embeddings** with transformer models
- **Vector similarity search** with FAISS + MMR
- **LangChain** document loaders and text splitters
- **Gradio** UI for ML apps
- **HuggingFace Spaces** deployment

---

## 👨‍💻 Author

**Lovnish Verma** · Project Engineer, NIELIT Ropar
GitHub: [@lovnishverma](https://github.com/lovnishverma)
YouTube: [@lovnishverma](https://youtube.com/@lovnishverma)
Portfolio: [lovnishverma.in](https://lovnishverma.in)