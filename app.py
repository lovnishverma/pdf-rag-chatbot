import gradio as gr
import torch
import re

# PDF Loading
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Embeddings & Vector Store
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# LLM integration via OpenAI client
from openai import OpenAI

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
EMBED_MODEL   = "BAAI/bge-small-en-v1.5"
LLM_MODEL     = "deepseek-ai/deepseek-v4-pro"
CHUNK_SIZE    = 800
CHUNK_OVERLAP = 100
TOP_K         = 3

# ─────────────────────────────────────────────
# GLOBAL STATE
# ─────────────────────────────────────────────
vectorstore = None
retriever   = None

# ─────────────────────────────────────────────
# LOAD MODELS AT STARTUP
# ─────────────────────────────────────────────
print("⏳ Loading embedding model...")
embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)
print("✅ Embeddings ready.")

print(f"⏳ Initializing NVIDIA API Client for {LLM_MODEL}...")
client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = "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"
)
print("✅ LLM Client ready.")

# ─────────────────────────────────────────────
# GREETING / SMALL-TALK DETECTION
# ─────────────────────────────────────────────
GREETINGS = {
    "hi", "hello", "hey", "hii", "helo", "howdy", "sup", "what's up",
    "whats up", "good morning", "good afternoon", "good evening",
    "good night", "how are you", "how r u", "how are u", "greetings",
    "namaste", "namaskar", "salaam", "yo", "hiya",
}

SMALL_TALK = {
    "thanks", "thank you", "thank you so much", "ok", "okay", "cool",
    "nice", "great", "awesome", "got it", "understood", "sure",
    "bye", "goodbye", "see you", "take care", "good job", "well done",
    "who are you", "what are you", "what can you do",
}

def is_greeting(text: str) -> bool:
    return text.strip().lower().rstrip("!.,?") in GREETINGS

def is_small_talk(text: str) -> bool:
    return text.strip().lower().rstrip("!.,?") in SMALL_TALK

def is_too_short(text: str) -> bool:
    words = text.strip().split()
    return len(words) <= 2 and "?" not in text

# ─────────────────────────────────────────────
# PROMPT BUILDER
# ─────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a document Q&A assistant. Your ONLY job is to answer questions "
    "using the EXACT information written in the CONTEXT below.\n"
    "Rules:\n"
    "1. Use ONLY facts explicitly stated in the context. Do NOT infer, guess, or assume.\n"
    "2. If the context does not contain the answer, reply: 'Not mentioned in the document.'\n"
    "3. Answer in 1-3 short sentences. No preamble, no repetition of the question.\n"
    "4. Never say 'we can assume' or 'it is likely'. Only state what is written."
)

# ─────────────────────────────────────────────
# ANSWER CLEANER
# ─────────────────────────────────────────────
STOP_PHRASES = [
    "we can assume", "it is likely", "it can be assumed", "it seems",
    "probably", "might have", "could have", "may have",
    "question:", "human:", "user:", "assistant:",
    "i am interested", "could you please", "can you please",
    "<|", "\n\n\n",
]

def clean_answer(raw: str) -> str:
    if not isinstance(raw, str):
        raw = str(raw)

    # Strip any stray <think>...</think> block just in case
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

    lower = raw.lower()
    for phrase in STOP_PHRASES:
        idx = lower.find(phrase)
        if idx > 20:
            raw = raw[:idx].strip()
            break

    raw = re.sub(r'\*{2,}', '', raw)
    raw = re.sub(r'\.{3,}', '...', raw)
    raw = raw.strip(" \t\n.,")

    # Keep max 3 sentences
    sentences = re.split(r'(?<=[.!?])\s+', raw)
    raw = " ".join(sentences[:3]).strip()

    return raw if raw else "Not mentioned in the document."

# ─────────────────────────────────────────────
# PROCESS PDF
# ─────────────────────────────────────────────
def process_pdf(pdf_file):
    global vectorstore, retriever

    if pdf_file is None:
        return "❌ Please upload a PDF file.", gr.update(interactive=False)

    try:
        loader    = PyPDFLoader(pdf_file.name)
        documents = loader.load()

        if not documents:
            return "❌ Could not extract text. Try another PDF.", gr.update(interactive=False)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        chunks = splitter.split_documents(documents)

        if not chunks:
            return "❌ No text chunks. PDF may be image-based (scanned).", gr.update(interactive=False)

        vectorstore = FAISS.from_documents(chunks, embeddings)
        retriever   = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": TOP_K, "fetch_k": TOP_K * 3}
        )

        return (
            f"✅ PDF processed!\n"
            f"📄 Pages: {len(documents)} | 🧩 Chunks: {len(chunks)}\n"
            f"💬 You can now ask questions about the document.",
            gr.update(interactive=True)
        )

    except Exception as e:
        return f"❌ Error: {str(e)}", gr.update(interactive=False)

# ─────────────────────────────────────────────
# CHAT
# ─────────────────────────────────────────────
def chat(user_message, history):
    global retriever

    history = history or []

    if not user_message.strip():
        return history, ""

    if is_greeting(user_message):
        history.append({"role": "user",      "content": user_message})
        history.append({"role": "assistant", "content":
            "👋 Hello! I'm your PDF assistant. Upload a PDF and ask me anything about it."})
        return history, ""

    if is_small_talk(user_message):
        history.append({"role": "user",      "content": user_message})
        history.append({"role": "assistant", "content":
            "😊 Happy to help! Ask any question about your uploaded PDF."})
        return history, ""

    if retriever is None:
        history.append({"role": "user",      "content": user_message})
        history.append({"role": "assistant", "content":
            "⚠️ Please upload and process a PDF first, then ask your question."})
        return history, ""

    if is_too_short(user_message):
        history.append({"role": "user",      "content": user_message})
        history.append({"role": "assistant", "content":
            "🤔 Could you ask a complete question? e.g. *\"What is the main topic of this document?\"*"})
        return history, ""

    # ── RAG pipeline ────────────────────────────────────────────────
    try:
        source_docs  = retriever.invoke(user_message)
        context_text = "\n\n".join(d.page_content for d in source_docs)

        # Prepare messages array for DeepSeek via OpenAI API
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"CONTEXT:\n{context_text}\n\nQUESTION: {user_message}"}
        ]

        completion = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=0.7, # Lowered from 1 slightly for better grounded RAG facts
            top_p=0.95,
            max_tokens=2048,
            extra_body={"chat_template_kwargs": {"thinking": False}},
            stream=True
        )

        raw = ""
        # Process the streaming response blocks
        for chunk in completion:
            if not getattr(chunk, "choices", None):
                continue
            if chunk.choices and chunk.choices[0].delta.content is not None:
                raw += chunk.choices[0].delta.content

        answer = clean_answer(raw)

        pages = sorted(set(
            doc.metadata.get("page", 0) + 1
            for doc in source_docs
            if isinstance(doc.metadata.get("page"), int)
        ))
        if pages:
            answer += f"\n\n📌 *Sources: Page(s) {', '.join(map(str, pages))}*"

        history.append({"role": "user",      "content": user_message})
        history.append({"role": "assistant", "content": answer})
        return history, ""

    except Exception as e:
        history.append({"role": "user",      "content": user_message})
        history.append({"role": "assistant", "content": f"❌ Error: {str(e)}"})
        return history, ""

def clear_chat():
    return [], ""

# ─────────────────────────────────────────────
# GRADIO UI
# ─────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;500;600&display=swap');

.gradio-container { font-family: 'Inter', sans-serif !important; max-width: 1200px !important; }

.nielit-header { background: #0f172a; border-bottom: 3px solid #0d9488; }
.nielit-footer { background: #0f172a; border-top: 2px solid #1e293b; }

.upload-box label { color: #1e293b !important; font-weight: 500 !important; }

.status-box textarea {
    background: #f0fdf4 !important;
    color: #064e3b !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.85rem !important;
    border: 1.5px solid #0d9488 !important;
    border-radius: 8px !important;
}
.status-box label { color: #1e293b !important; font-weight: 600 !important; }

.msg-input textarea {
    color: #111827 !important;
    background: #ffffff !important;
    font-size: 1rem !important;
}
.msg-input label { color: #374151 !important; }

.gradio-container .prose { color: #1e293b !important; }
.gradio-container p, .gradio-container li { color: #374151 !important; }
.gradio-container strong { color: #111827 !important; }
.gradio-container code { background: #f1f5f9 !important; color: #0f766e !important; padding: 2px 5px; border-radius: 4px; }

.process-btn { background: #0d9488 !important; color: white !important; font-weight: 700 !important; font-size: 1rem !important; }
.process-btn:hover { background: #0f766e !important; }

.send-btn { background: #0d9488 !important; color: white !important; font-weight: 700 !important; }
"""

with gr.Blocks(
    title="📄 PDF Q&A Chatbot — RAG",
    css=CSS,
    theme=gr.themes.Soft(
        primary_hue="teal",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter")
    )
) as demo:

    gr.HTML("""
    <div class="nielit-header" style="padding:12px 24px;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px;">
        <div style="display:flex;align-items:center;gap:14px;">
            <img src="https://www.nielit.gov.in/images/NIELIT_logo.jpg" alt="NIELIT Logo"
                 style="height:52px;border-radius:6px;background:white;padding:3px;"
                 onerror="this.style.display='none'">
            <div>
                <div style="font-family:'Space Mono',monospace;color:#14b8a6;font-size:0.78rem;font-weight:700;letter-spacing:0.05em;">
                    NATIONAL INSTITUTE OF ELECTRONICS &amp; INFORMATION TECHNOLOGY
                </div>
                <div style="color:#94a3b8;font-size:0.72rem;margin-top:1px;">
                    NIELIT Ropar &nbsp;·&nbsp; Deemed to be University &nbsp;·&nbsp; Ministry of Electronics &amp; IT, Govt. of India
                </div>
            </div>
        </div>
        <div style="color:#475569;font-size:0.72rem;text-align:right;font-family:'Space Mono',monospace;">
            NIELIT ROPAR<br>AI WITH ML Course
        </div>
    </div>
    <div style="text-align:center;padding:20px 0 8px;">
        <h1 style="font-family:'Space Mono',monospace;font-size:2rem;color:#0d9488;margin:0;">
            📄 PDF Q&amp;A Chatbot
        </h1>
        <p style="color:#475569;font-size:0.95rem;margin-top:8px;">
            Retrieval-Augmented Generation · FAISS · BGE Embeddings · DeepSeek-V4-Pro
        </p>
        <div style="display:flex;gap:8px;justify-content:center;margin-top:12px;flex-wrap:wrap;">
            <span style="background:#0d9488;color:white;padding:3px 12px;border-radius:20px;font-size:0.8rem;">🆓 API Ready</span>
            <span style="background:#f1f5f9;color:#475569;padding:3px 12px;border-radius:20px;font-size:0.8rem;border:1px solid #cbd5e1;">NVIDIA NIM</span>
        </div>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📁 Upload PDF")
            pdf_input = gr.File(label="Drop your PDF here", file_types=[".pdf"],
                                elem_classes=["upload-box"])
            process_btn = gr.Button("⚡ Process PDF", variant="primary", size="lg",
                                    elem_classes=["process-btn"])
            status_box  = gr.Textbox(label="📊 Status",
                                     value="⬆️ Upload a PDF to get started.",
                                     interactive=False, lines=4,
                                     elem_classes=["status-box"])
            gr.Markdown("""
---
**How it works:**
1. Upload any PDF
2. Click **Process PDF**
3. Ask questions below

**Stack:**
- 🧠 `BAAI/bge-small-en-v1.5` embeddings
- 📦 FAISS + MMR retrieval (local)
- 🤖 `deepseek-v4-pro` — via NVIDIA API
- 🔗 Strict grounding prompt
            """)

        with gr.Column(scale=2):
            gr.Markdown("### 💬 Ask Questions")
            chatbot = gr.Chatbot(label="Conversation", height=500,
                                 type="messages",
                                 show_label=True,
                                 elem_classes=["chat-window"])
            with gr.Row():
                msg_input = gr.Textbox(placeholder="Ask something about the PDF...",
                                       label="Your question",
                                       scale=4, interactive=False,
                                       elem_classes=["msg-input"])
                send_btn  = gr.Button("Send ➤", variant="primary", scale=1,
                                      elem_classes=["send-btn"])
            clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary", size="sm")

    process_btn.click(fn=process_pdf, inputs=[pdf_input], outputs=[status_box, msg_input])
    send_btn.click(fn=chat, inputs=[msg_input, chatbot], outputs=[chatbot, msg_input])
    msg_input.submit(fn=chat, inputs=[msg_input, chatbot], outputs=[chatbot, msg_input])
    clear_btn.click(fn=clear_chat, outputs=[chatbot, msg_input])

    gr.HTML("""
    <div style="background:#1e293b;border-top:1px solid #334155;margin-top:24px;padding:16px 24px;">
        <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px;">
            <div style="display:flex;align-items:center;gap:10px;">
                <img src="https://www.nielit.gov.in/images/NIELIT_logo.jpg" alt="NIELIT"
                     style="height:32px;border-radius:4px;background:white;padding:2px;"
                     onerror="this.style.display='none'">
                <span style="color:#475569;font-size:0.78rem;font-family:'Space Mono',monospace;">
                    &copy; 2026 NIELIT Ropar. All rights reserved.
                </span>
            </div>
            <div style="color:#475569;font-size:0.78rem;font-family:'Space Mono',monospace;text-align:right;">
                Project 1 — RAG Chatbot &nbsp;|&nbsp;
                Developed by <a href="https://github.com/lovnishverma" style="color:#0d9488;text-decoration:none;">Lovnish Verma</a>
            </div>
        </div>
    </div>
    """)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, show_api=False)