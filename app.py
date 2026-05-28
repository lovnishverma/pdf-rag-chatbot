import gradio as gr
import torch
import os

# PDF Loading
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Embeddings & Vector Store
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# LLM integration via Groq
from langchain_groq import ChatGroq

# ─────────────────────────────────────────────
# CONFIG & HYPERPARAMETERS
# ─────────────────────────────────────────────
EMBED_MODEL   = "BAAI/bge-small-en-v1.5"
LLM_MODEL     = "llama-3.3-70b-versatile" # High-tier reasoning, ultra-fast inference
CHUNK_SIZE    = 800
CHUNK_OVERLAP = 100
TOP_K         = 8 # Increased slightly to accommodate multi-doc synthesis

# ─────────────────────────────────────────────
# GLOBAL STATE
# ─────────────────────────────────────────────
vectorstore = None
retriever   = None

# ─────────────────────────────────────────────
# LOAD MODELS AT STARTUP
# ─────────────────────────────────────────────
print("⏳ Loading embedding model into memory...")
embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)
print("✅ Embeddings ready.")

print(f"⏳ Initializing API Client for {LLM_MODEL}...")

# FETCH API KEY FROM SECRETS
groq_api_key = os.environ.get("GROQ")

if not groq_api_key:
    print("⚠️ WARNING: 'GROQ' key not found in environment variables! Generation will fail.")

client = ChatGroq(
    model=LLM_MODEL,
    api_key=groq_api_key,
    temperature=0.5, # Tightly bounded for factual RAG tasks
    max_tokens=2048,
)
print("✅ LLM Client ready.")

# ─────────────────────────────────────────────
# EARLY EXIT ROUTING
# ─────────────────────────────────────────────
GREETINGS = {"hi", "hello", "hey", "howdy", "good morning", "good evening"}
SMALL_TALK = {"thanks", "thank you", "ok", "got it", "bye", "goodbye"}

def is_greeting(text: str) -> bool:
    return text.strip().lower().rstrip("!.,?") in GREETINGS

def is_small_talk(text: str) -> bool:
    return text.strip().lower().rstrip("!.,?") in SMALL_TALK

# ─────────────────────────────────────────────
# PROMPT BUILDER
# ─────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are an enterprise Q&A system. Answer questions using ONLY the facts "
    "provided in the CONTEXT below.\n"
    "Rules:\n"
    "1. If the user asks for a comparison, summary, or difference, synthesize the facts across the provided documents.\n"
    "2. Never hallucinate or use external knowledge.\n"
    "3. If the context entirely lacks the data to formulate an answer, reply strictly with: 'Not mentioned in the document.'\n"
)

# ─────────────────────────────────────────────
# BATCH PROCESS PDFs
# ─────────────────────────────────────────────
def process_pdfs(pdf_files):
    global vectorstore, retriever

    if not pdf_files:
        return "❌ Please upload at least one PDF file.", gr.update(interactive=False)

    try:
        all_chunks = []
        total_pages = 0
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", " ", ""]
        )

        # Batch process all uploaded files
        for pdf in pdf_files:
            loader = PyPDFLoader(pdf.name)
            documents = loader.load()
            total_pages += len(documents)
            
            chunks = splitter.split_documents(documents)
            # Tag metadata with filename for multi-doc clarity
            filename = os.path.basename(pdf.name)
            for chunk in chunks:
                chunk.metadata["source_file"] = filename
                
            all_chunks.extend(chunks)

        if not all_chunks:
            return "❌ No extractable text found. PDFs may be scanned images.", gr.update(interactive=False)

        # Build combined FAISS index
        vectorstore = FAISS.from_documents(all_chunks, embeddings)
        retriever   = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": TOP_K, "fetch_k": TOP_K * 3}
        )

        return (
            f"✅ Batch Processing Complete!\n"
            f"📄 Documents: {len(pdf_files)} | Pages: {total_pages}\n"
            f"🧩 Total Vector Chunks: {len(all_chunks)}\n"
            f"💬 Ready for multi-document queries.",
            gr.update(interactive=True)
        )

    except Exception as e:
        return f"❌ Pipeline Error: {str(e)}", gr.update(interactive=False)

# ─────────────────────────────────────────────
# CHAT GENERATOR (STREAMING)
# ─────────────────────────────────────────────
def chat(user_message, history):
    global retriever
    history = history or []

    if not user_message.strip():
        yield history, ""
        return

    history.append({"role": "user", "content": user_message})

    # Heuristic routing
    if is_greeting(user_message):
        history.append({"role": "assistant", "content": "👋 Systems active. Upload your PDFs and query the data."})
        yield history, ""
        return

    if is_small_talk(user_message):
        history.append({"role": "assistant", "content": "Acknowledged. Awaiting your next query."})
        yield history, ""
        return

    if retriever is None:
        history.append({"role": "assistant", "content": "⚠️ Vector space empty. Please upload and process documents first."})
        yield history, ""
        return

    # RAG Execution
    try:
        source_docs  = retriever.invoke(user_message)
        context_text = "\n\n".join(f"[Source: {d.metadata.get('source_file', 'Unknown')}, Page {d.metadata.get('page', 0)+1}]\n{d.page_content}" for d in source_docs)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"CONTEXT:\n{context_text}\n\nQUESTION: {user_message}"}
        ]

        history.append({"role": "assistant", "content": ""})
        yield history, "" 

        # Stream API output
        for chunk in client.stream(messages):
            if hasattr(chunk, "content") and chunk.content:
                history[-1]["content"] += chunk.content
                yield history, "" 

        # Construct citation footer
        citations = set()
        for doc in source_docs:
            fname = doc.metadata.get("source_file", "Doc")
            # Truncate long filenames for mobile UI
            if len(fname) > 20: fname = fname[:17] + "..."
            page = doc.metadata.get("page", 0) + 1
            citations.add(f"{fname} (Pg {page})")
            
        if citations:
            history[-1]["content"] += f"\n\n📌 *Sources: {', '.join(sorted(citations))}*"
            yield history, ""

    except Exception as e:
        if "api_key" in str(e).lower() or "401" in str(e):
            history[-1]["content"] = "❌ **Auth Error:** Invalid or missing GROQ key in environment."
        else:
            history[-1]["content"] = f"❌ **Inference Error:** {str(e)}"
        yield history, ""

def clear_chat():
    return [], ""

# ─────────────────────────────────────────────
# RESPONSIVE GRADIO UI
# ─────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

.gradio-container { font-family: 'Inter', sans-serif !important; max-width: 1200px !important; margin: 0 auto; padding: 0 16px; }

/* Mobile optimization constraints */
@media (max-width: 768px) {
    .gradio-container { padding: 0 8px; }
    .chat-window { height: 60vh !important; }
}

.upload-box { border: 2px dashed #0d9488 !important; border-radius: 8px; background: #f8fafc; }
.status-box textarea { background: #f0fdf4 !important; color: #064e3b !important; font-size: 0.85rem !important; border: 1.5px solid #0d9488 !important; border-radius: 8px !important; }

.process-btn { background: #0d9488 !important; color: white !important; font-weight: 600 !important; }
.send-btn { background: #0f172a !important; color: white !important; font-weight: 600 !important; }
"""

with gr.Blocks(title="📄 Enterprise RAG System", css=CSS, theme=gr.themes.Soft(primary_hue="teal")) as demo:
    
    gr.Markdown(
        "<div style='text-align:center; padding: 1rem 0;'>"
        "<h1 style='color:#0d9488; margin:0;'>📄 Multi-Document RAG System</h1>"
        "<p style='color:#64748b; margin-top:4px;'>Llama-3.3-70B · FAISS · Batch Processing</p>"
        "</div>"
    )

    # Default row (automatically handles responsive stacking via Column min_width)
    with gr.Row():
        
        # Left Panel: Ingestion
        with gr.Column(scale=1, min_width=300):
            gr.Markdown("### 1. Ingestion Pipeline")
            # ENABLED MULTIPLE FILES
            pdf_input = gr.File(label="Upload Documents", file_count="multiple", file_types=[".pdf"], elem_classes=["upload-box"])
            process_btn = gr.Button("⚡ Build Vector Space", variant="primary", size="lg", elem_classes=["process-btn"])
            status_box  = gr.Textbox(label="System Status", value="Awaiting data ingestion...", interactive=False, lines=4, elem_classes=["status-box"])

        # Right Panel: Interface
        with gr.Column(scale=2, min_width=300):
            gr.Markdown("### 2. Query Interface")
            chatbot = gr.Chatbot(label="Inference Console", height=550, type="messages", elem_classes=["chat-window"])
            
            with gr.Row():
                msg_input = gr.Textbox(placeholder="Execute query across processed documents...", label="Input", scale=4, interactive=True)
                send_btn  = gr.Button("Send", variant="primary", scale=1, elem_classes=["send-btn"])
            
            clear_btn = gr.Button("Clear Memory", variant="secondary", size="sm")

    # Event Wiring
    process_btn.click(fn=process_pdfs, inputs=[pdf_input], outputs=[status_box, msg_input])
    send_btn.click(fn=chat, inputs=[msg_input, chatbot], outputs=[chatbot, msg_input])
    msg_input.submit(fn=chat, inputs=[msg_input, chatbot], outputs=[chatbot, msg_input])
    clear_btn.click(fn=clear_chat, outputs=[chatbot, msg_input])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, show_api=False)
