# 🤖 Document AI Assistant (RAG-based)

An intelligent **Retrieval-Augmented Generation (RAG) Document AI Assistant** that turns your documents into interactive conversations.  
Upload **PDF, DOCX, TXT, PPTX, CSV, XLSX** files and ask questions in **English**.  
The assistant retrieves relevant context and provides **accurate, well-structured answers** using **LangChain, FAISS, HuggingFace, and Gradio**.

---

# ✨ Features
- 📂 **Multi-Format Support** → PDF, DOCX, TXT, PPTX, CSV, XLSX  
- 🗣️ **English Q&A** → Crisp answers grounded in your documents  
- ⚡ **RAG Pipeline** → Embeddings + FAISS for semantic retrieval  
- 🧠 **Conversational Memory** → Tracks chat history for context  
- 🎨 **Modern UI** → Gradio Blocks-based chat interface

> Note: The code contains stub functions for Urdu translation; this build is positioned as **English-only**.

---

# 🛠 Tech Stack
- Python 3.9+  
- LangChain (community/experimental/text splitters)  
- FAISS (CPU)  
- HuggingFace Embeddings (`sentence-transformers/all-MiniLM-L6-v2`)  
- Gradio  
- PDF/Office tooling: PyPDF2, PyMuPDF, pdfplumber, python-docx/docx2txt, python-pptx, pandas (openpyxl)

---

## 📂 Project Structure
document-ai-assistant-rag/
│── main.py # Core logic: loading, splitting, embeddings, RAG chain, chatbot
│── interface/
│ └── ui.py # Gradio UI (Blocks) with chat, file upload, controls
│── requirements.txt # Dependencies
│── .env.example # Environment variables template
│── README.md # This file

yaml
Copy code

---

# ⚙️ Setup & Installation
```bash
# 1) Clone
git clone https://github.com/<your-username>/document-ai-assistant-rag.git
cd document-ai-assistant-rag

# 2) Create & activate a virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
# source venv/bin/activate

# 3) Install dependencies
pip install -r requirements.txt

# 4) Create .env
cp .env.example .env
# Edit .env and paste your key(s)
🔑 Environment Variables
Create a .env file in the project root with:

ini
Copy code
GROQ_API_KEY=your_groq_api_key_here
🚀 Run the App
bash
Copy code
python main.py
Open the Gradio URL shown in the terminal (e.g., http://127.0.0.1:7860).

🧪 How It Works
Load documents from uploads (PDF/DOCX/TXT/PPTX/CSV/XLSX).

Extract text via robust readers (PyPDF2 → PyMuPDF → pdfplumber fallback).

Chunk using SemanticChunker (fallback to RecursiveCharacterTextSplitter).

Embed chunks with HuggingFace; index using FAISS.

Retrieve + Generate with a ConversationalRetrievalChain and memory.

Gradio UI provides a clean chat flow.

🗺️ Roadmap
Optional OCR for scanned PDFs (e.g., Tesseract)

Export chat + cited snippets

Pluggable vector stores and LLM backends

🧾 License
MIT

shell
Copy code

# requirements.txt
```txt
# Core framework
langchain>=0.2.0
langchain-community>=0.2.0
langchain-core>=0.2.0
langchain-experimental>=0.0.65
langchain-text-splitters>=0.2.0
langchain-huggingface>=0.1.0
langchain-groq>=0.1.5

# Retrieval / embeddings
faiss-cpu>=1.7.4
sentence-transformers>=2.7.0
huggingface-hub>=0.22.0

# UI
gradio>=4.0.0

# Documents I/O
PyPDF2>=3.0.1
pymupdf>=1.23.0
pdfplumber>=0.11.0
python-docx>=1.1.0
docx2txt>=0.8
python-pptx>=0.6.23
pandas>=2.1.0
openpyxl>=3.1.0

# Utilities
python-dotenv>=1.0.0
.env.example
env
Copy code
GROQ_API_KEY=your_groq_api_key_here
