# Legal Knowledge Graph Chatbot

An AI-powered legal assistant chatbot for Indian law built using a Knowledge Graph + Retrieval-Augmented Generation (Graph RAG) pipeline.

The system answers legal queries related to NDPS laws and legal SOPs using a structured legal knowledge graph, semantic retrieval, and a locally hosted LLM through Ollama.

---

## Features

- ⚖️ Legal question answering for Indian law
- 🧠 Graph RAG architecture
- 🔎 Knowledge graph retrieval
- 💬 Conversational memory support
- 🖥️ Custom chat UI
- 🚀 FastAPI backend
- 🦙 Local LLM inference using Ollama
- 📚 Triple-based legal knowledge representation
- 🔄 Pickle-to-triples JSON conversion pipeline

---

## Tech Stack

### Backend
- Python
- FastAPI

### AI / RAG
- Ollama
- Llama 3.2
- Custom Graph RAG pipeline

### Knowledge Graph
- NetworkX
- Triple-based graph retrieval

### Frontend
- HTML
- CSS
- JavaScript

---

## Architecture

User Query
↓
FastAPI Backend
↓
Knowledge Graph Retrieval
↓
Relevant Legal Facts Extraction
↓
LLM Context Injection (Ollama)
↓
Response Generation

---

## Project Structure

```text
project/
│
├── legal_kg_engine.py
├── legal_search_engine.py
├── models.py
├── rag_engine.py
├── gemini_client.py
├── convert_pkl_to_triples.py
├── index.html
├── main.py
├── knowlegegraphs/      # Ignored (confidential)
├── .env
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/rooprajmurmu06-creator/LegalChatbot.git
cd project
```

---

### 2. Create virtual environment

```bash
python -m venv venv
```

---

### 3. Activate virtual environment

#### Windows

```bash
venv\Scripts\activate
```

#### Linux / Mac

```bash
source venv/bin/activate
```

---

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Install Ollama

Download Ollama

Pull the model:

```bash
ollama pull llama3.2:3b
```

Start Ollama locally:

```bash
ollama serve
```

---

## Environment Variables

Create a `.env` file:

```env
OLLAMA_BASE_URL=http://localhost:11434
LLM_MODEL=llama3.2:3b
LLM_TIMEOUT=60

BACKEND_HOST=0.0.0.0
BACKEND_PORT=8000

KG_DATA_PATH=knowledge_graphs
```

---

## Running the Frontend

Open:

```text
index.html
```

in your browser.

---

## Knowledge Graph Pipeline

Run the main.py file.

Run:

```bash
python main.py
```

---

## Example Legal Domains

- NDPS Act
- Zero FIR SOP
- Crime Scene SOP
- Investigation Procedures

---
## Teammates

- Roopraj Murmu
- Rishabh Kumar Jha
- Sachin Kumar

---
## License

MIT License
