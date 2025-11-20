# Advanced RAG Chatbot with Agentic Routing and Streaming

## 📝 Project Overview

This project implements a high-performance **Retrieval-Augmented Generation (RAG) chatbot** using a sophisticated **Agentic Router** architecture. It leverages the open-source **Ollama** platform to host local Large Language Models (LLMs) and uses **FastAPI** for a fully streaming backend, providing real-time responses through a **Streamlit** user interface.

The core innovation is a **two-model routing system** that intelligently directs user queries to either a fast, simple model for chit-chat or a more powerful RAG pipeline for complex knowledge retrieval—optimizing both speed and accuracy.

---

## ✨ Key Features

- **Agentic Routing:** Uses a lightweight, fast model (`phi3:mini`) to classify queries as `SIMPLE` or `COMPLEX`.
- **Decoupled Architecture:** Clearly separates concerns with a FastAPI backend ("The Brain") and a Streamlit frontend ("The UI").
- **Streaming Responses:** Fully streamed communication (NDJSON), enabling a low-latency, real-time chat experience.
- **Advanced RAG Pipeline:**
  1. **ChromaDB** for rapid vector search.
  2. **Cross-Encoder Reranker** (`ms-marco-MiniLM-L-6-v2`) for strict answer relevance.
  3. **Llama 3** (`llama3:8b`) to synthesize grounded answers.
- **Dynamic Status Feedback:** Displays real-time status messages (e.g., "Routing to LLM," "Re-ranking search results") showing processing stages.
- **Metadata Citations:** Explicitly cites source files used to produce the answer, reducing hallucination risk.

---

## 🛠️ Architecture Stack

| Component             | Technology               | Role                                               |
| --------------------- | ------------------------ | -------------------------------------------------- |
| **LLM Orchestration** | **Ollama**               | Manages and serves local LLMs                      |
| **Backend API**       | **FastAPI**              | Provides scalable, asynchronous streaming API      |
| **Vector Database**   | **ChromaDB**             | Stores document embeddings for fast retrieval      |
| **Frontend UI**       | **Streamlit**            | Interactive Python web chat interface              |
| **Embedding Model**   | `all-MiniLM-L6-v2`       | Converts queries & docs to vectors                 |
| **Reranker Model**    | `ms-marco-MiniLM-L-6-v2` | Improves retrieval quality before answer synthesis |
| **Router LLM**        | `phi3:mini`              | Classifies query type (`simple` or `complex`)      |
| **RAG LLM**           | `llama3:8b`              | Generates grounded answers from context            |

---

## 📂 Project Structure

```plaintext
vrag/
├── rag_server/         # FastAPI backend
│   └── main.py         # Main server logic (Router + RAG)
├── client/             # Streamlit frontend
│   └── app.py          # UI logic
├── db/                 # ChromaDB vector store (generated)
├── ingest.py           # Script to build the vector database
├── requirements.txt    # Project dependencies
└── README.md
```

---

## 🚀 Installation and Setup

### 1. Prerequisites

- **Ollama:** Download and install [Ollama](https://ollama.com).
- **Python:** Version 3.10 or higher should be installed.

### 2. Install Ollama Models

Open your terminal and pull the required models using the Ollama CLI:

```bash
ollama pull phi3:mini
ollama pull llama3:8b
```

### 3. Setup Python Environment

In your project root:

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # Linux/macOS

# or (Windows)
.\venv\Scripts\activate
```

Then install required Python packages:

```bash
pip install -r requirements.txt
```

_If `requirements.txt` is missing, install: fastapi, uvicorn, ollama, chromadb, sentence-transformers, streamlit, requests, sse-starlette_

### 4. Ingest Data (Build the Knowledge Base)

Build the vector database:

```bash
python ingest.py
```

### 5. Run the Ollama Server (CRITICAL)

Start Ollama in a separate terminal session:

```bash
ollama serve
```

### 6. Run the FastAPI Server

```bash
cd rag_server
uvicorn main:app --reload
```

By default, the server runs at [http://127.0.0.1:8000](http://127.0.0.1:8000).

### 7. Run the Streamlit Client

In a third terminal:

```bash
cd client
streamlit run app.py
```

The chatbot UI will open in your browser.

---

## 🧠 How the Agentic Router Works

The `get_routed_response` function is the "brain" of the system:

1. **Query Input:** User sends a query (e.g. "hello" or "What is RAG?").
2. **Routing:** Query and a routing prompt are sent to the Router LLM (`phi3:mini`).
3. **Classification:**
   - If `SIMPLE`: Uses a fast model to answer directly, bypassing the RAG pipeline.
   - If `COMPLEX`: Triggers the full RAG process (retrieval + reranking + synthesis).
4. **Status Updates:** Streaming real-time feedback as the process unfolds.

---

## 📊 Status Messages Explained

During complex queries, users will see these statuses in real-time:

| Status Message            | Location            | Description                                 |
| ------------------------- | ------------------- | ------------------------------------------- |
| Getting to your query     | get_routed_response | Initializing the request flow               |
| Routing to the valid LLM  | get_routed_response | Phi-3 model classifies query type           |
| Processing the request    | get_routed_response | System routes to the correct pipeline       |
| Searching vector database | get_rag_response    | Performing initial ChromaDB vector search   |
| Re-ranking search results | get_rag_response    | Cross-Encoder sorts docs by relevance       |
| Getting the response      | get_rag_response    | Llama 3 generates the streaming text answer |

---
