# Advanced RAG Chatbot with Agentic Routing and Streaming

## 📝 Project Overview

This project implements a high-performance **Retrieval-Augmented Generation (RAG)** chatbot using a sophisticated **Agentic Router** architecture. It leverages the open-source **Ollama** platform to host local Large Language Models (LLMs) and uses **FastAPI** for a fully streaming backend, providing real-time responses to the user via a **Streamlit** user interface.

The core innovation is the **two-model routing system**, which intelligently directs user queries to either a fast, simple model for chit-chat or a more powerful RAG pipeline for complex knowledge retrieval, optimizing both speed and accuracy.

---

## ✨ Key Features

- **Agentic Routing:** Uses a small, fast model (`phi3:mini`) as a router to classify queries as `simple` or `complex`.
- **Streaming Responses:** The entire communication flow, from the FastAPI server to the Streamlit client, is fully streamed (NDJSON), providing a low-latency, real-time user experience.
- **Advanced RAG Pipeline:** For complex questions, the system uses:
  1.  **ChromaDB** for fast vector search.
  2.  A **Cross-Encoder Reranker** (`ms-marco-MiniLM-L-6-v2`) to improve the relevance of search results.
  3.  A powerful LLM (`llama3:8b`) to synthesize the final, grounded answer.
- **Dynamic Status Feedback:** Displays real-time status messages (e.g., "Routing to LLM," "Searching vector database," "Re-ranking search results") to inform the user about the current stage of processing.
- **Local LLM Execution:** All models are run locally via **Ollama**, ensuring privacy and independence from cloud APIs.

---

## 🛠️ Architecture Stack

| Component             | Technology               | Role                                                                |
| :-------------------- | :----------------------- | :------------------------------------------------------------------ |
| **LLM Orchestration** | **Ollama**               | Manages and serves local LLMs.                                      |
| **Backend API**       | **FastAPI**              | Provides the scalable, asynchronous streaming API (`/chat_router`). |
| **Vector Database**   | **ChromaDB**             | Stores document embeddings for fast retrieval.                      |
| **Frontend UI**       | **Streamlit**            | Interactive Python web interface for chat.                          |
| **Embedding Model**   | `all-MiniLM-L6-v2`       | Converts query and documents into vectors.                          |
| **Reranker Model**    | `ms-marco-MiniLM-L-6-v2` | Improves search result quality before synthesis.                    |
| **Router LLM**        | `phi3:mini`              | Classifies query type (`simple` or `complex`).                      |
| **RAG LLM**           | `llama3:8b`              | Generates final, grounded answers from context.                     |

---

## 🚀 Installation and Setup

### 1. Prerequisites

You must have the following installed and running:

- **Ollama:** Download and install the [Ollama software](https://ollama.com/download).
- **Python:** Version 3.10 or higher.
- **Virtual Environment:** Highly recommended (`venv` or `conda`).

### 2. Install Ollama Models

Open your terminal and pull the required models using the Ollama CLI.

```bash
ollama pull phi3:mini
ollama pull llama3:8b
3. Setup Python EnvironmentNavigate to the project root directory and set up your Python environment.Bash# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # Linux/macOS
.\venv\Scripts\activate   # Windows

# Install required Python packages (assuming the list is in a requirements.txt)
pip install -r requirements.txt
(If you don't have a requirements.txt, install: fastapi, uvicorn, ollama, chromadb, sentence-transformers, streamlit, requests)4. Run the Ollama Server (CRITICAL)The Ollama server must be running in a separate terminal session for the FastAPI application to work.Bashollama serve
5. Run the FastAPI ServerNavigate to the rag_server directory and start the FastAPI application with Uvicorn.Bashcd rag_server
uvicorn main:app --reload
The server will run at http://127.0.0.1:8000.6. Run the Streamlit ClientOpen a third terminal, navigate to the client directory, and start the Streamlit application.Bashcd client
streamlit run app.py
This will open the chatbot UI in your web browser.🧠 How the Agentic Router WorksThe get_routed_response function acts as the "brain" of the application:Query Input: The user sends a query (e.g., "hello" or "What is RAG?").Routing: The system sends the query and a routing prompt to the Router LLM (phi3:mini).Classification:If the router returns simple (e.g., for greetings, basic math), the system uses get_simple_response to answer directly using the fast model. This bypasses the slower RAG pipeline entirely.If the router returns complex (e.g., for detailed, knowledge-based questions), the system proceeds to the get_rag_response function, triggering the full vector search and re-ranking process.Status Updates: Status messages like "Routing to the valid LLM" and "Searching vector database" are streamed back to the user during this process for transparent feedback.📊 Status Messages ExplainedDuring a complex query, the user will see the following status messages streamed in real-time, corresponding to the steps in the agentic workflow:Status MessageLocationDescriptionGetting to your queryget_routed_responseInitializing the request flow.Routing to the valid LLMget_routed_responseThe Phi-3 model is classifying the query type.Processing the requestget_routed_responseThe classification is complete, and the system is initiating the correct pipeline.Searching vector databaseget_rag_responsePerforming the initial ChromaDB vector search.Re-ranking search resultsget_rag_responseThe Cross-Encoder is sorting the top documents by relevance.Getting the responseget_rag_response or get_simple_responseThe final LLM (Llama 3 or Phi-3) is generating the streaming text answer.
```
