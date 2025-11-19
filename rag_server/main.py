import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import ollama
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
import json
from typing import AsyncGenerator

# --- Configuration & Initialization ---

# Models
DB_DIR = "../db"  # Path relative to this file's location
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2" 
ROUTER_LLM_MODEL = "phi3:mini"      # Router model
RAG_LLM_MODEL = "llama3:8b"     # Powerful model for RAG answers

# Search parameters
TOP_K_SEARCH = 20
TOP_K_RERANK = 3

# Initialize app
app = FastAPI(title="Advanced RAG Server (Ollama Streaming)")

# --- Global Resources (Loaded at Startup) ---
embedding_model = None
rerank_model = None
db_client = None
collection = None
ollama_client = None

try:
    # Load vector models
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    rerank_model = CrossEncoder(RERANK_MODEL_NAME) 
    db_client = chromadb.PersistentClient(path=DB_DIR)
    collection = db_client.get_collection(name="rag_collection")
    
    # Initialize Ollama Client
    ollama_client = ollama.Client() 
    
    print("Models and DB client loaded successfully.")
    
    try:
        models = ollama_client.list().get('models', [])
        model_names = [m.get('name', 'UNKNOWN') for m in models]
        print(f"Ollama models available: {model_names}")
    except Exception as e:
        print(f"Warning: Could not list Ollama models (Ollama client is still initialized): {e}")

except Exception as e:
    print(f"CRITICAL ERROR loading resources: {e}")
    embedding_model = None
    rerank_model = None
    collection = None
    ollama_client = None

# --- Pydantic Models ---

class ChatRequest(BaseModel):
    query: str

# --- Internal RAG & Chat Logic (Refactored) ---

async def get_simple_response(query: str) -> AsyncGenerator[str, None]:
    """
    (Internal) Streams a simple, non-RAG response with a strict system prompt.
    """
    # Strict prompt to eliminate conversational filler (e.g., for "2+2")
    messages = [
        {"role": "system", "content": "You are an extremely concise assistant. Respond to the user's query with the shortest possible answer, containing ONLY the result or direct text requested. Do NOT include any greetings, conversational filler, or introductory phrases."},
        {"role": "user", "content": query}
    ]

    # 1. Send a "no sources" message
    yield json.dumps({"event": "sources", "data": []}) + "\n"
    
    # Status before final LLM call
    yield json.dumps({"event": "status", "data": "Getting the response"}) + "\n"

    # 2. Stream the LLM response
    stream = ollama_client.chat(
        model=ROUTER_LLM_MODEL, # Use the fast model
        messages=messages,      # Use the new messages list
        stream=True
    )
    
    for chunk in stream:
        if token := chunk['message']['content']:
            yield json.dumps({"event": "token", "data": token}) + "\n"

async def get_rag_response(query: str) -> AsyncGenerator[str, None]:
    """
    (Internal) Runs the full RAG pipeline and streams the response.
    """
    if not all([embedding_model, rerank_model, collection]):
        yield json.dumps({"event": "error", "data": "Server resources not initialized."}) + "\n"
        return

    # Status 1: Starting Search
    yield json.dumps({"event": "status", "data": "Searching vector database"}) + "\n"

    # --- Stage 1: "Dumb" Vector Search ---
    try:
        query_embedding = embedding_model.encode(query).tolist()
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=TOP_K_SEARCH
        )
    except Exception as e:
        print(f"Error querying ChromaDB: {e}")
        yield json.dumps({"event": "error", "data": f"Error searching vector store: {e}"}) + "\n"
        return

    search_hits = []
    if results['documents']:
        for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
            search_hits.append({"content": doc, "source": meta.get("source", "unknown")})

    if not search_hits:
        yield json.dumps({"event": "sources", "data": []}) + "\n"
        yield json.dumps({"event": "token", "data": "I couldn't find any relevant information."}) + "\n"
        return

    # Status 2: Starting Re-ranking
    yield json.dumps({"event": "status", "data": "Re-ranking search results"}) + "\n"

    # --- Stage 2: "Smart" Re-ranking ---
    rerank_pairs = [[query, hit['content']] for hit in search_hits]
    scores = rerank_model.predict(rerank_pairs)
    reranked_hits = sorted(zip(scores, search_hits), key=lambda x: x[0], reverse=True)
    final_context_hits = reranked_hits[:TOP_K_RERANK]
    
    # **Metadata Feature:** Get the unique sources
    sources = list(set(hit[1]['source'] for hit in final_context_hits))
    
    # 1. --- YIELD SOURCES (FIRST) ---
    yield json.dumps({"event": "sources", "data": sources}) + "\n"
    
    # Status 3: Starting Synthesis
    yield json.dumps({"event": "status", "data": "Getting the response"}) + "\n"

    # --- Synthesize Answer ---
    context_str = "\n\n---\n\n".join([hit[1]['content'] for hit in final_context_hits])
    
    system_prompt = """
    You are a helpful assistant. Answer the user's query based *only* on the
    provided context. If the context does not contain the answer, say so.
    Do not use any outside knowledge.
    """
    user_prompt = f"Context:\n{context_str}\n\nQuery:\n{query}"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    # 2. --- YIELD LLM TOKENS (SECOND) ---
    stream = ollama_client.chat(
        model=RAG_LLM_MODEL,
        messages=messages,
        stream=True
    )
    
    for chunk in stream:
        if token := chunk['message']['content']:
            yield json.dumps({"event": "token", "data": token}) + "\n"


async def get_routed_response(query: str) -> AsyncGenerator[str, None]:
    """
    (Internal) The "Agentic Brain."
    1. Classifies the query.
    2. Calls the correct internal function (simple or RAG).
    3. Yields the resulting stream.
    """
    # Status 0: Initializing
    yield json.dumps({"event": "status", "data": "Getting to your query"}) + "\n"
    
    if not ollama_client:
        yield json.dumps({"event": "error", "data": "Ollama client not initialized."}) + "\n"
        return

    # Status 1: Routing
    yield json.dumps({"event": "status", "data": "Routing to the valid LLM"}) + "\n"

    routing_prompt = f"""
    You are a routing agent. Your job is to classify a user query.
    Is the query a simple conversational greeting, chit-chat, or a generic question like 'how are you?' OR is it a complex question that requires searching the provided knowledge base?
    
    If the user is asking about the code, the project, or the files, respond with 'complex'.
    If the user is asking a social question, responding with 'simple'.
    
    Respond with only the word 'simple' or 'complex'.
    
    Query: "{query}"
    """
    
    try:
        response = ollama_client.chat(
            model=ROUTER_LLM_MODEL,
            messages=[{"role": "user", "content": routing_prompt}],
            stream=False, # We need this full response to make a decision
            options={"temperature": 0.0}
        )
        route = response['message']['content'].lower().strip()
    except Exception as e:
        print(f"Error calling routing LLM: {e}")
        route = "complex" # Default to complex on failure

    # Status 2: Processing
    yield json.dumps({"event": "status", "data": "Processing the request"}) + "\n"

    # 2. The "if" statement
    if "simple" in route:
        print(f"Routing query to: simple")
        async for chunk in get_simple_response(query):
            yield chunk
    else:
        print(f"Routing query to: RAG")
        async for chunk in get_rag_response(query):
            yield chunk

# --- API Endpoint ---

@app.post("/chat_router")
async def chat_router(request: ChatRequest):
    """
    The *only* endpoint for the client.
    It streams an ndjson (Newline DelimDited JSON) response.
    
    - First packet: `{"event": "sources", "data": [...]}`
    - Subsequent packets: `{"event": "token", "data": "..."}`
    - Status packets: `{"event": "status", "data": "..."}`
    - Error packets: `{"event": "error", "data": "..."}`
    """
    return StreamingResponse(
        get_routed_response(request.query), 
        media_type="application/x-ndjson"
    )

if __name__ == "__main__":
    import uvicorn
    print("Starting server... (Run from 'rag_server' dir with 'uvicorn main:app --reload')")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
