import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredFileLoader  # <- FIXED IMPORT
import chromadb
from sentence_transformers import SentenceTransformer

# Configuration
ROOT_DIR = "."  # Start scanning from the project root
DB_DIR = "db"   # Where to save the DB
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Define what to ingest ---
ALLOWED_EXTENSIONS = [".py", ".md", ".txt"]
IGNORE_DIRECTORIES = ["venv", "__pycache__", "db", ".git", "client", "rag_server"]

# Document loader ---
def load_documents(root_dir):
    """Loads all allowed files from the root directory, skipping ignored ones."""
    documents = []
    
    # Add root files
    for filename in os.listdir(root_dir):
        if any(filename.endswith(ext) for ext in ALLOWED_EXTENSIONS) and os.path.isfile(os.path.join(root_dir, filename)):
            file_path = os.path.join(root_dir, filename)
            relative_path = os.path.relpath(file_path, root_dir)
            try:
                loader = UnstructuredFileLoader(file_path)  # <- FIXED
                docs = loader.load()
                for doc in docs:
                    doc.metadata["source"] = relative_path
                documents.extend(docs)
                print(f"Loaded document: {relative_path}")
            except Exception as e:
                print(f"Error loading {relative_path}: {e}")

    # Walk through allowed subdirectories
    for dir_to_scan in ["client", "rag_server"]:
        dir_path = os.path.join(root_dir, dir_to_scan)
        if not os.path.exists(dir_path):
            continue
        for dirpath, dirnames, filenames in os.walk(dir_path):
            dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRECTORIES]
            for filename in filenames:
                if any(filename.endswith(ext) for ext in ALLOWED_EXTENSIONS):
                    file_path = os.path.join(dirpath, filename)
                    relative_path = os.path.relpath(file_path, root_dir)
                    try:
                        loader = UnstructuredFileLoader(file_path)
                        docs = loader.load()
                        for doc in docs:
                            doc.metadata["source"] = relative_path
                        documents.extend(docs)
                        print(f"Loaded document: {relative_path}")
                    except Exception as e:
                        print(f"Error loading {relative_path}: {e}")

    return documents

def chunk_documents(documents):
    """Splits loaded documents into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", "", "\n\nclass ", "\ndef ", "\nimport "]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks")
    return chunks

def setup_vector_store(chunks, clear_old=False):
    """Creates and persists a ChromaDB vector store."""
    print("Setting up vector store...")
    client = chromadb.PersistentClient(path=DB_DIR)
    collection_name = "rag_collection"

    if clear_old:
        try:
            print(f"Clearing old collection '{collection_name}'...")
            client.delete_collection(name=collection_name)
        except Exception as e:
            print(f"Collection didn't exist or error clearing: {e}")
    
    collection = client.get_or_create_collection(name=collection_name)
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    documents = []
    metadatas = []
    ids = []

    for i, chunk in enumerate(chunks):
        documents.append(chunk.page_content)
        metadatas.append({"source": chunk.metadata.get("source", "unknown")})
        ids.append(str(i))

    print(f"Generating embeddings for {len(documents)} chunks...")
    embeddings = embedding_model.encode(documents, show_progress_bar=True)
    
    collection.add(
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    print(f"Successfully added {len(ids)} chunks to the '{collection_name}' collection in {DB_DIR}")

def main():
    documents = load_documents(ROOT_DIR)
    if not documents:
        print("No documents loaded. Exiting.")
        return
    chunks = chunk_documents(documents)
    if not chunks:
        print("No chunks created. Exiting.")
        return
    setup_vector_store(chunks, clear_old=True)
    print("\nIngestion complete. Your bot can now talk about its own code.")

if __name__ == "__main__":
    main()
