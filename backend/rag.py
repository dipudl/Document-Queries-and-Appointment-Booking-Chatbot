import os
import chromadb
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from config import OLLAMA_BASE_URL, EMBED_MODEL, CHROMA_PERSIST_DIR, CHUNK_SIZE, CHUNK_OVERLAP


def get_embeddings():
    return OllamaEmbeddings(base_url=OLLAMA_BASE_URL, model=EMBED_MODEL)


def ingest_document(file_path: str, session_id: str) -> int:
    """Load a document, chunk it, embed it, and store in ChromaDB. Returns chunk count."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext in (".txt", ".md"):
        loader = TextLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}. Use PDF, TXT or MD.")

    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)

    if not chunks:
        return 0

    embedder = get_embeddings()
    texts = [chunk.page_content for chunk in chunks]
    metadatas = [{"source": os.path.basename(file_path), "page": chunk.metadata.get("page", 0)} for chunk in chunks]

    embeddings = embedder.embed_documents(texts)

    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    collection_name = f"docs_{session_id}"
    collection = client.get_or_create_collection(name=collection_name)

    ids = [f"{os.path.basename(file_path)}_{i}" for i in range(len(texts))]
    collection.upsert(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)

    return len(chunks)
