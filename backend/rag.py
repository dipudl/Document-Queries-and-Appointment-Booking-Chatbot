import os
import logging
import chromadb
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from config import OLLAMA_BASE_URL, EMBED_MODEL, CHROMA_PERSIST_DIR, CHUNK_SIZE, CHUNK_OVERLAP, LLM_PROVIDER, OPENAI_API_KEY


logger = logging.getLogger(__name__)


def get_embeddings():
    if LLM_PROVIDER == "openai":
        return OpenAIEmbeddings(model=EMBED_MODEL, api_key=OPENAI_API_KEY)
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
    logger.info("Loaded %s: %d pages", os.path.basename(file_path), len(documents))

    # Merge all pages into a single text so chunks can span across pages
    full_text = "\n".join(doc.page_content for doc in documents)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_text(full_text)

    if not chunks:
        return 0

    embedder = get_embeddings()
    texts = chunks
    source = os.path.basename(file_path)
    metadatas = [{"source": source} for _ in chunks]

    embeddings = embedder.embed_documents(texts)

    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    collection_name = f"docs_{session_id}"
    collection = client.get_or_create_collection(name=collection_name)

    ids = [f"{os.path.basename(file_path)}_{i}" for i in range(len(texts))]
    collection.upsert(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)
    logger.info("Stored %d chunks in collection %s", len(chunks), collection_name)

    return len(chunks)
