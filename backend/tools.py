import re
import dateparser
from dateparser.search import search_dates
from langchain_core.tools import tool
import chromadb
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from config import CHROMA_PERSIST_DIR, OLLAMA_BASE_URL, EMBED_MODEL, LLM_PROVIDER, OPENAI_API_KEY


@tool
def extract_date(text: str) -> dict:
    """Extract a date from natural language text and return in YYYY-MM-DD format.
    Handles inputs like 'next Monday', 'coming Tuesday', 'in two days', 'March 25th', etc.
    """
    parsed = dateparser.parse(text, settings={"PREFER_DATES_FROM": "future"})
    if parsed:
        return {"valid": True, "date": parsed.strftime("%Y-%m-%d")}

    # Fallback: search_dates handles "next tuesday", "coming monday" etc.
    results = search_dates(text, settings={"PREFER_DATES_FROM": "future"})
    if results:
        return {"valid": True, "date": results[0][1].strftime("%Y-%m-%d")}

    return {"valid": False, "error": f"Could not parse a date from: '{text}'"}


@tool
def validate_name(name: str) -> dict:
    """Validate a person's name. Must be non-empty, at least 2 characters, and contain only letters and spaces."""
    name = name.strip()
    if len(name) < 2:
        return {"valid": False, "error": "Name must be at least 2 characters long."}
    if not re.match(r"^[a-zA-Z\s]+$", name):
        return {"valid": False, "error": "Name must contain only letters and spaces."}
    return {"valid": True, "name": name}


@tool
def validate_phone(phone: str) -> dict:
    """Validate a phone number. Must contain at least 10 digits after stripping non-numeric characters."""
    digits = re.sub(r"\D", "", phone)
    if len(digits) < 10:
        return {"valid": False, "error": f"Phone number must have at least 10 digits. Got {len(digits)} digits."}
    return {"valid": True, "phone": digits}


@tool
def validate_email(email: str) -> dict:
    """Validate an email address format. Must contain @ with a domain that has a dot."""
    email = email.strip()
    if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email):
        return {"valid": False, "error": "Invalid email format. Please provide a valid email like user@example.com."}
    return {"valid": True, "email": email}


@tool
def search_documents(query: str, session_id: str) -> dict:
    """Search uploaded documents for content relevant to the query. Uses semantic similarity search."""
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    collection_name = f"docs_{session_id}"

    try:
        collection = client.get_collection(name=collection_name)
    except Exception:
        return {"results": [], "message": "No documents have been uploaded yet."}

    if LLM_PROVIDER == "openai":
        embedder = OpenAIEmbeddings(model=EMBED_MODEL, api_key=OPENAI_API_KEY)
    else:
        embedder = OllamaEmbeddings(base_url=OLLAMA_BASE_URL, model=EMBED_MODEL)
    query_embedding = embedder.embed_query(query)
    results = collection.query(query_embeddings=[query_embedding], n_results=5)

    if not results["documents"] or not results["documents"][0]:
        return {"results": [], "message": "No relevant content found in uploaded documents."}

    docs = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        docs.append({"content": doc, "source": meta.get("source", "unknown")})

    return {"results": docs}
