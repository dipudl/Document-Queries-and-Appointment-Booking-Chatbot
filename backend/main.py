import os
import uuid
import logging
import tempfile
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage
from graph import chatbot_graph
from rag import ingest_document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Document Queries & Appointment Booking Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store
sessions: dict[str, dict] = {}


def get_session(session_id: str) -> dict:
    if session_id not in sessions:
        sessions[session_id] = {
            "messages": [],
            "session_id": session_id,
            "intent": "",
            "appointment": {},
            "appointment_step": None,
            "retrieved_docs": [],
            "response": "",
        }
    return sessions[session_id]


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    response: str
    appointment: dict | None = None


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    logger.info("Chat request [session=%s]: %s", req.session_id, req.message[:100])
    session = get_session(req.session_id)

    # Add user message
    session["messages"].append(HumanMessage(content=req.message))

    # Run the graph
    result = chatbot_graph.invoke({
        "messages": session["messages"],
        "session_id": session["session_id"],
        "intent": session["intent"],
        "appointment": session["appointment"],
        "appointment_step": session["appointment_step"],
        "retrieved_docs": session["retrieved_docs"],
        "response": "",
    })

    # Update session state
    session["messages"] = result["messages"]
    session["intent"] = result.get("intent", "")
    session["appointment"] = result.get("appointment", {})
    session["appointment_step"] = result.get("appointment_step")
    session["retrieved_docs"] = result.get("retrieved_docs", [])

    response_text = result.get("response", "Sorry, I couldn't process that.")
    appointment = result.get("appointment") or None
    if appointment and not appointment:
        appointment = None

    return ChatResponse(response=response_text, appointment=appointment if appointment else None)


class UploadResponse(BaseModel):
    filename: str
    chunks: int


@app.get("/history/{session_id}")
async def history(session_id: str):
    session = get_session(session_id)
    msgs = []
    for msg in session["messages"]:
        if isinstance(msg, HumanMessage):
            msgs.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            msgs.append({"role": "bot", "content": msg.content})
    return msgs


ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md"}


@app.post("/upload", response_model=UploadResponse)
async def upload(session_id: str = Form(...), file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename)[1].lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}. Use PDF, TXT or MD.")

    # Save uploaded file to temp location
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        chunks = ingest_document(tmp_path, session_id)
    finally:
        os.unlink(tmp_path)

    logger.info("Uploaded [session=%s]: %s (%d chunks)", session_id, file.filename, chunks)
    return UploadResponse(filename=file.filename, chunks=chunks)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
