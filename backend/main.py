import os
import uuid
import tempfile
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from graph import chatbot_graph
from rag import ingest_document

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


@app.post("/upload", response_model=UploadResponse)
async def upload(session_id: str = Form(...), file: UploadFile = File(...)):
    # Save uploaded file to temp location
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        chunks = ingest_document(tmp_path, session_id)
    finally:
        os.unlink(tmp_path)

    return UploadResponse(filename=file.filename, chunks=chunks)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
