import re
from typing import Annotated, Optional
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from config import OLLAMA_BASE_URL, LLM_MODEL, LLM_PROVIDER, OPENAI_API_KEY
from tools import extract_date, validate_name, validate_phone, validate_email, search_documents
from prompts import INTENT_CLASSIFIER_MID_APPOINTMENT, INTENT_CLASSIFIER, RAG_SYSTEM, RAG_NO_DOCS


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    session_id: str
    intent: str
    appointment: dict
    appointment_step: Optional[str]
    retrieved_docs: list[str]
    response: str


def get_llm():
    if LLM_PROVIDER == "openai":
        return ChatOpenAI(model=LLM_MODEL, api_key=OPENAI_API_KEY, temperature=0, max_tokens=256)
    return ChatOllama(base_url=OLLAMA_BASE_URL, model=LLM_MODEL, temperature=0, num_predict=256)


# --- Nodes ---

def intent_router(state: State) -> State:
    """Classify user intent into doc_query, appointment, or general."""
    # If we're mid-appointment and user seems to be providing info, keep in appointment flow
    if state.get("appointment_step"):
        last_msg = state["messages"][-1].content if state["messages"] else ""
        # Quick check: if user is asking about documents, let them
        check_llm = get_llm()
        check_resp = check_llm.invoke([
            SystemMessage(content=INTENT_CLASSIFIER_MID_APPOINTMENT),
            HumanMessage(content=last_msg),
        ])
        intent = check_resp.content.strip().lower()
        if intent == "doc_query":
            return {**state, "intent": "doc_query"}
        return {**state, "intent": "appointment"}

    last_msg = state["messages"][-1].content if state["messages"] else ""
    llm = get_llm()
    resp = llm.invoke([
        SystemMessage(content=INTENT_CLASSIFIER),
        HumanMessage(content=last_msg),
    ])
    intent = resp.content.strip().lower()
    if intent != "appointment":
        intent = "doc_query"
    return {**state, "intent": intent}


def rag_node(state: State) -> State:
    """Answer user question using retrieved document context."""
    last_msg = state["messages"][-1].content if state["messages"] else ""
    session_id = state.get("session_id", "default")

    # Search documents
    result = search_documents.invoke({"query": last_msg, "session_id": session_id})

    if not result.get("results"):
        llm = get_llm()
        resp = llm.invoke([
            SystemMessage(content=RAG_NO_DOCS),
            HumanMessage(content=last_msg),
        ])
        return {
            **state,
            "retrieved_docs": [],
            "response": resp.content,
            "messages": [AIMessage(content=resp.content)],
        }

    context_parts = [doc["content"] for doc in result["results"]]
    context = "\n\n---\n\n".join(context_parts)

    llm = get_llm()
    resp = llm.invoke([
        SystemMessage(content=RAG_SYSTEM.format(context=context)),
        HumanMessage(content=last_msg),
    ])

    response = resp.content
    return {
        **state,
        "retrieved_docs": context_parts,
        "response": response,
        "messages": [AIMessage(content=response)],
    }


APPOINTMENT_TOOLS = [extract_date, validate_name, validate_phone, validate_email]
FIELD_ORDER = ["name", "phone", "email", "date"]
FIELD_PROMPTS = {
    "name": "What is your full name?",
    "phone": "What is your phone number?",
    "email": "What is your email address?",
    "date": "When would you like to schedule the appointment? (e.g., 'next Monday', 'March 25th', 'in two days')",
}
FIELD_TOOL = {
    "name": "validate_name",
    "phone": "validate_phone",
    "email": "validate_email",
    "date": "extract_date",
}


def appointment_node(state: State) -> State:
    """Handle appointment booking with tool-based validation."""
    appointment = state.get("appointment") or {}
    step = state.get("appointment_step")
    last_msg = state["messages"][-1].content if state["messages"] else ""

    # If no step yet, start the flow
    if not step:
        step = _next_missing_field(appointment)
        if not step:
            return _confirm_booking(state, appointment)
        prompt = f"I'd be happy to help you book an appointment! {FIELD_PROMPTS[step]}"
        return {
            **state,
            "appointment": appointment,
            "appointment_step": step,
            "response": prompt,
            "messages": [AIMessage(content=prompt)],
        }

    # We're collecting a field — validate user input via the appropriate tool
    tool_name = FIELD_TOOL[step]
    tool_map = {t.name: t for t in APPOINTMENT_TOOLS}
    tool_fn = tool_map[tool_name]

    if step == "date":
        result = tool_fn.invoke({"text": last_msg})
    elif step == "name":
        result = tool_fn.invoke({"name": last_msg})
    elif step == "phone":
        result = tool_fn.invoke({"phone": last_msg})
    elif step == "email":
        result = tool_fn.invoke({"email": last_msg})

    if not result.get("valid"):
        error = result.get("error", "Invalid input.")
        response = f"{error} Please try again. {FIELD_PROMPTS[step]}"
        return {
            **state,
            "appointment": appointment,
            "appointment_step": step,
            "response": response,
            "messages": [AIMessage(content=response)],
        }

    # Store validated value
    if step == "name":
        appointment["name"] = result["name"]
    elif step == "phone":
        appointment["phone"] = result["phone"]
    elif step == "email":
        appointment["email"] = result["email"]
    elif step == "date":
        appointment["date"] = result["date"]

    # Move to next field
    next_step = _next_missing_field(appointment)
    if not next_step:
        return _confirm_booking(state, appointment)

    response = f"Got it! {FIELD_PROMPTS[next_step]}"
    return {
        **state,
        "appointment": appointment,
        "appointment_step": next_step,
        "response": response,
        "messages": [AIMessage(content=response)],
    }


def _next_missing_field(appointment: dict) -> Optional[str]:
    for field in FIELD_ORDER:
        if field not in appointment:
            return field
    return None


def _confirm_booking(state: State, appointment: dict) -> State:
    response = (
        f"Your appointment has been booked successfully!\n\n"
        f"**Name:** {appointment['name']}\n"
        f"**Phone:** {appointment['phone']}\n"
        f"**Email:** {appointment['email']}\n"
        f"**Date:** {appointment['date']}\n\n"
        f"Is there anything else I can help you with?"
    )
    return {
        **state,
        "appointment": appointment,
        "appointment_step": None,
        "response": response,
        "messages": [AIMessage(content=response)],
    }


# --- Routing ---

def route_intent(state: State) -> str:
    intent = state.get("intent", "doc_query")
    if intent == "appointment":
        return "appointment_node"
    return "rag_node"


# --- Build Graph ---

def build_graph():
    graph = StateGraph(State)

    graph.add_node("intent_router", intent_router)
    graph.add_node("rag_node", rag_node)
    graph.add_node("appointment_node", appointment_node)

    graph.set_entry_point("intent_router")
    graph.add_conditional_edges("intent_router", route_intent, {
        "rag_node": "rag_node",
        "appointment_node": "appointment_node",
    })
    graph.add_edge("rag_node", END)
    graph.add_edge("appointment_node", END)

    return graph.compile()


chatbot_graph = build_graph()
