import os
from typing import List, Dict

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from openai import OpenAI


# -----------------------------
# Basic config
# -----------------------------

FAQ_FILE = "faq.txt"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "gpt-4o-mini"

TOP_K = 3
LOW_CONFIDENCE_THRESHOLD = 0.35

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI(title="Basic Support RAG Assistant")


# -----------------------------
# In-memory storage
# -----------------------------

chunks: List[str] = []
chunk_embeddings = None
chat_history: Dict[str, List[Dict[str, str]]] = {}

embedding_model = SentenceTransformer(EMBEDDING_MODEL)


# -----------------------------
# API models
# -----------------------------

class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    answer: str
    confidence: float
    escalated: bool
    sources: List[str]


# -----------------------------
# Load FAQ
# -----------------------------

def load_faq():
    global chunks, chunk_embeddings

    with open(FAQ_FILE, "r", encoding="utf-8") as f:
        text = f.read()

    # Simple chunking: split each FAQ by blank line
    chunks = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]

    chunk_embeddings = embedding_model.encode(
        chunks,
        normalize_embeddings=True
    )


# Load docs when server starts
load_faq()


# -----------------------------
# Retrieval
# -----------------------------

def retrieve(query: str):
    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True
    )[0]

    scores = np.dot(chunk_embeddings, query_embedding)

    top_indices = scores.argsort()[::-1][:TOP_K]

    results = []

    for idx in top_indices:
        results.append({
            "text": chunks[idx],
            "score": float(scores[idx])
        })

    return results


# -----------------------------
# LLM answering
# -----------------------------

def answer_from_context(question: str, retrieved_chunks: List[Dict]):
    context = "\n\n".join(
        [f"Source {i+1}:\n{item['text']}" for i, item in enumerate(retrieved_chunks)]
    )

    prompt = f"""
You are a support assistant.

Answer the user's question using only the context below.
If the context does not contain the answer, say:
"I don't have enough information to answer that."

Context:
{context}

Question:
{question}

Answer:
"""

    response = client.chat.completions.create(
        model=LLM_MODEL,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "You answer only from the provided context. Do not guess."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content


# -----------------------------
# Chat endpoint
# -----------------------------

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    session_id = request.session_id
    message = request.message

    # Save user message
    if session_id not in chat_history:
        chat_history[session_id] = []

    chat_history[session_id].append({
        "role": "user",
        "content": message
    })

    # Retrieve relevant FAQ chunks
    retrieved_chunks = retrieve(message)

    confidence = retrieved_chunks[0]["score"]

    # Escalate if retrieval confidence is low
    if confidence < LOW_CONFIDENCE_THRESHOLD:
        answer = "I don't have enough information to answer that. I am escalating this to a human support agent."

        chat_history[session_id].append({
            "role": "assistant",
            "content": answer
        })

        return ChatResponse(
            answer=answer,
            confidence=confidence,
            escalated=True,
            sources=[]
        )

    # Generate answer from retrieved context
    answer = answer_from_context(message, retrieved_chunks)

    # Escalate if model says it does not know
    if "don't have enough information" in answer.lower():
        escalated = True
        answer = answer + "\n\nEscalating this to a human support agent."
    else:
        escalated = False

    # Save assistant response
    chat_history[session_id].append({
        "role": "assistant",
        "content": answer
    })

    return ChatResponse(
        answer=answer,
        confidence=confidence,
        escalated=escalated,
        sources=[item["text"] for item in retrieved_chunks]
    )


# -----------------------------
# Health check
# -----------------------------

@app.get("/")
def home():
    return {
        "message": "Support RAG API is running",
        "num_chunks": len(chunks)
    }


'''
curl -X POST "http://127.0.0.1:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "abc123",
    "message": "How do I reset my password?"
  }'
'''

import os
from typing import List, Dict

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from openai import OpenAI


# -----------------------------
# Basic config
# -----------------------------

FAQ_FILE = "faq.txt"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "gpt-4o-mini"

TOP_K = 3
LOW_CONFIDENCE_THRESHOLD = 0.35

# Only keep recent conversation turns
MAX_HISTORY_MESSAGES = 6

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI(title="Basic Support RAG Assistant")


# -----------------------------
# In-memory storage
# -----------------------------

chunks: List[str] = []
chunk_embeddings = None

# session_id -> list of messages
chat_history: Dict[str, List[Dict[str, str]]] = {}

embedding_model = SentenceTransformer(EMBEDDING_MODEL)


# -----------------------------
# API models
# -----------------------------

class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    answer: str
    confidence: float
    escalated: bool
    sources: List[str]
    memory_used: List[Dict[str, str]]


# -----------------------------
# Load FAQ
# -----------------------------

def load_faq():
    global chunks, chunk_embeddings

    with open(FAQ_FILE, "r", encoding="utf-8") as f:
        text = f.read()

    # Simple chunking: split each FAQ by blank line
    chunks = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]

    chunk_embeddings = embedding_model.encode(
        chunks,
        normalize_embeddings=True
    )


load_faq()


# -----------------------------
# Conversation memory helpers
# -----------------------------

def get_recent_history(session_id: str) -> List[Dict[str, str]]:
    """
    Returns the most recent conversation messages for this session.
    """
    history = chat_history.get(session_id, [])
    return history[-MAX_HISTORY_MESSAGES:]


def format_history_for_prompt(history: List[Dict[str, str]]) -> str:
    """
    Converts chat history into plain text for the LLM prompt.
    """
    if not history:
        return "No previous conversation."

    lines = []

    for msg in history:
        role = msg["role"]
        content = msg["content"]
        lines.append(f"{role.upper()}: {content}")

    return "\n".join(lines)


def build_retrieval_query(message: str, history: List[Dict[str, str]]) -> str:
    """
    Uses recent conversation context plus the current message for retrieval.

    Example:
    Previous user: "Can I integrate with Salesforce?"
    Current user: "How do I set it up?"

    Searching only "How do I set it up?" is weak.
    Searching with recent history gives the retriever more context.
    """
    recent_context = format_history_for_prompt(history)

    retrieval_query = f"""
Recent conversation:
{recent_context}

Current user message:
{message}
"""

    return retrieval_query.strip()


def save_message(session_id: str, role: str, content: str):
    """
    Saves a message to the in-memory chat history.
    """
    if session_id not in chat_history:
        chat_history[session_id] = []

    chat_history[session_id].append({
        "role": role,
        "content": content
    })

    # Keep memory small
    chat_history[session_id] = chat_history[session_id][-MAX_HISTORY_MESSAGES:]


# -----------------------------
# Retrieval
# -----------------------------

def retrieve(query: str):
    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True
    )[0]

    scores = np.dot(chunk_embeddings, query_embedding)

    top_indices = scores.argsort()[::-1][:TOP_K]

    results = []

    for idx in top_indices:
        results.append({
            "text": chunks[idx],
            "score": float(scores[idx])
        })

    return results


# -----------------------------
# LLM answering
# -----------------------------

def answer_from_context(
    question: str,
    retrieved_chunks: List[Dict],
    history: List[Dict[str, str]]
):
    context = "\n\n".join(
        [
            f"Source {i + 1}:\n{item['text']}"
            for i, item in enumerate(retrieved_chunks)
        ]
    )

    conversation_memory = format_history_for_prompt(history)

    prompt = f"""
You are a support assistant.

You must answer using only the FAQ context below.

You may use the conversation history only to understand what the user is referring to.
Do not use the conversation history as a factual source.
Only the FAQ context is allowed as factual evidence.

If the FAQ context does not contain the answer, say:
"I don't have enough information to answer that."

Conversation history:
{conversation_memory}

FAQ context:
{context}

Current user question:
{question}

Answer:
"""

    response = client.chat.completions.create(
        model=LLM_MODEL,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "You answer only from the provided FAQ context. Do not guess."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content


# -----------------------------
# Chat endpoint
# -----------------------------

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    session_id = request.session_id
    message = request.message

    # 1. Get previous memory before adding current message
    recent_history = get_recent_history(session_id)

    # 2. Build a memory-aware retrieval query
    retrieval_query = build_retrieval_query(message, recent_history)

    # 3. Retrieve FAQ chunks
    retrieved_chunks = retrieve(retrieval_query)

    confidence = retrieved_chunks[0]["score"]

    # 4. If retrieval is weak, escalate
    if confidence < LOW_CONFIDENCE_THRESHOLD:
        answer = (
            "I don't have enough information to answer that. "
            "I am escalating this to a human support agent."
        )

        save_message(session_id, "user", message)
        save_message(session_id, "assistant", answer)

        return ChatResponse(
            answer=answer,
            confidence=confidence,
            escalated=True,
            sources=[],
            memory_used=recent_history
        )

    # 5. Generate answer using FAQ context + conversation memory
    answer = answer_from_context(
        question=message,
        retrieved_chunks=retrieved_chunks,
        history=recent_history
    )

    # 6. Escalate if the LLM still cannot answer from context
    if "don't have enough information" in answer.lower():
        escalated = True
        answer = answer + "\n\nEscalating this to a human support agent."
    else:
        escalated = False

    # 7. Save current turn to memory
    save_message(session_id, "user", message)
    save_message(session_id, "assistant", answer)

    return ChatResponse(
        answer=answer,
        confidence=confidence,
        escalated=escalated,
        sources=[item["text"] for item in retrieved_chunks],
        memory_used=recent_history
    )


# -----------------------------
# Basic endpoints
# -----------------------------

@app.get("/")
def home():
    return {
        "message": "Support RAG API is running",
        "num_chunks": len(chunks)
    }


@app.get("/history/{session_id}")
def get_history(session_id: str):
    return {
        "session_id": session_id,
        "history": chat_history.get(session_id, [])
    }


