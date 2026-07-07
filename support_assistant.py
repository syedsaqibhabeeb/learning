import os
import re
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Literal
from dataclasses import dataclass

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from openai import OpenAI


# ============================================================
# Config
# ============================================================

DOCS_DIR = os.getenv("DOCS_DIR", "faq_docs")

EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "sentence-transformers/all-MiniLM-L6-v2"
)

LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

TOP_K = int(os.getenv("TOP_K", "4"))

# Cosine similarity threshold.
# Tune this based on your documentation quality.
ESCALATION_THRESHOLD = float(os.getenv("ESCALATION_THRESHOLD", "0.35"))

MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "8"))

CHUNK_SIZE_CHARS = int(os.getenv("CHUNK_SIZE_CHARS", "900"))
CHUNK_OVERLAP_CHARS = int(os.getenv("CHUNK_OVERLAP_CHARS", "150"))


# ============================================================
# Data Models
# ============================================================

@dataclass
class RawDocument:
    source: str
    text: str


@dataclass
class Chunk:
    chunk_id: str
    source: str
    text: str


@dataclass
class SearchResult:
    chunk: Chunk
    score: float


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)


class SourceResponse(BaseModel):
    chunk_id: str
    source: str
    score: float
    text_preview: str


class ChatResponse(BaseModel):
    session_id: str
    status: Literal["answered", "escalated"]
    answer: str
    confidence: float
    sources: List[SourceResponse]
    escalation_reason: Optional[str] = None


# ============================================================
# Document Loading
# ============================================================

def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def load_faq_documents(docs_dir: str) -> List[RawDocument]:
    path = Path(docs_dir)

    if not path.exists():
        raise FileNotFoundError(
            f"Documentation folder '{docs_dir}' does not exist."
        )

    docs: List[RawDocument] = []

    for file_path in sorted(path.rglob("*")):
        if file_path.suffix.lower() not in {".txt", ".md"}:
            continue

        text = file_path.read_text(encoding="utf-8", errors="ignore")
        text = normalize_text(text)

        if text:
            docs.append(
                RawDocument(
                    source=str(file_path),
                    text=text
                )
            )

    if not docs:
        raise ValueError(
            f"No .txt or .md FAQ documents found inside '{docs_dir}'."
        )

    return docs


def chunk_document(
    doc: RawDocument,
    chunk_size_chars: int = CHUNK_SIZE_CHARS,
    overlap_chars: int = CHUNK_OVERLAP_CHARS
) -> List[Chunk]:
    paragraphs = [
        p.strip()
        for p in re.split(r"\n\s*\n", doc.text)
        if p.strip()
    ]

    chunks: List[Chunk] = []
    current = ""

    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph

        if len(candidate) <= chunk_size_chars:
            current = candidate
        else:
            if current:
                chunk_id = str(uuid.uuid4())[:8]
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        source=doc.source,
                        text=current
                    )
                )

                overlap = current[-overlap_chars:] if overlap_chars > 0 else ""
                current = f"{overlap}\n\n{paragraph}".strip()
            else:
                # Paragraph itself is too large, split it directly.
                start = 0
                while start < len(paragraph):
                    piece = paragraph[start:start + chunk_size_chars]
                    chunk_id = str(uuid.uuid4())[:8]
                    chunks.append(
                        Chunk(
                            chunk_id=chunk_id,
                            source=doc.source,
                            text=piece
                        )
                    )
                    start += chunk_size_chars - overlap_chars

                current = ""

    if current:
        chunk_id = str(uuid.uuid4())[:8]
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                source=doc.source,
                text=current
            )
        )

    return chunks


def build_chunks(docs: List[RawDocument]) -> List[Chunk]:
    chunks: List[Chunk] = []

    for doc in docs:
        chunks.extend(chunk_document(doc))

    return chunks


# ============================================================
# Vector Store
# ============================================================

class InMemoryVectorStore:
    """
    Lightweight semantic vector store.

    For production:
    - Replace this with FAISS, Chroma, Pinecone, Weaviate, Milvus, or pgvector.
    - Persist embeddings.
    - Add document versioning.
    """

    def __init__(self, chunks: List[Chunk], embedding_model: SentenceTransformer):
        if not chunks:
            raise ValueError("Cannot initialize vector store with zero chunks.")

        self.chunks = chunks
        self.embedding_model = embedding_model

        texts = [chunk.text for chunk in chunks]

        self.embeddings = self.embedding_model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype("float32")

    def search(self, query: str, top_k: int = TOP_K) -> List[SearchResult]:
        query_embedding = self.embedding_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype("float32")[0]

        scores = self.embeddings @ query_embedding

        top_indices = np.argsort(scores)[::-1][:top_k]

        results: List[SearchResult] = []

        for idx in top_indices:
            results.append(
                SearchResult(
                    chunk=self.chunks[int(idx)],
                    score=float(scores[int(idx)])
                )
            )

        return results


# ============================================================
# Conversation Memory
# ============================================================

class ConversationMemory:
    """
    Simple in-memory session history.

    For production:
    - Replace with Redis, Postgres, DynamoDB, or another durable store.
    - Add expiration.
    - Encrypt sensitive conversations.
    """

    def __init__(self):
        self.sessions: Dict[str, List[Dict[str, str]]] = {}

    def get(self, session_id: str) -> List[Dict[str, str]]:
        return self.sessions.get(session_id, [])

    def add(self, session_id: str, role: str, content: str) -> None:
        if session_id not in self.sessions:
            self.sessions[session_id] = []

        self.sessions[session_id].append(
            {
                "role": role,
                "content": content
            }
        )

        self.sessions[session_id] = self.sessions[session_id][-MAX_HISTORY_MESSAGES:]

    def build_retrieval_query(self, session_id: str, message: str) -> str:
        """
        Uses recent user turns to help resolve follow-up questions.

        Example:
        User: How do I reset my password?
        User: How long does the link last?

        The second query becomes more retrievable because memory is included.
        """
        history = self.get(session_id)

        recent_user_messages = [
            item["content"]
            for item in history
            if item["role"] == "user"
        ][-3:]

        return "\n".join(recent_user_messages + [message]).strip()

    def render_for_prompt(self, session_id: str) -> str:
        history = self.get(session_id)[-MAX_HISTORY_MESSAGES:]

        if not history:
            return "No prior conversation."

        return "\n".join(
            f"{item['role'].upper()}: {item['content']}"
            for item in history
        )


# ============================================================
# RAG Prompting
# ============================================================

def build_context_block(results: List[SearchResult]) -> str:
    blocks = []

    for result in results:
        blocks.append(
            f"[chunk_id: {result.chunk.chunk_id}]\n"
            f"[source: {result.chunk.source}]\n"
            f"[similarity_score: {result.score:.3f}]\n"
            f"{result.chunk.text}"
        )

    return "\n\n---\n\n".join(blocks)


def build_llm_messages(
    session_id: str,
    user_message: str,
    memory: ConversationMemory,
    results: List[SearchResult]
) -> List[Dict[str, str]]:

    context_block = build_context_block(results)
    history_block = memory.render_for_prompt(session_id)

    system_prompt = """
You are an internal support assistant.

You must follow these rules:

1. Answer only using the retrieved documentation context.
2. Do not use outside knowledge.
3. Conversation history may only be used to understand follow-up references.
4. Conversation history is not a source of factual truth.
5. If the retrieved context does not contain enough information, say:
   "I don't have enough information in the provided documentation to answer that."
6. Do not invent policies, links, procedures, prices, timelines, or names.
7. Include a short "Sources" section using the provided chunk IDs.
8. Keep the answer concise and helpful.
""".strip()

    user_prompt = f"""
Conversation history:
{history_block}

Retrieved documentation context:
{context_block}

Current user question:
{user_message}

Answer:
""".strip()

    return [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ]


def call_llm(messages: List[Dict[str, str]]) -> str:
    """
    Calls OpenAI-compatible chat model.

    Requires:
    export OPENAI_API_KEY="your_api_key"
    """

    if not os.getenv("OPENAI_API_KEY"):
        return (
            "I don't have enough information in the provided documentation to answer that.\n\n"
            "Sources: none\n\n"
            "Note: OPENAI_API_KEY is not configured, so the LLM could not generate a grounded answer."
        )

    client = OpenAI()

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=0.0
    )

    return response.choices[0].message.content.strip()


def should_escalate_before_generation(results: List[SearchResult]) -> Optional[str]:
    if not results:
        return "No relevant documentation chunks were retrieved."

    top_score = results[0].score

    if top_score < ESCALATION_THRESHOLD:
        return (
            f"Top retrieval score {top_score:.3f} is below confidence threshold "
            f"{ESCALATION_THRESHOLD:.3f}."
        )

    return None


def should_escalate_after_generation(answer: str) -> Optional[str]:
    normalized = answer.lower()

    insufficient_phrases = [
        "i don't have enough information",
        "provided documentation to answer that",
        "sources: none"
    ]

    if any(phrase in normalized for phrase in insufficient_phrases):
        return "The retrieved documentation did not contain enough information."

    return None


def build_sources(results: List[SearchResult]) -> List[SourceResponse]:
    return [
        SourceResponse(
            chunk_id=result.chunk.chunk_id,
            source=result.chunk.source,
            score=round(result.score, 4),
            text_preview=result.chunk.text[:240].replace("\n", " ") + "..."
        )
        for result in results
    ]


# ============================================================
# App Startup
# ============================================================

app = FastAPI(
    title="Internal Documentation Support Assistant",
    version="1.0.0"
)

memory = ConversationMemory()

embedding_model: Optional[SentenceTransformer] = None
vector_store: Optional[InMemoryVectorStore] = None


@app.on_event("startup")
def startup_event():
    global embedding_model, vector_store

    print("Loading embedding model...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print(f"Loading documents from: {DOCS_DIR}")
    docs = load_faq_documents(DOCS_DIR)

    print(f"Loaded {len(docs)} documents.")

    chunks = build_chunks(docs)

    print(f"Built {len(chunks)} chunks.")

    vector_store = InMemoryVectorStore(
        chunks=chunks,
        embedding_model=embedding_model
    )

    print("Support assistant is ready.")


# ============================================================
# API Routes
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "docs_dir": DOCS_DIR,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "llm_model": LLM_MODEL,
        "top_k": TOP_K,
        "escalation_threshold": ESCALATION_THRESHOLD
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if vector_store is None:
        raise HTTPException(
            status_code=503,
            detail="Vector store is not initialized."
        )

    clean_message = request.message.strip()

    retrieval_query = memory.build_retrieval_query(
        session_id=request.session_id,
        message=clean_message
    )

    results = vector_store.search(
        query=retrieval_query,
        top_k=TOP_K
    )

    sources = build_sources(results)

    pre_generation_escalation_reason = should_escalate_before_generation(results)

    memory.add(
        session_id=request.session_id,
        role="user",
        content=clean_message
    )

    if pre_generation_escalation_reason:
        answer = (
            "I don't have enough information in the provided documentation to answer that.\n\n"
            "I recommend escalating this to a human support owner because the matching documentation confidence is low."
        )

        memory.add(
            session_id=request.session_id,
            role="assistant",
            content=answer
        )

        return ChatResponse(
            session_id=request.session_id,
            status="escalated",
            answer=answer,
            confidence=round(results[0].score if results else 0.0, 4),
            sources=sources,
            escalation_reason=pre_generation_escalation_reason
        )

    messages = build_llm_messages(
        session_id=request.session_id,
        user_message=clean_message,
        memory=memory,
        results=results
    )

    answer = call_llm(messages)

    post_generation_escalation_reason = should_escalate_after_generation(answer)

    if post_generation_escalation_reason:
        status: Literal["answered", "escalated"] = "escalated"
        escalation_reason = post_generation_escalation_reason
    else:
        status = "answered"
        escalation_reason = None

    memory.add(
        session_id=request.session_id,
        role="assistant",
        content=answer
    )

    return ChatResponse(
        session_id=request.session_id,
        status=status,
        answer=answer,
        confidence=round(results[0].score if results else 0.0, 4),
        sources=sources,
        escalation_reason=escalation_reason
    )


# uvicorn app:app --reload

# curl -X POST "http://127.0.0.1:8000/chat" \
#   -H "Content-Type: application/json" \
#   -d '{
#     "session_id": "user-123",
#     "message": "How do I reset my password?"
#   }'