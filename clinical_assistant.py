"""
Standalone Medical Document RAG API

Requirements:
    pip install fastapi uvicorn python-dotenv pydantic
    pip install langchain langchain-community langchain-openai langchain-huggingface langchain-text-splitters
    pip install sentence-transformers faiss-cpu pypdf

Usage:
    1. Put 5–10 PDF/TXT files inside: ./data/raw/

    2. Add your OpenAI API key:
        export OPENAI_API_KEY="your_key_here"

       Or create a .env file:
        OPENAI_API_KEY=your_key_here

    3. Build the FAISS index:
        python medical_rag_api.py --ingest

    4. Run the API:
        uvicorn medical_rag_api:app --reload

    5. Test:
        POST http://127.0.0.1:8000/ask

        {
            "question": "What does the document say about diabetes symptoms?",
            "k": 5
        }
"""

import os
import argparse
from pathlib import Path
from functools import lru_cache
from typing import List, Optional, Tuple

from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data" / "raw"
STORAGE_DIR = BASE_DIR / "storage"
FAISS_INDEX_DIR = STORAGE_DIR / "faiss_index"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)

TOP_K = int(os.getenv("TOP_K", "5"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

# FAISS score is distance. Lower is better.
# Tune this based on your corpus.
MAX_FAISS_DISTANCE = float(os.getenv("MAX_FAISS_DISTANCE", "1.20"))

SUPPORTED_EXTENSIONS = {".pdf", ".txt"}

INSUFFICIENT_ANSWER = (
    "I don't have enough information in the provided medical document corpus to answer that."
)


# ---------------------------------------------------------------------
# API Schemas
# ---------------------------------------------------------------------

class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        description="Medical question to answer using the document corpus.",
    )
    k: Optional[int] = Field(
        default=None,
        ge=1,
        le=10,
        description="Number of chunks to retrieve.",
    )


class SourceDocument(BaseModel):
    file_name: str
    page: Optional[int]
    chunk_id: Optional[str]
    score: float
    excerpt: str


class AskResponse(BaseModel):
    answer: str
    source_documents: List[SourceDocument]


# ---------------------------------------------------------------------
# Document Loading
# ---------------------------------------------------------------------

def find_documents(data_dir: Path) -> List[Path]:
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Data directory does not exist: {data_dir}. "
            "Create it and add 5–10 PDF/TXT files."
        )

    files = [
        path for path in data_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not 5 <= len(files) <= 10:
        raise ValueError(
            f"Expected 5–10 PDF/TXT documents in {data_dir}, found {len(files)}."
        )

    return files


def load_single_document(path: Path) -> List[Document]:
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        loader = PyPDFLoader(str(path))
        documents = loader.load()

    elif suffix == ".txt":
        loader = TextLoader(str(path), encoding="utf-8")
        documents = loader.load()

    else:
        raise ValueError(f"Unsupported file type: {path}")

    for doc in documents:
        doc.metadata["file_name"] = path.name
        doc.metadata["source_path"] = str(path)

        if "page" not in doc.metadata:
            doc.metadata["page"] = None

    return documents


def load_all_documents() -> List[Document]:
    files = find_documents(DATA_DIR)

    all_documents: List[Document] = []

    for file_path in files:
        docs = load_single_document(file_path)
        all_documents.extend(docs)

    return all_documents


# ---------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------

def chunk_documents(documents: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"chunk_{index}"

    return chunks


# ---------------------------------------------------------------------
# Embeddings + FAISS
# ---------------------------------------------------------------------

def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )


def ingest_documents() -> None:
    print("=" * 80)
    print("Starting ingestion")
    print("=" * 80)

    print(f"Loading documents from: {DATA_DIR}")
    documents = load_all_documents()
    print(f"Loaded {len(documents)} document records/pages.")

    print("Chunking documents...")
    chunks = chunk_documents(documents)
    print(f"Created {len(chunks)} chunks.")

    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    embeddings = get_embeddings()

    print("Building FAISS index...")
    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings,
    )

    FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Saving FAISS index to: {FAISS_INDEX_DIR}")
    vectorstore.save_local(str(FAISS_INDEX_DIR))

    print("Ingestion complete.")
    print("=" * 80)


def load_vectorstore() -> FAISS:
    if not FAISS_INDEX_DIR.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {FAISS_INDEX_DIR}. "
            "Run: python medical_rag_api.py --ingest"
        )

    embeddings = get_embeddings()

    # Safe only when loading an index that you created locally.
    return FAISS.load_local(
        str(FAISS_INDEX_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )


# ---------------------------------------------------------------------
# RAG Engine
# ---------------------------------------------------------------------

SYSTEM_PROMPT = """
You are a medical document question-answering assistant.

Rules:
1. Answer only using the retrieved context.
2. Do not use outside medical knowledge.
3. Do not guess.
4. If the retrieved context does not clearly contain the answer, respond exactly with:
   "I don't have enough information in the provided medical document corpus to answer that."
5. Keep the answer concise and clinically careful.
6. Do not provide diagnosis, treatment, or medical advice beyond what the documents state.
"""


USER_PROMPT_TEMPLATE = """
Question:
{question}

Retrieved Context:
{context}

Answer:
"""


class MedicalRAG:
    def __init__(self):
        if not OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY is missing. Set it in your environment or .env file."
            )

        self.vectorstore = load_vectorstore()

        self.llm = ChatOpenAI(
            model=OPENAI_MODEL,
            temperature=0,
        )

    def retrieve(self, question: str, k: int) -> List[Tuple[Document, float]]:
        return self.vectorstore.similarity_search_with_score(
            question,
            k=k,
        )

    def is_context_insufficient(
        self,
        docs_with_scores: List[Tuple[Document, float]],
    ) -> bool:
        if not docs_with_scores:
            return True

        best_score = docs_with_scores[0][1]

        # FAISS returns distance. Lower is better.
        if best_score > MAX_FAISS_DISTANCE:
            return True

        return False

    def format_context(
        self,
        docs_with_scores: List[Tuple[Document, float]],
    ) -> str:
        context_blocks = []

        for i, (doc, score) in enumerate(docs_with_scores, start=1):
            file_name = doc.metadata.get("file_name", "unknown")
            page = doc.metadata.get("page", None)
            chunk_id = doc.metadata.get("chunk_id", None)

            block = f"""
[Source {i}]
file_name: {file_name}
page: {page}
chunk_id: {chunk_id}
score: {score}

content:
{doc.page_content}
"""
            context_blocks.append(block)

        return "\n\n".join(context_blocks)

    def build_sources(
        self,
        docs_with_scores: List[Tuple[Document, float]],
    ) -> List[SourceDocument]:
        sources = []

        for doc, score in docs_with_scores:
            excerpt = doc.page_content.strip().replace("\n", " ")

            if len(excerpt) > 500:
                excerpt = excerpt[:500] + "..."

            source = SourceDocument(
                file_name=doc.metadata.get("file_name", "unknown"),
                page=doc.metadata.get("page", None),
                chunk_id=doc.metadata.get("chunk_id", None),
                score=float(score),
                excerpt=excerpt,
            )

            sources.append(source)

        return sources

    def ask(self, question: str, k: Optional[int] = None) -> AskResponse:
        retrieval_k = k or TOP_K

        docs_with_scores = self.retrieve(question, retrieval_k)
        sources = self.build_sources(docs_with_scores)

        if self.is_context_insufficient(docs_with_scores):
            return AskResponse(
                answer=INSUFFICIENT_ANSWER,
                source_documents=sources,
            )

        context = self.format_context(docs_with_scores)

        user_prompt = USER_PROMPT_TEMPLATE.format(
            question=question,
            context=context,
        )

        response = self.llm.invoke(
            [
                ("system", SYSTEM_PROMPT),
                ("user", user_prompt),
            ]
        )

        answer = response.content.strip()

        if not answer:
            answer = INSUFFICIENT_ANSWER

        return AskResponse(
            answer=answer,
            source_documents=sources,
        )


# ---------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------

app = FastAPI(
    title="Medical Document RAG API",
    description="Answers questions from a provided medical document corpus.",
    version="1.0.0",
)


@lru_cache
def get_rag_engine() -> MedicalRAG:
    return MedicalRAG()


@app.get("/")
def root():
    return {
        "message": "Medical Document RAG API is running.",
        "docs": "/docs",
        "ask_endpoint": "POST /ask",
    }


@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest):
    try:
        rag = get_rag_engine()

        response = rag.ask(
            question=request.question,
            k=request.k,
        )

        return response

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e),
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {str(e)}",
        )


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Standalone Medical RAG API"
    )

    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Load documents, chunk them, embed them, and build FAISS index.",
    )

    args = parser.parse_args()

    if args.ingest:
        ingest_documents()
    else:
        print("No CLI action selected.")
        print("To ingest documents:")
        print("  python medical_rag_api.py --ingest")
        print()
        print("To run the API:")
        print("  uvicorn medical_rag_api:app --reload")


if __name__ == "__main__":
    main()


# curl -X POST "http://127.0.0.1:8000/ask" \
#   -H "Content-Type: application/json" \
#   -d '{
#     "question": "What does the document say about diabetes symptoms?",
#     "k": 5
#   }'
