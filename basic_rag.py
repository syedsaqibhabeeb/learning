import os
import glob
import json
from typing import List, Dict, Any, Tuple

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI


# ---------------------------------------------------------
# 1. Setup
# ---------------------------------------------------------

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("RAG_CHAT_MODEL", "gpt-5.5")


# ---------------------------------------------------------
# 2. Load documents
# ---------------------------------------------------------

def load_text_files(folder_path: str) -> List[Dict[str, Any]]:
    """
    Loads all .txt files from a folder.

    Returns:
        [
            {
                "source": "docs/file.txt",
                "text": "full document text"
            }
        ]
    """
    documents = []

    file_paths = glob.glob(os.path.join(folder_path, "*.txt"))

    for file_path in file_paths:
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()

        if text.strip():
            documents.append({
                "source": file_path,
                "text": text
            })

    return documents


# ---------------------------------------------------------
# 3. Chunk documents
# ---------------------------------------------------------

def chunk_text(
    text: str,
    chunk_size_words: int = 220,
    overlap_words: int = 40
) -> List[str]:
    """
    Splits text into overlapping chunks.

    Why chunk?
    - LLMs have context limits.
    - Retrieval works better on focused chunks.
    - Smaller chunks reduce irrelevant context.

    This is a simple word-based chunker.
    In production, you may use section-aware or semantic chunking.
    """
    words = text.split()

    if not words:
        return []

    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size_words
        chunk = " ".join(words[start:end])
        chunks.append(chunk)

        start += chunk_size_words - overlap_words

    return chunks


def build_chunks(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Converts full documents into retrievable chunks.

    Each chunk keeps metadata:
    - source file
    - chunk id
    - text
    """
    all_chunks = []

    for doc in documents:
        chunks = chunk_text(doc["text"])

        for idx, chunk in enumerate(chunks):
            all_chunks.append({
                "chunk_id": f"{doc['source']}::chunk_{idx}",
                "source": doc["source"],
                "chunk_index": idx,
                "text": chunk
            })

    return all_chunks


# ---------------------------------------------------------
# 4. Embeddings
# ---------------------------------------------------------

def get_embedding(text: str) -> List[float]:
    """
    Converts text into an embedding vector.
    """
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )

    return response.data[0].embedding


def embed_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Adds embedding vectors to each chunk.
    """
    embedded_chunks = []

    for chunk in chunks:
        embedding = get_embedding(chunk["text"])

        embedded_chunks.append({
            **chunk,
            "embedding": embedding
        })

    return embedded_chunks


# ---------------------------------------------------------
# 5. Save/load index locally
# ---------------------------------------------------------

def save_index(embedded_chunks: List[Dict[str, Any]], path: str = "rag_index.json") -> None:
    """
    Saves the RAG index to disk so you do not need to re-embed every time.
    """
    with open(path, "w", encoding="utf-8") as file:
        json.dump(embedded_chunks, file)


def load_index(path: str = "rag_index.json") -> List[Dict[str, Any]]:
    """
    Loads previously saved embedded chunks.
    """
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


# ---------------------------------------------------------
# 6. Similarity search
# ---------------------------------------------------------

def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """
    Computes cosine similarity between two vectors.

    Higher score = more semantically similar.
    """
    a = np.array(vec_a)
    b = np.array(vec_b)

    denominator = np.linalg.norm(a) * np.linalg.norm(b)

    if denominator == 0:
        return 0.0

    return float(np.dot(a, b) / denominator)


def retrieve(
    query: str,
    embedded_chunks: List[Dict[str, Any]],
    top_k: int = 4
) -> List[Dict[str, Any]]:
    """
    Retrieves the top_k most relevant chunks for the query.
    """
    query_embedding = get_embedding(query)

    scored_chunks = []

    for chunk in embedded_chunks:
        score = cosine_similarity(query_embedding, chunk["embedding"])

        scored_chunks.append({
            **chunk,
            "similarity_score": score
        })

    scored_chunks.sort(key=lambda x: x["similarity_score"], reverse=True)

    return scored_chunks[:top_k]


# ---------------------------------------------------------
# 7. Build prompt/context
# ---------------------------------------------------------

def build_context(retrieved_chunks: List[Dict[str, Any]]) -> str:
    """
    Converts retrieved chunks into a context block for the LLM.
    """
    context_blocks = []

    for i, chunk in enumerate(retrieved_chunks, start=1):
        block = f"""
[Source {i}]
File: {chunk["source"]}
Chunk ID: {chunk["chunk_id"]}
Similarity Score: {chunk["similarity_score"]:.4f}

Content:
{chunk["text"]}
"""
        context_blocks.append(block)

    return "\n".join(context_blocks)


# ---------------------------------------------------------
# 8. Generate answer
# ---------------------------------------------------------

def generate_answer(query: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
    """
    Sends retrieved context + user question to the LLM.
    """
    context = build_context(retrieved_chunks)

    prompt = f"""
You are a helpful RAG assistant.

Answer the user's question using ONLY the provided context.

Rules:
1. If the context does not contain the answer, say: "I don't know based on the provided documents."
2. Do not make up facts.
3. Cite the source file names you used.
4. Be concise but complete.

Context:
{context}

User question:
{query}
"""

    response = client.responses.create(
        model=CHAT_MODEL,
        input=prompt
    )

    return response.output_text


# ---------------------------------------------------------
# 9. Full RAG pipeline
# ---------------------------------------------------------

def build_rag_index(
    docs_folder: str = "docs",
    index_path: str = "rag_index.json",
    rebuild: bool = False
) -> List[Dict[str, Any]]:
    """
    Builds or loads the RAG index.

    If rebuild=True, it re-loads docs and re-embeds everything.
    If rebuild=False and index exists, it loads from disk.
    """
    if os.path.exists(index_path) and not rebuild:
        print(f"Loading existing index from {index_path}")
        return load_index(index_path)

    print("Loading documents...")
    documents = load_text_files(docs_folder)

    if not documents:
        raise ValueError(f"No .txt documents found in folder: {docs_folder}")

    print(f"Loaded {len(documents)} documents.")

    print("Chunking documents...")
    chunks = build_chunks(documents)
    print(f"Created {len(chunks)} chunks.")

    print("Embedding chunks...")
    embedded_chunks = embed_chunks(chunks)

    print(f"Saving index to {index_path}")
    save_index(embedded_chunks, index_path)

    return embedded_chunks


def ask(query: str, embedded_chunks: List[Dict[str, Any]], top_k: int = 4) -> None:
    """
    Runs retrieval + generation and prints the result.
    """
    print("\nRetrieving relevant chunks...")
    retrieved_chunks = retrieve(query, embedded_chunks, top_k=top_k)

    print("\nTop retrieved chunks:")
    for chunk in retrieved_chunks:
        print(
            f"- {chunk['source']} | "
            f"chunk {chunk['chunk_index']} | "
            f"score={chunk['similarity_score']:.4f}"
        )

    print("\nGenerating answer...")
    answer = generate_answer(query, retrieved_chunks)

    print("\nAnswer:")
    print(answer)


# ---------------------------------------------------------
# 10. CLI entry point
# ---------------------------------------------------------

if __name__ == "__main__":
    index = build_rag_index(
        docs_folder="docs",
        index_path="rag_index.json",
        rebuild=False
    )

    while True:
        user_query = input("\nAsk a question, or type 'exit': ").strip()

        if user_query.lower() in ["exit", "quit"]:
            break

        if not user_query:
            continue

        ask(user_query, index, top_k=4)