#!/usr/bin/env python3
"""
rag.py

A Retrieval-Augmented Generation (RAG) script that:
  1. Reads a given .txt document.
  2. Splits it into semantically coherent chunks using a token-based sliding window.
  3. Generates a short contextual summary for each chunk.
  4. Computes an embedding for each contextualized chunk using OpenAI's embeddings model.
  5. Stores each chunk (with metadata such as the original file name, chunk number, and processing date) in an SQLite database.
  6. Checks if a file has already been chunked to avoid reprocessing.
  7. In query mode, retrieves the top-K chunks for a user query by computing cosine similarity over stored embeddings.

Usage:
  # To index a document:
  python rag.py --mode index --document_path path/to/document.txt --db_path chunks.db

  # To query:
  python rag.py --mode query --query "What was ACME's Q2 2023 revenue?" --db_path chunks.db --top_k 5
"""
import os

from dotenv import load_dotenv
load_dotenv(override=True)  # Force override of existing environment variables

import argparse
import datetime
import hashlib
import pickle
import re
import sqlite3
from typing import List
from tqdm import tqdm  # Import tqdm for progress bars

import numpy as np
import openai
import tiktoken  # For accurate token counting

# Import configuration values from config.py
from config import (
    OPENAI_API_KEY, 
    EMBEDDING_MODEL, 
    CHAT_MODEL,
    DEFAULT_MAX_TOKENS,
    DEFAULT_OVERLAP,
    DEFAULT_TOP_K
)

# Set OpenAI API key from config
openai.api_key = OPENAI_API_KEY

from openai import OpenAI
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ---------------------------
# Database Setup and Helpers
# ---------------------------
def init_db(db_path: str) -> sqlite3.Connection:
    """
    Initialize the SQLite database with two tables:
      - documents: stores file hash, file name, and processing date.
      - chunks: stores each chunk's text, context, embedding (as a pickle blob), chunk number, and a foreign key to documents.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            doc_id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_hash TEXT UNIQUE,
            file_name TEXT,
            processing_date TEXT
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id INTEGER,
            chunk_number INTEGER,
            text TEXT,
            context TEXT,
            contextualized_text TEXT,
            embedding BLOB,
            tokens INTEGER,
            FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
        )
        """
    )
    conn.commit()
    return conn

def compute_file_hash(file_path: str) -> str:
    """
    Compute a SHA256 hash of the file contents.
    """
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

# ---------------------------
# Chunking and Embedding Helpers
# ---------------------------
def count_tokens(text: str, model: str = EMBEDDING_MODEL) -> int:
    """
    Count tokens using the tiktoken library, which matches OpenAI's tokenization.
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except Exception:
        encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    return len(tokens)

def chunk_document_tokens(document: str, max_tokens: int = DEFAULT_MAX_TOKENS, overlap: int = DEFAULT_OVERLAP) -> List[str]:
    """
    Split the document into chunks using a sliding window over tokens.
    Each chunk will have at most max_tokens tokens, with the specified overlap between consecutive chunks.
    This approach guarantees that no chunk exceeds the max_tokens limit.
    """
    try:
        encoding = tiktoken.encoding_for_model(EMBEDDING_MODEL)
    except Exception:
        encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(document)
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + max_tokens
        chunk_tokens = tokens[start:end]
        chunks.append(encoding.decode(chunk_tokens).strip())
        start = end - overlap  # Slide window with overlap
    return chunks

def generate_context(chunk_text: str, chunk_number: int) -> str:
    """
    Generate a contextual summary for a chunk.
    For simplicity, this returns the first sentence (or first 20 words) of the chunk.
    """
    sentences = re.split(r'(?<=[.!?]) +', chunk_text)
    first_sentence = sentences[0] if sentences else chunk_text
    words = first_sentence.split()
    context = " ".join(words[:20])
    return f"Context for chunk {chunk_number}: {context}"

def get_openai_embedding(text: str, model: str = EMBEDDING_MODEL) -> np.ndarray:
    """
    Get the embedding for a given text using OpenAI's embeddings model.
    """
    response = openai.embeddings.create(
        input=[text],
        model=model,
        encoding_format="float"
    )
    vector = response.data[0].embedding
    return np.array(vector)

def cosine_similarity(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Compute the cosine similarity between a query vector and each row in the matrix.
    """
    query_norm = np.linalg.norm(query_vec)
    matrix_norm = np.linalg.norm(matrix, axis=1)
    # Avoid division by zero
    if query_norm == 0:
        return np.zeros(matrix.shape[0])
    similarities = np.dot(matrix, query_vec) / (matrix_norm * query_norm + 1e-10)
    return similarities

def simulate_generation(query: str, retrieved_chunks: List[dict]) -> str:
    """
    Simulate generating an answer using the retrieved chunks.
    """
    combined_context = "\n---\n".join(
        [f"{chunk['contextualized_text']}\n(File: {chunk['file_name']}, Chunk: {chunk['chunk_number']}, Date: {chunk['processing_date']})"
         for chunk in retrieved_chunks]
    )
    response = (
        f"Query: {query}\n\nRetrieved Context:\n{combined_context}\n\n"
        "[Simulated Answer based on the above context]"
    )
    return response

# ---------------------------
# Indexing and Querying Functions
# ---------------------------
def index_document(document_path: str, db_path: str, max_tokens: int = DEFAULT_MAX_TOKENS, overlap: int = DEFAULT_OVERLAP):
    """
    Index a document by splitting it into chunks using a token-based sliding window,
    generating embeddings for each chunk, and storing them along with metadata in the database.
    Skips reprocessing if the document (by its hash) already exists.
    """
    if not os.path.exists(document_path):
        raise FileNotFoundError(f"Document not found: {document_path}")

    # Compute file hash and read the document.
    file_hash = compute_file_hash(document_path)
    with open(document_path, "r", encoding="utf-8") as f:
        document = f.read()
    file_name = os.path.basename(document_path)
    processing_date = datetime.datetime.now().isoformat()

    # Initialize database.
    conn = init_db(db_path)
    cursor = conn.cursor()

    # Check if this document has already been processed.
    cursor.execute("SELECT doc_id FROM documents WHERE file_hash = ?", (file_hash,))
    result = cursor.fetchone()
    if result:
        print(f"Document '{file_name}' has already been processed (doc_id={result[0]}). Skipping indexing.")
        conn.close()
        return

    # Insert the new document record.
    cursor.execute(
        "INSERT INTO documents (file_hash, file_name, processing_date) VALUES (?, ?, ?)",
        (file_hash, file_name, processing_date),
    )
    doc_id = cursor.lastrowid
    conn.commit()

    raw_chunks = chunk_document_tokens(document, max_tokens=max_tokens, overlap=overlap)
    print(f"Created {len(raw_chunks)} chunks.")

    # Add tqdm progress bar to show indexing progress
    for idx, raw_chunk in tqdm(enumerate(raw_chunks), total=len(raw_chunks), desc=f"Processing {file_name}", unit="chunk"):
        context = generate_context(raw_chunk, idx)
        contextualized_text = f"{context}\n{raw_chunk}"
        tokens = count_tokens(raw_chunk)
        embedding = get_openai_embedding(contextualized_text)
        # Store the embedding as a pickle blob.
        embedding_blob = pickle.dumps(embedding)
        cursor.execute(
            """
            INSERT INTO chunks (doc_id, chunk_number, text, context, contextualized_text, embedding, tokens)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (doc_id, idx, raw_chunk, context, contextualized_text, embedding_blob, tokens),
        )
    
    conn.commit()
    conn.close()
    print(f"Document '{file_name}' indexed and stored in the database.")

def query_index(query: str, db_path: str, top_k: int = DEFAULT_TOP_K) -> str:
    """
    Query the database by computing the query embedding using OpenAI's model, then calculating cosine similarity
    between the query embedding and all stored chunk embeddings. Returns a simulated answer based on the top-K chunks.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT c.chunk_number, c.text, c.context, c.contextualized_text, c.embedding, d.file_name, d.processing_date
        FROM chunks c
        JOIN documents d ON c.doc_id = d.doc_id
        """
    )
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return "No chunks found in the database. Please index a document first."

    # Reconstruct embeddings and metadata.
    chunks = []
    embedding_list = []
    for row in rows:
        chunk_number, text, context, contextualized_text, embedding_blob, file_name, processing_date = row
        embedding = pickle.loads(embedding_blob)
        embedding_list.append(embedding)
        chunks.append({
            "chunk_number": chunk_number,
            "text": text,
            "context": context,
            "contextualized_text": contextualized_text,
            "file_name": file_name,
            "processing_date": processing_date
        })

    embedding_matrix = np.vstack(embedding_list)

    query_vec = get_openai_embedding(query)

    similarities = cosine_similarity(query_vec, embedding_matrix)

    # Get the top_k chunks.
    top_indices = np.argsort(similarities)[::-1][:top_k]
    retrieved_chunks = [chunks[i] for i in top_indices]
    
    # Log retrieved chunks information to terminal
    print("\n=== Retrieved Chunks ===")
    for i, idx in enumerate(top_indices):
        print(f"{i+1}. File: '{chunks[idx]['file_name']}', Chunk: {chunks[idx]['chunk_number']}, Similarity: {similarities[idx]:.3f}")
    print("=====================\n")

    answer = simulate_generation(query, retrieved_chunks)
    return answer

def generate_answer(query: str, retrieved_chunks: str) -> str:
    """
    Generate an answer to the user query using the retrieved chunks as context.
    """
    combined_context = retrieved_chunks 
    prompt = (
        f"Answer the following question using only the provided context. If the context does not contain enough information, say so.\n\n"
        f"Context:\n{combined_context}\n\n"
        f"Question: {query}\n\n"
        f"Answer:"
    )
    
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that answers questions based on the provided context. Every argument or statement should have a reference to the file where it came from."
            },
            {"role": "user", "content": prompt}
        ],
        max_tokens=16384,
        temperature=0.7,
    )
    answer = response.choices[0].message.content.strip()
    return answer

# ---------------------------
# Main CLI
# ---------------------------
def main():
    test = True  # Set to False for production mode
    import argparse

    if test:
        print("Test mode enabled")
        args = argparse.Namespace()
        args.db_path = "chunks.db"
        args.mode = "index"
        args.folder_path = "extracted_texts/paper_2_intro" # None
        args.document_path = None#"extracted_texts/paper_2_intro/Buskop et al. - 2024 - Amplifying exploration of regional climate risks .txt"
        args.query = "Buskop et al introduces a multi model plausibilistic framework for regional climate risk assessment. What are the key components of this framework and what have other storyline papers approached the creation of storylines?"	
        args.top_k = 20
        
        # Force the correct value from the environment variable
        max_tokens = int(os.environ.get("DEFAULT_MAX_TOKENS", "300"))
        overlap = int(os.environ.get("DEFAULT_OVERLAP", "20"))
    else:
        parser = argparse.ArgumentParser(
            description="RAG Script with Database Storage: Index a document or query the database"
        )
        parser.add_argument("--mode", choices=["index", "query"], required=True,
                            help="Mode: index a document/folder or query the database")
        parser.add_argument("--document_path", type=str,
                            help="Path to the .txt document to index (if indexing a single file)")
        parser.add_argument("--folder_path", type=str,
                            help="Path to a folder containing .txt documents to index (if indexing multiple files)")
        parser.add_argument("--db_path", type=str, required=True,
                            help="Path to the SQLite database file")
        parser.add_argument("--query", type=str,
                            help="User query (required in query mode)")
        parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K,
                            help="Number of top chunks to retrieve (for query mode)")
        args = parser.parse_args()

    # Ensure OpenAI API key is set.
    if not OPENAI_API_KEY:
        raise EnvironmentError("Please set the OPENAI_API_KEY in config.py or as an environment variable.")

    if args.mode == "index":
        if not args.document_path and not args.folder_path:
            raise ValueError("For indexing, you must provide either --document_path or --folder_path.")

        if args.folder_path:
            if not os.path.isdir(args.folder_path):
                raise ValueError(f"Folder path '{args.folder_path}' does not exist or is not a directory.")
            for root, dirs, files in os.walk(args.folder_path):
                for file in files:
                    if file.endswith(".txt"):
                        file_path = os.path.join(root, file)
                        print(f"Indexing file: {file_path}")
                        index_document(file_path, args.db_path, max_tokens=DEFAULT_MAX_TOKENS, overlap=DEFAULT_OVERLAP)
        else:
            # Force use of the correct values for testing
            if test:
                max_tokens = int(os.environ.get("DEFAULT_MAX_TOKENS", "300"))
                overlap = int(os.environ.get("DEFAULT_OVERLAP", "20"))
                index_document(args.document_path, args.db_path, max_tokens=max_tokens, overlap=overlap)
            else:
                index_document(args.document_path, args.db_path, max_tokens=DEFAULT_MAX_TOKENS, overlap=DEFAULT_OVERLAP)

    elif args.mode == "query":
        if not args.query:
            raise ValueError("Query must be provided in query mode.")
        answer = query_index(args.query, args.db_path, top_k=args.top_k)
        print("\n=== Final Answer ===")
        print(answer)

    # Optional test mode: if test is enabled, you can run an indexing and query cycle.
    if test and args.mode == "index":
        print("\nRunning test query after indexing...")
        args.mode = "query"
        relevant_chunks = query_index(args.query, args.db_path, top_k=args.top_k)
        answer = generate_answer(args.query, relevant_chunks)
        print("\n=== Final Answer ===")
        print(answer)

if __name__ == "__main__":
    main()
