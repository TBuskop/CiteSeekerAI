#!/usr/bin/env python3
"""
rag.py

A Retrieval-Augmented Generation (RAG) script with deep linking that:
  1. Reads a given .txt document.
  2. Splits it into semantically coherent chunks using a token-based sliding window,
     capturing token offsets for each chunk.
  3. Generates a short contextual summary for each chunk using the full document context.
  4. Computes an embedding for each contextualized chunk using OpenAI's embeddings model.
  5. Stores each chunk (with metadata including file name, chunk number, token offset range, and processing date)
     in an SQLite database.
  6. Checks if a file has already been chunked to avoid reprocessing.
  7. In query mode, retrieves the top-K chunks by computing cosine similarity over stored embeddings.
  8. In iterative query mode, first expands the original query into subqueries, retrieves chunks for each,
     and then generates the final answer using the combined context and deep linking metadata.
     
Usage:
  # To index a document:
  python rag.py --mode index --document_path path/to/document.txt --db_path chunks.db

  # To query iteratively:
  python rag.py --mode query --query "What was ACME's Q2 2023 revenue?" --db_path chunks.db --top_k 5
"""

import os
import argparse
import datetime
import hashlib
import pickle
import sqlite3
from typing import List, Dict, Any
from tqdm import tqdm

import numpy as np
import openai
import tiktoken

# Import configuration values from config.py
from config import (
    OPENAI_API_KEY,
    GEMINI_API_KEY, 
    EMBEDDING_MODEL,
    CHUNK_CONTEXT_MODEL,
    SUBQUERY_MODEL, 
    CHAT_MODEL,
    DEFAULT_MAX_TOKENS,
    DEFAULT_OVERLAP,
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_TOP_K
)

# Set OpenAI API key from config and initialize OpenAI client
openai.api_key = OPENAI_API_KEY
from openai import OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)

# ---------------------------
# Initialize Gemini Client
# ---------------------------
from google import genai
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# ---------------------------
# Database Setup and Helpers
# ---------------------------
def init_db(db_path: str) -> sqlite3.Connection:
    """
    Initialize the SQLite database with two tables:
      - documents: stores file hash, file name, and processing date.
      - chunks: stores each chunk's text, context, token offset range,
                embedding (as a pickle blob), chunk number, and a foreign key to documents.
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
            start_token INTEGER,
            end_token INTEGER,
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
    Count tokens using the tiktoken library.
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except Exception:
        encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    return len(tokens)

def chunk_document_tokens(document: str,
                          max_tokens: int = DEFAULT_MAX_TOKENS,
                          overlap: int = DEFAULT_OVERLAP) -> List[tuple]:
    """
    Split the document into chunks using a sliding window over tokens.
    Returns a list of tuples: (chunk_text, start_token, end_token).
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
        chunk_text = encoding.decode(chunk_tokens).strip()
        chunks.append((chunk_text, start, min(end, len(tokens))))
        start = end - overlap  # Slide window with overlap
    return chunks

def truncate_text(text: str, token_limit: int, model: str = CHAT_MODEL) -> str:
    try:
        encoding = tiktoken.encoding_for_model(model)
    except Exception:
        encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    if len(tokens) > token_limit:
        tokens = tokens[:token_limit]
        text = encoding.decode(tokens)
    return text

def generate_chunk_context(document: str, chunk: str, token_limit: int = 30000,
                           context_length: int = DEFAULT_CONTEXT_LENGTH,
                           model: str = CHUNK_CONTEXT_MODEL) -> str:
    """
    Generate a succinct, chunk-specific context using the full document.
    """
    if count_tokens(document) > token_limit:
        document = truncate_text(document, token_limit)
        
    prompt = (
        f"<document>\n{document}\n</document>\n"
        f"Here is the chunk we want to situate within the whole document\n"
        f"<chunk>\n{chunk}\n</chunk>\n"
        "Please give a short succinct context to situate this chunk within the overall document "
        "for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."
    )
    
    return generate_llm_response(prompt, context_length, temperature=0.5, model=model)

def get_embedding(text: str, model: str = EMBEDDING_MODEL, task_type=None) -> np.ndarray:
    """
    Get the embedding for a given text using OpenAI's embeddings model.
    """
    if task_type:
        if task_type not in ["retrieval_document", "retrieval_query"]:
            print ("Task type not supported")
            

    if EMBEDDING_MODEL in ["text-embedding-3-small"]:
        response = openai.embeddings.create(
            input=[text],
            model=model,
            encoding_format="float"
        )
        vector = response.data[0].embedding
        return np.array(vector)
    
    elif EMBEDDING_MODEL in ["embedding-001"]:
        response = gemini_client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=text,
            task_type=task_type,
        )
        vector = response["embedding"]
        return np.array(vector)
    
    elif EMBEDDING_MODEL in ["text-embedding-004"]:
        response = gemini_client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=text,
        )
        vector = response.embeddings[0].values
        return np.array(vector)

def cosine_similarity(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Compute the cosine similarity between a query vector and each row in the matrix.
    """
    query_norm = np.linalg.norm(query_vec)
    matrix_norm = np.linalg.norm(matrix, axis=1)
    if query_norm == 0:
        return np.zeros(matrix.shape[0])
    similarities = np.dot(matrix, query_vec) / (matrix_norm * query_norm + 1e-10)
    return similarities

def simulate_generation(query: str, retrieved_chunks: List[dict]) -> str:
    """
    Simulate generating an answer using the retrieved chunks,
    including deep linking references.
    """
    combined_context = "\n---\n".join(
        [
            f"{chunk['contextualized_text']}\n"
            f"[Deep Link: File={chunk['file_name']}, Chunk={chunk['chunk_number']}, "
            f"Tokens={chunk['start_token']}-{chunk['end_token']}, Date={chunk['processing_date']}]"
            for chunk in retrieved_chunks
        ]
    )
    response = (
        f"Query: {query}\n\n"
        f"Retrieved Context:\n{combined_context}\n\n"
        "[Simulated Answer based on the above context]"
    )
    return response

# ---------------------------
# Indexing Functions
# ---------------------------
def index_document(document_path: str, db_path: str,
                   max_tokens: int = DEFAULT_MAX_TOKENS, overlap: int = DEFAULT_OVERLAP):
    """
    Index a document by splitting it into chunks, generating embeddings,
    and storing the data (including deep linking token offsets) in the database.
    """
    if not os.path.exists(document_path):
        raise FileNotFoundError(f"Document not found: {document_path}")

    file_hash = compute_file_hash(document_path)
    with open(document_path, "r", encoding="utf-8") as f:
        document = f.read()
    file_name = os.path.basename(document_path)
    processing_date = datetime.datetime.now().isoformat()

    conn = init_db(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT doc_id FROM documents WHERE file_hash = ?", (file_hash,))
    result = cursor.fetchone()
    if result:
        print(f"Document '{file_name}' has already been processed (doc_id={result[0]}). Skipping indexing.")
        conn.close()
        return

    cursor.execute(
        "INSERT INTO documents (file_hash, file_name, processing_date) VALUES (?, ?, ?)",
        (file_hash, file_name, processing_date),
    )
    doc_id = cursor.lastrowid
    conn.commit()

    raw_chunks = chunk_document_tokens(document, max_tokens=max_tokens, overlap=overlap)
    print(f"Created {len(raw_chunks)} chunks.")

    # --- CHANGES MADE ---
    # Removed global file summary generation (generate_context) and now generate a chunk-specific context.
    for idx, (raw_chunk, start_tok, end_tok) in tqdm(
        enumerate(raw_chunks),
        total=len(raw_chunks),
        desc=f"Processing {file_name[0:30]}...",
        unit="chunk"
    ):
        chunk_context = generate_chunk_context(document, raw_chunk)
        contextualized_text = f"{chunk_context}\n{raw_chunk}"
        tokens = count_tokens(raw_chunk)
        embedding = get_embedding(contextualized_text, task_type="retrieval_document")
        embedding_blob = pickle.dumps(embedding)
        cursor.execute(
            """
            INSERT INTO chunks (
                doc_id, chunk_number, start_token, end_token,
                text, context, contextualized_text, embedding, tokens
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                doc_id,
                idx,
                start_tok,
                end_tok,
                raw_chunk,
                chunk_context,
                contextualized_text,
                embedding_blob,
                tokens,
            )
        )

    conn.commit()
    conn.close()
    print(f"Document '{file_name}' indexed and stored in the database.")

# ---------------------------
# Retrieval and Iterative Query Functions
# ---------------------------
def generate_llm_response(prompt: str, max_tokens: int, temperature: float = 1, model=None) -> str:
    """
    Generate a response from the configured LLM provider.
    """
    if model is None:
        raise ValueError("A model must be specified")
    if model.lower() in ["gpt-3.5-turbo", "gpt-4", "gpt-4o"]:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()
    elif model.lower() in ["gemini-2.0-flash", "gemini-2.0-flash-lite"]:
        generate_content_config = genai.types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            response_mime_type="text/plain",
        )


        response = gemini_client.models.generate_content(
            model=model,
            contents=prompt,
            config=generate_content_config
        )
        return response.text.strip()
    else:
        raise ValueError(f"Unsupported model: {model}")

def generate_subqueries(initial_query: str, model: str = SUBQUERY_MODEL) -> List[str]:
    """
    Generate a set of expanded or alternative queries based on the original query.
    """
    n_queries=5
    prompt = (
        f"You are an assistant tasked with expanding a user's academic query to retrieve more context. "
        f"Based on the original question, provide a list of {n_queries} expanded or alternative queries that capture different aspects or keywords. Just return the questions in a list, nothing else.\n\n"
        f"Original Query: {initial_query}\nPlease provide a list of expanded queries."
    )
    
    response_text = generate_llm_response(prompt, max_tokens=150, temperature=0.7, model=model)
    subqueries = [line.strip("- ").strip() for line in response_text.split("\n") if line.strip()]
    return subqueries

def retrieve_chunks_for_query(query: str, db_path: str, top_k: int) -> List[dict]:
    """
    Retrieve the top-K chunks from the database relevant to a given query,
    including deep linking metadata.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT
            c.chunk_number,
            c.start_token,
            c.end_token,
            c.text,
            c.context,
            c.contextualized_text,
            c.embedding,
            d.file_name,
            d.processing_date
        FROM chunks c
        JOIN documents d ON c.doc_id = d.doc_id
        """
    )
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return []

    chunks = []
    embedding_list = []
    for row in rows:
        (chunk_number, start_token, end_token, text, context,
         contextualized_text, embedding_blob, file_name, processing_date) = row
        embedding = pickle.loads(embedding_blob)
        embedding_list.append(embedding)
        chunks.append({
            "chunk_number": chunk_number,
            "start_token": start_token,
            "end_token": end_token,
            "text": text,
            "context": context,
            "contextualized_text": contextualized_text,
            "file_name": file_name,
            "processing_date": processing_date
        })
    embedding_matrix = np.vstack(embedding_list)
    query_vec = get_embedding(query, task_type="retrieval_query")
    similarities = cosine_similarity(query_vec, embedding_matrix)
    top_indices = np.argsort(similarities)[::-1][:top_k]
    retrieved_chunks = [chunks[i] for i in top_indices]
    return retrieved_chunks

def generate_answer(query: str, combined_context: str, retrieved_chunks: List[dict],
                    model: str = CHAT_MODEL) -> str:
    """
    Generate an answer to the user query using the provided combined context,
    embedding deep link metadata so the user can see precisely where each chunk came from.
    """
    references = "\n".join(
        f"- {chunk['file_name']} [chunk #{chunk['chunk_number']}, tokens {chunk['start_token']}–{chunk['end_token']}]"
        for chunk in retrieved_chunks
    )
    prompt = (
        f"Answer the following question using only the provided context. You may elaborate a bit and provide some explanations on the statement. If the context does not contain enough information, say so.\n\n"
        f"Context:\n{combined_context}\n\n"
        f"Question: {query}\n\n"
        f"Please include references in your answer using the format [Source: filename, chunk number, token range].\n"
        f"\nSources:\n{references}\n"
        f"\nAnswer:"
    )

    # wrap the prompt to a txt file
    with open('prompt.txt', 'w', encoding='utf-8') as f:
        f.write(prompt)
    
    return generate_llm_response(prompt, max_tokens=16384, temperature=1, model=model)

def iterative_rag_query(initial_query: str, db_path: str, top_k: int = DEFAULT_TOP_K,
                        subquery_model: str = SUBQUERY_MODEL,
                        answer_model: str = CHAT_MODEL) -> str:
    """
    Implements the iterative retrieval process with deep linking:
      1. Generate subqueries from the original query using the specified model.
      2. Retrieve chunks for each subquery.
      3. Deduplicate and combine the retrieved contexts.
      4. Generate the final answer using the original query and aggregated context with the specified model.
    """
    subqueries = generate_subqueries(initial_query, model=subquery_model)
    print("Generated subqueries:")
    for idx, subq in enumerate(subqueries, 1):
        print(f"  {idx}. {subq}")
    
    all_retrieved_chunks = []
    for subq in subqueries:
        retrieved = retrieve_chunks_for_query(subq, db_path, top_k)
        all_retrieved_chunks.extend(retrieved)
    
    if not all_retrieved_chunks:
        return "No relevant chunks found in the database. Please index a document first."

    # Deduplicate chunks using file name and chunk number as unique key.
    unique_chunks: Dict[tuple, Dict[str, Any]] = {
        (chunk["file_name"], chunk["chunk_number"]): chunk 
        for chunk in all_retrieved_chunks
    }
    unique_chunks_list: List[Dict[str, Any]] = list(unique_chunks.values())
    print(f"Retrieved {len(unique_chunks_list)} unique chunks.")
    print("\n=== Unique Retrieved Chunks ===")
    for i, chunk in enumerate(unique_chunks_list, 1):
        print(f"{i}. File: '{chunk['file_name']}', Chunk: {chunk['chunk_number']}")
    print("=====================\n")
    
    combined_context = "\n====\n".join(
        f"Context for chunk {chunk['chunk_number']} from '{chunk['file_name']}':\n"
        f"{chunk.get('contextualized_text', chunk.get('text', ''))}\n(Date: {chunk['processing_date']})"
        for chunk in unique_chunks_list
    )
    # Optionally, save combined context to a file.
    with open('combined_context.txt', 'w', encoding='utf-8') as f:
        f.write(combined_context)

    combined_query = f"{initial_query}\n\nSubqueries:\n" + "\n".join(subqueries)

    answer = generate_answer(combined_query, combined_context, unique_chunks_list, model=answer_model)
    return answer

def query_index(query: str, db_path: str, top_k: int = DEFAULT_TOP_K) -> str:
    """
    (Legacy) Query the database using the original query and return a simulated answer,
    including deep linking references.
    """
    retrieved_chunks = retrieve_chunks_for_query(query, db_path, top_k)
    if not retrieved_chunks:
        return "No chunks found in the database. Please index a document first."
    print("\n=== Retrieved Chunks ===")
    for i, chunk in enumerate(retrieved_chunks, 1):
        print(f"{i}. File: '{chunk['file_name']}', Chunk: {chunk['chunk_number']}")
    print("=====================\n")
    answer = simulate_generation(query, retrieved_chunks)
    return answer

# ---------------------------
# Main CLI
# ---------------------------
def main():
    test = False  # Set to False for production mode
    max_tokens = DEFAULT_MAX_TOKENS
    overlap = DEFAULT_OVERLAP
    if test:
        print("Test mode enabled")
        args = argparse.Namespace()
        args.db_path = "chunks.db"
        args.mode = "index"
        args.folder_path = "extracted_texts/test"  # For folder indexing
        args.document_path = None  # Or set a specific file path
        args.query = "How do cascading risks work?"
        args.top_k = DEFAULT_TOP_K
    else:
        parser = argparse.ArgumentParser(
            description="RAG Script with Deep Linking: Index a document/folder or query the database"
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

    if not OPENAI_API_KEY:
        raise EnvironmentError("Please set the OPENAI_API_KEY in config.py or as an environment variable.")

    if args.mode == "index":
        if not args.document_path and not args.folder_path:
            raise ValueError("For indexing, provide either --document_path or --folder_path.")
        if args.folder_path:
            if not os.path.isdir(args.folder_path):
                raise ValueError(f"Folder path '{args.folder_path}' does not exist or is not a directory.")
            for root, dirs, files in os.walk(args.folder_path):
                for file in files:
                    if file.endswith(".txt"):
                        file_path = os.path.join(root, file)
                        print(f"Indexing file: {file_path}")
                        index_document(file_path, args.db_path, max_tokens=max_tokens, overlap=overlap)
        else:
            index_document(args.document_path, args.db_path, max_tokens=max_tokens, overlap=overlap)

    elif args.mode == "query":
        if not args.query:
            raise ValueError("Query must be provided in query mode.")
        final_answer = iterative_rag_query(args.query, args.db_path, top_k=args.top_k)
        print("\n=== Final Answer ===")
        print(final_answer)

    if test and args.mode == "index":
        print("\nRunning test iterative query after indexing...")
        final_answer = iterative_rag_query(args.query, args.db_path, top_k=args.top_k)
        print("\n=== Final Answer ===")
        print(final_answer)

if __name__ == "__main__":
    main()
