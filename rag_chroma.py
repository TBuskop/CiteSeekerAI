#!/usr/bin/env python3
"""
rag_chroma.py (Corrected & Parallelized Indexing)

A Retrieval-Augmented Generation (RAG) script with deep linking that:
  1. Reads given .txt documents (supports single file or folder).
  2. **Processes files in parallel during indexing.**
  3. Splits each document into semantically coherent chunks using a token-based sliding window,
     capturing token offsets for each chunk.
  4. Generates a short contextual summary for each chunk using the full document context.
  5. Computes an embedding for each contextualized chunk using configured embeddings model.
  6. Stores each chunk (with metadata including file name, chunk number, token offset range,
     processing date, and contextualized text) in a ChromaDB collection.
  7. Checks if a file has already been chunked to avoid reprocessing.
  8. In query mode, retrieves the top-K chunks using ChromaDB's similarity search.
  9. In iterative query mode, first expands the original query into subqueries, retrieves chunks for each,
     and then generates the final answer using the combined context and deep linking metadata.

Usage:
  # To index documents from a folder in parallel:
  python rag_chroma_parallel.py --mode index --folder_path path/to/documents --db_path ./chroma_db --collection_name my_docs --workers 4

  # To index a single document:
  python rag_chroma_parallel.py --mode index --document_path path/to/document.txt --db_path ./chroma_db --collection_name my_docs

  # To query iteratively:
  python rag_chroma_parallel.py --mode query --query "What was ACME's Q2 2023 revenue?" --db_path ./chroma_db --collection_name my_docs --top_k 5
"""

import os
import argparse
import datetime
import hashlib
from typing import List, Dict, Any, Tuple # Added Tuple
from tqdm import tqdm
from google import genai

import numpy as np
# Removed direct openai import here, handled later if needed
# import openai
import tiktoken
import chromadb # Added ChromaDB

# --- Imports for Parallelization ---
import multiprocessing
import functools
# ---------------------------------

# Import configuration values from config.py
# Ensure config.py exists and has the required variables set
try:
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
except ImportError:
    print("Error: config.py not found or missing required variables.")
    print("Please create config.py with necessary API keys and model names.")
    # Provide default fallback values or raise a more specific error
    # For example:
    # OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", None)
    # GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", None)
    # EMBEDDING_MODEL = "text-embedding-3-small"
    # CHUNK_CONTEXT_MODEL = "gpt-3.5-turbo"
    # SUBQUERY_MODEL = "gpt-3.5-turbo"
    # CHAT_MODEL = "gpt-4o"
    # DEFAULT_MAX_TOKENS = 512
    # DEFAULT_OVERLAP = 50
    # DEFAULT_CONTEXT_LENGTH = 150
    # DEFAULT_TOP_K = 5
    print("Warning: Using default configuration values as config.py was not fully loaded.")
    exit(1) # Exit if config is critical and missing

# --- ChromaDB Configuration ---
DEFAULT_CHROMA_COLLECTION_NAME = "rag_chunks_default"
# -----------------------------

# --- Global Variables for API Clients (Initialize later if needed) ---
# It's often safer for multiprocessing workers to initialize their own clients,
# but we can try passing them or making them global first.
# Let's initialize them globally and see if they work well across processes.
# If issues arise (e.g., pickling errors, connection state problems),
# we'll move initialization into the worker function.
openai_client = None
gemini_client = None

def initialize_clients():
    """Initializes API clients based on config keys."""
    global openai_client, gemini_client

    # Set OpenAI API key from config and initialize OpenAI client
    if OPENAI_API_KEY:
        try:
            from openai import OpenAI # Import here
            openai_client = OpenAI(api_key=OPENAI_API_KEY)
            # print("OpenAI client initialized.") # Avoid printing in worker setup
        except ImportError:
            print("Warning: OpenAI library not installed (`pip install openai`). OpenAI models will be unavailable.")
        except Exception as e:
            print(f"Warning: Failed to initialize OpenAI client: {e}")
    else:
        pass # Don't print warning repeatedly in workers print("Warning: OPENAI_API_KEY not found. OpenAI models will be unavailable.")

    # Initialize Gemini Client
    if GEMINI_API_KEY:
        try:
            from google import genai # Import here
            # Consider adding retry/error handling around client creation if needed
            # Test connection/authentication if possible, e.g., list models (can be slow)
            # genai.configure(api_key=GEMINI_API_KEY) # Alternative way to configure
            gemini_client = genai.Client(api_key=GEMINI_API_KEY)
            # print("Gemini client configured.") # Avoid printing in worker setup
        except ImportError:
            print("Warning: Google Generative AI library not installed (`pip install google-generativeai`). Gemini models will be unavailable.")
        except Exception as e:
            print(f"Warning: Failed to configure Gemini client: {e}")
    else:
        pass # Don't print warning repeatedly in workers print("Warning: GEMINI_API_KEY not found. Gemini models will be unavailable.")

# Call initialization once at the start
initialize_clients()
# --- End API Client Initialization ---


# ---------------------------
# ChromaDB Client Setup
# ---------------------------
# Made collection_name a required argument
# This function might be called by multiple processes.
# PersistentClient is generally designed to handle this, as the DB file handles locking.
def get_chroma_collection(db_path: str, collection_name: str) -> chromadb.Collection:
    """
    Initialize the ChromaDB client and get/create the specified collection.
    Uses persistent storage at db_path. SAFE FOR MULTIPROCESSING (usually).
    """
    if not collection_name:
        raise ValueError("ChromaDB collection name cannot be empty.")
    try:
        # Each process gets its own client instance connecting to the same path
        chroma_client = chromadb.PersistentClient(path=db_path)
        collection = chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"} # Cosine is often good for text embeddings
        )
        # Avoid printing connection messages repeatedly from workers
        # print(f"Process {os.getpid()} connected to ChromaDB collection '{collection_name}'.")
        return collection
    except Exception as e:
        # Log error specific to the process trying to connect
        print(f"!!! Process {os.getpid()} Error connecting/creating ChromaDB collection '{collection_name}' at path '{db_path}': {e}")
        raise # Re-raise to signal failure in the worker


# ---------------------------
# File Hashing (Unchanged)
# ---------------------------
def compute_file_hash(file_path: str) -> str:
    """
    Compute a SHA256 hash of the file contents.
    """
    hasher = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            while True:
                chunk = f.read(4096) # Read in chunks for large files
                if not chunk:
                    break
                hasher.update(chunk)
        return hasher.hexdigest()
    except FileNotFoundError:
        print(f"Error: File not found during hashing: {file_path}")
        raise
    except Exception as e:
        print(f"Error hashing file {file_path}: {e}")
        raise


# ---------------------------
# Chunking and Embedding Helpers (Unchanged logic, Ensure they use global clients correctly)
# ---------------------------
# ... (count_tokens, chunk_document_tokens, truncate_text, generate_chunk_context, get_embedding) ...
# Make sure these functions correctly use the globally initialized 'openai_client' and 'gemini_client'.
# If issues arise, pass clients as arguments or initialize them within the functions/worker.

def count_tokens(text: str, model: str = EMBEDDING_MODEL) -> int:
    """
    Count tokens using the tiktoken library. Falls back to basic split.
    """
    try:
        # Caching encoding can speed up repeated calls within a process
        if not hasattr(count_tokens, "_encoding_cache"):
            count_tokens._encoding_cache = {}
        if model not in count_tokens._encoding_cache:
             try:
                 count_tokens._encoding_cache[model] = tiktoken.encoding_for_model(model)
             except Exception:
                 try:
                     count_tokens._encoding_cache[model] = tiktoken.get_encoding("cl100k_base")
                 except Exception as e:
                      print(f"Error getting tiktoken encoding: {e}. Using simple space split.")
                      count_tokens._encoding_cache[model] = None # Mark as failed

        encoding = count_tokens._encoding_cache.get(model)
        if encoding:
            tokens = encoding.encode(text)
            return len(tokens)
        else:
             # Fallback if encoding failed
             return len(text.split())
    except Exception as e:
        # Catch errors during the encoding process itself
        print(f"Error encoding text with tiktoken: {e}. Using simple space split.")
        return len(text.split())

def chunk_document_tokens(document: str,
                          max_tokens: int = DEFAULT_MAX_TOKENS,
                          overlap: int = DEFAULT_OVERLAP) -> List[tuple]:
    """
    Split the document into chunks using a sliding window over tokens.
    Returns a list of tuples: (chunk_text, start_token, end_token).
    Handles potential encoding errors.
    """
    if max_tokens <= 0:
         raise ValueError("max_tokens must be positive.")
    if overlap < 0:
         raise ValueError("overlap cannot be negative.")
    if overlap >= max_tokens:
        # print(f"Warning: Overlap ({overlap}) is greater than or equal to max_tokens ({max_tokens}). Setting overlap to {max_tokens // 2}.")
        overlap = max_tokens // 2 # Adjust overlap to prevent issues

    try:
        # Use cached encoding if possible
        encoding = tiktoken.encoding_for_model(EMBEDDING_MODEL)
    except Exception:
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
             raise ValueError(f"Could not get tiktoken encoding for chunking: {e}")

    try:
        tokens = encoding.encode(document)
    except Exception as e:
        print(f"Error encoding document for chunking: {e}. Returning empty list.")
        return []

    if not tokens:
        return []

    chunks = []
    start_token_idx = 0
    total_tokens = len(tokens)
    effective_overlap = overlap

    while start_token_idx < total_tokens:
        end_token_idx = min(start_token_idx + max_tokens, total_tokens)
        chunk_tokens = tokens[start_token_idx:end_token_idx]

        if not chunk_tokens: # Should not happen if logic is correct, but safeguard
            break

        try:
            # Using errors='replace' during decode can help with potential issues
            chunk_text = encoding.decode(chunk_tokens, errors='replace').strip()
        except Exception as e:
            print(f"Error decoding tokens for chunk starting at {start_token_idx}: {e}. Skipping chunk.")
            next_start_token_idx = start_token_idx + max_tokens - effective_overlap # Calculate step normally
            if next_start_token_idx <= start_token_idx: # Prevent infinite loop if step is non-positive
                 start_token_idx += 1 # Force advance by at least one token
            else:
                start_token_idx = next_start_token_idx
            continue # Skip appending this problematic chunk

        if chunk_text: # Only add non-empty chunks
            chunks.append((chunk_text, start_token_idx, end_token_idx))

        # Calculate next start position
        next_start_token_idx = start_token_idx + max_tokens - effective_overlap

        # Prevent infinite loop if step size is non-positive or we are at the end
        if next_start_token_idx <= start_token_idx:
             if start_token_idx + 1 < total_tokens: # Move forward by at least one token if possible
                 start_token_idx += 1
             else:
                 break # Cannot advance further
        else:
            start_token_idx = next_start_token_idx

    return chunks


def truncate_text(text: str, token_limit: int, model: str = CHAT_MODEL) -> str:
    """Truncates text to a specified token limit."""
    if token_limit <= 0:
        return ""
    try:
        # Caching logic similar to count_tokens can be added if called frequently with same model
        encoding = tiktoken.encoding_for_model(model)
    except Exception:
        encoding = tiktoken.get_encoding("cl100k_base")
    try:
        tokens = encoding.encode(text)
        if len(tokens) > token_limit:
            tokens = tokens[:token_limit]
            # Decode potentially truncated tokens, replacing errors
            text = encoding.decode(tokens, errors='replace')
        return text
    except Exception as e:
        print(f"Error truncating text: {e}. Returning original text (potentially too long).")
        return text # Or handle more gracefully

def generate_chunk_context(document: str, chunk: str, token_limit: int = 30000,
                           context_length: int = DEFAULT_CONTEXT_LENGTH,
                           model: str = CHUNK_CONTEXT_MODEL) -> str:
    """
    Generate a succinct, chunk-specific context using the full document.
    Handles potential LLM errors. Uses global API clients.
    """
    global openai_client, gemini_client # Explicitly state usage of global clients

    if not document.strip():
        # print("Warning: Document is empty. Cannot generate context.") # Less verbose
        return "Document context unavailable."
    if not chunk.strip():
        # print("Warning: Chunk is empty. Cannot generate context.") # Less verbose
        return "Chunk context unavailable."

    doc_token_count = count_tokens(document, model=model)
    if doc_token_count > token_limit:
        # print(f"Warning: Document ({doc_token_count} tokens) exceeds limit ({token_limit}). Truncating.")
        document = truncate_text(document, token_limit, model=model)

    prompt = (
        f"<document>\n{document}\n</document>\n"
        f"Here is the chunk we want to situate within the whole document\n"
        f"<chunk>\n{chunk}\n</chunk>\n"
        "Provide a short, succinct context (1-2 sentences) describing where this chunk fits within the overall document. "
        "This context will be used to improve search retrieval. Focus on the surrounding topic or section. "
        "Answer ONLY with the succinct context itself, nothing else."
    )

    try:
        if context_length <= 0: context_length = 50
        # Call generate_llm_response which handles client selection
        response = generate_llm_response(prompt, context_length, temperature=0.5, model=model)
        response = response.replace("Context:", "").strip()
        return response
    except Exception as e:
        print(f"Error generating chunk context with model {model}: {e}")
        return f"Error generating context." # Return simpler error message


# Assume generate_llm_response, get_embedding, simulate_generation are defined as before
# and correctly use the global 'openai_client' and 'gemini_client'.
def get_embedding(text: str, model: str = EMBEDDING_MODEL, task_type=None) -> np.ndarray | None:
    """
    Get embedding for text using OpenAI or Gemini models. Simplified structure.
    Returns None on failure.
    """
    # 1. Input Validation
    if not text or not text.strip():
        print("Warning: Attempting to embed empty or whitespace-only text. Returning None.")
        return None
        text = text.replace("\n", " ")

    try:
        # 2. Determine Provider and Check Client Availability
        provider = None
        if model in ["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"]:
            provider = "openai"
            if not openai_client:
                print(f"Error: OpenAI client not available for model {model}.")
                return None
        elif model in ["embedding-001", "models/embedding-001", "text-embedding-004", "models/text-embedding-004"]:
            provider = "gemini"
            if not gemini_client:
                print(f"Error: Gemini client not available for model {model}.")
                return None
        else:
            print(f"Error: Unsupported or unknown embedding model '{model}'.")
            return None

        # 3. Call Appropriate API
        # print(f"Using {provider.capitalize()} model: {model}")

        if provider == "openai":
            response = openai_client.embeddings.create(input=[text], model=model)
            vector = response.data[0].embedding
            return np.array(vector)

        elif provider == "gemini":
            # --- Gemini Model Specific Logic ---
            api_model_name = model if model.startswith("models/") else f"models/{model}"

            if model in ["embedding-001", "models/embedding-001"]:
                # Validate task_type specifically for embedding-001
                valid_task_types = ["retrieval_document", "retrieval_query", "semantic_similarity", "classification", "clustering"]
                if task_type and task_type not in valid_task_types:
                    print(f"Warning: Task type '{task_type}' invalid for {api_model_name}. Using 'retrieval_document'.")
                    task_type = "retrieval_document"
                elif not task_type:
                    task_type = "retrieval_document" # Default task type

                response = gemini_client.models.embed_content(
                    model=api_model_name,
                    contents=text, # Use 'content'
                    task_type=task_type
                )
                if "embedding" in response:
                    return np.array(response["embedding"])
                else:
                    print(f"Error: Unexpected response structure from {api_model_name}: {response}")
                    return None

            elif model in ["text-embedding-004", "models/text-embedding-004"]:
                # Assumes text-embedding-004 does not use task_type based on prior context
                response = gemini_client.models.embed_content(
                    model=api_model_name,
                    contents=text # Use 'content'
                )
                # Check for different potential response structures for Gemini embeddings
                if hasattr(response, 'embedding') and response.embedding and hasattr(response.embedding, 'values'):
                    return np.array(response.embedding.values)
                elif hasattr(response, 'embeddings') and response.embeddings and hasattr(response.embeddings[0], 'values'):
                    return np.array(response.embeddings[0].values)
                else:
                    print(f"Error: Unexpected response structure from {api_model_name}: {response}")
                    return None
            else:
                # This case should technically not be reached due to the initial model check,
                # but added as a safeguard.
                print(f"Error: Internal logic error - reached Gemini block with unhandled model: {model}")
                return None
        # Note: No 'else' needed here as provider check covers all supported cases

    # 4. Catch-all Exception Handling
    except Exception as e:
        print(f"!!! Critical Error during embedding generation for model {model}: {e}")
        print(f"    Text length: {len(text)} chars, approx {count_tokens(text, model)} tokens.")
        return None

    # Should not be reachable if logic is correct, but prevents potential 'None' return without error
    print(f"Error: Reached end of get_embedding function unexpectedly for model {model}.")
    return None

def generate_llm_response(prompt: str, max_tokens: int, temperature: float = 1.0, model=None) -> str:
    """
    Generate a response from the configured LLM provider.
    Handles OpenAI and Gemini, includes basic error checking.
    """
    if not model:
        raise ValueError("A model name must be specified for LLM generation.")
    if max_tokens <= 0:
        print("Warning: max_tokens must be positive. Using default 100.")
        max_tokens = 100

    model_id_lower = model.lower()

    try:
        # --- OpenAI ---
        if any(prefix in model_id_lower for prefix in ["gpt-4", "gpt-3.5-turbo"]):
            if not openai_client:
                raise RuntimeError("OpenAI client not initialized. Cannot use GPT models.")
            print(f"--- Calling OpenAI model: {model} ---")
            response = openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content
            print("--- OpenAI response received ---")
            return content.strip() if content else ""

        # --- Gemini ---
        elif "gemini" in model_id_lower:
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
            content = response.text
            # print("--- Gemini response received ---")
            return content.strip() if content else ""

        # --- Unsupported ---
        else:
            raise ValueError(f"Unsupported model type or client not configured: {model}")

    except Exception as e:
        print(f"!!! Error calling LLM model {model}: {e}")
        # Consider logging the prompt that caused the error
        # try:
        #     with open('failed_prompt.txt', 'w', encoding='utf-8') as f: f.write(prompt)
        # except: pass
        return f"[Error generating response with {model}]" # Return error message
# ---------------------------
# Indexing Function (Original - will be called by the worker)
# ---------------------------
# Modified slightly to be less verbose and potentially return status
def index_document(document_path: str, db_path: str, collection_name: str,
                   max_tokens: int = DEFAULT_MAX_TOKENS, overlap: int = DEFAULT_OVERLAP) -> bool:
    """
    Index a single document: checks existence, chunks, embeds, adds to collection.
    Returns True on success, False on failure. Designed to be called by a worker process.
    Uses global API clients and gets its own Chroma collection instance.
    """
    file_name = os.path.basename(document_path)
    # Less verbose printing inside the function, main process will show overall progress
    # print(f"Worker {os.getpid()} processing: {file_name}")

    if not os.path.exists(document_path):
        print(f"Error (Worker {os.getpid()}): Document not found: {document_path}")
        return False

    try:
        file_hash = compute_file_hash(document_path)
    except Exception as e:
        print(f"Error (Worker {os.getpid()}) hashing file {file_name}, skipping: {e}")
        return False

    processing_date = datetime.datetime.now().isoformat()

    try:
        collection = get_chroma_collection(db_path, collection_name)
        existing = collection.get(where={"file_hash": file_hash}, limit=1, include=[]) # Don't need data, just check existence
        if existing and existing.get('ids'):
            # print(f"Info (Worker {os.getpid()}): Document '{file_name}' already indexed. Skipping.")
            return True # Treat as success if already done
    except Exception as e:
        print(f"Warning (Worker {os.getpid()}): Error checking ChromaDB for hash {file_hash[:8]}...: {e}. Proceeding.")
        # Decide if this is a fatal error for the worker. Let's proceed.

    try:
        with open(document_path, "r", encoding="utf-8", errors='replace') as f:
            document = f.read()
    except Exception as e:
        print(f"Error (Worker {os.getpid()}) reading file {file_name}, skipping: {e}")
        return False

    if not document.strip():
        # print(f"Info (Worker {os.getpid()}): Document '{file_name}' is empty. Skipping.")
        return True # Treat as success (nothing to index)

    # print(f"Worker {os.getpid()} chunking {file_name}...") # Can be verbose
    raw_chunks = chunk_document_tokens(document, max_tokens=max_tokens, overlap=overlap)
    if not raw_chunks:
        # print(f"Info (Worker {os.getpid()}): No chunks generated for '{file_name}'.")
        return True # Treat as success (nothing to index)

    chunk_ids_batch = []
    embeddings_batch = []
    metadatas_batch = []
    # No tqdm here, main process handles overall progress
    for idx, (raw_chunk, start_tok, end_tok) in enumerate(raw_chunks):
        if not raw_chunk.strip(): continue

        # Use functions relying on global clients
        chunk_context = generate_chunk_context(document, raw_chunk)
        contextualized_text = f"{chunk_context}\n{raw_chunk}"
        tokens = count_tokens(raw_chunk)
        embedding_vector = get_embedding(contextualized_text, model=EMBEDDING_MODEL, task_type="retrieval_document")

        if embedding_vector is None:
            print(f"\nWarning (Worker {os.getpid()}): Skipping chunk {idx} in {file_name} due to embedding failure.")
            continue
        if embedding_vector.ndim != 1 or embedding_vector.size == 0:
             print(f"\nWarning (Worker {os.getpid()}): Embedding for chunk {idx} in {file_name} has bad shape {embedding_vector.shape}. Skipping.")
             continue

        chunk_id = f"{file_hash}_{idx}"
        metadata = {
            "file_hash": file_hash, "file_name": file_name, "processing_date": processing_date,
            "chunk_number": idx, "start_token": start_tok, "end_token": end_tok,
            "text": raw_chunk, "context": chunk_context,
            "contextualized_text": contextualized_text, "tokens": tokens
        }
        chunk_ids_batch.append(chunk_id)
        embeddings_batch.append(embedding_vector.tolist())
        metadatas_batch.append(metadata)

    if chunk_ids_batch:
        try:
            # Get collection instance again (or reuse if safe, but getting new one is safer)
            collection = get_chroma_collection(db_path, collection_name)
            # print(f"Worker {os.getpid()} adding {len(chunk_ids_batch)} chunks for {file_name}...") # Verbose
            collection.add(ids=chunk_ids_batch, embeddings=embeddings_batch, metadatas=metadatas_batch)
            # print(f"Worker {os.getpid()} finished indexing {file_name}.") # Verbose
            return True
        except Exception as e:
            print(f"\n!!! Error (Worker {os.getpid()}) adding batch to ChromaDB for {file_name}: {e}")
            return False # Indicate failure
    else:
        # print(f"Info (Worker {os.getpid()}): No valid chunks embedded for '{file_name}'.")
        return True # Success, just nothing to add

# ---------------------------
# Worker Function for Parallel Processing
# ---------------------------
def process_single_file_wrapper(file_path: str, db_path: str, collection_name: str,
                                max_tokens: int, overlap: int) -> Tuple[str, bool, str | None]:
    """
    Wrapper function for multiprocessing pool. Calls index_document and handles exceptions.
    Returns (file_path, success_status, error_message).
    """
    file_name = os.path.basename(file_path)
    try:
        # Optional: Re-initialize clients per process if global ones cause issues
        # initialize_clients() # Uncomment if needed

        # Check if clients needed for indexing are available in this worker
        # (This check might be redundant if initialize_clients() handles it, but good safety)
        needs_openai = any("gpt" in m for m in [EMBEDDING_MODEL, CHUNK_CONTEXT_MODEL])
        needs_gemini = any("gemini" in m or "embedding" in m for m in [EMBEDDING_MODEL, CHUNK_CONTEXT_MODEL])

        # Use the globally initialized clients (assuming they are process-safe enough)
        global openai_client, gemini_client
        if needs_openai and not openai_client:
            return (file_path, False, "OpenAI client not initialized in worker process.")
        if needs_gemini and not gemini_client:
             # Try initializing Gemini again just in case - sometimes needed per-process
             try:
                 if GEMINI_API_KEY:
                     from google import genai
                     gemini_client = genai.Client(api_key=GEMINI_API_KEY)
                     # print(f"Worker {os.getpid()} re-initialized Gemini client.")
                     if not gemini_client: raise RuntimeError("Failed re-init")
                 else:
                      return (file_path, False, "Gemini client not initialized (no API key) in worker process.")
             except Exception as init_err:
                  return (file_path, False, f"Gemini client failed to initialize in worker: {init_err}")

        # Call the actual indexing logic
        success = index_document(
            document_path=file_path,
            db_path=db_path,
            collection_name=collection_name,
            max_tokens=max_tokens,
            overlap=overlap
        )
        return (file_path, success, None) # No error message if success or handled failure
    except Exception as e:
        import traceback
        err_msg = f"CRITICAL Error processing {file_name} in worker {os.getpid()}: {e}\n{traceback.format_exc()}"
        print(err_msg) # Print critical errors immediately
        return (file_path, False, str(e)) # Return file path, failure status, and error message

# ---------------------------
# Retrieval and Query Functions (Largely Unchanged)
# ---------------------------
# ... (generate_subqueries, retrieve_chunks_for_query, generate_answer, iterative_rag_query, query_index) ...
# Ensure they use global clients correctly or adapt as needed.
def retrieve_chunks_for_query(query: str, db_path: str, collection_name: str, # Added collection_name
                              top_k: int) -> List[dict]:
    """
    Retrieve the top-K chunks from ChromaDB for a query, returning their metadata.
    Handles embedding errors and query errors.
    """
    if top_k <= 0: top_k = 1

    try:
        # Get collection - should be safe as it's likely read-only access here
        collection = get_chroma_collection(db_path, collection_name)

        # Get query embedding using global client
        query_vec = get_embedding(query, model=EMBEDDING_MODEL, task_type="retrieval_query")
        if query_vec is None:
            print(f"Error: Failed to embed query: '{query[:100]}...'")
            return []
        if not isinstance(query_vec, np.ndarray) or query_vec.ndim != 1:
             print(f"Error: Query embedding has unexpected type/shape: {type(query_vec)}/{query_vec.shape if isinstance(query_vec, np.ndarray) else 'N/A'}")
             return []


        # Perform ChromaDB query
        results = collection.query(
            query_embeddings=[query_vec.tolist()], # Chroma expects list of lists or np.ndarray
            n_results=top_k,
            include=['metadatas', 'distances'] # Essential data + similarity score
        )

        retrieved_chunks = []
        if results and results.get('ids') and results['ids'][0]: # Check if any results came back
             metadatas = results.get('metadatas', [[]])[0]
             distances = results.get('distances', [[]])[0]

             for i, meta in enumerate(metadatas):
                 if meta is None: continue # Skip if metadata is missing for some reason
                 # Add distance and calculated similarity (1 - cosine distance)
                 meta['distance'] = distances[i] if i < len(distances) else None
                 meta['similarity'] = (1.0 - distances[i]) if meta['distance'] is not None else None
                 retrieved_chunks.append(meta) # Add metadata dict to the list
        # else: # No results found is not an error, just return empty list
             # print(f"Debug: No results found for query '{query[:50]}...'")


        return retrieved_chunks

    except Exception as e:
        print(f"!!! Error querying ChromaDB collection '{collection_name}': {e}")
        import traceback
        traceback.print_exc() # Print stack trace for query errors
        return [] # Return empty list on error

def generate_subqueries(initial_query: str, model: str = SUBQUERY_MODEL) -> List[str]:
    """
    Generate expanded/alternative queries. Handles LLM errors and parsing variations.
    """
    n_queries=5
    prompt = (
        f"You are an expert query generator. Based on the user's question, create a list of exactly {n_queries} alternative or expanded queries. "
        f"These queries should explore different facets, keywords, or potential interpretations of the original question to improve document retrieval. "
        f"Return ONLY the list of queries, one per line. Do not include numbering, bullet points, or any introductory/concluding text.\n\n"
        f"Original Question: {initial_query}\n\n"
        f"Generated Queries:"
    )

    try:
        # Uses global client via generate_llm_response
        response_text = generate_llm_response(prompt, max_tokens=250, temperature=0.7, model=model)
        if "[Error generating response" in response_text: # Check for error from generate_llm_response
            raise RuntimeError("LLM failed to generate subqueries.")

        lines = response_text.splitlines()
        subqueries = [line.strip() for line in lines if line.strip()]

        cleaned_subqueries = []
        for q in subqueries:
            if len(q) > 2 and q[0].isdigit() and q[1] == '.': q = q.split('.', 1)[-1].strip()
            elif len(q) > 1 and q[0] in ['-', '*']: q = q[1:].strip()
            if q: cleaned_subqueries.append(q)

        if not cleaned_subqueries:
            print("Warning: Subquery generation returned empty result. Using original query only.")
            return [initial_query]

        return cleaned_subqueries[:n_queries]

    except Exception as e:
        print(f"Error generating subqueries with model {model}: {e}. Using original query only.")
        return [initial_query] # Fallback

def generate_answer(query: str, combined_context: str, retrieved_chunks: List[dict],
                    model: str = CHAT_MODEL) -> str:
    """
    Generate an answer using provided context and chunks, citing sources.
    Handles empty context and LLM errors. Uses global clients.
    """
    if not combined_context or not combined_context.strip():
         print("Warning: Combined context is empty for answer generation.")
         # return "Could not generate an answer because no relevant context was found."

    references = "\n".join(
        f"- {chunk.get('file_name', '?')} [Chunk #{chunk.get('chunk_number', '?')}, ~Toks {chunk.get('start_token', '?')}-{chunk.get('end_token', '?')}] (Sim: {chunk.get('similarity', 0.0):.3f})"
        for chunk in retrieved_chunks if chunk
    )

    MODEL_CONTEXT_LIMITS = { "gpt-4": 8000, "gpt-4o": 128000, "gpt-3.5-turbo": 16000, "gemini-1.5-flash": 1000000, "gemini-1.0-pro": 30720 } # Add Gemini Pro
    prompt_allowance_ratio = 0.75
    # Use model name directly, handle potential 'models/' prefix for Gemini
    clean_model_name = model.split('/')[-1] if '/' in model else model
    model_token_limit = MODEL_CONTEXT_LIMITS.get(clean_model_name, 8000) # Default if model unknown
    max_context_tokens_for_prompt = int(model_token_limit * prompt_allowance_ratio)


    context_token_count = count_tokens(combined_context, model=model)
    if context_token_count > max_context_tokens_for_prompt:
        print(f"Warning: Combined context ({context_token_count} tokens) exceeds estimated limit ({max_context_tokens_for_prompt}) for model {model}. Truncating.")
        combined_context = truncate_text(combined_context, max_context_tokens_for_prompt, model=model)

    prompt = (
        f"You are an AI assistant answering questions based *only* on the provided context. Follow these instructions carefully:\n"
        f"1. Analyze the 'Context' section below to find information relevant to the 'Question'.\n"
        f"2. Synthesize the relevant information into a coherent answer.\n"
        f"3. **Cite your sources** for every piece of information used in the answer using the format [Source: file_name, Chunk #N]. Refer to the 'Sources Available' list for details.\n"
        f"4. If the context does not contain enough information to answer the question completely, state that clearly. Do not add information not present in the context.\n"
        f"5. Format the answer clearly.\n\n"
        f"---\n"
        f"Context:\n{combined_context}\n\n"
        f"---\n"
        f"Sources Available:\n{references}\n\n"
        f"---\n"
        f"Question: {query}\n\n"
        f"---\n"
        f"Answer (with citations):"
    )

    try:
        with open('final_prompt.txt', 'w', encoding='utf-8') as f: f.write(prompt)
    except Exception as e:
        print(f"Warning: Could not write final_prompt.txt: {e}")

    prompt_tokens = count_tokens(prompt, model=model)
    model_total_limit = model_token_limit
    answer_max_tokens = max(100, model_total_limit - prompt_tokens - 200) # Safety buffer
    answer_max_tokens = min(answer_max_tokens, 4096) # Reasonable cap

    print(f"Generating final answer using {model} (max answer tokens: {answer_max_tokens})...")
    # Uses global client via generate_llm_response
    return generate_llm_response(prompt, max_tokens=answer_max_tokens, temperature=0.1, model=model)


def iterative_rag_query(initial_query: str, db_path: str, collection_name: str, # Added collection_name
                        top_k: int = DEFAULT_TOP_K,
                        subquery_model: str = SUBQUERY_MODEL,
                        answer_model: str = CHAT_MODEL) -> str:
    """
    Implements iterative RAG: generate subqueries, retrieve for all, deduplicate, generate final answer.
    """
    # 1. Generate Queries
    all_queries = [initial_query]
    generated_subqueries = generate_subqueries(initial_query, model=subquery_model)
    if generated_subqueries and generated_subqueries != [initial_query]:
        print("--- Generated Subqueries ---")
        for idx, subq in enumerate(generated_subqueries, 1): print(f"  {idx}. {subq}")
        all_queries.extend(generated_subqueries)
    else:
        print("--- Using only the initial query ---")

    # 2. Retrieve Chunks
    all_retrieved_chunks_dict: Dict[str, Dict[str, Any]] = {} # Deduplicate by unique chunk ID
    print("\n--- Retrieving Chunks ---")
    for q_idx, current_query in enumerate(all_queries):
         print(f"  Querying ({q_idx+1}/{len(all_queries)}): \"{current_query[:100]}...\"")
         retrieved = retrieve_chunks_for_query(current_query, db_path, collection_name, top_k)
         # print(f"    Retrieved {len(retrieved)} chunks for this query.") # Can be verbose
         for chunk in retrieved:
             file_hash = chunk.get('file_hash')
             chunk_number = chunk.get('chunk_number')
             if file_hash is not None and chunk_number is not None:
                 chunk_id = f"{file_hash}_{chunk_number}"
                 current_distance = chunk.get('distance')
                 existing_distance = all_retrieved_chunks_dict.get(chunk_id, {}).get('distance')

                 # Add if new, or replace if the new chunk is 'closer' (lower distance)
                 if chunk_id not in all_retrieved_chunks_dict or \
                    (current_distance is not None and (existing_distance is None or current_distance < existing_distance)):
                     all_retrieved_chunks_dict[chunk_id] = chunk
             else:
                 print(f"Warning: Retrieved chunk missing hash/number: {chunk.get('file_name')}")

    # 3. Process Retrieved Chunks
    unique_chunks_list = list(all_retrieved_chunks_dict.values())
    if not unique_chunks_list:
        return "No relevant chunks found in the database. Please index documents or refine your query."

    unique_chunks_list.sort(key=lambda c: c.get('distance', float('inf'))) # Sort by distance (lower is better)

    print(f"\n--- Processing Retrieved Data ---")
    print(f"Total unique chunks retrieved: {len(unique_chunks_list)}")
    # print("Top 5 Unique Chunks (by similarity):")
    # for i, chunk in enumerate(unique_chunks_list[:5], 1):
    #     print(f"  {i}. File: '{chunk.get('file_name', 'N/A')}', Chunk: {chunk.get('chunk_number', 'N/A')} (Similarity: {chunk.get('similarity', 0.0):.4f})")
    # print("--------------------------------")

    # 4. Combine Context
    combined_context = "\n\n---\n\n".join(
        f"Source Document: {chunk.get('file_name', 'N/A')}\n"
        f"Source Chunk Number: {chunk.get('chunk_number', '?')}\n"
        f"Content:\n{chunk.get('contextualized_text', chunk.get('text', ''))}"
        for chunk in unique_chunks_list
    )
    try:
        with open('combined_context.txt', 'w', encoding='utf-8') as f: f.write(combined_context)
    except Exception as e:
        print(f"Warning: Could not write combined_context.txt: {e}")

    # 5. Generate Final Answer
    print("\n--- Generating Final Answer ---")
    final_answer = generate_answer(initial_query, combined_context, unique_chunks_list, model=answer_model)
    return final_answer

def query_index(query: str, db_path: str, collection_name: str, # Added collection_name
                top_k: int = DEFAULT_TOP_K, answer_model: str = CHAT_MODEL) -> str:
    """
    Performs a direct query (no subqueries), retrieves chunks, and generates an answer.
    """
    print(f"--- Running Direct Query ---")
    print(f"Query: {query}")
    print(f"Retrieving Top K: {top_k}")

    retrieved_chunks = retrieve_chunks_for_query(query, db_path, collection_name, top_k)

    if not retrieved_chunks:
        return "No relevant chunks found in the database for this query."

    print("\n--- Retrieved Chunks (Direct Query) ---")
    # for i, chunk in enumerate(retrieved_chunks, 1):
    #     print(f"{i}. File: '{chunk.get('file_name', 'N/A')}', Chunk: {chunk.get('chunk_number', 'N/A')} (Similarity: {chunk.get('similarity', 0.0):.4f})")
    # print("--------------------------------------\n")

    combined_context = "\n\n---\n\n".join(
        f"Source Document: {chunk.get('file_name', 'N/A')}\n"
        f"Source Chunk Number: {chunk.get('chunk_number', '?')}\n"
        f"Content:\n{chunk.get('contextualized_text', chunk.get('text', ''))}"
        for chunk in retrieved_chunks
    )
    try:
        with open('direct_query_context.txt', 'w', encoding='utf-8') as f: f.write(combined_context)
    except Exception as e:
        print(f"Warning: Could not write direct_query_context.txt: {e}")

    print("--- Generating Answer (Direct Query) ---")
    answer = generate_answer(query, combined_context, retrieved_chunks, model=answer_model)
    return answer
# ---------------------------
# Main CLI
# ---------------------------
def main():
    test_mode_enabled = True # Set to False for normal use

    parser = argparse.ArgumentParser(
        description="RAG Script using ChromaDB: Index documents (in parallel) or query.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--mode", choices=["index", "query", "query_direct"], required=True,
                        help="Operation mode.")
    parser.add_argument("--document_path", type=str, default=None,
                        help="Path to a single .txt document to index.")
    parser.add_argument("--folder_path", type=str, default=None,
                        help="Path to a folder containing .txt documents to index.")
    parser.add_argument("--db_path", type=str, required=True,
                        help="Path for ChromaDB persistent storage.")
    parser.add_argument("--query", type=str, default=None,
                        help="User query (for query modes).")
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K,
                        help="Number of chunks to retrieve.")
    parser.add_argument("--collection_name", type=str, default=DEFAULT_CHROMA_COLLECTION_NAME,
                        help="Name of the ChromaDB collection.")
    # --- Argument for Parallelization ---
    parser.add_argument("--workers", type=int, default=os.cpu_count(), # Default to number of CPU cores
                        help="Number of parallel worker processes for indexing mode.")
    # ------------------------------------

    if test_mode_enabled:
        # ... (test mode setup as before, potentially add --workers to test_args) ...
        print("--- Running in Test Mode ---")
        test_args = {
            "mode": "index",
            "document_path": None,
            "folder_path": "cleaned_text/test",
            "db_path": "chunk_database/chroma_db_parallel", # Use separate DB for parallel test
            "query": "What are plausibilistic storylines?",
            "top_k": 3,
            "collection_name": "test_parallel_collection",
            "workers": 2 # Set specific worker count for testing
        }
        args = argparse.Namespace(**test_args)
        # Setup dummy files/folders if needed
        if args.folder_path and not os.path.exists(args.folder_path):
             print(f"Creating dummy test folder: {args.folder_path}")
             os.makedirs(args.folder_path)
             with open(os.path.join(args.folder_path, "doc1_parallel.txt"), "w") as f: f.write("Apples are red.")
             with open(os.path.join(args.folder_path, "doc2_parallel.txt"), "w") as f: f.write("Oranges are orange.")
             with open(os.path.join(args.folder_path, "doc3_parallel.txt"), "w") as f: f.write("Bananas are yellow and grow in bunches.")
             with open(os.path.join(args.folder_path, "doc4_parallel.txt"), "w") as f: f.write("Grapes can be green or purple.")
        os.makedirs(os.path.dirname(args.db_path), exist_ok=True)

    else:
        args = parser.parse_args()

    # --- Configuration & Validation ---
    # Initialize clients (moved global initialization earlier)
    # initialize_clients() # Already called globally

    # Check required clients *before* starting expensive operations
    required_openai = any("gpt" in m for m in [EMBEDDING_MODEL, CHUNK_CONTEXT_MODEL, SUBQUERY_MODEL, CHAT_MODEL])
    required_gemini = any("gemini" in m or "embedding" in m for m in [EMBEDDING_MODEL, CHUNK_CONTEXT_MODEL, SUBQUERY_MODEL, CHAT_MODEL])

    # Use the globally initialized clients for the check
    global openai_client, gemini_client
    if required_openai and not openai_client:
        parser.error("OpenAI API Key/Client missing/failed, but OpenAI models are configured.")
    if required_gemini and not gemini_client:
        parser.error("Google API Key/Client missing/failed, but Gemini models are configured.")

    print(f"\n--- Configuration ---")
    print(f"Mode: {args.mode}")
    print(f"ChromaDB Path: {args.db_path}")
    print(f"Collection Name: {args.collection_name}")
    if args.mode != 'index': print(f"Query: {args.query}")
    if args.mode != 'index': print(f"Top K: {args.top_k}")
    if args.mode == 'index': print(f"Parallel Workers: {args.workers}")
    print(f"Embedding Model: {EMBEDDING_MODEL}")
    print(f"Chat/Answer Model: {CHAT_MODEL}")
    print(f"--------------------\n")

    # Validate mode-specific arguments
    if args.mode == "index":
        if not args.document_path and not args.folder_path:
            parser.error("For 'index' mode, provide --document_path or --folder_path.")
        if args.document_path and args.folder_path:
             print("Warning: Both --document_path and --folder_path provided. Indexing folder only.")
             args.document_path = None
        if args.workers <= 0:
             print("Warning: Number of workers must be positive. Setting to 1.")
             args.workers = 1
    elif args.mode in ["query", "query_direct"]:
        if not args.query:
            parser.error("For query modes, --query must be provided.")

    # --- Execute Mode ---
    try:
        if args.mode == "index":
            try:
                os.makedirs(args.db_path, exist_ok=True)
            except OSError as e:
                 raise OSError(f"Cannot create ChromaDB directory '{args.db_path}': {e}")

            files_to_index = []
            if args.folder_path:
                if not os.path.isdir(args.folder_path):
                    raise ValueError(f"Folder path '{args.folder_path}' is not a directory.")
                print(f"Scanning folder: {args.folder_path}")
                for root, _, files in os.walk(args.folder_path):
                    for file in files:
                        if file.lower().endswith(".txt"):
                            files_to_index.append(os.path.join(root, file))
            elif args.document_path:
                 if not os.path.isfile(args.document_path):
                     raise ValueError(f"Document path '{args.document_path}' is not a file.")
                 if args.document_path.lower().endswith(".txt"):
                     files_to_index.append(args.document_path)
                 else:
                      print(f"Warning: Skipping non-txt file: {args.document_path}")

            if not files_to_index:
                 print("No .txt files found to index.")
                 return

            print(f"Found {len(files_to_index)} file(s) for indexing using up to {args.workers} workers.")

            # --- Parallel Processing Setup ---
            successful_indexes = 0
            failed_files = []

            # Use functools.partial to preset arguments for the worker function
            worker_func = functools.partial(
                process_single_file_wrapper,
                db_path=args.db_path,
                collection_name=args.collection_name,
                max_tokens=DEFAULT_MAX_TOKENS,
                overlap=DEFAULT_OVERLAP
            )

            # Create the pool and map the work
            # Using 'with' ensures pool is closed properly
            # Using imap_unordered to get results as they finish
            with multiprocessing.Pool(processes=args.workers) as pool:
                results_iterator = pool.imap_unordered(worker_func, files_to_index)

                # Process results with tqdm progress bar
                print("--- Starting Parallel Indexing ---")
                for file_path, success, error_msg in tqdm(results_iterator, total=len(files_to_index), desc="Indexing Files", unit="file"):
                    if success:
                        successful_indexes += 1
                    else:
                        failed_files.append((os.path.basename(file_path), error_msg))
            # --- End Parallel Processing ---

            print(f"\n--- Indexing Summary ---")
            print(f"Attempted to index {len(files_to_index)} files.")
            print(f"Successfully processed/indexed: {successful_indexes}")
            print(f"Failed attempts: {len(failed_files)}")
            if failed_files:
                print("Failed files:")
                for fname, err in failed_files:
                    print(f"  - {fname}: {err}")
            print(f"------------------------\n")

        elif args.mode == "query" or args.mode == "query_direct":
            if not os.path.isdir(args.db_path):
                 print(f"Error: ChromaDB directory '{args.db_path}' not found.")
                 return

            if args.mode == "query":
                final_answer = iterative_rag_query(args.query, args.db_path, args.collection_name,
                                                   top_k=args.top_k,
                                                   subquery_model=SUBQUERY_MODEL,
                                                   answer_model=CHAT_MODEL)
            else: # Direct query
                final_answer = query_index(args.query, args.db_path, args.collection_name,
                                           top_k=args.top_k,
                                           answer_model=CHAT_MODEL)

            print("\n" + "="*20 + " Final Answer " + "="*20)
            print(final_answer)
            print("="*54 + "\n")

    except Exception as e:
        print(f"\n\n!!! An unexpected error occurred in main execution: {e}")
        import traceback
        print("Traceback:")
        traceback.print_exc()

    # --- Post-Indexing Test Query (Only if test_mode_enabled) ---
    if test_mode_enabled and args.mode == "index":
        print("\n--- Running Test Query After Parallel Indexing (Test Mode) ---")
        test_query_mode = "query" # Or "query_direct"
        try:
            if test_query_mode == "query":
                 print("Running test iterative query...")
                 final_answer = iterative_rag_query(args.query, args.db_path, args.collection_name, top_k=args.top_k)
            else:
                 print("Running test direct query...")
                 final_answer = query_index(args.query, args.db_path, args.collection_name, top_k=args.top_k)

            print("\n=== Test Query Final Answer ===")
            print(final_answer)
            print("=============================")
        except Exception as test_e:
            print(f"\n!!! Error during post-indexing test query: {test_e}")


if __name__ == "__main__":
    # Important for multiprocessing on some OS (like Windows)
    multiprocessing.freeze_support() # Good practice, though often only needed for frozen apps
    main()