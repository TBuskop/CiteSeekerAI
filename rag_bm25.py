#!/usr/bin/env python3
"""
rag_chroma.py (Corrected & Parallelized Indexing with BM25)

A Retrieval-Augmented Generation (RAG) script with deep linking that:
  1. Reads given .txt documents (supports single file or folder).
  2. **Processes files in parallel during indexing.**
  3. Splits each document into semantically coherent chunks using a token-based sliding window,
     capturing token offsets for each chunk.
  4. Generates a short contextual summary for each chunk using the full document context.
  5. Computes an embedding for each contextualized chunk using configured embeddings model.
  6. **Tokenizes raw chunks for BM25.**
  7. **Gathers all processed chunk data (embeddings, metadata, BM25 tokens) in the main process.**
  8. **Builds and saves a BM25 index.**
  9. **Stores chunks (with metadata) in a ChromaDB collection.**
  10. Checks if a file has already been chunked (basic check, less robust with deferred indexing).
  11. In query mode, retrieves chunks using **hybrid search** (ChromaDB vector + BM25 lexical)
      and combines results using Reciprocal Rank Fusion (RRF).
  12. In iterative query mode, first expands the original query into subqueries, performs hybrid
      retrieval for each, combines and re-ranks all results, and then generates the final
      answer using the combined context and deep linking metadata.

Usage:
  # To index documents from a folder in parallel:
  python rag_chroma_bm25.py --mode index --folder_path path/to/documents --db_path ./hybrid_db --collection_name my_hybrid_docs --workers 4

  # To index a single document:
  python rag_chroma_bm25.py --mode index --document_path path/to/document.txt --db_path ./hybrid_db --collection_name my_hybrid_docs

  # To query iteratively (hybrid search):
  python rag_chroma_bm25.py --mode query --query "What about unique terms like BM25?" --db_path ./hybrid_db --collection_name my_hybrid_docs --top_k 5

  # To query directly (hybrid search):
  python rag_chroma_bm25.py --mode query_direct --query "Exact keyword match" --db_path ./hybrid_db --collection_name my_hybrid_docs --top_k 3
"""

import os
import argparse
import datetime
import hashlib
from typing import List, Dict, Any, Tuple, Optional # Added Optional
from tqdm import tqdm
from google import genai

import numpy as np
import tiktoken
import chromadb # Added ChromaDB
import pickle # For saving BM25 index
import re # For BM25 tokenization
from rank_bm25 import BM25Okapi # For lexical search

# --- Imports for Parallelization ---
import multiprocessing
import functools
# ---------------------------------

# --- NLTK for Tokenization (Optional but Recommended) ---
try:
    import nltk
    from nltk.corpus import stopwords
    # Attempt to load stopwords, download if necessary
    try:
        stop_words_english = set(stopwords.words('english'))
    except LookupError:
        print("NLTK stopwords not found. Downloading...")
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            stop_words_english = set(stopwords.words('english'))
            print("NLTK data downloaded.")
        except Exception as dl_err:
            print(f"Warning: Failed to download NLTK data: {dl_err}. Proceeding without stopwords.")
            stop_words_english = set() # Use empty set if download fails
except ImportError:
    print("Warning: NLTK not installed (`pip install nltk rank_bm25`). Using basic tokenization without stopwords.")
    stop_words_english = set() # Use empty set if NLTK not available
# ------------------------------------------------------

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
    print("Warning: Using default configuration values as config.py was not fully loaded.")
    exit(1) # Exit if config is critical and missing

# --- ChromaDB Configuration ---
DEFAULT_CHROMA_COLLECTION_NAME = "rag_chunks_hybrid_default"
# -----------------------------

# --- BM25 Configuration ---
# Define paths for saving/loading the BM25 index relative to the db_path
def get_bm25_paths(db_path: str, collection_name: str) -> Tuple[str, str]:
    """Gets the file paths for the BM25 index and ID mapping."""
    base_path = os.path.join(db_path, f"{collection_name}_bm25")
    index_path = f"{base_path}_index.pkl"
    map_path = f"{base_path}_ids.pkl"
    return index_path, map_path

# Global variables to cache loaded BM25 index (optional, for query performance)
bm25_index_cache = {}
bm25_ids_cache = {}
# -------------------------


# --- Global Variables for API Clients (Initialize later if needed) ---
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
        except ImportError:
            print("Warning: OpenAI library not installed (`pip install openai`). OpenAI models will be unavailable.")
        except Exception as e:
            print(f"Warning: Failed to initialize OpenAI client: {e}")

    # Initialize Gemini Client
    if GEMINI_API_KEY:
        try:
            from google import genai # Import here
            gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        except ImportError:
            print("Warning: Google Generative AI library not installed (`pip install google-generativeai`). Gemini models will be unavailable.")
        except Exception as e:
            print(f"Warning: Failed to configure Gemini client: {e}")

# Call initialization once at the start
initialize_clients()
# --- End API Client Initialization ---


# ---------------------------
# ChromaDB Client Setup
# ---------------------------
def get_chroma_collection(db_path: str, collection_name: str) -> chromadb.Collection:
    """
    Initialize the ChromaDB client and get/create the specified collection.
    Uses persistent storage at db_path. SAFE FOR MULTIPROCESSING (usually).
    """
    if not collection_name:
        raise ValueError("ChromaDB collection name cannot be empty.")
    try:
        chroma_client = chromadb.PersistentClient(path=db_path)
        collection = chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        return collection
    except Exception as e:
        print(f"!!! Process {os.getpid()} Error connecting/creating ChromaDB collection '{collection_name}' at path '{db_path}': {e}")
        raise


# ---------------------------
# File Hashing (Unchanged)
# ---------------------------
def compute_file_hash(file_path: str) -> str:
    """Compute a SHA256 hash of the file contents."""
    hasher = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            while True:
                chunk = f.read(4096)
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
# Chunking and Embedding Helpers (Unchanged logic)
# ---------------------------
def count_tokens(text: str, model: str = EMBEDDING_MODEL) -> int:
    """Count tokens using tiktoken, fallback to space split."""
    try:
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
                      count_tokens._encoding_cache[model] = None

        encoding = count_tokens._encoding_cache.get(model)
        if encoding:
            return len(encoding.encode(text))
        else:
             return len(text.split())
    except Exception as e:
        print(f"Error encoding text with tiktoken: {e}. Using simple space split.")
        return len(text.split())

def chunk_document_tokens(document: str,
                          max_tokens: int = DEFAULT_MAX_TOKENS,
                          overlap: int = DEFAULT_OVERLAP) -> List[tuple]:
    """Split document into token-based chunks with overlap."""
    if max_tokens <= 0: raise ValueError("max_tokens must be positive.")
    if overlap < 0: raise ValueError("overlap cannot be negative.")
    if overlap >= max_tokens: overlap = max_tokens // 2

    try:
        encoding = tiktoken.encoding_for_model(EMBEDDING_MODEL)
    except Exception:
        try: encoding = tiktoken.get_encoding("cl100k_base")
        except Exception as e: raise ValueError(f"Could not get tiktoken encoding: {e}")

    try: tokens = encoding.encode(document)
    except Exception as e: print(f"Error encoding doc for chunking: {e}."); return []
    if not tokens: return []

    chunks = []
    start_token_idx = 0
    total_tokens = len(tokens)
    while start_token_idx < total_tokens:
        end_token_idx = min(start_token_idx + max_tokens, total_tokens)
        chunk_tokens = tokens[start_token_idx:end_token_idx]
        if not chunk_tokens: break
        try:
            chunk_text = encoding.decode(chunk_tokens, errors='replace').strip()
        except Exception as e:
            print(f"Error decoding tokens at {start_token_idx}: {e}. Skipping.")
            next_start = start_token_idx + max_tokens - overlap
            start_token_idx = next_start if next_start > start_token_idx else start_token_idx + 1
            continue
        if chunk_text:
            chunks.append((chunk_text, start_token_idx, end_token_idx))
        next_start = start_token_idx + max_tokens - overlap
        start_token_idx = next_start if next_start > start_token_idx else start_token_idx + 1
    return chunks

def truncate_text(text: str, token_limit: int, model: str = CHAT_MODEL) -> str:
    """Truncates text to a specified token limit."""
    if token_limit <= 0: return ""
    try: encoding = tiktoken.encoding_for_model(model)
    except Exception: encoding = tiktoken.get_encoding("cl100k_base")
    try:
        tokens = encoding.encode(text)
        if len(tokens) > token_limit:
            tokens = tokens[:token_limit]
            text = encoding.decode(tokens, errors='replace')
        return text
    except Exception as e: print(f"Error truncating text: {e}. Returning original."); return text

def generate_chunk_context(document: str, chunk: str, token_limit: int = 30000,
                           context_length: int = DEFAULT_CONTEXT_LENGTH,
                           model: str = CHUNK_CONTEXT_MODEL) -> str:
    """Generate succinct context for a chunk within the document."""
    global openai_client, gemini_client
    if not document.strip() or not chunk.strip(): return "Context unavailable."

    doc_token_count = count_tokens(document, model=model)
    if doc_token_count > token_limit:
        document = truncate_text(document, token_limit, model=model)

    prompt = (
        f"<document>\n{document}\n</document>\n"
        f"<chunk>\n{chunk}\n</chunk>\n"
        "Provide a short, succinct context (1-2 sentences) describing where this chunk fits "
        "within the overall document. Focus on the surrounding topic or section. "
        "Answer ONLY with the succinct context itself."
    )
    try:
        if context_length <= 0: context_length = 50
        response = generate_llm_response(prompt, context_length, temperature=0.5, model=model)
        return response.replace("Context:", "").strip()
    except Exception as e:
        print(f"Error generating chunk context with {model}: {e}")
        return "Error generating context."

def get_embedding(text: str, model: str = EMBEDDING_MODEL, task_type=None) -> Optional[np.ndarray]:
    """Get embedding using OpenAI or Gemini models."""
    if not text or not text.strip():
        print("Warning: Attempting to embed empty text. Returning None.")
        return None
    text = text.replace("\n", " ")

    try:
        provider = None
        if model in ["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"]:
            provider = "openai"
            if not openai_client: raise RuntimeError("OpenAI client unavailable.")
        elif model in ["embedding-001", "models/embedding-001", "text-embedding-004", "models/text-embedding-004"]:
            provider = "gemini"
            if not gemini_client: raise RuntimeError("Gemini client unavailable.")
        else: raise ValueError(f"Unsupported embedding model '{model}'.")

        if provider == "openai":
            response = openai_client.embeddings.create(input=[text], model=model)
            return np.array(response.data[0].embedding)
        elif provider == "gemini":
            api_model_name = model if model.startswith("models/") else f"models/{model}"
            if api_model_name in ["models/embedding-001"]:
                valid_tasks = ["retrieval_document", "retrieval_query", "semantic_similarity", "classification", "clustering"]
                task_type = task_type if task_type in valid_tasks else "retrieval_document"
                response = gemini_client.models.embed_content(model=api_model_name, contents=text, task_type=task_type)
            else: # Assumes text-embedding-004 etc. don't need task_type
                response = gemini_client.models.embed_content(model=api_model_name, contents=text)

            # Handle potential response structures
            if hasattr(response, 'embedding') and response.embedding and hasattr(response.embedding, 'values'):
                return np.array(response.embedding.values)
            elif hasattr(response, 'embeddings') and response.embeddings and hasattr(response.embeddings[0], 'values'):
                return np.array(response.embeddings[0].values)
            elif "embedding" in response: # Older structure?
                 return np.array(response["embedding"])
            else:
                print(f"Error: Unexpected Gemini response structure for {api_model_name}: {response}")
                return None
    except Exception as e:
        print(f"!!! Critical Error during embedding generation for model {model}: {e}")
        return None
    print(f"Error: Reached end of get_embedding unexpectedly for model {model}.")
    return None

def generate_llm_response(prompt: str, max_tokens: int, temperature: float = 1.0, model=None) -> str:
    """Generate LLM response using OpenAI or Gemini."""
    if not model: raise ValueError("Model name required for LLM generation.")
    if max_tokens <= 0: max_tokens = 100

    model_id_lower = model.lower()
    try:
        if any(prefix in model_id_lower for prefix in ["gpt-4", "gpt-3.5-turbo"]):
            if not openai_client: raise RuntimeError("OpenAI client unavailable.")
            response = openai_client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": prompt}],
                temperature=temperature, max_tokens=max_tokens,
            )
            content = response.choices[0].message.content
            return content.strip() if content else ""
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
        else:
            raise ValueError(f"Unsupported LLM model type: {model}")
    except Exception as e:
        print(f"!!! Error calling LLM model {model}: {e}")
        return f"[Error generating response with {model}]"

# ---------------------------
# BM25 Specific Helpers
# ---------------------------
def tokenize_text_bm25(text: str) -> List[str]:
    """Simple tokenizer for BM25: lowercase, alphanumeric, split, remove stopwords."""
    if not isinstance(text, str): return []
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    tokens = text.split()
    # Use globally loaded stopwords
    tokens = [word for word in tokens if word and word not in stop_words_english]
    return tokens

# Global placeholders for loaded BM25 index
bm25_instance: Optional[BM25Okapi] = None
bm25_chunk_ids_ordered: Optional[List[str]] = None

def load_bm25_index(db_path: str, collection_name: str) -> bool:
    """Loads the BM25 index and ID mapping from disk into global vars. Returns True on success."""
    global bm25_instance, bm25_chunk_ids_ordered
    cache_key = (db_path, collection_name)

    if cache_key in bm25_index_cache:
        bm25_instance = bm25_index_cache[cache_key]
        bm25_chunk_ids_ordered = bm25_ids_cache[cache_key]
        # print("Using cached BM25 index.") # Debug
        return True

    index_path, map_path = get_bm25_paths(db_path, collection_name)

    if os.path.exists(index_path) and os.path.exists(map_path):
        try:
            print(f"Loading BM25 index from: {index_path}")
            with open(index_path, 'rb') as f_idx:
                bm25_instance = pickle.load(f_idx)
            with open(map_path, 'rb') as f_map:
                bm25_chunk_ids_ordered = pickle.load(f_map)

            bm25_index_cache[cache_key] = bm25_instance
            bm25_ids_cache[cache_key] = bm25_chunk_ids_ordered
            print(f"BM25 index loaded successfully ({len(bm25_chunk_ids_ordered)} documents).")
            return True
        except Exception as e:
            print(f"!!! Error loading BM25 index files: {e}")
            bm25_instance = None
            bm25_chunk_ids_ordered = None
            return False
    else:
        # print(f"Warning: BM25 index files not found at {index_path} or {map_path}.")
        bm25_instance = None
        bm25_chunk_ids_ordered = None
        return False

def retrieve_chunks_bm25(query: str, db_path: str, collection_name: str, top_k: int) -> List[Tuple[str, float]]:
    """Retrieves top-k chunk IDs and BM25 scores. Returns list of (chunk_id, score)."""
    if bm25_instance is None or bm25_chunk_ids_ordered is None:
        if not load_bm25_index(db_path, collection_name):
            print("BM25 index is not available for querying.")
            return []

    tokenized_query = tokenize_text_bm25(query)
    if not tokenized_query:
        print("Warning: BM25 query empty after tokenization.")
        return []

    try:
        # Get scores for ALL documents in the index
        # Ensure bm25_instance and bm25_chunk_ids_ordered are not None (checked by load_bm25_index)
        scores = bm25_instance.get_scores(tokenized_query) # type: ignore

        # Combine scores with their original indices -> chunk IDs
        scored_indices = list(enumerate(scores))
        scored_indices.sort(key=lambda item: item[1], reverse=True)

        # Get top K results (chunk_id, score), filtering non-positive scores
        results = []
        for idx, score in scored_indices:
            if len(results) >= top_k: break
            if score > 0:
                 if idx < len(bm25_chunk_ids_ordered): # type: ignore # Safety check
                    chunk_id = bm25_chunk_ids_ordered[idx] # type: ignore
                    results.append((chunk_id, score))
                 else:
                    print(f"Warning: BM25 index {idx} out of bounds for chunk IDs (len {len(bm25_chunk_ids_ordered)}).") # type: ignore

        return results
    except Exception as e:
        print(f"!!! Error during BM25 scoring/ranking: {e}")
        import traceback; traceback.print_exc()
        return []
# ---------------------------
# End BM25 Specific Helpers
# ---------------------------


# ---------------------------
# Indexing Function (Worker Task)
# ---------------------------
def index_document(document_path: str,
                   max_tokens: int = DEFAULT_MAX_TOKENS,
                   overlap: int = DEFAULT_OVERLAP) -> List[Dict]:
    """
    Processes a single document: reads, chunks, gets context/embedding, tokenizes for BM25.
    Returns a list of dictionaries for each chunk, ready for central aggregation.
    Returns empty list on failure or if skipped.
    """
    file_name = os.path.basename(document_path)
    processed_chunk_data = []

    if not os.path.exists(document_path):
        print(f"Error (Worker {os.getpid()}): Doc not found: {document_path}")
        return []

    try: file_hash = compute_file_hash(document_path)
    except Exception as e: print(f"Error (Worker {os.getpid()}) hashing {file_name}: {e}"); return []

    processing_date = datetime.datetime.now().isoformat()

    try:
        with open(document_path, "r", encoding="utf-8", errors='replace') as f:
            document = f.read()
    except Exception as e: print(f"Error (Worker {os.getpid()}) reading {file_name}: {e}"); return []
    if not document.strip(): return [] # Skip empty

    raw_chunks = chunk_document_tokens(document, max_tokens=max_tokens, overlap=overlap)
    if not raw_chunks: return []

    for idx, (raw_chunk, start_tok, end_tok) in enumerate(raw_chunks):
        if not raw_chunk.strip(): continue

        chunk_context = generate_chunk_context(document, raw_chunk)
        contextualized_text = f"{chunk_context}\n{raw_chunk}"
        tokens_count = count_tokens(raw_chunk)
        embedding_vector = get_embedding(contextualized_text, model=EMBEDDING_MODEL, task_type="retrieval_document")

        if embedding_vector is None:
            print(f"Warning (Worker {os.getpid()}): Skipping chunk {idx} in {file_name} (embedding failed).")
            continue
        if embedding_vector.ndim != 1 or embedding_vector.size == 0:
             print(f"Warning (Worker {os.getpid()}): Skipping chunk {idx} in {file_name} (bad embed shape {embedding_vector.shape}).")
             continue

        chunk_id = f"{file_hash}_{idx}"
        metadata = {
            "file_hash": file_hash, "file_name": file_name, "processing_date": processing_date,
            "chunk_number": idx, "start_token": start_tok, "end_token": end_tok,
            "text": raw_chunk, "context": chunk_context,
            "contextualized_text": contextualized_text, "tokens": tokens_count
        }
        bm25_tokens = tokenize_text_bm25(raw_chunk) # Tokenize for BM25

        processed_chunk_data.append({
            "id": chunk_id,
            "embedding": embedding_vector.tolist(), # Convert numpy array
            "metadata": metadata,
            "bm25_tokens": bm25_tokens # Add tokens for central BM25 index building
        })

    return processed_chunk_data

# ---------------------------
# Worker Function Wrapper for Parallel Processing
# ---------------------------
def process_single_file_wrapper(file_path: str,
                                max_tokens: int, overlap: int) -> Tuple[str, bool, str | None, List[Dict]]:
    """
    Wrapper for multiprocessing pool. Calls index_document.
    Returns (file_path, success_status, error_message, list_of_processed_chunk_data).
    """
    file_name = os.path.basename(file_path)
    processed_data = []
    try:
        # Client initialization should happen globally or be robust to multiprocessing.
        # If issues arise, uncommenting re-initialization *might* help but can be slow.
        # initialize_clients()

        # Check needed clients (redundant if global init worked, but safe)
        needs_openai = any("gpt" in m for m in [EMBEDDING_MODEL, CHUNK_CONTEXT_MODEL])
        needs_gemini = any("gemini" in m or "embedding" in m for m in [EMBEDDING_MODEL, CHUNK_CONTEXT_MODEL])
        global openai_client, gemini_client
        if needs_openai and not openai_client: return (file_path, False, "OpenAI client missing in worker.", [])
        if needs_gemini and not gemini_client:
             try: # Try re-init Gemini in worker if needed
                 if GEMINI_API_KEY: gemini_client = genai.Client(api_key=GEMINI_API_KEY)
                 if not gemini_client: raise RuntimeError("Failed Gemini re-init")
             except Exception as init_err: return (file_path, False, f"Gemini client fail in worker: {init_err}", [])

        # Call the actual processing logic
        processed_data = index_document(
            document_path=file_path,
            max_tokens=max_tokens,
            overlap=overlap
        )
        # Success means no critical wrapper error, even if file yielded no chunks
        return (file_path, True, None, processed_data)
    except Exception as e:
        import traceback
        err_msg = f"CRITICAL Error processing {file_name} in worker {os.getpid()}: {e}\n{traceback.format_exc()}"
        print(err_msg)
        return (file_path, False, str(e), []) # Return failure and empty list

# ---------------------------
# Retrieval and Query Functions
# ---------------------------
def retrieve_chunks_for_query(query: str, db_path: str, collection_name: str,
                              top_k: int) -> List[dict]:
    """Retrieve top-K chunks from ChromaDB (vector search), returning metadata."""
    if top_k <= 0: top_k = 1
    retrieved_chunks = []
    try:
        collection = get_chroma_collection(db_path, collection_name)
        query_vec = get_embedding(query, model=EMBEDDING_MODEL, task_type="retrieval_query")
        if query_vec is None: print(f"Error: Failed to embed query."); return []
        if not isinstance(query_vec, np.ndarray) or query_vec.ndim != 1:
             print(f"Error: Query embedding bad shape {type(query_vec)}."); return []

        results = collection.query(
            query_embeddings=[query_vec.tolist()],
            n_results=top_k,
            include=['metadatas', 'distances']
        )

        if results and results.get('ids') and results['ids'][0]:
             metadatas = results.get('metadatas', [[]])[0]
             distances = results.get('distances', [[]])[0]
             for i, meta in enumerate(metadatas):
                 if meta is None: continue
                 dist = distances[i] if i < len(distances) else None
                 meta['distance'] = dist
                 meta['similarity'] = (1.0 - dist) if dist is not None else None
                 retrieved_chunks.append(meta)
        return retrieved_chunks
    except Exception as e:
        print(f"!!! Error querying ChromaDB '{collection_name}': {e}")
        import traceback; traceback.print_exc()
        return []

# --- Helper for RRF Combination ---
def combine_results_rrf(vector_results: List[Dict], bm25_results: List[Tuple[str, float]],
                        db_path: str, collection_name: str, k_rrf: int = 60) -> List[Dict]:
    """
    Combines vector and BM25 results using Reciprocal Rank Fusion (RRF).
    Fetches metadata for BM25-only results from ChromaDB.
    Returns a sorted list of unique chunk metadata dictionaries.
    """
    combined_scores: Dict[str, float] = {}
    chunk_metadata_cache: Dict[str, Dict] = {} # Cache metadata fetched or from vector results

    # Process vector results (rank based on order/distance)
    # Vector results already contain metadata
    vector_results.sort(key=lambda c: c.get('distance', float('inf'))) # Ensure sorted by distance
    for rank, chunk_meta in enumerate(vector_results):
        file_hash = chunk_meta.get('file_hash')
        chunk_number = chunk_meta.get('chunk_number')
        if file_hash is None or chunk_number is None: continue
        chunk_id = f"{file_hash}_{chunk_number}"

        score = 1.0 / (k_rrf + rank)
        combined_scores[chunk_id] = combined_scores.get(chunk_id, 0.0) + score
        if chunk_id not in chunk_metadata_cache:
            chunk_metadata_cache[chunk_id] = chunk_meta

    # Process BM25 results (rank based on order)
    bm25_ids_to_fetch = []
    for rank, (chunk_id, _) in enumerate(bm25_results): # BM25 score itself isn't used in RRF, only rank
        score = 1.0 / (k_rrf + rank)
        combined_scores[chunk_id] = combined_scores.get(chunk_id, 0.0) + score
        if chunk_id not in chunk_metadata_cache:
            bm25_ids_to_fetch.append(chunk_id)

    # Fetch metadata for BM25-only results if needed
    if bm25_ids_to_fetch:
        unique_ids_to_fetch = list(set(bm25_ids_to_fetch))
        # print(f"Fetching metadata for {len(unique_ids_to_fetch)} BM25-specific chunks...") # Debug
        if unique_ids_to_fetch:
            try:
                collection = get_chroma_collection(db_path, collection_name)
                fetched_data = collection.get(ids=unique_ids_to_fetch, include=['metadatas'])
                if fetched_data and fetched_data.get('ids'):
                    for i, fetched_id in enumerate(fetched_data['ids']):
                        if fetched_id in combined_scores: # Check if relevant
                            chunk_metadata_cache[fetched_id] = fetched_data['metadatas'][i]
                            # Add placeholder distance/similarity if needed later
                            chunk_metadata_cache[fetched_id]['distance'] = float('inf') # Indicate it wasn't from vector match
                            chunk_metadata_cache[fetched_id]['similarity'] = -1.0
            except Exception as fetch_err:
                print(f"!!! Error fetching metadata for BM25 results: {fetch_err}")

    # Sort chunk IDs by their combined RRF score
    sorted_chunk_ids = sorted(combined_scores.keys(), key=lambda cid: combined_scores[cid], reverse=True)

    # Create the final list of unique metadata dictionaries in RRF order
    final_results = []
    for chunk_id in sorted_chunk_ids:
        if chunk_id in chunk_metadata_cache:
            metadata = chunk_metadata_cache[chunk_id].copy() # Avoid modifying cache
            metadata['rrf_score'] = combined_scores[chunk_id]
            final_results.append(metadata)
        else:
             print(f"Warning: Metadata not found for chunk {chunk_id} after fetching.") # Should not happen

    return final_results

def generate_subqueries(initial_query: str, model: str = SUBQUERY_MODEL) -> List[str]:
    """Generate expanded/alternative queries."""
    n_queries=5
    prompt = (
        f"Generate {n_queries} alternative or expanded queries based on:\n"
        f"'{initial_query}'\n"
        f"Return ONLY the queries, one per line. No numbering or introduction."
    )
    try:
        response_text = generate_llm_response(prompt, max_tokens=250, temperature=0.7, model=model)
        if "[Error generating response" in response_text: raise RuntimeError("LLM failed.")
        lines = response_text.strip().splitlines()
        subqueries = [line.strip().lstrip('-* ').lstrip('0123456789. ') for line in lines if line.strip()]
        return subqueries[:n_queries] if subqueries else [initial_query]
    except Exception as e:
        print(f"Error generating subqueries with {model}: {e}. Using original query."); return [initial_query]

def generate_answer(query: str, combined_context: str, retrieved_chunks: List[dict],
                    model: str = CHAT_MODEL) -> str:
    """Generate answer using context and chunks, citing sources."""
    if not combined_context or not combined_context.strip():
         return "Could not generate an answer: no relevant context found."

    # Sort chunks for citation consistency (e.g., by RRF score or file/chunk#)
    retrieved_chunks.sort(key=lambda c: (c.get('file_name', ''), c.get('chunk_number', 0)))

    references = "\n".join(
        f"- {chunk.get('file_name', '?')} [Chunk #{chunk.get('chunk_number', '?')}] (RRF: {chunk.get('rrf_score', 0.0):.3f})"
        for chunk in retrieved_chunks if chunk
    )

    # Truncate context based on model limit
    MODEL_CONTEXT_LIMITS = { "gpt-4": 8000, "gpt-4o": 128000, "gpt-3.5-turbo": 16000, "gemini-1.5-flash": 1000000, "gemini-1.0-pro": 30720 }
    clean_model_name = model.split('/')[-1] if '/' in model else model
    model_token_limit = MODEL_CONTEXT_LIMITS.get(clean_model_name, 8000)
    max_context_tokens = int(model_token_limit * 0.75) # Leave room for prompt/answer

    context_tokens = count_tokens(combined_context, model=model)
    if context_tokens > max_context_tokens:
        print(f"Warning: Combined context ({context_tokens} tokens) too long for {model} ({max_context_tokens}). Truncating.")
        combined_context = truncate_text(combined_context, max_context_tokens, model=model)

    prompt = (
        f"Answer the following question based *only* on the provided context. "
        f"Cite sources using [Source: file_name, Chunk #N] format for every claim, referring to the 'Sources Available' list.\n\n"
        f"Context:\n{combined_context}\n\n"
        f"Sources Available:\n{references}\n\n"
        f"Question: {query}\n\n"
        f"Answer (with citations):"
    )

    try: 
        with open('final_prompt.txt', 'w', encoding='utf-8') as f: f.write(prompt)
    except Exception as e: print(f"Warning: Could not write final_prompt.txt: {e}")

    prompt_tokens = count_tokens(prompt, model=model)
    answer_max_tokens = max(150, model_token_limit - prompt_tokens - 200) # Safety buffer
    answer_max_tokens = min(answer_max_tokens, 4096) # Cap

    print(f"Generating final answer using {model} (max answer tokens: {answer_max_tokens})...")
    return generate_llm_response(prompt, max_tokens=answer_max_tokens, temperature=0.1, model=model)

def iterative_rag_query(initial_query: str, db_path: str, collection_name: str,
                        top_k: int = DEFAULT_TOP_K,
                        subquery_model: str = SUBQUERY_MODEL,
                        answer_model: str = CHAT_MODEL) -> str:
    """Iterative RAG with HYBRID retrieval (Vector + BM25 + RRF)."""
    # 1. Generate Queries
    all_queries = [initial_query]
    generated_subqueries = generate_subqueries(initial_query, model=subquery_model)
    if generated_subqueries and generated_subqueries != [initial_query]:
        print("--- Generated Subqueries ---")
        for idx, subq in enumerate(generated_subqueries, 1): print(f"  {idx}. {subq}")
        all_queries.extend(generated_subqueries)
    else: print("--- Using only the initial query ---")

    # 2. Retrieve Chunks (Hybrid) for all queries
    vector_results_all = []
    bm25_results_all = []
    fetch_k = max(top_k, int(top_k * 1.5)) # Fetch more candidates for RRF

    print("\n--- Retrieving Chunks (Hybrid Vector + BM25) ---")
    for q_idx, current_query in enumerate(all_queries):
         print(f"  Querying ({q_idx+1}/{len(all_queries)}): \"{current_query[:100]}...\"")
         vector_results_all.extend(retrieve_chunks_for_query(current_query, db_path, collection_name, fetch_k))
         bm25_results_all.extend(retrieve_chunks_bm25(current_query, db_path, collection_name, fetch_k))

    # 3. Deduplicate and Combine using RRF
    print(f"\n--- Combining {len(vector_results_all)} Vector & {len(bm25_results_all)} BM25 candidate results ---")

    # Deduplicate vector results (keep lowest distance)
    deduped_vector_results_dict = {}
    for chunk_meta in vector_results_all:
        chunk_id = f"{chunk_meta.get('file_hash')}_{chunk_meta.get('chunk_number')}"
        current_dist = chunk_meta.get('distance', float('inf'))
        if chunk_id not in deduped_vector_results_dict or current_dist < deduped_vector_results_dict[chunk_id].get('distance', float('inf')):
             deduped_vector_results_dict[chunk_id] = chunk_meta
    deduped_vector_results = list(deduped_vector_results_dict.values())

    # Deduplicate BM25 results (keep highest score)
    deduped_bm25_results_dict = {}
    for chunk_id, score in bm25_results_all:
         if chunk_id not in deduped_bm25_results_dict or score > deduped_bm25_results_dict[chunk_id]:
              deduped_bm25_results_dict[chunk_id] = score
    deduped_bm25_results = list(deduped_bm25_results_dict.items()) # List of (id, score)

    print(f"Unique Vector candidates: {len(deduped_vector_results)}, Unique BM25 candidates: {len(deduped_bm25_results)}")

    # Combine using RRF
    unique_chunks_list = combine_results_rrf(
        deduped_vector_results, deduped_bm25_results, db_path, collection_name
    )
    unique_chunks_list = unique_chunks_list[:top_k] # Limit to final top_k

    if not unique_chunks_list:
        return "No relevant chunks found in the database using hybrid iterative search."

    # 4. Process Final Chunks & Generate Context
    print(f"\n--- Processing Top {len(unique_chunks_list)} Combined Chunks ---")
    combined_context = "\n\n---\n\n".join(
        f"Source Document: {chunk.get('file_name', 'N/A')}\n"
        f"Source Chunk Number: {chunk.get('chunk_number', '?')}\n"
        f"Content:\n{chunk.get('contextualized_text', chunk.get('text', ''))}"
        for chunk in unique_chunks_list
    )
    try: 
        with open('combined_context.txt', 'w', encoding='utf-8') as f: f.write(combined_context)
    except Exception as e: print(f"Warning: Could not write combined_context.txt: {e}")

    # 5. Generate Final Answer
    print("\n--- Generating Final Answer (Iterative Hybrid Retrieval) ---")
    final_answer = generate_answer(initial_query, combined_context, unique_chunks_list, model=answer_model)
    return final_answer

def query_index(query: str, db_path: str, collection_name: str,
                top_k: int = DEFAULT_TOP_K, answer_model: str = CHAT_MODEL) -> str:
    """Direct query using HYBRID retrieval (Vector + BM25 + RRF)."""
    print(f"--- Running Direct Hybrid Query ---")
    print(f"Query: {query}")
    fetch_k = max(top_k, int(top_k * 1.5)) # Fetch more candidates

    # 1. Retrieve Chunks (Hybrid)
    vector_results = retrieve_chunks_for_query(query, db_path, collection_name, fetch_k)
    bm25_results = retrieve_chunks_bm25(query, db_path, collection_name, fetch_k)
    print(f"Retrieved {len(vector_results)} vector candidates, {len(bm25_results)} BM25 candidates.")

    # 2. Combine using RRF
    unique_chunks_list = combine_results_rrf(
        vector_results, bm25_results, db_path, collection_name
    )
    unique_chunks_list = unique_chunks_list[:top_k] # Limit to final top_k

    if not unique_chunks_list:
        return "No relevant chunks found in the database using direct hybrid search."

    # 3. Process Final Chunks & Generate Context
    print(f"\n--- Processing Top {len(unique_chunks_list)} Combined Chunks (Direct Query) ---")
    combined_context = "\n\n---\n\n".join(
        f"Source Document: {chunk.get('file_name', 'N/A')}\n"
        f"Source Chunk Number: {chunk.get('chunk_number', '?')}\n"
        f"Content:\n{chunk.get('contextualized_text', chunk.get('text', ''))}"
        for chunk in unique_chunks_list
    )
    try: 
        with open('direct_query_context.txt', 'w', encoding='utf-8') as f: f.write(combined_context)
    except Exception as e: print(f"Warning: Could not write direct_query_context.txt: {e}")

    # 4. Generate Final Answer
    print("--- Generating Answer (Direct Hybrid Query) ---")
    answer = generate_answer(query, combined_context, unique_chunks_list, model=answer_model)
    return answer

# ---------------------------
# Main CLI
# ---------------------------
def main():
    test_mode_enabled = True # Set to False for normal use # <<< CHANGE THIS BACK TO False FOR NORMAL USE

    parser = argparse.ArgumentParser(
        description="RAG Script using ChromaDB & BM25: Index documents (parallel) or query (hybrid).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # ... (Keep parser arguments as they are) ...
    parser.add_argument("--force_reindex", action="store_true",
                        help="Force reprocessing of all files, even if already indexed.") # <<< ADDED ARGUMENT

    if test_mode_enabled:
        print("--- Running in Test Mode ---")
        # Ensure the test directory exists for output
        test_db_path = "chunk_database/chunks" # <<< Adjusted for consistency
        test_folder = "cleaned_text/paper_2_intro"     # <<< Adjusted for consistency
        os.makedirs(os.path.dirname(test_db_path), exist_ok=True)
        os.makedirs(test_folder, exist_ok=True)

        test_args = {
            "mode": "index", # Start with indexing
            "document_path": None,
            "folder_path": test_folder,
            "db_path": test_db_path,
            "query": "What are plausibilistic storylines and how are they different from other climate storylines?", # Query for later test
            "top_k": 10,
            "collection_name": "test_hybrid_collection",
            "workers": 4,
            "force_reindex": False # Test without forcing initially
        }
        args = argparse.Namespace(**test_args)
        # Setup dummy files if they don't exist or force_reindex is True (simplified for test)
        # A more robust test might clear the DB/files first
        if not os.listdir(test_folder) or args.force_reindex:
             print("Setting up dummy files for test...")
             # Clear existing test files if forcing reindex might be good too
             for f in os.listdir(test_folder): os.remove(os.path.join(test_folder, f))
             with open(os.path.join(args.folder_path, "doc1_hybrid.txt"), "w") as f: f.write("Apples are red. BM25 is a lexical search algorithm.")
             with open(os.path.join(args.folder_path, "doc2_hybrid.txt"), "w") as f: f.write("Oranges are orange. Vector search finds semantic similarity.")
             with open(os.path.join(args.folder_path, "doc3_hybrid.txt"), "w") as f: f.write("Bananas are yellow and grow in bunches. RRF combines results.")
    else:
        args = parser.parse_args()

    # --- Configuration & Validation ---
    # ... (Keep client initialization and checks as they are) ...

    print(f"\n--- Configuration ---")
    print(f"Mode: {args.mode}")
    print(f"DB/Index Path: {args.db_path}")
    print(f"Collection Name: {args.collection_name}")
    if args.mode != 'index': print(f"Query: {args.query} | Top K: {args.top_k}")
    if args.mode == 'index': print(f"Parallel Workers: {args.workers} | Force Re-index: {args.force_reindex}") # <<< UPDATED PRINT
    print(f"Embedding Model: {EMBEDDING_MODEL} | Chat Model: {CHAT_MODEL}")
    print(f"--------------------\n")

    # ... (Keep mode-specific argument validation as it is) ...

    # --- Execute Mode ---
    try:
        if args.mode == "index":
            # --- 1. Identify all potential files ---
            potential_files_to_index = []
            if args.folder_path:
                if not os.path.isdir(args.folder_path): raise ValueError(f"Folder not found: {args.folder_path}")
                print(f"Scanning folder: {args.folder_path}")
                for root, _, files in os.walk(args.folder_path):
                    for file in files:
                        if file.lower().endswith(".txt"):
                            potential_files_to_index.append(os.path.join(root, file))
            elif args.document_path:
                 if not os.path.isfile(args.document_path): raise ValueError(f"File not found: {args.document_path}")
                 if args.document_path.lower().endswith(".txt"): potential_files_to_index.append(args.document_path)
                 else: print(f"Warning: Skipping non-txt file: {args.document_path}")

            if not potential_files_to_index:
                print("No source .txt files found.")
                return

            print(f"Found {len(potential_files_to_index)} potential source file(s).")

            # --- 2. Check against existing hashes in ChromaDB (unless forcing reindex) ---
            files_to_process = []
            skipped_files_count = 0
            existing_hashes = set()

            if not args.force_reindex:
                print("Checking ChromaDB for already indexed files...")
                try:
                    # Get collection (will create if doesn't exist, that's fine)
                    collection = get_chroma_collection(args.db_path, args.collection_name)
                    # Fetch existing metadata - fetch in batches if collection is huge
                    # For simplicity, fetch all metadata if collection isn't excessively large.
                    # Note: .get() without filters can be slow/memory-intensive on huge collections.
                    # A more scalable approach might involve querying with specific hashes
                    # if you could pre-compute hashes efficiently, but let's use .get() for now.
                    # Only fetch metadatas containing 'file_hash'. Limit if possible, but difficult without IDs.
                    # Fetching just 'metadatas'
                    # Reduce payload by getting only necessary metadata? Chroma `get` includes all by default.
                    # We might need to iterate or use a more targeted query if scale becomes an issue.
                    existing_data = collection.get(include=['metadatas']) # Get all metadatas
                    if existing_data and existing_data.get('metadatas'):
                        for meta in existing_data['metadatas']:
                            if meta and 'file_hash' in meta:
                                existing_hashes.add(meta['file_hash'])
                    print(f"Found {len(existing_hashes)} unique file hashes already in ChromaDB.")
                except Exception as e:
                    print(f"Warning: Could not check existing files in ChromaDB: {e}. Proceeding without skipping.")
                    # Optionally, force reindex or halt if check is critical
                    # args.force_reindex = True # Or exit

            print("Calculating hashes and filtering files...")
            files_to_process = []
            skipped_files = []
            for file_path in tqdm(potential_files_to_index, desc="Checking files", unit="file"):
                try:
                    file_hash = compute_file_hash(file_path)
                    if not args.force_reindex and file_hash in existing_hashes:
                        skipped_files.append(os.path.basename(file_path))
                        skipped_files_count += 1
                    else:
                        files_to_process.append(file_path)
                except Exception as e:
                    print(f"\nError hashing file {os.path.basename(file_path)}, skipping: {e}")
                    # Decide how to handle hash errors - skip or attempt processing?
                    # Let's skip it for safety.
                    skipped_files.append(f"{os.path.basename(file_path)} (hash error)")
                    skipped_files_count += 1

            if skipped_files_count > 0:
                 print(f"Skipping {skipped_files_count} file(s) already indexed (or had hash errors).")
                 # Optional: print skipped filenames if list isn't too long
                 # if len(skipped_files) < 20:
                 #    for fname in skipped_files: print(f"  - Skipped: {fname}")

            if not files_to_process:
                print("No new files found to process.")

            print(f"Preparing to process {len(files_to_process)} new file(s) using up to {args.workers} workers.")

            # --- 3. Parallel Processing (Only for files_to_process) ---
            all_processed_chunks = []
            successful_files = 0
            failed_files = []

            worker_func = functools.partial(
                process_single_file_wrapper,
                max_tokens=DEFAULT_MAX_TOKENS,
                overlap=DEFAULT_OVERLAP
            )

            print("--- Starting Parallel Chunk Processing for New Files ---")
            with multiprocessing.Pool(processes=args.workers) as pool:
                # Process only the filtered list: files_to_process
                results_iterator = pool.imap_unordered(worker_func, files_to_process)
                # Adjust tqdm total to the number of files actually being processed
                for file_path, success, error_msg, processed_data in tqdm(results_iterator, total=len(files_to_process), desc="Processing New Files", unit="file"):
                    if success:
                        # Check if worker actually returned data (might have skipped internally)
                        if processed_data:
                           all_processed_chunks.extend(processed_data)
                           successful_files += 1 # Count files that produced chunks
                        # else: consider if a file producing no chunks is still "successful"
                    else:
                        failed_files.append((os.path.basename(file_path), error_msg))
            # --- End Parallel Processing ---

            print(f"\n--- Post-Processing and Indexing ---")
            print(f"Successfully processed {successful_files} new files (producing chunks).")
            print(f"Skipped files (already indexed or hash error): {skipped_files_count}")
            if failed_files: print(f"Failed processing attempts: {len(failed_files)}")
            for fname, err in failed_files: print(f"  - {fname}: {err}")

            if not all_processed_chunks:
                 print("No valid new chunks were generated. Nothing to index.")
                 # Check if BM25/Chroma additions should still happen if only existing data matters
                 # For now, exit if no *new* chunks.
                 
            print(f"Total new chunks generated: {len(all_processed_chunks)}")

            # --- 4. Build/Update BM25 Index ---
            # Option 1: Rebuild BM25 from scratch using ALL data (new + old) - Simpler but potentially slow
            # Option 2: Update existing BM25 index (more complex, library might not support easily)
            # Option 3: Build BM25 ONLY from new chunks (Inconsistent search results)
            # Let's go with Option 1 (Rebuild) for simplicity, assuming it's acceptable performance-wise.
            # Requires fetching ALL existing metadata+text OR storing tokenized text in Chroma metadata.
            # Storing tokenized text in metadata is problematic (size, queryability).
            # Re-fetching all text from Chroma is needed to rebuild BM25 correctly.

            print("Rebuilding BM25 index with all current data (new + existing)...")
            all_final_chunk_ids = []
            all_final_tokenized_corpus = []
            try:
                 collection = get_chroma_collection(args.db_path, args.collection_name)
                 # Add the NEW chunks first
                 if all_processed_chunks:
                     print(f"Adding {len(all_processed_chunks)} new chunks to ChromaDB collection '{args.collection_name}'...")
                     new_chroma_ids = [chunk['id'] for chunk in all_processed_chunks]
                     new_chroma_embeddings = [chunk['embedding'] for chunk in all_processed_chunks]
                     new_chroma_metadatas = [chunk['metadata'] for chunk in all_processed_chunks]
                     batch_size = 100
                     num_batches = (len(new_chroma_ids) + batch_size - 1) // batch_size
                     for i in tqdm(range(0, len(new_chroma_ids), batch_size), total=num_batches, desc="Adding New Chunks to ChromaDB"):
                         batch_ids = new_chroma_ids[i : i + batch_size]
                         batch_embeddings = new_chroma_embeddings[i : i + batch_size]
                         batch_metadatas = new_chroma_metadatas[i : i + batch_size]
                         if not batch_ids: continue
                         collection.add(ids=batch_ids, embeddings=batch_embeddings, metadatas=batch_metadatas)
                     print("Finished adding new chunks to ChromaDB.")

                 # Now fetch ALL data (including newly added) for BM25 rebuild
                 print("Fetching all data from ChromaDB for BM25 index rebuild...")
                 # Fetch ids and the 'text' field from metadata
                 all_data = collection.get(include=['metadatas']) # Fetch all metadata again
                 if all_data and all_data.get('ids'):
                      all_final_chunk_ids = all_data['ids']
                      metadatas = all_data.get('metadatas', [])
                      print(f"Tokenizing {len(all_final_chunk_ids)} total chunks for BM25...")
                      all_final_tokenized_corpus = [
                           tokenize_text_bm25(meta.get('text', '')) # Tokenize raw text from metadata
                           for meta in tqdm(metadatas, desc="Tokenizing all chunks") if meta # Check meta exists
                      ]

                      if len(all_final_chunk_ids) != len(all_final_tokenized_corpus):
                           print(f"Warning: Mismatch between chunk IDs ({len(all_final_chunk_ids)}) and tokenized texts ({len(all_final_tokenized_corpus)}) fetched. BM25 index may be incomplete.")
                           # Attempt to proceed with matched items? Difficult to align.
                           # Fallback: Use only newly processed chunks for BM25? Less ideal.
                           # For now, log warning and proceed with potentially mismatched lists.
                           # A better solution involves ensuring alignment during fetch/processing.

                 else:
                      print("Warning: Could not fetch data from ChromaDB for BM25 rebuild.")

                 # Build and save BM25 index using all_final_tokenized_corpus and all_final_chunk_ids
                 if all_final_tokenized_corpus:
                     print(f"Building final BM25 index for {len(all_final_tokenized_corpus)} total chunks...")
                     bm25_index = BM25Okapi(all_final_tokenized_corpus)
                     bm25_index_path, bm25_mapping_path = get_bm25_paths(args.db_path, args.collection_name)
                     print(f"Saving final BM25 index to: {bm25_index_path}")
                     with open(bm25_index_path, 'wb') as f_idx: pickle.dump(bm25_index, f_idx)
                     print(f"Saving final BM25 ID mapping to: {bm25_mapping_path}")
                     # Ensure the mapping corresponds exactly to the corpus order
                     with open(bm25_mapping_path, 'wb') as f_map: pickle.dump(all_final_chunk_ids, f_map)
                     print("Final BM25 index saved.")
                 else:
                     print("No corpus generated for BM25 index. Skipping BM25 build.")

            except Exception as e:
                 print(f"!!! Error during ChromaDB add or BM25 rebuild: {e}")
                 import traceback; traceback.print_exc()

        elif args.mode == "query" or args.mode == "query_direct":
             # --- Querying Logic (Remains the same) ---
             # ... (Existing query logic: load_bm25_index, iterative_rag_query, query_index) ...
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
        print("\n--- Running Test Query After Indexing (Test Mode - Hybrid) ---")
        # ... (Keep test query logic as it is) ...
        test_query_mode = "query" # Or "query_direct"
        try:
            print(f"Running test {test_query_mode} query: '{args.query}'")
            if test_query_mode == "query":
                 final_answer = iterative_rag_query(args.query, args.db_path, args.collection_name, top_k=args.top_k)
            else:
                 final_answer = query_index(args.query, args.db_path, args.collection_name, top_k=args.top_k)

            print("\n=== Test Query Final Answer ===")
            print(final_answer)
            print("=============================")
        except Exception as test_e:
            print(f"\n!!! Error during post-indexing test query: {test_e}")
            import traceback; traceback.print_exc()


if __name__ == "__main__":
    multiprocessing.freeze_support() # Good practice for multiprocessing
    main()