#!/usr/bin/env python3
"""
rag_bm25.py (Two-Phase Indexing: Chunk+Store, then Embed)

A Retrieval-Augmented Generation (RAG) script with deep linking that:
  Phase 1 (Index Mode):
    1. Reads given .txt documents (supports single file or folder).
    2. Processes files sequentially.
    3. Splits each document into semantically coherent chunks using a token-based sliding window.
    4. Generates a short contextual summary for each chunk.
    5. Stores raw chunks and metadata (including 'has_embedding': False) in ChromaDB.
    6. Tokenizes raw chunks for BM25.
    7. Gathers processed chunk data (metadata, BM25 tokens).
    8. Builds and saves a BM25 index based on the stored raw text.
    9. Checks for already indexed files based on hash (unless --force_reindex).

  Phase 2 (Embed Mode):
    1. Connects to the ChromaDB collection.
    2. Finds chunks where 'has_embedding' is False.
    3. Computes embeddings for these chunks sequentially.
    4. Updates ChromaDB with embeddings and sets 'has_embedding' to True.

  Query Modes (Query / Query Direct):
    1. Retrieves chunks using hybrid search (Vector + BM25 + RRF).
    2. Generates answers based on retrieved context.

Usage:
  # 1. Index documents (Chunk & Store raw data + Build BM25):
  python rag_chroma.py --mode index --folder_path path/to/documents --db_path ./hybrid_db --collection_name my_hybrid_docs

  # 2. Embed the stored chunks:
  python rag_chroma.py --mode embed --db_path ./hybrid_db --collection_name my_hybrid_docs

  # 3. Query iteratively (hybrid search):
  python rag_chroma.py --mode query --query "What about unique terms?" --db_path ./hybrid_db --collection_name my_hybrid_docs --top_k 5

  # Force re-chunking/storing (will reset 'has_embedding'):
  python rag_chroma.py --mode index --folder_path path/to/docs --db_path ./hybrid_db --collection_name my_hybrid_docs --force_reindex
  # Then re-embed:
  python rag_chroma.py --mode embed --db_path ./hybrid_db --collection_name my_hybrid_docs
"""

import os
import argparse
import datetime
import hashlib
from typing import List, Dict, Any, Tuple, Optional
from tqdm import tqdm
import numpy as np
import tiktoken
import chromadb
from chromadb.utils import embedding_functions
from chromadb import Documents, EmbeddingFunction, Embeddings # Added EmbeddingFunction, Documents, Embeddings
import pickle
import re
from rank_bm25 import BM25Okapi
from google import genai
from google.genai.types import EmbedContentConfig
import traceback # For more detailed error logging


# Removed: multiprocessing, functools
import time # For embed delay

# --- NLTK Imports ---
try:
    import nltk
    from nltk.corpus import stopwords
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
            stop_words_english = set()
except ImportError:
    print("Warning: NLTK not installed (`pip install nltk rank_bm25`). Using basic tokenization without stopwords.")
    stop_words_english = set()
# ------------------------------------------------------

# --- Config Imports ---
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
    DEFAULT_TOP_K,
    OUTPUT_EMBEDDING_DIMENSION,
)


# --- ChromaDB Configuration ---
DEFAULT_CHROMA_COLLECTION_NAME = "rag_chunks_hybrid_default"

# --- BM25 Configuration ---
def get_bm25_paths(db_path: str, collection_name: str) -> Tuple[str, str]:
    base_path = os.path.join(db_path, f"{collection_name}_bm25")
    index_path = f"{base_path}_index.pkl"
    map_path = f"{base_path}_ids.pkl"
    return index_path, map_path

bm25_index_cache = {}
bm25_ids_cache = {}

# --- Global Variables for API Clients (initialized by main process) ---
gemini_client = None

def initialize_clients():
    """Initializes API clients based on config keys."""
    global gemini_client
    # Reset clients first in case this is called multiple times
    gemini_client = None

    if GEMINI_API_KEY:
        try:
            from google import genai
            
            gemini_client = genai.Client(api_key=GEMINI_API_KEY) # Use the configured module
            # print("Google GenAI client configured.") # Verbose logging
        except ImportError:
            print("Warning - Google Generative AI library not installed.")
        except Exception as e:
            print(f"Warning - Failed to configure Gemini client: {e}")


class ConfigurableEmbeddingFunction(EmbeddingFunction):
    """
    A ChromaDB EmbeddingFunction that uses the script's configured
    get_embedding function and respects the OUTPUT_EMBEDDING_DIMENSION.
    """
    def __init__(self,
                 model_name: Optional[str] = None, # Allow setting model later if needed
                 output_dimension_override: Optional[int] = None, # Optional explicit override
                 task_type: str = "retrieval_document"):
        """
        Initializes the embedding function.

        Args:
            model_name: The embedding model to use. Defaults to global EMBEDDING_MODEL.
            output_dimension_override: Explicitly set dimension, otherwise uses global OUTPUT_EMBEDDING_DIMENSION.
            task_type: Default task type for embedding calls.
        """
        # Use passed model_name or fall back to global default
        self._model_name = model_name if model_name else EMBEDDING_MODEL

        # Determine the dimension: Use override if provided, otherwise use the globally loaded value
        if output_dimension_override is not None:
             self._output_dimension = output_dimension_override
             print(f"DEBUG (EmbeddingFunction Init): Using override dimension {self._output_dimension}")
        else:
             # --- Access the global variable HERE, inside the method ---
             self._output_dimension = OUTPUT_EMBEDDING_DIMENSION # This should now be defined
            #  if self._output_dimension is not None:
            #       print(f"DEBUG (EmbeddingFunction Init): Using global dimension {self._output_dimension}")
            #  else:
            #       print(f"DEBUG (EmbeddingFunction Init): Using model default dimension.")


        self._task_type = task_type
        # API clients (gemini_client) are assumed to be
        # initialized globally before this function is used by ChromaDB.
        if not self._model_name:
             raise ValueError("Embedding model name must be provided either during init or globally via EMBEDDING_MODEL.")


    def __call__(self, input_texts: Documents) -> Embeddings:
        """
        Generates embeddings for the given input texts.
        (Keep the rest of the __call__ method implementation as before)
        """
        # ... (previous __call__ logic remains the same) ...
        all_embeddings: Embeddings = []
        for text in input_texts:
            try:
                embedding = get_embedding(
                    text=text,
                    model=self._model_name,
                    task_type=self._task_type
                    # get_embedding internally uses the global OUTPUT_EMBEDDING_DIMENSION
                )
                # ... (rest of the error handling and appending logic) ...
                if embedding is not None and isinstance(embedding, np.ndarray):
                    # Optional check: verify dimension if explicitly set
                    if self._output_dimension is not None and len(embedding) != self._output_dimension:
                         print(f"WARNING (EmbeddingFunction): Dimension mismatch! Expected {self._output_dimension}, got {len(embedding)} for model {self._model_name}. Using the vector anyway.")
                    all_embeddings.append(embedding.tolist())
                else:
                    print(f"ERROR (EmbeddingFunction): Failed to get embedding for text: '{text[:50]}...'. Skipping.")
                    all_embeddings.append([]) # Append empty list to signify failure
            except Exception as e:
                print(f"ERROR (EmbeddingFunction): Exception during embedding for text '{text[:50]}...': {e}")
                all_embeddings.append([]) # Append empty list on error

        # Filter/handle failures as before if necessary
        # ...

        return all_embeddings

# --- ChromaDB Client Setup ---
def get_chroma_collection(db_path: str, collection_name: str) -> chromadb.Collection:
    """
    Gets or creates a ChromaDB collection, explicitly setting the
    embedding function based on global configuration.
    """
    # ... (checks remain the same) ...
    try:
        # Instantiate the custom embedding function.
        # It will now internally fetch the global OUTPUT_EMBEDDING_DIMENSION.
        # We pass the global model name explicitly for clarity.
        emb_func = ConfigurableEmbeddingFunction(
            model_name=EMBEDDING_MODEL
            # No need to pass output_dimension here anymore,
            # unless you wanted to specifically override the global one
            # for this collection instance (e.g., output_dimension_override=512)
        )

        # ... (rest of the function remains the same: PersistentClient, get_or_create_collection) ...
        os.makedirs(db_path, exist_ok=True)
        chroma_client = chromadb.PersistentClient(path=db_path)

        # print(f"DEBUG: Getting/Creating collection '{collection_name}' with EmbeddingFunction for model '{EMBEDDING_MODEL}' and configured dimension.")

        collection = chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=emb_func,
            metadata={"hnsw:space": "cosine"}
        )
        return collection

    except Exception as e:
        # ... (error handling remains the same) ...
        print(f"!!! Error connecting/creating ChromaDB collection '{collection_name}' at path '{db_path}': {e}")
        import traceback; traceback.print_exc()
        raise

# --- File Hashing ---
def compute_file_hash(file_path: str) -> str:
    hasher = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            while True:
                chunk = f.read(4096)
                if not chunk: break
                hasher.update(chunk)
        return hasher.hexdigest()
    except FileNotFoundError: raise
    except Exception as e: raise RuntimeError(f"Error hashing file {file_path}: {e}") from e

# --- Chunking and Text Helpers ---
def count_tokens(text: str, model: str = EMBEDDING_MODEL) -> int:
    try:
        if not hasattr(count_tokens, "_encoding_cache"): count_tokens._encoding_cache = {}
        if model not in count_tokens._encoding_cache:
             try: count_tokens._encoding_cache[model] = tiktoken.encoding_for_model(model)
             except Exception:
                 try: count_tokens._encoding_cache[model] = tiktoken.get_encoding("cl100k_base")
                 except Exception: count_tokens._encoding_cache[model] = None
        encoding = count_tokens._encoding_cache.get(model)
        return len(encoding.encode(text)) if encoding else len(text.split())
    except Exception: return len(text.split())

def chunk_document_tokens(document: str, max_tokens: int = DEFAULT_MAX_TOKENS, overlap: int = DEFAULT_OVERLAP) -> List[tuple]:
    if max_tokens <= 0: raise ValueError("max_tokens must be positive.")
    if overlap < 0: raise ValueError("overlap cannot be negative.")
    if overlap >= max_tokens: overlap = max_tokens // 2
    try: encoding = tiktoken.encoding_for_model(EMBEDDING_MODEL)
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
        try: chunk_text = encoding.decode(chunk_tokens, errors='replace').strip()
        except Exception as e:
            # print(f"Debug: Error decoding tokens {start_token_idx}-{end_token_idx}: {e}") # More verbose debug
            next_start = start_token_idx + max_tokens - overlap
            start_token_idx = next_start if next_start > start_token_idx else start_token_idx + 1; continue
        if chunk_text: chunks.append((chunk_text, start_token_idx, end_token_idx))
        next_start = start_token_idx + max_tokens - overlap
        start_token_idx = next_start if next_start > start_token_idx else start_token_idx + 1
    return chunks

def truncate_text(text: str, token_limit: int, model: str = CHAT_MODEL) -> str:
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

 #LLM Interaction Functions (with client checks)

def generate_chunk_context(document: str, chunk: str, token_limit: int = 30000,
                           context_length: int = DEFAULT_CONTEXT_LENGTH,
                           model: str = CHUNK_CONTEXT_MODEL) -> str:
    """
    Generate a succinct, chunk-specific context using the full document.
    Relies on generate_llm_response which handles client/module checks.
    """
    # Input validation (optional but good practice)
    if not document.strip() or not chunk.strip():
        return "Context unavailable (empty input)."

    doc_token_count = count_tokens(document) # Use default model for counting if model specific fails
    if doc_token_count > token_limit:
        print(f"Truncating document for context generation (limit: {token_limit} tokens)")
        document = truncate_text(document, token_limit) # Use default model for truncating

    prompt = (
        f"<document>\n{document}\n</document>\n\n"
        f"Here is a specific chunk from the document above:\n"
        f"<chunk>\n{chunk}\n</chunk>\n\n"
        "Provide a very short, succinct context (1-2 sentences maximum) that describes the immediate topic or section surrounding this chunk within the overall document. "
        "This context is intended to help improve search retrieval for the chunk later. "
        "Focus only on the local context. Do not summarize the chunk itself. "
        "Answer ONLY with the succinct context itself, without any preamble like 'Context:'."
    )

    try:
        # Ensure context_length is reasonable
        if context_length <= 0 or context_length > 512:
             context_length = DEFAULT_CONTEXT_LENGTH # Reset to default if invalid

        # Call the generic response generator
        response = generate_llm_response(prompt, context_length, temperature=0.5, model=model)

        # Basic check for errors returned by generate_llm_response
        if response.startswith("[Error") or response.startswith("[Blocked"):
             print(f"Warning: Failed to generate context via LLM: {response}")
             return "Error generating context." # Return a generic error

        # Clean up potential LLM preamble if necessary (though prompt asks not to include it)
        response = response.replace("Context:", "").replace("Here is the context:", "").strip()
        return response if response else "Context could not be generated."

    except Exception as e:
        # Catch unexpected errors during the call
        print(f"Error in generate_chunk_context function with model {model}: {e}")
        import traceback
        traceback.print_exc() # Print stack trace for debugging
        return "Error generating context."

    except Exception as e:
        print(f"!!! Critical Error during embedding generation for model {model}: {e}")
        # import traceback; traceback.print_exc() # Enable for deep debugging
        return None
    return None # Should not be reached normally

def get_embedding(text: str, model: str = EMBEDDING_MODEL, embedding_dimension: int = OUTPUT_EMBEDDING_DIMENSION, task_type: Optional[str] = None) -> Optional[np.ndarray]:
    """
    Get embedding using Google,
    following the structure provided in the template.
    """
    # 1. Basic Input Validation
    if not text or not text.strip():
        # print("Warning: Attempting to embed empty text. Returning None.")
        return None
    text = text.replace("\n", " ") # Consistent preprocessing

    try:
        # 2. Determine provider and check client/module availability
        # Using the 'model' parameter passed to the function
        model_lower = model.lower() # Use lowercase for checks

        # --- Gemini Handling ---
        if any(m in model_lower for m in ["embedding-001", "text-embedding-004"]):
             if not genai: # Check if the genai module is configured
                 raise RuntimeError("Google GenAI library not available or not configured. Cannot get Gemini embedding.")

             # Ensure model name has 'models/' prefix for Gemini API
             api_model_name = model if model.startswith("models/") else f"models/{model}"

             # Prepare arguments using genai.embed_content
             kwargs = {'model': api_model_name, 'content': text}

             # Handle task_type based on model (adapted from template logic)
             if api_model_name == "models/embedding-001":
                 # Template passes task_type for this model
                 valid_tasks_001 = ["retrieval_document", "retrieval_query", "semantic_similarity", "classification", "clustering"]
                 # Use provided task_type if valid, otherwise default might be implicit or required
                 if task_type and task_type in valid_tasks_001:
                     kwargs['task_type'] = task_type
                 elif task_type:
                      print(f"Warning: Unsupported task_type '{task_type}' for {api_model_name}. Proceeding without it or with default.")
                 # else: task_type is None or invalid, rely on API default or potentially error if required

             elif api_model_name == "models/text-embedding-004":
                  # Template *doesn't* pass task_type for this model, but API *does* support it (usually uppercase)
                  # Let's add optional uppercase task_type handling based on common practice
                  valid_tasks_004 = ["RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY", "SEMANTIC_SIMILARITY", "CLASSIFICATION", "CLUSTERING"]
                  default_task = "RETRIEVAL_DOCUMENT"
                  task_type_map = { "retrieval_document": "RETRIEVAL_DOCUMENT", "retrieval_query": "RETRIEVAL_QUERY" } # Example mapping
                  final_task_type = task_type_map.get(task_type, default_task) if task_type else default_task
                  if final_task_type in valid_tasks_004:
                      kwargs['task_type'] = final_task_type
                  # If user explicitly provides an uppercase task_type, use it if valid
                  elif task_type and task_type in valid_tasks_004:
                       kwargs['task_type'] = task_type
                  # else: don't pass task_type if not provided or invalid for this model

             # Make the corrected Gemini API call
             response = gemini_client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=text,
                config=EmbedContentConfig(
                    output_dimensionality=embedding_dimension,  # Optional
                ),
            )

             # Parse the standard Gemini response structure
             if isinstance(response, dict) and 'embedding' in response:
                 vector = response['embedding']
                 if isinstance(vector, list) and len(vector) > 0:
                      return np.array(vector)
                 else:
                      print(f"Error: Gemini embedding response contained empty or invalid embedding list for {api_model_name}.")
                      return None
             else:
                 # Handle the specific structure mentioned in template for -004 IF the standard one fails
                 # This is less likely with the current library version but included as a fallback based on template hint
                 if api_model_name == "models/text-embedding-004" and hasattr(response, 'embeddings') and response.embeddings and hasattr(response.embeddings[0], 'values'):
                    #   print("Note: Parsing Gemini response using alternative structure '.embeddings[0].values'.") # Inform user
                      vector = response.embeddings[0].values
                      if isinstance(vector, list) and len(vector) > 0:
                           return np.array(vector)
                      else:
                           print(f"Error: Alternative Gemini structure parsing failed for {api_model_name}.")
                           return None
                 else:
                      print(f"Error: Unexpected Gemini embedding response structure for {api_model_name}: {type(response)}")
                      return None

        # --- Unsupported Model ---
        else:
            raise ValueError(f"Embedding model '{model}' is not supported by this function.")

    # 3. Catch All Exceptions
    except Exception as e:
        print(f"!!! Critical Error during embedding generation for model {model}: {e}")
        import traceback
        traceback.print_exc() # Print stack trace for debugging
        return None # Return None on any error

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


# ---------------------------
# BM25 Specific Helpers
# ---------------------------
def tokenize_text_bm25(text: str) -> List[str]:
    if not isinstance(text, str): return []
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.split()
    tokens = [word for word in tokens if word and word not in stop_words_english]
    return tokens

bm25_instance: Optional[BM25Okapi] = None
bm25_chunk_ids_ordered: Optional[List[str]] = None

def load_bm25_index(db_path: str, collection_name: str) -> bool:
    global bm25_instance, bm25_chunk_ids_ordered
    cache_key = (db_path, collection_name)
    if cache_key in bm25_index_cache:
        bm25_instance = bm25_index_cache[cache_key]
        bm25_chunk_ids_ordered = bm25_ids_cache[cache_key]
        return True
    index_path, map_path = get_bm25_paths(db_path, collection_name)
    if os.path.exists(index_path) and os.path.exists(map_path):
        try:
            # print(f"Loading BM25 index from: {index_path}") # Verbose
            with open(index_path, 'rb') as f_idx: bm25_instance = pickle.load(f_idx)
            with open(map_path, 'rb') as f_map: bm25_chunk_ids_ordered = pickle.load(f_map)
            bm25_index_cache[cache_key] = bm25_instance
            bm25_ids_cache[cache_key] = bm25_chunk_ids_ordered
            print(f"BM25 index loaded successfully ({len(bm25_chunk_ids_ordered or [])} documents).")
            return True
        except Exception as e:
            print(f"!!! Error loading BM25 index files: {e}")
            bm25_instance, bm25_chunk_ids_ordered = None, None; return False
    else:
        # print(f"BM25 index files not found at {index_path}/{map_path}. Will be built if indexing.") # Informative
        bm25_instance, bm25_chunk_ids_ordered = None, None; return False

def retrieve_chunks_bm25(query: str, db_path: str, collection_name: str, top_k: int) -> List[Tuple[str, float]]:
    if bm25_instance is None or bm25_chunk_ids_ordered is None:
        if not load_bm25_index(db_path, collection_name):
            # print("BM25 index is not available for querying.") # Already printed by load_bm25_index
            return []
    tokenized_query = tokenize_text_bm25(query)
    if not tokenized_query: print("Warning: BM25 query empty after tokenization."); return []
    try:
        # Ensure bm25_instance and bm25_chunk_ids_ordered are not None (checked by load_bm25_index)
        scores = bm25_instance.get_scores(tokenized_query) # type: ignore
        # Combine scores with original chunk IDs (indices map to bm25_chunk_ids_ordered)
        # Ensure lengths match - crucial if index/map are out of sync
        if len(scores) != len(bm25_chunk_ids_ordered): # type: ignore
             print(f"!!! Error: BM25 score count ({len(scores)}) mismatch with ID map count ({len(bm25_chunk_ids_ordered)}). Index may be corrupt.")
             return []
        scored_results = zip(bm25_chunk_ids_ordered, scores) # type: ignore
        # Sort by score descending
        sorted_results = sorted(scored_results, key=lambda item: item[1], reverse=True)
        # Get top K results with positive scores
        results = [(chunk_id, score) for chunk_id, score in sorted_results if score > 0][:top_k]
        return results
    except Exception as e:
        print(f"!!! Error during BM25 scoring/ranking: {e}"); return []

# ---------------------------
# Indexing Function (PHASE 1: Chunk & Store Raw - runs sequentially)
# ---------------------------
def index_document_phase1(document_path: str,
                          max_tokens: int = DEFAULT_MAX_TOKENS,
                          overlap: int = DEFAULT_OVERLAP) -> List[Dict]:
    """
    Processes a single document for Phase 1: reads, chunks, gets context, tokenizes for BM25.
    DOES NOT generate embeddings. Returns list of dicts {id, metadata, bm25_tokens}.
    Metadata includes 'has_embedding': False.
    """
    file_name = os.path.basename(document_path)
    processed_chunk_data = []

    if not os.path.exists(document_path):
        print(f"Error: Doc not found: {document_path}")
        return []

    try: file_hash = compute_file_hash(document_path)
    except Exception as e: print(f"Error hashing {file_name}: {e}"); return []

    processing_date = datetime.datetime.now().isoformat()

    try:
        with open(document_path, "r", encoding="utf-8", errors='replace') as f:
            document = f.read()
    except Exception as e: print(f"Error reading {file_name}: {e}"); return []
    if not document.strip(): return [] # Skip empty files

    raw_chunks = chunk_document_tokens(document, max_tokens=max_tokens, overlap=overlap)
    if not raw_chunks: return []

    for idx, (raw_chunk, start_tok, end_tok) in enumerate(raw_chunks):
        if not raw_chunk.strip(): continue

        # Generate context (still useful for embedding later and display)
        # generate_chunk_context will handle client check internally
        chunk_context = generate_chunk_context(document, raw_chunk, model=CHUNK_CONTEXT_MODEL)
        if "Error:" in chunk_context: # Check if context generation failed due to client issues etc.
             print(f"Warning: Skipping chunk {idx} in {file_name} due to context generation failure: {chunk_context}")
             continue # Skip this chunk if context failed significantly

        contextualized_text = f"{chunk_context}\n{raw_chunk}"
        tokens_count = count_tokens(raw_chunk)

        # *** NO EMBEDDING GENERATED HERE ***

        chunk_id = f"{file_hash}_{idx}"
        metadata = {
            "file_hash": file_hash, "file_name": file_name, "processing_date": processing_date,
            "chunk_number": idx, "start_token": start_tok, "end_token": end_tok,
            "text": raw_chunk, # Store raw chunk text
            "context": chunk_context, # Store generated context
            "contextualized_text": contextualized_text, # Store combined text for embedding later
            "tokens": tokens_count,
            "has_embedding": False # <<< FLAG: Mark as needing embedding
        }
        bm25_tokens = tokenize_text_bm25(raw_chunk) # Tokenize raw text for BM25

        processed_chunk_data.append({
            "id": chunk_id,
            # "embedding": NO EMBEDDING FIELD
            "metadata": metadata,
            "bm25_tokens": bm25_tokens
        })

    return processed_chunk_data

# Removed: Worker Function Wrapper (process_single_file_wrapper_phase1)


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

        # Embed the query (uses main process client)
        query_vec = get_embedding(query, model=EMBEDDING_MODEL, task_type="retrieval_query")
        if query_vec is None:
            print(f"Error: Failed to embed query '{query[:50]}...'")
            return []
        if not isinstance(query_vec, np.ndarray) or query_vec.ndim != 1:
             print(f"Error: Query embedding has unexpected shape {type(query_vec)}.")
             return []

        # Query ChromaDB - implicitly searches only chunks with embeddings
        results = collection.query(
            query_embeddings=[query_vec.tolist()],
            n_results=top_k,
            include=['metadatas', 'distances']
            # Optional: Filter for 'has_embedding: True' if paranoid, but query should only match embedded items
            # where={"has_embedding": True}
        )

        if results and results.get('ids') and results['ids'][0]:
             metadatas = results.get('metadatas', [[]])[0]
             distances = results.get('distances', [[]])[0]
             # ids = results.get('ids', [[]])[0] # For debug
             for i, meta in enumerate(metadatas):
                 if meta is None: continue # Skip if metadata is missing
                 # Ensure necessary fields exist before adding
                 if 'file_hash' in meta and 'chunk_number' in meta:
                     dist = distances[i] if i < len(distances) else None
                     meta['distance'] = dist
                     meta['similarity'] = (1.0 - dist) if dist is not None and dist <= 1.0 else 0.0 # Clamp similarity
                     retrieved_chunks.append(meta)
                 else:
                      print(f"Warning: Vector result metadata missing key fields (e.g., file_hash, chunk_number). Skipping.")

        # else: print(f"No vector results found for query '{query[:50]}...'") # Debug

        return retrieved_chunks
    except Exception as e:
        print(f"!!! Error querying ChromaDB '{collection_name}': {e}")
        import traceback; traceback.print_exc() # More detail on error
        return []

# --- RRF Combination Helper ---
def combine_results_rrf(vector_results: List[Dict], bm25_results: List[Tuple[str, float]],
                        db_path: str, collection_name: str, k_rrf: int = 60) -> List[Dict]:
    """Combines vector and BM25 results using Reciprocal Rank Fusion (RRF)."""
    combined_scores: Dict[str, float] = {}
    chunk_metadata_cache: Dict[str, Dict] = {}

    # Process vector results (already have metadata)
    vector_results.sort(key=lambda c: c.get('distance', float('inf'))) # Sort by distance (lower is better)
    for rank, chunk_meta in enumerate(vector_results):
        # Construct ID from metadata to ensure consistency
        file_hash = chunk_meta.get('file_hash')
        chunk_number = chunk_meta.get('chunk_number')
        if file_hash is None or chunk_number is None: continue # Skip if ID parts missing
        chunk_id = f"{file_hash}_{chunk_number}"

        score = 1.0 / (k_rrf + rank)
        combined_scores[chunk_id] = combined_scores.get(chunk_id, 0.0) + score
        if chunk_id not in chunk_metadata_cache:
             # Store a copy to avoid modifying the original list results
             chunk_metadata_cache[chunk_id] = chunk_meta.copy()

    # Process BM25 results (need to potentially fetch metadata)
    bm25_ids_to_fetch = []
    for rank, (chunk_id, _) in enumerate(bm25_results): # BM25 score itself isn't used in RRF, only rank
        if not chunk_id: continue # Skip empty IDs if they somehow occur
        score = 1.0 / (k_rrf + rank)
        combined_scores[chunk_id] = combined_scores.get(chunk_id, 0.0) + score
        if chunk_id not in chunk_metadata_cache:
            # Only need to fetch if not already present from vector results
            bm25_ids_to_fetch.append(chunk_id)

    # Fetch metadata for BM25-only results if needed
    if bm25_ids_to_fetch:
        unique_ids_to_fetch = list(set(bm25_ids_to_fetch))
        # print(f"Fetching metadata for {len(unique_ids_to_fetch)} BM25-specific chunks...") # Debug
        if unique_ids_to_fetch:
            try:
                collection = get_chroma_collection(db_path, collection_name)
                # Fetch in batches for potentially large number of IDs
                batch_size_fetch = 200 # Adjust as needed
                num_batches_fetch = (len(unique_ids_to_fetch) + batch_size_fetch - 1) // batch_size_fetch
                for i in range(num_batches_fetch):
                     batch_ids = unique_ids_to_fetch[i*batch_size_fetch : (i+1)*batch_size_fetch]
                     if not batch_ids: continue
                     try:
                         fetched_data = collection.get(ids=batch_ids, include=['metadatas'])
                         if fetched_data and fetched_data.get('ids'):
                              for j, fetched_id in enumerate(fetched_data['ids']):
                                  # Check if ID is relevant (still in combined_scores) and metadata exists
                                  if fetched_id in combined_scores and fetched_data['metadatas'] and j < len(fetched_data['metadatas']):
                                      meta = fetched_data['metadatas'][j]
                                      if meta: # Ensure metadata is not None
                                          # Add placeholder distance/similarity for consistency if needed later
                                          meta['distance'] = float('inf') # Indicate it wasn't from vector match
                                          meta['similarity'] = -1.0
                                          chunk_metadata_cache[fetched_id] = meta
                                      # else: print(f"Warning: Null metadata fetched for BM25 chunk ID {fetched_id}") # Debug
                     except Exception as batch_fetch_err:
                          print(f"!!! Error fetching metadata batch for BM25 results (IDs: {batch_ids[:5]}...): {batch_fetch_err}")

            except Exception as fetch_err:
                print(f"!!! Error accessing collection for fetching BM25 metadata: {fetch_err}")

    # Sort chunk IDs by their combined RRF score
    sorted_chunk_ids = sorted(combined_scores.keys(), key=lambda cid: combined_scores[cid], reverse=True)

    # Create the final list of unique metadata dictionaries in RRF order
    final_results = []
    for chunk_id in sorted_chunk_ids:
        if chunk_id in chunk_metadata_cache:
            metadata = chunk_metadata_cache[chunk_id] # Already a copy or freshly fetched
            metadata['rrf_score'] = combined_scores[chunk_id] # Add RRF score
            final_results.append(metadata)
        else:
             # This should be less common now with batch fetching and checks
             print(f"Warning: Metadata unexpectedly missing for chunk {chunk_id} after fetching attempt.")

    return final_results


# --- Subquery and Answer Generation ---
def generate_subqueries(initial_query: str, model: str = SUBQUERY_MODEL) -> List[str]:
    n_queries=1
    prompt = (f"Generate {n_queries} alternative or expanded queries based on:\n"
              f"'{initial_query}'\n"
              f"Return ONLY the queries, one per line. No numbering or introduction.")
    try:
        # Uses main process client
        response_text = generate_llm_response(prompt, max_tokens=250, temperature=0.7, model=model)
        if "[Error generating response" in response_text or "[Blocked" in response_text:
             raise RuntimeError(f"Subquery LLM failed or blocked: {response_text}")
        lines = response_text.strip().splitlines()
        # Clean up potential list markers
        subqueries = [re.sub(r"^\s*[\d\.\-\*]+\s*", "", line).strip() for line in lines if line.strip()]
        return subqueries[:n_queries] if subqueries else [initial_query]
    except Exception as e:
        print(f"Error generating subqueries with {model}: {e}. Using original query."); return [initial_query]

def generate_answer(query: str, combined_context: str, retrieved_chunks: List[dict],
                    model: str = CHAT_MODEL) -> str:
    """Generate answer using context and chunks, citing sources."""
    if not combined_context or not combined_context.strip():
         return "Could not generate an answer: no relevant context found."

    # Sort chunks for citation consistency (e.g., by RRF score descending)
    retrieved_chunks.sort(key=lambda c: c.get('rrf_score', 0.0), reverse=True)

    # Create citation list
    references = "\n".join(
        f"- {chunk.get('file_name', '?')} [Chunk #{chunk.get('chunk_number', '?')}] (RRF: {chunk.get('rrf_score', 0.0):.4f})"
        for i, chunk in enumerate(retrieved_chunks) if chunk # Add index if needed: f"{i+1}. {chunk...}"
    )

    # Truncate context based on model limit (use a conservative estimate)
    MODEL_CONTEXT_LIMITS = { "gpt-4": 8000, "gpt-4o": 128000, "gpt-3.5-turbo": 16000, "gemini-1.5-flash": 1000000, "gemini-1.0-pro": 30720 }
    clean_model_name = model.split('/')[-1] if '/' in model else model # Handle model names like 'models/gemini...'
    model_token_limit = MODEL_CONTEXT_LIMITS.get(clean_model_name, 8000) # Default to 8k if unknown
    # Leave ample room for prompt instructions, query, citations, and the answer itself
    max_context_tokens = int(model_token_limit * 0.70) # Reduced multiplier for safety

    context_tokens = count_tokens(combined_context, model=model)
    if context_tokens > max_context_tokens:
        print(f"Warning: Combined context ({context_tokens} tokens) exceeds estimated limit ({max_context_tokens}) for {model}. Truncating.")
        combined_context = truncate_text(combined_context, max_context_tokens, model=model)
        # Re-count after truncation for calculating answer length
        context_tokens = count_tokens(combined_context, model=model)

    # Construct the final prompt
    # open system_prompt.txt 
    with open('final_answer_system_prompt.txt', 'r', encoding='utf-8') as f:
        system_prompt = f.read()

    prompt = (
        f"{system_prompt}\n\n"
        f"Context:\n---\n{combined_context}\n---\n\n"
        f"Sources Available:\n{references}\n\n"
        f"Question: {query}\n\n"
        f"Answer (with citations):"
    )

    # save prompt to file
    with open('final_prompt.txt', 'w', encoding='utf-8') as f: f.write(prompt)


    # Estimate prompt tokens to calculate max answer tokens
    prompt_base_tokens = count_tokens(prompt.replace(combined_context, ""), model=model) # Approx tokens without context
    prompt_total_tokens = prompt_base_tokens + context_tokens # Estimate total prompt tokens
    # Calculate max tokens for the answer, leave buffer
    answer_max_tokens = max(150, model_token_limit - prompt_total_tokens - 200) # Min 150, buffer 200
    answer_max_tokens = min(answer_max_tokens, 4096) # Hard cap answer length

    # Optional: Save final prompt for debugging
    # try: with open('final_prompt.txt', 'w', encoding='utf-8') as f: f.write(prompt)
    # except Exception as e: print(f"Warning: Could not write final_prompt.txt: {e}")

    print(f"Generating final answer using {model} (context tokens: ~{context_tokens}, prompt tokens: ~{prompt_total_tokens}, max answer tokens: {answer_max_tokens})...")
    # Use main process client
    final_answer = generate_llm_response(prompt, max_tokens=answer_max_tokens, temperature=0.1, model=model)

    # Basic check if generation failed
    if "[Error generating response" in final_answer or "[Blocked" in final_answer:
         print(f"Warning: Final answer generation failed or was blocked.")
         # Return a more informative error message
         return f"Failed to generate the final answer. Details: {final_answer}"

    return final_answer


# --- Iterative and Direct Query Functions ---
def iterative_rag_query(initial_query: str, db_path: str, collection_name: str,
                        top_k: int = DEFAULT_TOP_K,
                        subquery_model: str = SUBQUERY_MODEL,
                        answer_model: str = CHAT_MODEL) -> str:
    """Iterative RAG with HYBRID retrieval (Vector + BM25 + RRF)."""
    # 1. Generate Queries
    all_queries = [initial_query]
    # Ensure subquery model client is available (uses main process)
    if not gemini_client:
        print(f"Warning: Client for subquery model '{subquery_model}' not available. Using only initial query.")
    else:
        generated_subqueries = generate_subqueries(initial_query, model=subquery_model)
        if generated_subqueries and generated_subqueries != [initial_query]:
            print("--- Generated Subqueries ---")
            for idx, subq in enumerate(generated_subqueries): print(f"  {idx+1}. {subq}")
            all_queries.extend(generated_subqueries)
        else: print("--- Using only the initial query (no distinct subqueries generated) ---")

    # 2. Retrieve Chunks (Hybrid) for all queries
    vector_results_all, bm25_results_all = [], []
    fetch_k = max(top_k, int(top_k * 1.5)) # Fetch more candidates for RRF

    print("\n--- Retrieving Chunks (Hybrid Vector + BM25) ---")
    # Ensure embedding client is available for query embedding (uses main process)
    if not gemini_client:
         return f"Error: Client for embedding model '{EMBEDDING_MODEL}' needed for query is not available."

    for q_idx, current_query in enumerate(all_queries):
         print(f"  Querying ({q_idx+1}/{len(all_queries)}): \"{current_query[:100]}...\"")
         # retrieve_chunks_for_query embeds the query
         vector_results_all.extend(retrieve_chunks_for_query(current_query, db_path, collection_name, fetch_k))
         # retrieve_chunks_bm25 uses the loaded index
         bm25_results_all.extend(retrieve_chunks_bm25(current_query, db_path, collection_name, fetch_k))

    # 3. Deduplicate and Combine using RRF
    print(f"\n--- Combining {len(vector_results_all)} Vector & {len(bm25_results_all)} BM25 candidate results ---")

    # Deduplicate vector results (keep best score/lowest distance for each unique chunk ID)
    deduped_vector_results_dict: Dict[str, Dict] = {}
    for chunk_meta in vector_results_all:
        file_hash = chunk_meta.get('file_hash')
        chunk_number = chunk_meta.get('chunk_number')
        if file_hash is None or chunk_number is None: continue
        chunk_id = f"{file_hash}_{chunk_number}"
        current_dist = chunk_meta.get('distance', float('inf'))
        # Keep the one with the smallest distance if ID collision occurs
        if chunk_id not in deduped_vector_results_dict or current_dist < deduped_vector_results_dict[chunk_id].get('distance', float('inf')):
             deduped_vector_results_dict[chunk_id] = chunk_meta
    deduped_vector_results = list(deduped_vector_results_dict.values())

    # Deduplicate BM25 results (keep highest score for each unique chunk ID)
    deduped_bm25_results_dict: Dict[str, float] = {}
    for chunk_id, score in bm25_results_all:
         if chunk_id and (chunk_id not in deduped_bm25_results_dict or score > deduped_bm25_results_dict[chunk_id]):
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
        # Use contextualized text if available, otherwise fall back to raw text
        f"Content:\n{chunk.get('contextualized_text', chunk.get('text', ''))}"
        for chunk in unique_chunks_list
    )
    # Optional: Save combined context for debugging
    # try: with open('combined_context_iterative.txt', 'w', encoding='utf-8') as f: f.write(combined_context)
    # except Exception as e: print(f"Warning: Could not write combined_context_iterative.txt: {e}")

    # 5. Generate Final Answer
    print("\n--- Generating Final Answer (Iterative Hybrid Retrieval) ---")
    # Ensure answer model client is available (uses main process)
    if not gemini_client:
        return f"Error: Client for answer model '{answer_model}' is not available."

    final_answer = generate_answer(initial_query, combined_context, unique_chunks_list, model=answer_model)
    return final_answer

def query_index(query: str, db_path: str, collection_name: str,
                top_k: int = DEFAULT_TOP_K, answer_model: str = CHAT_MODEL) -> str:
    """Direct query using HYBRID retrieval (Vector + BM25 + RRF)."""
    print(f"--- Running Direct Hybrid Query ---")
    print(f"Query: {query}")
    fetch_k = max(top_k, int(top_k * 1.5)) # Fetch more candidates

    # 1. Retrieve Chunks (Hybrid)
    # Ensure embedding client is available (uses main process)
    if not gemini_client:
         return f"Error: Client for embedding model '{EMBEDDING_MODEL}' needed for query is not available."

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
    # Optional: Save combined context for debugging
    # try: with open('direct_query_context.txt', 'w', encoding='utf-8') as f: f.write(combined_context)
    # except Exception as e: print(f"Warning: Could not write direct_query_context.txt: {e}")

    # 4. Generate Final Answer
    print("\n--- Generating Final Answer (Direct Hybrid Query) ---")
     # Ensure answer model client is available (uses main process)
    if not gemini_client:
        return f"Error: Client for answer model '{answer_model}' is not available."

    answer = generate_answer(query, combined_context, unique_chunks_list, model=answer_model)
    return answer

def setup_test_mode(test_folder="cleaned_text/test_docs", test_db_path="chunk_database/test_rag_db", test_collection="test_collection", query="what color are apples?"):
    """Creates test directory and dummy files if needed."""
    print("--- Running in Test Mode ---")
    os.makedirs(test_folder, exist_ok=True)
    if not os.listdir(test_folder):
        with open(os.path.join(test_folder, "test1.txt"), "w") as f: f.write("Red apples are sweet. BM25 scores terms.")
        with open(os.path.join(test_folder, "test2.txt"), "w") as f: f.write("Vector search finds similar concepts like fruit.")
        with open(os.path.join(test_folder, "test3.txt"), "w") as f: f.write("Apples can be green too. Climate index storylines distractor")
        print("Created dummy test files.")
    return query, test_folder, test_db_path, test_collection

def get_test_args(phase, query, test_folder, test_db_path, test_collection, top_k=3):
    """Returns the argument list for a specific test phase."""
    if phase == "index":
        print("\n--- TEST PHASE: INDEX ---")
        return ["--mode", "index", "--folder_path", test_folder, "--db_path", test_db_path, "--collection_name", test_collection, "--force_reindex"]
    elif phase == "embed":
        print("\n--- TEST PHASE: EMBED ---")
        return ["--mode", "embed", "--db_path", test_db_path, "--collection_name", test_collection]
    elif phase == "query":
        print("\n--- TEST PHASE: QUERY ---")
        return ["--mode", "query", "--query", query, "--db_path", test_db_path, "--collection_name", test_collection, "--top_k", str(top_k)]
    else: # Default to index if phase unknown
        print("\n--- TEST PHASE: INDEX (Default) ---")
        return ["--mode", "index", "--folder_path", test_folder, "--db_path", test_db_path, "--collection_name", test_collection]

def parse_arguments(test_mode_enabled=True):
    """Parses command-line arguments, handling test mode overrides."""
    parser = argparse.ArgumentParser(
        description="RAG Script (Two-Phase Indexing: index -> embed). Handles .txt files sequentially.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--mode", required=True, choices=["index", "embed", "query", "query_direct"],
                        help="'index': Chunk files, store raw data, build BM25. "
                             "'embed': Generate embeddings for stored chunks lacking them. "
                             "'query': Iterative hybrid query. "
                             "'query_direct': Direct hybrid query.")
    parser.add_argument("--document_path", type=str, default=None, help="Path to a single .txt document (index mode).")
    parser.add_argument("--folder_path", type=str, default=None, help="Path to a folder containing .txt documents (index mode).")
    parser.add_argument("--db_path", type=str, default="./rag_db", help="Path to the persistent database directory.")
    parser.add_argument("--collection_name", type=str, default=DEFAULT_CHROMA_COLLECTION_NAME, help="Name of the ChromaDB collection.")
    parser.add_argument("--query", type=str, default="What is the main topic?", help="Query string (query modes).")
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K, help="Number of top results to retrieve/use (query modes).")
    parser.add_argument("--force_reindex", action="store_true",
                        help="Force reprocessing (chunking/storing) of all source files in 'index' mode, "
                             "resetting 'has_embedding' flag. Does not automatically trigger 'embed'.")
    parser.add_argument("--embed_batch_size", type=int, default=50,
                        help="Number of chunks to process in each batch during 'embed' mode.")
    parser.add_argument("--embed_delay", type=float, default=0.1,
                         help="Optional delay (seconds) between embedding batches in 'embed' mode (to avoid rate limits).")

    if test_mode_enabled:
        query, test_folder, test_db_path, test_collection = setup_test_mode()
        test_phase = "query" # Control phase: "index", "embed", "query"
        args_list = get_test_args(test_phase, query, test_folder, test_db_path, test_collection, DEFAULT_TOP_K)
        if args_list:
            args = parser.parse_args(args_list)
        else:
            print("Error: Test mode enabled, but failed to determine test arguments.")
            exit(1) # Exit if something went wrong determining the test phase args
    else:
        args = parser.parse_args()

    return args

def print_and_validate_config(args):
    """Prints the current configuration and performs basic validation."""
    print(f"\n--- Configuration ---")
    print(f"Mode: {args.mode}")
    print(f"DB Path: {args.db_path}")
    print(f"Collection Name: {args.collection_name}")
    if args.mode in ['query', 'query_direct']: print(f"Query: '{args.query}' | Top K: {args.top_k}")
    if args.mode == 'index': print(f"Source Folder: {args.folder_path} | Source File: {args.document_path} | Force Re-index: {args.force_reindex}")
    if args.mode == 'embed': print(f"Embedding Batch Size: {args.embed_batch_size} | Batch Delay: {args.embed_delay}s")
    print(f"Embedding Model: {EMBEDDING_MODEL} | Context Model: {CHUNK_CONTEXT_MODEL} | Chat Model: {CHAT_MODEL}")
    print(f"--------------------\n")

    # Basic validation
    if args.mode == "index" and not args.document_path and not args.folder_path:
        raise ValueError("--document_path or --folder_path is required for 'index' mode.")
    if args.mode in ["query", "query_direct"] and not args.query:
        raise ValueError("--query cannot be empty for query modes.")

# --- Index Mode Functions ---

def find_files_to_index(folder_path, document_path):
    """Identifies potential .txt files to index from folder or single path."""
    potential_files = []
    if folder_path:
        if not os.path.isdir(folder_path): raise ValueError(f"Folder not found: {folder_path}")
        print(f"Scanning folder: {folder_path}")
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(".txt"):
                    potential_files.append(os.path.join(root, file))
    elif document_path:
        if not os.path.isfile(document_path): raise ValueError(f"File not found: {document_path}")
        if document_path.lower().endswith(".txt"): potential_files.append(document_path)
        else: print(f"Warning: Skipping non-txt file: {document_path}")

    if not potential_files: print("No source .txt files found to index.")
    else: print(f"Found {len(potential_files)} potential source file(s).")
    return potential_files

def filter_files_for_processing(potential_files, db_path, collection_name, force_reindex):
    """Checks ChromaDB for existing file hashes and filters the list of files."""
    files_to_process = []
    skipped_files_count = 0
    existing_hashes = set()

    if not force_reindex:
        print("Checking ChromaDB for already indexed files (based on hash)...")
        try:
            collection = get_chroma_collection(db_path, collection_name)
            existing_data = collection.get(include=['metadatas']) # Fetch metadata
            if existing_data and existing_data.get('metadatas'):
                count = 0
                for meta in existing_data['metadatas']:
                    if meta and 'file_hash' in meta:
                        existing_hashes.add(meta['file_hash'])
                        count += 1
                print(f"Found {len(existing_hashes)} unique file hashes from {count} existing chunks in ChromaDB.")
        except Exception as e:
            if "does not exist" in str(e) or "not found" in str(e).lower(): # Adapt based on actual ChromaDB exception message
                 print("Collection not found or empty, assuming no files are indexed yet.")
            else: print(f"Warning: Could not check existing files in ChromaDB: {e}. Processing all files.")
    else:
        print("Force re-index enabled, processing all found files.")

    print("Filtering files to process...")
    skipped_files = []
    for file_path in tqdm(potential_files, desc="Checking files", unit="file"):
        try:
            file_hash = compute_file_hash(file_path)
            if not force_reindex and file_hash in existing_hashes:
                skipped_files.append(os.path.basename(file_path)); skipped_files_count += 1
            else: files_to_process.append(file_path)
        except FileNotFoundError: print(f"\nWarning: File not found during check: {file_path}. Skipping.")
        except Exception as e: print(f"\nError hashing file {os.path.basename(file_path)}, skipping: {e}"); skipped_files_count += 1

    if skipped_files_count > 0:
        print(f"Skipping {skipped_files_count} file(s) already indexed or with hash errors.")

    return files_to_process, skipped_files_count

def process_files_sequentially(files_to_process):
    """Processes a list of files sequentially for Phase 1 (chunking)."""
    all_phase1_chunks = []
    successful_files, failed_files_info = 0, []

    print("--- Starting Sequential Chunk Processing (Phase 1) ---")
    for file_path in tqdm(files_to_process, desc="Processing Files (Phase 1)", unit="file"):
        try:
            processed_data = index_document_phase1(
                document_path=file_path,
                max_tokens=DEFAULT_MAX_TOKENS,
                overlap=DEFAULT_OVERLAP
            )
            if processed_data:
                all_phase1_chunks.extend(processed_data)
                successful_files += 1
        except Exception as e:
            err_msg = f"CRITICAL Error processing {os.path.basename(file_path)} (Phase 1): {e}\n{traceback.format_exc()}"
            print(f"\n{err_msg}")
            failed_files_info.append((os.path.basename(file_path), str(e)))

    print(f"\n--- Phase 1 Processing Summary ---")
    print(f"Successfully processed files (yielding chunks): {successful_files}")
    if failed_files_info:
        print(f"Failed processing attempts: {len(failed_files_info)}")
        for fname, err in failed_files_info[:5]: print(f"  - Example Failure: {fname}: {err}")
        if len(failed_files_info) > 5: print("  ...")

    return all_phase1_chunks, failed_files_info

def update_chromadb_raw_chunks(collection, all_phase1_chunks):
    """Adds/Updates raw chunk data (Phase 1) to ChromaDB."""
    if not all_phase1_chunks:
        print("No new raw chunks to add/update in ChromaDB.")
        return

    print(f"Adding/Updating {len(all_phase1_chunks)} raw chunks in ChromaDB...")
    chroma_ids = [chunk['id'] for chunk in all_phase1_chunks]
    chroma_metadatas = [chunk['metadata'] for chunk in all_phase1_chunks]
    chroma_documents = [chunk['metadata'].get('text', '') for chunk in all_phase1_chunks] # Ensure text is included

    batch_size = 500
    num_batches = (len(chroma_ids) + batch_size - 1) // batch_size
    for i in tqdm(range(num_batches), desc="Adding/Upserting Raw Chunks to ChromaDB"):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        batch_ids = chroma_ids[start_idx:end_idx]
        batch_metadatas = chroma_metadatas[start_idx:end_idx]
        batch_documents = chroma_documents[start_idx:end_idx]

        if not batch_ids: continue
        try:
            collection.upsert(
                ids=batch_ids,
                metadatas=batch_metadatas,
                documents=batch_documents # Pass raw text for upsert
            )
        except Exception as upsert_err:
            print(f"\n!!! Error upserting batch {i+1}/{num_batches} to ChromaDB: {upsert_err}")
            # Consider logging failed IDs or retrying?

    print("Finished adding/upserting raw chunks to ChromaDB.")

def rebuild_bm25_index(collection, db_path, collection_name):
    """Rebuilds and saves the BM25 index using all current data in ChromaDB."""
    print("Rebuilding BM25 index using all data currently in ChromaDB...")
    try:
        all_data = collection.get(include=['metadatas']) # Fetch all needed data
        if not all_data or not all_data.get('ids'):
            print("Warning: Could not fetch any data from ChromaDB for BM25 rebuild. Collection might be empty.")
            return

        all_current_ids = all_data['ids']
        all_current_metadatas = all_data.get('metadatas', [])
        valid_entries = [(id, meta) for id, meta in zip(all_current_ids, all_current_metadatas) if meta and 'text' in meta and meta['text'].strip()]

        if len(valid_entries) != len(all_current_ids):
            print(f"Warning: Found {len(all_current_ids) - len(valid_entries)} entries with missing/invalid/empty text metadata. Excluding them from BM25.")

        if not valid_entries:
            print("No valid chunk text found in ChromaDB metadata. Skipping BM25 build.")
            return

        valid_ids = [id for id, meta in valid_entries]
        valid_texts = [meta['text'] for id, meta in valid_entries]
        print(f"Tokenizing {len(valid_ids)} total valid chunks for BM25...")

        all_final_tokenized_corpus = [
            tokenize_text_bm25(text)
            for text in tqdm(valid_texts, desc="Tokenizing all chunks")
        ]

        if not any(all_final_tokenized_corpus):
            print("Warning: Tokenization resulted in empty corpus. Skipping BM25 build.")
            return

        print(f"Building final BM25 index...")
        bm25_index = BM25Okapi(all_final_tokenized_corpus)
        bm25_index_path, bm25_mapping_path = get_bm25_paths(db_path, collection_name)

        print(f"Saving final BM25 index ({len(valid_ids)} docs) to: {bm25_index_path}")
        os.makedirs(os.path.dirname(bm25_index_path), exist_ok=True)
        with open(bm25_index_path, 'wb') as f_idx: pickle.dump(bm25_index, f_idx)

        print(f"Saving final BM25 ID mapping to: {bm25_mapping_path}")
        with open(bm25_mapping_path, 'wb') as f_map: pickle.dump(valid_ids, f_map)

        print("Final BM25 index saved.")

    except Exception as bm25_err:
        print(f"!!! Error during BM25 rebuild phase: {bm25_err}")
        traceback.print_exc()

def run_index_mode(args):
    """Handles the logic for the 'index' mode."""
    print("--- Running Index Mode (Phase 1) ---")
    # 1. Find potential files
    potential_files = find_files_to_index(args.folder_path, args.document_path)
    if not potential_files: return

    # 2. Filter based on existing hashes (unless forced)
    files_to_process, skipped_count = filter_files_for_processing(
        potential_files, args.db_path, args.collection_name, args.force_reindex
    )
    if not files_to_process:
        print("No new files need processing.")
        # If forcing, we might still want to rebuild BM25 even if no *new* files
        if args.force_reindex:
             print("Force re-index requested, proceeding to BM25 rebuild with existing data.")
             collection = get_chroma_collection(args.db_path, args.collection_name)
             rebuild_bm25_index(collection, args.db_path, args.collection_name)
        return

    # 3. Process files sequentially
    all_phase1_chunks, _ = process_files_sequentially(files_to_process)

    # 4 & 5. Update DB and Rebuild BM25
    if not all_phase1_chunks and not args.force_reindex:
        print("No valid new chunks were generated. Nothing to add to DB or index.")
        return

    try:
        collection = get_chroma_collection(args.db_path, args.collection_name)
        update_chromadb_raw_chunks(collection, all_phase1_chunks)
        rebuild_bm25_index(collection, args.db_path, args.collection_name)
    except Exception as db_bm25_err:
        print(f"!!! Error during ChromaDB update or BM25 build phase: {db_bm25_err}")
        traceback.print_exc()

    print("\n--- Index Mode (Phase 1) Complete ---")
    print("Raw chunks stored/updated. Run '--mode embed' to generate embeddings for any missing ones.")


# --- Embed Mode Functions ---

def find_chunks_to_embed(collection):
    """Finds chunks in the collection marked with 'has_embedding': False."""
    print(f"Checking collection for chunks needing embedding...")
    try:
        results = collection.get(
            where={"has_embedding": False},
            include=['metadatas'] # Only need metadata to get text
        )
        ids_to_embed = results.get('ids', [])
        metadatas_to_embed = results.get('metadatas', [])

        if not ids_to_embed:
            print("No chunks found needing embedding in this collection.")
            return [], []
        else:
            print(f"Found {len(ids_to_embed)} chunks to embed.")
            return ids_to_embed, metadatas_to_embed
    except Exception as e:
        print(f"Error querying ChromaDB for chunks to embed: {e}")
        return [], []

def generate_embeddings_in_batches(ids_to_embed, metadatas_to_embed, batch_size, delay):
    """Generates embeddings for the given chunks in batches using the client's batch API."""
    embeddings_to_update = []
    ids_to_update = []
    metadatas_to_update = []
    failed_ids = []

    num_batches = (len(ids_to_embed) + batch_size - 1) // batch_size
    print(f"Processing {len(ids_to_embed)} chunks in {num_batches} batches of size {batch_size}...")

    # --- Ensure Client is Available ---
    # Check if the client needed for the embedding model is available
    client_available = False
    model_name_lower = EMBEDDING_MODEL.lower()
    if any(m in model_name_lower for m in ["embedding-001", "text-embedding-004"]):
        if gemini_client:
            client_available = True
        else:
            print("ERROR: Google GenAI client (gemini_client) is not initialized. Cannot generate embeddings.")
            # Mark all as failed if client is missing
            return [], [], [], ids_to_embed
    # Add checks for other potential providers/clients if needed
    # elif 'openai' in model_name_lower:
    #     if openai_client: client_available = True ...

    if not client_available:
        print(f"ERROR: Appropriate client for model '{EMBEDDING_MODEL}' not found.")
        return [], [], [], ids_to_embed
    # ---------------------------------


    for i in tqdm(range(num_batches), desc="Generating Embeddings", unit="batch"):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        batch_ids = ids_to_embed[start_idx:end_idx]
        batch_metadatas = metadatas_to_embed[start_idx:end_idx]

        # --- Prepare Batch Data ---
        texts_in_batch = []
        valid_indices_in_batch = [] # Keep track of original index within the batch slice
        ids_in_batch_map = {} # Map valid_index -> chunk_id
        metadatas_in_batch_map = {} # Map valid_index -> original metadata

        for idx_in_slice, (chunk_id, meta) in enumerate(zip(batch_ids, batch_metadatas)):
            if not meta:
                print(f"Warning: Missing metadata for chunk {chunk_id} in batch {i+1}. Skipping.")
                failed_ids.append(chunk_id)
                continue

            text_to_embed = meta.get('contextualized_text', meta.get('text'))
            if not text_to_embed or not text_to_embed.strip():
                print(f"Warning: Empty text for chunk {chunk_id} in batch {i+1}. Skipping.")
                failed_ids.append(chunk_id)
                continue

            # Preprocess text (as done in get_embedding)
            text_to_embed = text_to_embed.replace("\n", " ").strip()
            if not text_to_embed: # Check again after preprocessing
                 print(f"Warning: Text became empty after preprocessing for chunk {chunk_id} in batch {i+1}. Skipping.")
                 failed_ids.append(chunk_id)
                 continue

            texts_in_batch.append(text_to_embed)
            valid_indices_in_batch.append(idx_in_slice) # Store the original position within the slice
            ids_in_batch_map[idx_in_slice] = chunk_id
            metadatas_in_batch_map[idx_in_slice] = meta


        if not texts_in_batch:
            print(f"Skipping batch {i+1} as it contains no valid texts to embed.")
            continue # Skip to the next batch

        # --- Make Batch Embedding API Call ---
        try:
            # Ensure model name has prefix if needed (Gemini often does)
            api_model_name = EMBEDDING_MODEL if EMBEDDING_MODEL.startswith("models/") else f"models/{EMBEDDING_MODEL}"

            # Configure embedding options
            # Use uppercase task_type for v1beta API (as shown in user example)
            # retrieval_document is a good default for storing chunks
            task_type_upper = "RETRIEVAL_DOCUMENT" # Explicitly set for embedding stored chunks

            # Prepare config, only include dimension if it's set globally
            embed_config_args = {'task_type': task_type_upper}
            if OUTPUT_EMBEDDING_DIMENSION is not None:
                embed_config_args['output_dimensionality'] = OUTPUT_EMBEDDING_DIMENSION

            config = EmbedContentConfig(**embed_config_args)

            # print(f"DEBUG: Calling embed_content for batch {i+1} with {len(texts_in_batch)} texts. Config: {config}")

            # Call the batch embedding endpoint
            response = gemini_client.models.embed_content(
                model=api_model_name,
                contents=texts_in_batch,
                config=config
            )

            # --- Process Batch Response ---
            if not response or not hasattr(response, 'embeddings'):
                 print(f"\nError: Invalid or empty response received from embedding API for batch {i+1}. Failing all chunks in this batch.")
                 # Mark all successfully *prepared* items in this batch as failed
                 for idx_in_slice in valid_indices_in_batch:
                     failed_ids.append(ids_in_batch_map[idx_in_slice])
                 continue # Move to next batch

            if len(response.embeddings) != len(texts_in_batch):
                print(f"\nError: Mismatch between requested texts ({len(texts_in_batch)}) and received embeddings ({len(response.embeddings)}) for batch {i+1}. Failing all chunks in this batch.")
                for idx_in_slice in valid_indices_in_batch:
                     failed_ids.append(ids_in_batch_map[idx_in_slice])
                continue # Move to next batch

            # Iterate through the embeddings received, correlating them back to the original chunks
            for response_idx, embedding_result in enumerate(response.embeddings):
                # Get the original index within the batch slice
                original_slice_idx = valid_indices_in_batch[response_idx]
                chunk_id = ids_in_batch_map[original_slice_idx]
                original_meta = metadatas_in_batch_map[original_slice_idx]

                if hasattr(embedding_result, 'values') and isinstance(embedding_result.values, list) and len(embedding_result.values) > 0:
                    # Successfully embedded
                    embedding_vector = embedding_result.values
                    embeddings_to_update.append(embedding_vector) # Already a list

                    # Update metadata
                    updated_meta = original_meta.copy()
                    updated_meta['has_embedding'] = True
                    metadatas_to_update.append(updated_meta)

                    # Add ID to the list of chunks successfully processed
                    ids_to_update.append(chunk_id)
                else:
                    # Embedding failed for this specific chunk in the batch
                    print(f"\nWarning: Failed to get embedding values for chunk {chunk_id} within batch {i+1}. Skipping update for this chunk.")
                    failed_ids.append(chunk_id)

        except Exception as embed_err:
             print(f"\n!!! Critical Error during batch embedding API call for batch {i+1}: {embed_err}")
             print(traceback.format_exc()) # Print stack trace for debugging
             # Mark all successfully prepared items in this batch as failed
             for idx_in_slice in valid_indices_in_batch:
                  failed_ids.append(ids_in_batch_map[idx_in_slice])
             # Optional: break loop if one batch fails critically? Or continue? Current: continue.


        # --- Optional Delay Between Batches ---
        if delay > 0 and i < num_batches - 1:
            # print(f"Waiting {delay}s before next batch...") # Debug
            time.sleep(delay)

    # --- Return results ---
    # Make failed_ids unique
    unique_failed_ids = list(set(failed_ids))
    print(f"Embedding generation complete. Successful: {len(ids_to_update)}, Failed: {len(unique_failed_ids)}")

    return ids_to_update, embeddings_to_update, metadatas_to_update, unique_failed_ids

def update_embedded_chunks_in_chromadb(collection, ids, embeddings, metadatas):
    """Updates chunks in ChromaDB with their generated embeddings."""
    if not ids:
        print("No successful embeddings were generated to update in the database.")
        return

    print(f"\nUpdating {len(ids)} chunks in ChromaDB with new embeddings...")
    update_batch_size = 500
    failed_update_ids = [] # Track failures during DB update
    num_update_batches = (len(ids) + update_batch_size - 1) // update_batch_size

    for i in tqdm(range(num_update_batches), desc="Updating ChromaDB", unit="batch"):
        start_idx = i * update_batch_size
        end_idx = start_idx + update_batch_size
        batch_ids = ids[start_idx:end_idx]
        batch_embeddings = embeddings[start_idx:end_idx]
        batch_metadatas = metadatas[start_idx:end_idx]

        if not batch_ids: continue
        try:
            collection.update(
                ids=batch_ids,
                embeddings=batch_embeddings,
                metadatas=batch_metadatas
            )
        except Exception as db_update_err:
            print(f"\n!!! Error updating batch {i+1}/{num_update_batches} in ChromaDB: {db_update_err}")
            failed_update_ids.extend(batch_ids) # Mark these as failed

    print("Finished updating chunks in ChromaDB.")
    return failed_update_ids


def run_embed_mode(args):
    """Handles the logic for the 'embed' mode."""
    print("--- Running Embed Mode (Phase 2) ---")
    # Check if necessary client is available
    provider = "Unknown"
    client_ok = False
    if EMBEDDING_MODEL in ["embedding-001", "models/embedding-001", "text-embedding-004", "models/text-embedding-004"]: provider = "Gemini"; client_ok = bool(gemini_client) # Add checks for other providers if needed

    if not client_ok:
        print(f"Error: {provider} client needed for embedding model '{EMBEDDING_MODEL}' is not initialized. Cannot proceed.")
        return

    try:
        collection = get_chroma_collection(args.db_path, args.collection_name)

        # 1. Find chunks needing embedding
        ids_to_embed, metadatas_to_embed = find_chunks_to_embed(collection)
        if not ids_to_embed: return

        # 2. Generate embeddings in batches
        ids_ok, embeddings_ok, metadatas_ok, failed_embed_ids = generate_embeddings_in_batches(
            ids_to_embed, metadatas_to_embed, args.embed_batch_size, args.embed_delay
        )

        # 3. Update ChromaDB
        failed_update_ids = update_embedded_chunks_in_chromadb(collection, ids_ok, embeddings_ok, metadatas_ok)

        # 4. Report failures
        all_failed_ids = set(failed_embed_ids + failed_update_ids)
        if all_failed_ids:
            print(f"\nWarning: Failed to process/embed/update {len(all_failed_ids)} unique chunks.")
            print("These chunks may remain marked as 'has_embedding': False and can be retried.")
            # Optional: Log failed_ids to a file

    except Exception as e:
        print(f"!!! Error during embed mode execution: {e}")
        traceback.print_exc()

    print("\n--- Embed Mode (Phase 2) Complete ---")


# --- Query Mode Function ---

def run_query_mode(args):
    """Handles the logic for 'query' and 'query_direct' modes."""
    print(f"--- Running Query Mode ({args.mode}) ---")

    # Check clients needed
    query_embed_client_ok = bool(gemini_client) # Assuming Gemini for default embedding
    subquery_client_ok = bool(gemini_client) if args.mode == "query" else True
    answer_client_ok = bool(gemini_client) # Assuming Gemini for default chat

    if not query_embed_client_ok: print(f"Error: Client for query embedding model '{EMBEDDING_MODEL}' not available."); return
    if not subquery_client_ok: print(f"Error: Client for subquery model '{SUBQUERY_MODEL}' not available."); return
    if not answer_client_ok: print(f"Error: Client for answer model '{CHAT_MODEL}' not available."); return

    # Ensure BM25 index is loaded for hybrid search
    if not load_bm25_index(args.db_path, args.collection_name):
        print("Warning: BM25 index not found or failed to load. Lexical search part of hybrid query will not work.")
        # Query can proceed but RRF might behave differently

    # Execute the appropriate query function
    final_answer = ""
    if args.mode == "query":
            final_answer = iterative_rag_query(args.query, args.db_path, args.collection_name,
                                               top_k=args.top_k,
                                               subquery_model=SUBQUERY_MODEL,
                                               answer_model=CHAT_MODEL)
    else: # query_direct
            final_answer = query_index(args.query, args.db_path, args.collection_name,
                                       top_k=args.top_k,
                                       answer_model=CHAT_MODEL)

    # Print the final result
    print("\n" + "="*20 + " Final Answer " + "="*20)
    print(final_answer)
    print("="*54 + "\n")


# --- Main Orchestrator ---

def main():
    """Main function to orchestrate the RAG script."""
    test_mode_enabled = True # Set to True/False as needed

    try:
        # 1. Parse Arguments (handles test mode internally)
        args = parse_arguments(test_mode_enabled)

        # 2. Initialize API Clients (essential for all modes)
        print(f"Main process initializing clients...")
        initialize_clients() # Should ideally raise error if fails

        # 3. Configuration & Validation
        print_and_validate_config(args) # Raises ValueError on validation failure

        # 4. Execute Mode-Specific Logic
        if args.mode == "index":
            run_index_mode(args)
        elif args.mode == "embed":
            run_embed_mode(args)
        elif args.mode in ["query", "query_direct"]:
            run_query_mode(args)
        else:
            # Should not happen due to choices in argparse, but good practice
            print(f"Error: Unknown mode '{args.mode}'")

    except ValueError as ve:
        # Catch specific configuration/validation errors
        print(f"\n!!! Configuration Error: {ve}")
        exit(1) # Exit on configuration errors
    except RuntimeError as rte:
         # Catch client initialization errors if initialize_clients raises them
        print(f"\n!!! Runtime Error (likely client initialization): {rte}")
        exit(1)
    except Exception as e:
        # Catch any other unexpected errors during main execution
        print(f"\n\n!!! An unexpected error occurred: {e}")
        print("Traceback:")
        traceback.print_exc()
        exit(1) # Exit on unexpected errors

if __name__ == "__main__":
    main()