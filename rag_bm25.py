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
from google.api_core import exceptions as google_exceptions # Import google exceptions for specific checking
from google.genai.types import EmbedContentConfig
import traceback # For more detailed error logging

try:
    from sentence_transformers import CrossEncoder
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("Warning: sentence-transformers not installed (`pip install sentence-transformers`). Re-ranking will be disabled.")
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    CrossEncoder = None # Define as None to avoid NameError

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
    RERANKER_MODEL,
    DEFAULT_RERANK_CANDIDATE_COUNT,
    DEFAULT_TOTAL_CONTEXT_WINDOW,
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
    def __init__(self,
                 model_name: Optional[str] = None,
                 output_dimension_override: Optional[int] = None,
                 task_type: str = "retrieval_document",
                 current_mode: str = "unknown"): # Add mode parameter
        """
        Initializes the embedding function, sets dimension, and stores execution mode.
        """
        self._model_name = model_name if model_name else EMBEDDING_MODEL
        if not self._model_name:
             raise ValueError("Embedding model name must be provided either during init or globally via EMBEDDING_MODEL.")

        self._mode = current_mode # Store the mode

        determined_dimension: Optional[int] = None
        if output_dimension_override is not None:
             determined_dimension = output_dimension_override
        else:
             determined_dimension = OUTPUT_EMBEDDING_DIMENSION

        self.dimension = determined_dimension
        self._output_dimension = determined_dimension # Keep internal if needed

        self._task_type = task_type

        if self.dimension is None and self._mode != "index":
             # Dimension is crucial if we need to generate real or zero vectors
             raise ValueError("Embedding dimension must be set via config or override unless in 'index' mode where it might be inferred later (though risky).")
        elif self.dimension is None and self._mode == "index":
             print("WARNING (EmbeddingFunction Init): Dimension not set in 'index' mode. Upsert might fail if zero vectors are needed.")


    def __call__(self, input_texts: Documents) -> Embeddings:
        """
        Generates embeddings or placeholders based on the mode.
        """
        print(f"DEBUG: ConfigurableEmbeddingFunction.__call__ invoked in mode '{self._mode}' for {len(input_texts)} texts!")

        # --- MODE: index ---
        # If called during index mode (likely via implicit upsert), return zeros.
        if self._mode == "index":
            if self.dimension is None:
                # This should ideally not happen if init logic is correct, but as a safeguard:
                print("ERROR (EmbeddingFunction): Cannot generate zero vectors in 'index' mode without a known dimension!")
                # Raising an error might be better than letting ChromaDB fail later
                raise ValueError("Dimension required for placeholder vectors in index mode.")
            print(f"INFO (EmbeddingFunction): In 'index' mode, returning zero vectors ({self.dimension}d) instead of calling API.")
            zero_vector = [0.0] * self.dimension
            return [zero_vector for _ in input_texts]

        # --- MODES: embed, query, other ---
        # Proceed with actual embedding generation
        all_embeddings: Embeddings = []
        client_available = False # Check client availability again
        model_name_lower = self._model_name.lower()
        if any(m in model_name_lower for m in ["embedding-001", "text-embedding-004"]):
            if gemini_client: client_available = True

        if not client_available:
            print(f"ERROR (EmbeddingFunction __call__): Client for model '{self._model_name}' not found in mode '{self._mode}'. Returning zero vectors.")
            if self.dimension is None: raise ValueError("Dimension required for placeholder vectors on client failure.")
            zero_vector = [0.0] * self.dimension
            return [zero_vector for _ in input_texts] # Return zeros on client failure

        # Generate real embeddings
        for text in input_texts:
            embedding = None # Initialize embedding for this text
            try:
                embedding = get_embedding(
                    text=text,
                    model=self._model_name,
                    task_type=self._task_type,
                    embedding_dimension=self.dimension
                )

                if embedding is not None and isinstance(embedding, np.ndarray) and embedding.size > 0:
                    # Optional dimension check
                    if self.dimension is not None and len(embedding) != self.dimension:
                         print(f"WARNING (EmbeddingFunction): Dimension mismatch! Expected {self.dimension}, got {len(embedding)}. Using vector anyway.")
                    all_embeddings.append(embedding.tolist())
                else:
                    # Failed to get a valid embedding (API error handled in get_embedding, or empty result)
                    print(f"ERROR (EmbeddingFunction): Failed to get valid embedding for text: '{text[:50]}...'. Appending zero vector.")
                    if self.dimension is None: raise ValueError("Dimension required for placeholder zero vector on embedding failure.")
                    all_embeddings.append([0.0] * self.dimension) # Append ZEROS

            except Exception as e:
                # Catch errors from get_embedding itself (like the 429)
                print(f"ERROR (EmbeddingFunction): Exception during embedding generation for text '{text[:50]}...': {e}. Appending zero vector.")
                # traceback.print_exc() # Optionally print traceback here too
                if self.dimension is None: raise ValueError("Dimension required for placeholder zero vector on exception.")
                all_embeddings.append([0.0] * self.dimension) # Append ZEROS

        return all_embeddings

# --- ChromaDB Client Setup ---
def get_chroma_collection(db_path: str, collection_name: str, execution_mode: str) -> chromadb.Collection:
    """
    Gets or creates a ChromaDB collection, passing the execution mode
    to the embedding function to control its behavior (e.g., return zeros
    during index phase). Ensures the collection is configured with the
    correct dimension hint from the embedding function instance.

    Args:
        db_path: Path to the persistent database directory.
        collection_name: Name of the ChromaDB collection.
        execution_mode: The current script mode ('index', 'embed', 'query', etc.).

    Returns:
        A chromadb.Collection instance.

    Raises:
        ValueError: If the embedding function cannot determine the dimension
                    when required by the execution mode.
        Exception: Propagates exceptions from ChromaDB connection/creation.
    """
    try:
        # Instantiate the custom embedding function, passing the mode
        # It will determine the dimension and store the mode internally.
        emb_func = ConfigurableEmbeddingFunction(
            model_name=EMBEDDING_MODEL, # Assumes EMBEDDING_MODEL is accessible
            current_mode=execution_mode
            # OUTPUT_EMBEDDING_DIMENSION is read globally inside ConfigurableEmbeddingFunction.__init__
        )

        # Log the dimension the function instance holds
        print(f"DEBUG: Instantiated ConfigurableEmbeddingFunction for mode '{execution_mode}' with dimension: {emb_func.dimension}")

        # Ensure the database directory exists
        os.makedirs(db_path, exist_ok=True)

        # Create the persistent client
        chroma_client = chromadb.PersistentClient(path=db_path)

        print(f"DEBUG: Getting/Creating collection '{collection_name}' with EmbeddingFunction instance.")

        # Get or create the collection, passing the embedding function instance.
        # ChromaDB should use the .dimension attribute (if available) from emb_func
        # during creation to set the index parameters correctly.
        # The emb_func.__call__ method will behave based on the stored 'execution_mode'.
        collection = chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=emb_func, # Pass the mode-aware instance
            metadata={"hnsw:space": "cosine"} # Example metadata, adjust if needed
        )

        print(f"DEBUG: Successfully retrieved/created collection '{collection_name}'.")
        return collection

    except ValueError as ve:
        # Catch potential dimension errors from embedding function init
        print(f"!!! Configuration Error for collection '{collection_name}': {ve}")
        raise # Re-raise config error
    except Exception as e:
        # Catch ChromaDB client/collection errors
        print(f"!!! Error connecting/creating ChromaDB collection '{collection_name}' at path '{db_path}': {e}")
        traceback.print_exc() # Print detailed traceback
        raise # Re-raise the exception to signal failure

def _update_db_batch(collection: chromadb.Collection, ids: List[str], embeddings: List[List[float]], metadatas: List[Dict]):
    """Internal helper to update a single batch in ChromaDB."""
    if not ids:
        return # Nothing to update
    try:
        print(f"DEBUG: Updating {len(ids)} successfully embedded chunks in DB.")
        collection.update(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas
        )
    except Exception as db_update_err:
        print(f"\n!!! Error updating batch in ChromaDB (IDs: {ids[:5]}...): {db_update_err}")
        # Consider how critical this is - should we add these IDs back to a failed list?
        # For now, just log the error. The metadata won't be marked 'has_embedding: True'.

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

def generate_chunk_context(
    full_document_text: str,
    chunk_text: str,
    start_token_idx: int,
    end_token_idx: int,
    total_window_tokens: int = DEFAULT_TOTAL_CONTEXT_WINDOW, # Use total window size
    summary_max_tokens: int = DEFAULT_CONTEXT_LENGTH,
    context_model: str = CHUNK_CONTEXT_MODEL,
    encoding_model: str = EMBEDDING_MODEL
) -> str:
    """
    Generates a succinct, chunk-specific context using a sliding token-based
    window of a fixed total size around the chunk.

    Args:
        full_document_text: The entire original document text.
        chunk_text: The text of the specific chunk.
        start_token_idx: The starting token index of the chunk in the full document.
        end_token_idx: The ending token index of the chunk in the full document.
        total_window_tokens: The total desired size of the context window.
        summary_max_tokens: Max tokens for the generated context summary output.
        context_model: The LLM model to use for generating the context summary.
        encoding_model: The model used for tokenization to define the window.

    Returns:
        A succinct contextual description (1-2 sentences) or an error message.
    """
    if not full_document_text.strip() or not chunk_text.strip():
        return "Context unavailable (empty input)."
    if start_token_idx is None or end_token_idx is None or start_token_idx < 0 or end_token_idx <= start_token_idx:
         print(f"Warning: Invalid token indices provided (start={start_token_idx}, end={end_token_idx}). Cannot extract window.")
         return "Error generating context (invalid indices)."
    if total_window_tokens <= 0:
         print(f"Warning: Non-positive total_window_tokens ({total_window_tokens}) requested. Cannot extract window.")
         return "Error generating context (invalid window size)."

    try:
        # 1. Tokenize the full document
        try:
             encoding = tiktoken.get_encoding("cl100k_base")
        except Exception:
             print(f"Warning: Using fallback 'cl100k_base' encoding for context window.")
             encoding = tiktoken.get_encoding("cl100k_base")

        document_tokens = encoding.encode(full_document_text)
        total_doc_tokens = len(document_tokens)

        # Handle case where document is smaller than window request
        if total_doc_tokens <= total_window_tokens:
            window_start_idx = 0
            window_end_idx = total_doc_tokens
            print(f"Info: Document ({total_doc_tokens} tokens) is smaller than requested window ({total_window_tokens}). Using full document as context window.")
        else:
            # 2. Calculate initial window boundaries trying to center the chunk
            chunk_len = end_token_idx - start_token_idx
            # Calculate tokens available *before* and *after* the ideal centered window space (excluding chunk itself)
            context_tokens_available = total_window_tokens - chunk_len
            if context_tokens_available < 0:
                # Chunk itself is larger than the window, just use the chunk bounds (or slightly expanded if possible)
                print(f"Warning: Chunk ({chunk_len} tokens) is larger than requested window ({total_window_tokens}). Expanding window slightly if possible.")
                expand_by = max(0, (total_window_tokens - chunk_len) // 2) # Try to add a little padding if window allows
                window_start_idx = max(0, start_token_idx - expand_by)
                window_end_idx = min(total_doc_tokens, end_token_idx + expand_by)
                # Recalculate actual size and adjust if still too small/large compared to total_window_tokens
                actual_size = window_end_idx - window_start_idx
                if actual_size < total_window_tokens:
                    # Try expanding more towards the end if possible
                    needed = total_window_tokens - actual_size
                    window_end_idx = min(total_doc_tokens, window_end_idx + needed)
                    actual_size = window_end_idx - window_start_idx # Recalc
                    if actual_size < total_window_tokens: # Still too small? Expand towards beginning
                        needed = total_window_tokens - actual_size
                        window_start_idx = max(0, window_start_idx - needed)

            else:
                # Ideal split of remaining context tokens
                ideal_before = context_tokens_available // 2
                ideal_after = context_tokens_available - ideal_before # Handles odd numbers

                # Calculate desired start/end based on ideal split
                desired_start = start_token_idx - ideal_before
                desired_end = end_token_idx + ideal_after

                # 3. Adjust ("slide") the window based on document boundaries
                overflow_start = 0 - desired_start if desired_start < 0 else 0
                overflow_end = desired_end - total_doc_tokens if desired_end > total_doc_tokens else 0

                if overflow_start > 0:
                    # Hit beginning: shift window right
                    window_start_idx = 0
                    # Add the overflow amount to the end, respecting document boundary
                    window_end_idx = min(total_doc_tokens, desired_end + overflow_start)
                elif overflow_end > 0:
                    # Hit end: shift window left
                    window_end_idx = total_doc_tokens
                    # Subtract the overflow amount from the start, respecting document boundary
                    window_start_idx = max(0, desired_start - overflow_end)
                else:
                    # No overflow, window fits perfectly
                    window_start_idx = desired_start
                    window_end_idx = desired_end

                # Final sanity check to ensure window size doesn't exceed total_window_tokens due to rounding/shifts near boundaries
                # This can happen if chunk is very close to edge and shifting causes window to be slightly smaller
                actual_window_size = window_end_idx - window_start_idx
                if actual_window_size < total_window_tokens and total_doc_tokens > total_window_tokens:
                    # Try to expand to fill the gap, prioritizing side away from boundary
                    needed = total_window_tokens - actual_window_size
                    if window_start_idx == 0: # At beginning, expand end
                         window_end_idx = min(total_doc_tokens, window_end_idx + needed)
                    elif window_end_idx == total_doc_tokens: # At end, expand start
                         window_start_idx = max(0, window_start_idx - needed)
                    else: # In middle, split expansion (though unlikely to hit this if initial calc was correct)
                         expand_start = needed // 2
                         expand_end = needed - expand_start
                         window_start_idx = max(0, window_start_idx - expand_start)
                         window_end_idx = min(total_doc_tokens, window_end_idx + expand_end)


        # 4. Extract the window text
        if window_start_idx >= window_end_idx:
             # This might happen if total_window_tokens is very small or zero after adjustments
             print(f"Warning: Calculated context window is empty or invalid (start={window_start_idx}, end={window_end_idx}). Using only chunk text for context prompt.")
             text_window = chunk_text # Fallback: use only the chunk itself
        else:
             window_tokens = document_tokens[window_start_idx:window_end_idx]
             text_window = encoding.decode(window_tokens, errors='replace').strip()
             if not text_window:
                 print(f"Warning: Decoded context window text is empty. Using only chunk text for context prompt.")
                 text_window = chunk_text # Fallback

    except Exception as e:
        print(f"Error during context window extraction: {e}")
        traceback.print_exc()
        return "Error generating context (window extraction failed)."

    # 5. Construct the prompt (remains the same as previous windowed version)
    prompt = f"""
        You are an AI assistant helping to create contextual summaries for text chunks.
        You will be given a window of text from a larger document, and the specific chunk within that window that needs context.

        **Provided Text Window:**
        <text_window>
        {text_window}
        </text_window>

        **Chunk Requiring Context (located within the window above):**
        <chunk>
        {chunk_text}
        </chunk>

        **Instructions:**
        1. Read the **Provided Text Window**.
        2. Identify the **Chunk Requiring Context** within the window.
        3. Analyze the text *immediately surrounding* the chunk **within the window**.
        4. Determine the primary subject or theme being discussed in that immediate vicinity. Think about the specific topic this chunk contributes to within the window's scope.
        5. Synthesize this surrounding subject into a highly succinct contextual description (maximum 1-2 sentences).
        6. This description should act as a high-level "topic tag" or "situational context" specifically relevant to the chunk's placement *within the provided window*.

        **Critical Requirements:**
        - The description MUST focus on the **surrounding context within the window**, NOT just the chunk itself, and NOT the broader document (which you haven't seen).
        - Do NOT summarize the chunk. Do NOT summarize the entire window. Focus on the *local* topic around the chunk.
        - The output MUST be **only** the 1-2 sentence contextual description. No extra text, explanations, or labels like "Context:".

        **Generated Contextual Description (Output Only):**
        """

    try:
        # 6. Call the LLM
        if summary_max_tokens <= 0 or summary_max_tokens > 512:
             summary_max_tokens = DEFAULT_CONTEXT_LENGTH

        response = generate_llm_response(prompt, summary_max_tokens, temperature=0.5, model=context_model)

        # 7. Process the response
        if response.startswith("[Error") or response.startswith("[Blocked"):
             print(f"Warning: Failed to generate context via LLM: {response}")
             return "Error generating context."

        response = response.replace("Context:", "").replace("Here is the context:", "").strip()
        return response if response else "Context could not be generated."

    except Exception as e:
        print(f"Error in generate_chunk_context function with model {context_model}: {e}")
        traceback.print_exc()
        return "Error generating context."
        print(f"Error in generate_chunk_context function with model {context_model}: {e}")
        traceback.print_exc()
        return "Error generating context."

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
    supported_models =  ["gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-1.5-flash-8b", "gemini-2.5-pro-exp-03-25"]

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
    
    elif model.lower() in supported_models:
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
        raise ValueError(f"Unsupported model: {model}, supported modes: {supported_models}")


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
                          overlap: int = DEFAULT_OVERLAP,
                          # Optionally add control for the total window size here
                          context_total_window: int = DEFAULT_TOTAL_CONTEXT_WINDOW
                         ) -> List[Dict]:
    """
    Processes a single document for Phase 1: reads, chunks, gets SLIDING WINDOW context, tokenizes for BM25.
    DOES NOT generate embeddings. Returns list of dicts {id, metadata, bm25_tokens}.
    Metadata includes 'has_embedding': False.
    """
    # ... (file reading, hashing, etc. - same as before) ...
    try:
        with open(document_path, "r", encoding="utf-8", errors='replace') as f:
            document = f.read()
    except Exception as e: print(f"Error reading {os.path.basename(document_path)}: {e}"); return []
    if not document.strip(): return []

    raw_chunks_with_indices = chunk_document_tokens(document, max_tokens=max_tokens, overlap=overlap)
    if not raw_chunks_with_indices: return []

    processed_chunk_data = [] # Initialize here
    file_hash = compute_file_hash(document_path) # Ensure hash is computed
    processing_date = datetime.datetime.now().isoformat() # Ensure date is set

    for idx, (raw_chunk, start_tok, end_tok) in enumerate(raw_chunks_with_indices):
        if not raw_chunk.strip(): continue

        # *** Generate context using the sliding window function ***
        chunk_context = generate_chunk_context(
            full_document_text=document,
            chunk_text=raw_chunk,
            start_token_idx=start_tok,
            end_token_idx=end_tok,
            total_window_tokens=context_total_window, # Pass the total window size
            context_model=CHUNK_CONTEXT_MODEL
        )

        if "Error generating context" in chunk_context:
             print(f"Warning: Skipping chunk {idx} in {os.path.basename(document_path)} due to context generation failure: {chunk_context}")
             # Decide how to handle: skip or use placeholder context
             chunk_context = "Context generation failed." # Using placeholder

        contextualized_text = f"{chunk_context}\n{raw_chunk}"
        tokens_count = count_tokens(raw_chunk)

        chunk_id = f"{file_hash}_{idx}"
        metadata = {
            "file_hash": file_hash, "file_name": os.path.basename(document_path), "processing_date": processing_date,
            "chunk_number": idx, "start_token": start_tok, "end_token": end_tok,
            "text": raw_chunk,
            "context": chunk_context, # Store generated context (from sliding window)
            "contextualized_text": contextualized_text,
            "tokens": tokens_count,
            "has_embedding": False
        }
        bm25_tokens = tokenize_text_bm25(raw_chunk)

        processed_chunk_data.append({
            "id": chunk_id,
            "metadata": metadata,
            "bm25_tokens": bm25_tokens
        })

    return processed_chunk_data

# Removed: Worker Function Wrapper (process_single_file_wrapper_phase1)


# ---------------------------
# Retrieval and Query Functions
# ---------------------------
def retrieve_chunks_for_query(query: str, db_path: str, collection_name: str,
                              top_k: int, execution_mode: str) -> List[dict]:
    """Retrieve top-K chunks from ChromaDB (vector search), returning metadata."""
    if top_k <= 0: top_k = 1
    retrieved_chunks = []
    try:
        collection = get_chroma_collection(db_path, collection_name, execution_mode=execution_mode)

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
                        db_path: str, collection_name: str, execution_mode: str, k_rrf: int = 60) -> List[Dict]:
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
                collection = get_chroma_collection(db_path, collection_name, execution_mode=execution_mode)
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

# Global cache for the reranker model
reranker_model_cache = {}

def rerank_chunks(query: str, chunks: List[Dict], model_name: Optional[str], top_n: int) -> List[Dict]:
    """
    Re-ranks a list of chunk dictionaries based on relevance to the query using a cross-encoder.

    Args:
        query: The original query string.
        chunks: A list of candidate chunk dictionaries (must contain 'text' or 'contextualized_text').
        model_name: The name of the CrossEncoder model (e.g., 'cross-encoder/ms-marco-MiniLM-L-6-v2')
                    or None to disable re-ranking.
        top_n: The final number of chunks to return after re-ranking.

    Returns:
        A list of chunk dictionaries sorted by the re-ranker score (highest first), limited to top_n.
        Returns the original chunks (limited to top_n) if re-ranking is disabled or fails.
    """
    if not model_name or not SENTENCE_TRANSFORMERS_AVAILABLE or not CrossEncoder or not chunks:
        if model_name and not SENTENCE_TRANSFORMERS_AVAILABLE:
             print("Info: Re-ranking skipped (sentence-transformers not installed).")
        # Return top_n based on original order (likely RRF score if applicable)
        return sorted(chunks, key=lambda c: c.get('rrf_score', 0.0), reverse=True)[:top_n]

    print(f"--- Re-ranking top {len(chunks)} candidates using {model_name} ---")

    try:
        # Load model from cache or instantiate
        if model_name not in reranker_model_cache:
            print(f"Loading re-ranker model: {model_name}...")
            # Consider adding model_args={'device': 'cuda'} if GPU is available
            reranker_model_cache[model_name] = CrossEncoder(model_name)
            print("Re-ranker model loaded.")
        model = reranker_model_cache[model_name]

        # Prepare query-chunk pairs for the model
        query_chunk_pairs = []
        valid_chunks_for_reranking = [] # Store chunks that have valid text
        for chunk in chunks:
            # Prioritize contextualized text if available, fall back to raw text
            text_to_rank = chunk.get('contextualized_text', chunk.get('text', ''))
            if text_to_rank and isinstance(text_to_rank, str) and text_to_rank.strip():
                query_chunk_pairs.append([query, text_to_rank.strip()])
                valid_chunks_for_reranking.append(chunk)
            else:
                print(f"Warning: Skipping chunk {chunk.get('file_hash', '')}_{chunk.get('chunk_number', '?')} for re-ranking due to missing/empty text.")

        if not valid_chunks_for_reranking:
            print("Warning: No valid chunks found to re-rank.")
            return sorted(chunks, key=lambda c: c.get('rrf_score', 0.0), reverse=True)[:top_n] # Fallback

        # Get scores from the cross-encoder
        scores = model.predict(query_chunk_pairs, show_progress_bar=False) # Set show_progress_bar=True for large batches

        # Add scores to the valid chunks
        for i, chunk in enumerate(valid_chunks_for_reranking):
            chunk['rerank_score'] = float(scores[i]) # Ensure it's a float

        # Sort the valid chunks by the new rerank_score (higher is better)
        reranked_chunks = sorted(valid_chunks_for_reranking, key=lambda c: c['rerank_score'], reverse=True)

        # Add back any chunks that were skipped (e.g., empty text), placing them at the end
        skipped_chunk_ids = {f"{c.get('file_hash')}_{c.get('chunk_number')}" for c in valid_chunks_for_reranking}
        original_chunk_ids = {f"{c.get('file_hash')}_{c.get('chunk_number')}" for c in chunks}
        missing_ids = original_chunk_ids - skipped_chunk_ids
        if missing_ids:
             print(f"Adding {len(missing_ids)} chunks skipped during re-ranking back to the end of the list.")
             for chunk in chunks:
                 chunk_id = f"{chunk.get('file_hash')}_{chunk.get('chunk_number')}"
                 if chunk_id in missing_ids:
                     chunk['rerank_score'] = -float('inf') # Ensure they appear last if sorted later
                     reranked_chunks.append(chunk)

        print(f"Re-ranking complete. Returning top {top_n} based on cross-encoder scores.")
        return reranked_chunks[:top_n]

    except Exception as e:
        print(f"!!! Error during re-ranking with {model_name}: {e}")
        print(traceback.format_exc())
        print("Warning: Falling back to original chunk order (before re-ranking).")
        # Fallback: return top_n based on the order they came in (likely RRF sorted)
        return sorted(chunks, key=lambda c: c.get('rrf_score', 0.0), reverse=True)[:top_n]

# --- Subquery and Answer Generation ---
def generate_subqueries(initial_query: str, model: str = SUBQUERY_MODEL) -> List[str]:
    # Recommendation: Use n_queries = 3-5
    n_queries = 4 # Example value

    prompt = f"""You are an expert research assistant specializing in climate science literature search using academic databases.
                Your task is to decompose the user's original query into {n_queries} specific and diverse sub-queries suitable for retrieving relevant academic papers on climate science, climate change, and climate impacts.

                These sub-queries should help a Retrieval-Augmented Generation (RAG) system find distinct pieces of evidence, data, or analysis from scientific literature that collectively address the original query.

                Consider breaking down the query by:
                *   Specific climate phenomena (e.g., sea level rise, extreme heat events, ocean acidification)
                *   Geographic regions or ecosystems (e.g., Arctic, Sahel region, coral reefs)
                *   Time scales or periods (e.g., paleo-climate, future projections, specific decades)
                *   Methodologies (e.g., climate modeling, observational data analysis, impact assessments)
                *   Specific impacts (e.g., impacts on agriculture, human health, biodiversity)
                *   Underlying mechanisms or drivers (e.g., greenhouse gas emissions, aerosol effects, climate feedbacks)

                Original Query:
                '{initial_query}'

                Generate exactly {n_queries} focused sub-queries based on the Original Query, suitable for searching academic climate literature.
                Return ONLY the sub-queries, each on a new line. Do not include numbering, bullet points, explanations, or introductory text. Avoid overly broad reformulations of the original query.
                """
    try:
        # Uses main process client
        response_text = generate_llm_response(prompt, max_tokens=250, temperature=0.7, model=model)
        if "[Error generating response" in response_text or "[Blocked" in response_text:
             raise RuntimeError(f"Subquery LLM failed or blocked: {response_text}")
        lines = response_text.strip().splitlines()

        # save response to file for debugging
        with open('generated_subqueries.txt', 'w', encoding='utf-8') as f: f.write(response_text)
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
    MODEL_CONTEXT_LIMITS = { "gpt-4": 8000, "gpt-4o": 128000, "gpt-3.5-turbo": 16000, "gemini-1.5-flash": 1000000, "gemini-1.5-pro": 2000000, "gemini-2.0-flash": 1000000, "gemini-2.0-flash-lite": 1000000, "gemini-2.5-pro-exp-03-25": 1000000 }
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
    answer_max_tokens = min(answer_max_tokens, 10000) # Hard cap answer length

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
                        answer_model: str = CHAT_MODEL,
                        reranker_model: Optional[str] = RERANKER_MODEL, # Use config default
                        rerank_candidate_count: int = DEFAULT_RERANK_CANDIDATE_COUNT,
                        execution_mode: str = "query") -> str: # Use config default
    """Iterative RAG with HYBRID retrieval (Vector + BM25 + RRF) and optional Re-ranking."""
    # 1. Generate Queries
    # --- *** FIX: Initialize all_queries HERE, before the if/else *** ---
    all_queries = [initial_query]
    # --------------------------------------------------------------------

    # Ensure subquery model client is available (uses main process)
    if not gemini_client: # Adjust client check if using different model provider
        print(f"Warning: Client for subquery model '{subquery_model}' not available. Using only initial query.")
    else:
        # ... (existing logic to generate and print subqueries) ...
        generated_subqueries = generate_subqueries(initial_query, model=subquery_model)
        if generated_subqueries and generated_subqueries != [initial_query]:
            print("--- Generated Subqueries ---")
            for idx, subq in enumerate(generated_subqueries): print(f"  {idx+1}. {subq}")
            all_queries.extend(generated_subqueries)
        else: print("--- Using only the initial query (no distinct subqueries generated) ---")


    # 2. Retrieve Chunks (Hybrid) for all queries
    vector_results_all, bm25_results_all = [], []
    # Determine how many candidates to fetch initially for reranking
    fetch_k = rerank_candidate_count if reranker_model and SENTENCE_TRANSFORMERS_AVAILABLE else max(top_k, int(top_k * 1.5))
    print(f"\n--- Retrieving Top {fetch_k} Initial Candidates (Hybrid Vector + BM25) ---")

    # Ensure embedding client is available for query embedding (uses main process)
    if not gemini_client: # Adjust client check if necessary
         return f"Error: Client for embedding model '{EMBEDDING_MODEL}' needed for query is not available."

    for q_idx, current_query in enumerate(all_queries):
         print(f"  Querying ({q_idx+1}/{len(all_queries)}): \"{current_query[:100]}...\"")
         vector_results_all.extend(retrieve_chunks_for_query(current_query, db_path, collection_name, fetch_k, execution_mode=execution_mode))
         bm25_results_all.extend(retrieve_chunks_bm25(current_query, db_path, collection_name, fetch_k))

    # 3. Deduplicate and Combine using RRF
    print(f"\n--- Combining {len(vector_results_all)} Vector & {len(bm25_results_all)} BM25 candidate results via RRF ---")
    # ... (keep existing deduplication logic) ...
    # Deduplicate vector results (keep best score/lowest distance for each unique chunk ID)
    deduped_vector_results_dict: Dict[str, Dict] = {}
    for chunk_meta in vector_results_all:
        file_hash = chunk_meta.get('file_hash')
        chunk_number = chunk_meta.get('chunk_number')
        if file_hash is None or chunk_number is None: continue
        chunk_id = f"{file_hash}_{chunk_number}"
        current_dist = chunk_meta.get('distance', float('inf'))
        if chunk_id not in deduped_vector_results_dict or current_dist < deduped_vector_results_dict[chunk_id].get('distance', float('inf')):
             deduped_vector_results_dict[chunk_id] = chunk_meta
    deduped_vector_results = list(deduped_vector_results_dict.values())

    # Deduplicate BM25 results (keep highest score for each unique chunk ID)
    deduped_bm25_results_dict: Dict[str, float] = {}
    for chunk_id, score in bm25_results_all:
         if chunk_id and (chunk_id not in deduped_bm25_results_dict or score > deduped_bm25_results_dict[chunk_id]):
              deduped_bm25_results_dict[chunk_id] = score
    deduped_bm25_results = list(deduped_bm25_results_dict.items())

    print(f"Unique Vector candidates: {len(deduped_vector_results)}, Unique BM25 candidates: {len(deduped_bm25_results)}")

    # Combine using RRF - Get up to fetch_k combined results for potential reranking
    combined_chunks_rrf = combine_results_rrf(
        deduped_vector_results, deduped_bm25_results, db_path, collection_name, execution_mode=execution_mode
    )
    combined_chunks_rrf = combined_chunks_rrf[:fetch_k] # Limit to candidates for reranking

    # ---------------------------------------------
    # 3.5. *** OPTIONAL RE-RANKING STEP ***
    # ---------------------------------------------
    if reranker_model and SENTENCE_TRANSFORMERS_AVAILABLE:
         final_chunks_list = rerank_chunks(
             initial_query, # Re-rank based on the *original* query
             combined_chunks_rrf,
             reranker_model,
             top_k # Return the final desired number of chunks
         )
    else:
         # If no reranker, just take the top_k from RRF results
         final_chunks_list = combined_chunks_rrf[:top_k]
         if reranker_model: # Only print warning if reranker was configured but unavailable
              print("Info: Skipping re-ranking step.")
    # ---------------------------------------------

    if not final_chunks_list:
        return "No relevant chunks found after retrieval and potential re-ranking."

    # 4. Process Final Chunks & Generate Context
    # Sort by final score (rerank or rrf) for context generation consistency
    final_chunks_list.sort(key=lambda c: c.get('rerank_score', c.get('rrf_score', 0.0)), reverse=True)

    print(f"\n--- Processing Top {len(final_chunks_list)} Final Chunks for Context ---")
    combined_context = "\n\n---\n\n".join(
        f"Source Document: {chunk.get('file_name', 'N/A')}\n"
        f"Source Chunk Number: {chunk.get('chunk_number', '?')}\n"
        # Use contextualized text if available, otherwise fall back to raw text
        f"Content:\n{chunk.get('contextualized_text', chunk.get('text', ''))}"
        for chunk in final_chunks_list
    )
    # ... (optional save combined context) ...

    # 5. Generate Final Answer
    print("\n--- Generating Final Answer (Iterative Hybrid Retrieval) ---")
    # Ensure answer model client is available (uses main process)
    if not gemini_client: # Adjust client check if necessary
        return f"Error: Client for answer model '{answer_model}' is not available."

    final_answer = generate_answer(initial_query, combined_context, final_chunks_list, model=answer_model)
    return final_answer

def query_index(query: str, db_path: str, collection_name: str,
                top_k: int = DEFAULT_TOP_K,
                answer_model: str = CHAT_MODEL,
                reranker_model: Optional[str] = RERANKER_MODEL, # Use config default
                rerank_candidate_count: int = DEFAULT_RERANK_CANDIDATE_COUNT,
                execution_mode: str = "query_direct") -> str: # Use config default
    """Direct query using HYBRID retrieval (Vector + BM25 + RRF) and optional Re-ranking."""
    print(f"--- Running Direct Hybrid Query ---")
    print(f"Query: {query}")

    # Determine how many candidates to fetch initially for reranking
    fetch_k = rerank_candidate_count if reranker_model and SENTENCE_TRANSFORMERS_AVAILABLE else max(top_k, int(top_k * 1.5))
    print(f"Retrieving Top {fetch_k} initial candidates...")

    # 1. Retrieve Chunks (Hybrid)
    # Ensure embedding client is available (uses main process)
    if not gemini_client: # Adjust client check if necessary
         return f"Error: Client for embedding model '{EMBEDDING_MODEL}' needed for query is not available."

    vector_results = retrieve_chunks_for_query(query, db_path, collection_name, fetch_k, execution_mode=execution_mode)
    bm25_results = retrieve_chunks_bm25(query, db_path, collection_name, fetch_k)
    print(f"Retrieved {len(vector_results)} vector candidates, {len(bm25_results)} BM25 candidates.")

    # 2. Combine using RRF
    print("Combining via RRF...")
    combined_chunks_rrf = combine_results_rrf(
        vector_results, bm25_results, db_path, collection_name, execution_mode=execution_mode
    )
    combined_chunks_rrf = combined_chunks_rrf[:fetch_k] # Limit to candidates for reranking

    # ---------------------------------------------
    # 2.5. *** OPTIONAL RE-RANKING STEP ***
    # ---------------------------------------------
    if reranker_model and SENTENCE_TRANSFORMERS_AVAILABLE:
         final_chunks_list = rerank_chunks(
             query, # Re-rank based on the query
             combined_chunks_rrf,
             reranker_model,
             top_k # Return the final desired number of chunks
         )
    else:
         # If no reranker, just take the top_k from RRF results
         final_chunks_list = combined_chunks_rrf[:top_k]
         if reranker_model: # Only print warning if reranker was configured but unavailable
             print("Info: Skipping re-ranking step.")
    # ---------------------------------------------

    if not final_chunks_list:
        return "No relevant chunks found after retrieval and potential re-ranking."

    # 3. Process Final Chunks & Generate Context
    # Sort by final score (rerank or rrf) for context generation consistency
    final_chunks_list.sort(key=lambda c: c.get('rerank_score', c.get('rrf_score', 0.0)), reverse=True)

    print(f"\n--- Processing Top {len(final_chunks_list)} Final Chunks (Direct Query) ---")
    combined_context = "\n\n---\n\n".join(
        f"Source Document: {chunk.get('file_name', 'N/A')}\n"
        f"Source Chunk Number: {chunk.get('chunk_number', '?')}\n"
        f"Content:\n{chunk.get('contextualized_text', chunk.get('text', ''))}"
        for chunk in final_chunks_list
    )
    # ... (optional save context) ...

    # 4. Generate Final Answer
    print("\n--- Generating Final Answer (Direct Hybrid Query) ---")
    # Ensure answer model client is available (uses main process)
    if not gemini_client: # Adjust client check if necessary
        return f"Error: Client for answer model '{answer_model}' is not available."

    answer = generate_answer(query, combined_context, final_chunks_list, model=answer_model)
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
        description="RAG Script (Two-Phase Indexing + Optional Reranking).",
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
    parser.add_argument("--embed_batch_size", type=int, default=140,
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

def filter_files_for_processing(potential_files, db_path, collection_name, force_reindex, execution_mode: str):
    """Checks ChromaDB for existing file hashes and filters the list of files."""
    files_to_process = []
    skipped_files_count = 0
    existing_hashes = set()

    if not force_reindex:
        print("Checking ChromaDB for already indexed files (based on hash)...")
        try:
            collection = get_chroma_collection(db_path, collection_name, execution_mode=execution_mode)
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
    """Adds/Updates raw chunk data (Phase 1) to ChromaDB, omitting embeddings."""
    if not all_phase1_chunks:
        print("No new raw chunks to add/update in ChromaDB.")
        return

    print(f"Adding/Updating {len(all_phase1_chunks)} raw chunks in ChromaDB...")
    chroma_ids = [chunk['id'] for chunk in all_phase1_chunks]
    chroma_metadatas = [chunk['metadata'] for chunk in all_phase1_chunks]
    # Ensure documents contain the raw text for storage/BM25 later
    chroma_documents = [chunk['metadata'].get('text', '') for chunk in all_phase1_chunks]

    batch_size = 140
    num_batches = (len(chroma_ids) + batch_size - 1) // batch_size
    for i in tqdm(range(num_batches), desc="Adding/Upserting Raw Chunks to ChromaDB"):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        batch_ids = chroma_ids[start_idx:end_idx]
        batch_metadatas = chroma_metadatas[start_idx:end_idx]
        batch_documents = chroma_documents[start_idx:end_idx]

        if not batch_ids: continue
        try:
            # *** REVERTED CHANGE: OMIT the embeddings parameter entirely ***
            # ChromaDB should now use the dimension established during collection creation
            # and NOT call the embedding function just because we provide documents.
            print(f"DEBUG: Calling upsert for batch {i+1} (omitting embeddings parameter)") # Add log
            collection.upsert(
                ids=batch_ids,
                metadatas=batch_metadatas,
                documents=batch_documents
                # NO 'embeddings=' parameter here!
            )
            print(f"DEBUG: Upsert for batch {i+1} completed.") # Add log
        except Exception as upsert_err:
            print(f"\n!!! Error upserting batch {i+1}/{num_batches} to ChromaDB: {upsert_err}")
            # Add traceback to see the full error if it still happens
            import traceback
            traceback.print_exc()

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
        potential_files, args.db_path, args.collection_name, args.force_reindex,  execution_mode="index"
    )
    if not files_to_process:
        print("No new files need processing.")
        # If forcing, we might still want to rebuild BM25 even if no *new* files
        if args.force_reindex:
             print("Force re-index requested, proceeding to BM25 rebuild with existing data.")
             collection = get_chroma_collection(args.db_path, args.collection_name, execution_mode="index")
             rebuild_bm25_index(collection, args.db_path, args.collection_name)
        return

    # 3. Process files sequentially
    all_phase1_chunks, _ = process_files_sequentially(files_to_process)

    # 4 & 5. Update DB and Rebuild BM25
    if not all_phase1_chunks and not args.force_reindex:
        print("No valid new chunks were generated. Nothing to add to DB or index.")
        return

    try:
        collection = get_chroma_collection(args.db_path, args.collection_name, execution_mode="index")
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
            return [], []
        else:
            print(f"Found {len(ids_to_embed)} chunks to embed.")
            return ids_to_embed, metadatas_to_embed
    except Exception as e:
        print(f"Error querying ChromaDB for chunks to embed: {e}")
        return [], []

def generate_embeddings_in_batches(
    collection: chromadb.Collection, # Pass the collection object directly
    ids_to_embed: List[str],
    metadatas_to_embed: List[Dict],
    batch_size: int,
    delay: float
) -> List[str]: # Return only the list of failed IDs
    """
    Generates embeddings and updates ChromaDB incrementally after each successful batch.
    Stops processing if a rate limit error (429) is encountered.

    Args:
        collection: The ChromaDB collection object to update.
        ids_to_embed: List of chunk IDs needing embedding.
        metadatas_to_embed: Corresponding metadata dictionaries.
        batch_size: Number of chunks per API batch call.
        delay: Delay in seconds between batch API calls.

    Returns:
        A list of unique chunk IDs that failed during the process (due to errors or rate limits).
    """
    all_failed_ids = [] # Track all failures across batches
    total_processed_count = 0

    num_batches = (len(ids_to_embed) + batch_size - 1) // batch_size
    print(f"Processing {len(ids_to_embed)} chunks in {num_batches} batches of size {batch_size}...")

    # --- Ensure Client is Available ---
    client_available = False
    model_name_lower = EMBEDDING_MODEL.lower()
    if any(m in model_name_lower for m in ["embedding-001", "text-embedding-004"]):
        if gemini_client: client_available = True
    # Add checks for other providers if needed...

    if not client_available:
        print(f"ERROR: Appropriate client for model '{EMBEDDING_MODEL}' not found. Cannot generate embeddings.")
        return ids_to_embed # All failed if client is missing

    # --- Process Batches ---
    for i in tqdm(range(num_batches), desc="Generating Embeddings & Updating DB", unit="batch"):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        current_batch_ids = ids_to_embed[start_idx:end_idx]
        current_batch_metadatas = metadatas_to_embed[start_idx:end_idx]

        batch_texts = []
        batch_valid_indices = []
        batch_ids_map = {}
        batch_metadatas_map = {}
        batch_initial_failed_ids = [] # IDs failing *before* API call in this batch

        # --- Prepare Batch Data (Text Extraction & Preprocessing) ---
        for idx_in_slice, (chunk_id, meta) in enumerate(zip(current_batch_ids, current_batch_metadatas)):
            if not meta:
                # print(f"Warning: Missing metadata for chunk {chunk_id} in batch {i+1}. Skipping.")
                batch_initial_failed_ids.append(chunk_id)
                continue

            text_to_embed = meta.get('contextualized_text', meta.get('text'))
            if not text_to_embed or not text_to_embed.strip():
                # print(f"Warning: Empty text for chunk {chunk_id} in batch {i+1}. Skipping.")
                batch_initial_failed_ids.append(chunk_id)
                continue

            text_to_embed = text_to_embed.replace("\n", " ").strip()
            if not text_to_embed:
                # print(f"Warning: Text became empty after preprocessing for chunk {chunk_id} in batch {i+1}. Skipping.")
                batch_initial_failed_ids.append(chunk_id)
                continue

            batch_texts.append(text_to_embed)
            batch_valid_indices.append(idx_in_slice)
            batch_ids_map[idx_in_slice] = chunk_id
            batch_metadatas_map[idx_in_slice] = meta

        # Add pre-API call failures to the main list
        all_failed_ids.extend(batch_initial_failed_ids)

        if not batch_texts:
            # print(f"Skipping batch {i+1} as it contains no valid texts to embed.")
            # Apply delay even if batch is skipped to maintain overall rate
            if delay > 0 and i < num_batches - 1: time.sleep(delay)
            continue # Skip to the next batch

        # --- Make Batch Embedding API Call ---
        batch_embeddings_ok = []
        batch_ids_ok = []
        batch_metadatas_ok = []
        batch_api_failed_ids = [] # IDs failing during/after API call in this batch
        rate_limit_hit = False

        try:
            # Prepare API call parameters
            api_model_name = EMBEDDING_MODEL if EMBEDDING_MODEL.startswith("models/") else f"models/{EMBEDDING_MODEL}"
            task_type_upper = "RETRIEVAL_DOCUMENT"
            embed_config_args = {'task_type': task_type_upper}
            if OUTPUT_EMBEDDING_DIMENSION is not None:
                embed_config_args['output_dimensionality'] = OUTPUT_EMBEDDING_DIMENSION
            config = EmbedContentConfig(**embed_config_args)

            # print(f"DEBUG: Calling embed_content for batch {i+1} with {len(batch_texts)} texts...")
            response = gemini_client.models.embed_content(
                model=api_model_name,
                contents=batch_texts,
                config=config
            )

            # --- Process Batch Response ---
            if not response or not hasattr(response, 'embeddings'):
                 print(f"\nError: Invalid/empty API response for batch {i+1}. Failing batch.")
                 # Mark all successfully *prepared* items in this batch as failed
                 for idx_in_slice in batch_valid_indices: batch_api_failed_ids.append(batch_ids_map[idx_in_slice])

            elif len(response.embeddings) != len(batch_texts):
                print(f"\nError: Mismatch text/embedding count for batch {i+1}. Failing batch.")
                for idx_in_slice in batch_valid_indices: batch_api_failed_ids.append(batch_ids_map[idx_in_slice])

            else:
                # Process successful response
                for response_idx, embedding_result in enumerate(response.embeddings):
                    original_slice_idx = batch_valid_indices[response_idx]
                    chunk_id = batch_ids_map[original_slice_idx]
                    original_meta = batch_metadatas_map[original_slice_idx]

                    if hasattr(embedding_result, 'values') and isinstance(embedding_result.values, list) and len(embedding_result.values) > 0:
                        batch_embeddings_ok.append(embedding_result.values)
                        updated_meta = original_meta.copy()
                        updated_meta['has_embedding'] = True # Mark as successfully embedded
                        batch_metadatas_ok.append(updated_meta)
                        batch_ids_ok.append(chunk_id)
                    else:
                        # print(f"\nWarning: Failed to get values for chunk {chunk_id} in batch {i+1}.")
                        batch_api_failed_ids.append(chunk_id)

        except google_exceptions.ResourceExhausted as rate_limit_error:
            # --- Rate Limit Specific Handling ---
            rate_limit_hit = True
            print(f"\n!!! Rate Limit Error (429) encountered during batch {i+1}: {rate_limit_error}")
            print("Stopping further embedding in this run.")
            # Mark all items prepared for *this specific batch* as failed because the API call failed
            for idx_in_slice in batch_valid_indices: batch_api_failed_ids.append(batch_ids_map[idx_in_slice])

        except Exception as embed_err:
             # --- Other Critical Error Handling ---
             print(f"\n!!! Critical Error during API call for batch {i+1}: {embed_err}")
             traceback.print_exc()
             # Mark all items prepared for this batch as failed
             for idx_in_slice in batch_valid_indices: batch_api_failed_ids.append(batch_ids_map[idx_in_slice])

        # --- Post-API Call ---

        # Add API-related failures for this batch to the main list
        all_failed_ids.extend(batch_api_failed_ids)

        # --- Update DB with successful results from THIS batch ---
        if batch_ids_ok:
            _update_db_batch(collection, batch_ids_ok, batch_embeddings_ok, batch_metadatas_ok)
            total_processed_count += len(batch_ids_ok)

        # --- Check if we need to stop due to rate limit ---
        if rate_limit_hit:
            break # Exit the loop immediately

        # --- Apply Delay before next batch ---
        if delay > 0 and i < num_batches - 1:
            # print(f"Waiting {delay}s before next batch...")
            time.sleep(delay)

    # --- End of Loop ---
    print(f"\nEmbedding generation loop finished.")
    print(f"Successfully processed and updated {total_processed_count} chunks in the database during this run.")
    unique_failed_ids = list(set(all_failed_ids))
    if unique_failed_ids:
        print(f"Encountered {len(unique_failed_ids)} unique failures (check logs). These chunks were not updated.")

    return unique_failed_ids # Return list of all unique failed IDs

def update_embedded_chunks_in_chromadb(collection, ids, embeddings, metadatas):
    """Updates chunks in ChromaDB with their generated embeddings."""
    if not ids:
        print("No successful embeddings were generated to update in the database.")
        return

    print(f"\nUpdating {len(ids)} chunks in ChromaDB with new embeddings...")
    update_batch_size = 140
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
    """
    Handles the logic for the 'embed' mode. It finds chunks lacking
    embeddings, generates them in batches (updating the DB incrementally),
    handles rate limits gracefully, and reports failures.

    Args:
        args: Parsed command-line arguments, expected to have attributes like
              db_path, collection_name, embed_batch_size, embed_delay.
    """
    print("--- Running Embed Mode (Phase 2) ---")

    # --- Client Availability Check ---
    provider = "Unknown"
    client_ok = False # Initialize client_ok to False
    try:
        # Access the global EMBEDDING_MODEL (ensure it's defined)
        global EMBEDDING_MODEL
        model_name_lower = EMBEDDING_MODEL.lower() # Use lowercase for checks

        # --- Check for Gemini ---
        if any(m in model_name_lower for m in ["embedding-001", "text-embedding-004"]):
            provider = "Gemini"
            try:
                # Access the global gemini_client (ensure it's defined)
                global gemini_client
                if gemini_client: # Check if the client object exists and is not None
                    client_ok = True
                    print(f"DEBUG: {provider} client seems available.")
                else:
                    print(f"DEBUG: {provider} client (gemini_client) is None.")
            except NameError:
                print(f"DEBUG: Global variable 'gemini_client' not found.")

        # --- Add checks for other potential providers here ---
        # Example for OpenAI (adjust variable names as needed)
        # elif 'gpt' in model_name_lower or 'text-embedding-ada-002' in model_name_lower:
        #     provider = "OpenAI"
        #     try:
        #         global openai_client
        #         if openai_client:
        #             client_ok = True
        #             print(f"DEBUG: {provider} client seems available.")
        #         else:
        #             print(f"DEBUG: {provider} client (openai_client) is None.")
        #     except NameError:
        #         print(f"DEBUG: Global variable 'openai_client' not found.")

        # --- Final check ---
        if not client_ok:
            print(f"\nError: {provider} client needed for embedding model '{EMBEDDING_MODEL}' is not available or not initialized.")
            print("Please ensure the required API key is set and the client was initialized successfully.")
            return # Exit embed mode if the required client is not ready

    except NameError as ne:
        print(f"\nError: Required global variable not found during client check: {ne}")
        print("This might indicate an issue with config loading or client initialization.")
        return # Exit if essential config like EMBEDDING_MODEL is missing
    except Exception as client_check_err:
        print(f"\nAn unexpected error occurred during client check: {client_check_err}")
        traceback.print_exc()
        return # Exit on unexpected errors during check
    # --- End of Client Availability Check ---


    # --- Main Embed Mode Logic ---
    try:
        # Get the collection object, passing 'embed' mode
        # This ensures the embedding function behaves correctly if called
        collection = get_chroma_collection(args.db_path, args.collection_name, execution_mode="embed")

        # 1. Find chunks needing embedding (those with has_embedding: False)
        print("Finding chunks that need embeddings...")
        ids_to_embed, metadatas_to_embed = find_chunks_to_embed(collection)

        if not ids_to_embed:
            print("No chunks found needing embedding in this collection.")
            print("\n--- Embed Mode (Phase 2) Complete ---")
            return # Nothing more to do

        print(f"Found {len(ids_to_embed)} chunks to process.")

        # 2. Generate embeddings and update DB incrementally
        # This function now handles batching, API calls, rate limits, and DB updates
        failed_embed_ids = generate_embeddings_in_batches(
            collection=collection, # Pass the collection object
            ids_to_embed=ids_to_embed,
            metadatas_to_embed=metadatas_to_embed,
            batch_size=args.embed_batch_size,
            delay=args.embed_delay
        )

        # 3. Report final failures (DB update happened inside the function)
        if failed_embed_ids:
            print(f"\nSummary: Failed to process/embed {len(failed_embed_ids)} unique chunks during this run.")
            print("These chunks were not updated and likely remain marked as 'has_embedding': False.")
            print("You can re-run '--mode embed' to retry processing them.")

            # Optional: Log failed_ids to a file for detailed review
            log_filename = "embedding_failures.log"
            try:
                with open(log_filename, "a", encoding="utf-8") as f:
                    timestamp = datetime.datetime.now().isoformat()
                    f.write(f"--- Run at {timestamp} ---\n")
                    f.write(f"Failed IDs ({len(failed_embed_ids)}):\n")
                    for fid in failed_embed_ids:
                         f.write(f"{fid}\n")
                    f.write("-" * 20 + "\n")
                print(f"List of failed chunk IDs appended to '{log_filename}'.")
            except Exception as log_err:
                print(f"Warning: Could not write failed IDs to log file '{log_filename}': {log_err}")
        else:
            print("\nSummary: All found chunks requiring embeddings were processed successfully.")

    except Exception as e:
        # Catch any broad errors during the embed process (e.g., DB connection issues)
        print(f"\n!!! An error occurred during the main embed mode execution: {e}")
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
                                               answer_model=CHAT_MODEL,
                                               execution_mode=args.mode)
    else: # query_direct
            final_answer = query_index(args.query, args.db_path, args.collection_name,
                                       top_k=args.top_k,
                                       answer_model=CHAT_MODEL,
                                       execution_mode=args.mode)

    # Print the final result
    print("\n" + "="*20 + " Final Answer " + "="*20)
    print(final_answer)
    # save file
    with open("final_answer.txt", "w", encoding="utf-8") as f:
        f.write(final_answer)
    print("="*54 + "\n")


# --- Main Orchestrator ---

def main():
    """Main function to orchestrate the RAG script."""
    test_mode_enabled = False # Set to True/False as needed

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