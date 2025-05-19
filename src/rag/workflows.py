"""
workflows.py - Contains the core logic for different execution modes (index, embed, query).
"""

import os
import traceback
from typing import Dict, Any, Optional

import sys
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# --- Local Imports ---
# Assume this file is run within the context where 'rag' package is accessible
import config # General configuration
from config import (
    HYPE, HYPE_SUFFIX, HYPE_SOURCE_COLLECTION_NAME # Add HYPE_SUFFIX
)
from src.my_utils import llm_interface # API client initialization and calls
from src.rag import chroma_manager # ChromaDB setup and interaction
from src.rag import bm25_manager # BM25 setup and interaction
from src.rag import indexing # Index mode logic
from src.rag import embedding # Embed mode logic
from src.rag import querying # Query modes logic
import chromadb # Direct ChromaDB client import for checks
from chromadb.config import Settings # Import Settings for type hinting if needed

# --- Add top-level genai import for consistent type checking ---
try:
    import google.genai as genai
    GENAI_TYPES_AVAILABLE = True
except ImportError:
    genai = None # Define as None if import fails
    GENAI_TYPES_AVAILABLE = False

# --- Mode Execution Logic (Moved from main.py) ---

def run_index_mode(config_params: Dict[str, Any]):
    """Handles the logic for the 'index' mode using a config dictionary."""
    print("--- Running Index Mode (Phase 1) ---")
    folder_path = config_params.get('folder_path')
    document_path = config_params.get('document_path')
    db_path = config_params.get('db_path', './rag_db')
    collection_name = config_params.get('collection_name', config.DEFAULT_CHROMA_COLLECTION_NAME)
    force_reindex = config_params.get('force_reindex', False)
    # Determine effective collection name (no HYPE suffix for indexing raw chunks)
    effective_collection_name = collection_name
    add_context = config_params.get('add_chunk_context', False) # Get the new flag

    potential_files = indexing.find_files_to_index(folder_path, document_path)
    if not potential_files: return

    files_to_process, skipped_count = indexing.filter_files_for_processing(
        potential_files, db_path, collection_name, force_reindex
        # Note: filter_files_for_processing likely uses the base name to check existence
    )

    if not files_to_process and not force_reindex:
        print("No new files need processing.")
        return
    elif not files_to_process and force_reindex:
        print("No new files found, but force_reindex is enabled. Will proceed to rebuild BM25 index from existing ChromaDB data.")
        try:
            # Use base name for getting collection to rebuild BM25
            collection = chroma_manager.get_chroma_collection(db_path, effective_collection_name, execution_mode="index")
            indexing.rebuild_bm25_index_from_chroma(collection, db_path, effective_collection_name)
        except Exception as db_err:
             print(f"!!! Error accessing ChromaDB or rebuilding BM25 index: {db_err}")
             traceback.print_exc()
        print("\n--- Index Mode (Phase 1) Complete (BM25 Rebuild Only) ---")
        return

    # Pass add_context to process_files_sequentially
    all_phase1_chunks, _ = indexing.process_files_sequentially(files_to_process, add_context=add_context)

    if not all_phase1_chunks and not force_reindex:
        print("No valid new chunks were generated. Nothing to add to DB or index.")
        return
    elif not all_phase1_chunks and force_reindex:
        print("No valid new chunks generated, but force_reindex is enabled. Rebuilding BM25 from existing data.")

    try:
        # Use base name for getting collection to update raw chunks
        collection = chroma_manager.get_chroma_collection(db_path, effective_collection_name, execution_mode="index")
        if all_phase1_chunks:
             indexing.update_chromadb_raw_chunks(collection, all_phase1_chunks)
        else:
             print("Skipping ChromaDB upsert as no new chunks were generated.")
        # Rebuild BM25 for the base collection
        indexing.rebuild_bm25_index_from_chroma(collection, db_path, effective_collection_name)
    except Exception as db_bm25_err:
        print(f"!!! Error during ChromaDB update or BM25 build phase: {db_bm25_err}")
        traceback.print_exc()

    print("\n--- Index Mode (Phase 1) Complete ---")
    if all_phase1_chunks: print("Raw chunks stored/updated.")
    print("BM25 index rebuilt.")
    print("Run mode 'embed' to generate embeddings for any missing ones.")


def run_embed_mode(config_params: Dict[str, Any], client: Optional[Any]):
    """Handles the logic for the 'embed' mode using a config dictionary."""
    print("--- Running Embed Mode (Phase 2) ---")
    # Determine which collection to embed (base or hype)
    collection_name = config_params.get('collection_name', config.DEFAULT_CHROMA_COLLECTION_NAME)
    use_hype = config_params.get('use_hype', config.HYPE)
    effective_collection_name = f"{collection_name}{HYPE_SUFFIX}" if use_hype else collection_name
    if use_hype:
        print(f"Info: Running Embed Mode on Hype collection '{effective_collection_name}'")
    else:
        print(f"Info: Running Embed Mode on base collection '{effective_collection_name}'")
    # Preliminary checks for Gemini SDK and client
    try:
        import google.genai
    except ImportError:
        print("Error: Google Generative AI SDK not installed. Please install 'google-generativeai' to use embeddings.")
        return
    if client is None:
        print(f"Error: Gemini client not initialized. Ensure GEMINI_API_KEY is set and valid in config.")
        return
    db_path = config_params.get('db_path', './rag_db')

    provider = "Unknown"
    client_ok = False
    model_name_lower = config.EMBEDDING_MODEL.lower()

    if any(m in model_name_lower for m in ["embedding-001", "text-embedding-004"]):
        provider = "Gemini"
        # Use the top-level genai for consistent isinstance check
        if GENAI_TYPES_AVAILABLE and llm_interface.GOOGLE_GENAI_AVAILABLE and isinstance(client, genai.Client):
            client_ok = True
        elif not GENAI_TYPES_AVAILABLE:
             print("Warning: google.genai library not found, cannot verify client type for Gemini model.")

    if not client_ok:
        print(f"\nError: {provider} client needed for embedding model '{config.EMBEDDING_MODEL}' is not available or not initialized correctly.")
        return

    try:
        # Get the correct collection (base or hype) for embedding
        collection = chroma_manager.get_chroma_collection(db_path, effective_collection_name, execution_mode="embed")
        # Pass the client to the logic function
        embedding.run_embed_mode_logic(config_params, collection, client)
    except Exception as e:
        print(f"\n!!! An error occurred during the main embed mode execution: {e}")
        traceback.print_exc()

    print("\n--- Embed Mode (Phase 2) Complete ---")


def run_query_mode(config_params: Dict[str, Any]):
    """Handles the logic for 'query' and 'query_direct' modes using a config dictionary."""
    mode = config_params['mode']
    print(f"--- Running Query Mode ({mode}) ---")
    db_path = config_params.get('db_path', './rag_db')
    collection_name = config_params.get('collection_name', config.DEFAULT_CHROMA_COLLECTION_NAME)
    use_hype = config_params.get('use_hype', config.HYPE)
    db_settings = config_params.get("db_settings") # Extract db_settings
    # Determine effective collection name for querying
    effective_collection_name = f"{collection_name}{HYPE_SUFFIX}" if use_hype else collection_name
    if use_hype:
        print(f"Info: Using Hype embeddings collection: {effective_collection_name}")
    else:
        print(f"Info: Using base collection: {effective_collection_name}")
    query = config_params['query']
    top_k = config_params.get('top_k', config.DEFAULT_TOP_K)
    reranker_model = config_params.get('reranker', config.RERANKER_MODEL)
    if isinstance(reranker_model, str) and reranker_model.strip().lower() in ['none', '']:
        reranker_model = None
    rerank_candidates = config_params.get('rerank_candidates', config.DEFAULT_RERANK_CANDIDATE_COUNT)
    output_dir = config_params.get('output_dir') # Get the output directory
    query_index = config_params.get('query_index') # Get the query index

    # --- Add Collection Verification Step ---
    print(f"\n--- Verifying Collection State ---")
    print(f"Checking collection '{effective_collection_name}' at path '{db_path}'...")
    try:
        # Create a temporary client just for checking, using the passed settings
        check_client = chromadb.PersistentClient(path=db_path, settings=db_settings if db_settings else Settings())
        try:
            check_collection = check_client.get_collection(name=effective_collection_name)
            collection_count = check_collection.count()
            print(f"Collection '{effective_collection_name}' found. Total items: {collection_count}")

            if collection_count > 0:
                # Peek at a few items to check for embeddings
                peek_results = check_collection.peek(limit=5)
                items_with_embeddings = 0
                if peek_results and peek_results.get('ids'):
                    print(f"Peeking at up to {len(peek_results['ids'])} items:")
                    for i, item_id in enumerate(peek_results['ids']):
                        meta = peek_results['metadatas'][i] if peek_results.get('metadatas') and i < len(peek_results['metadatas']) else {}
                        has_embedding_flag = meta.get('has_embedding', 'N/A')
                        print(f"  - Item ID: {item_id}, has_embedding: {has_embedding_flag}")
                        if has_embedding_flag is True:
                            items_with_embeddings += 1
                    if items_with_embeddings > 0:
                        print(f"Found at least {items_with_embeddings} item(s) with has_embedding=True in the sample.")
                    else:
                        print(f"WARNING: No items with has_embedding=True found in the sample. Embeddings might be missing.")
                else:
                    print("Could not peek at items.")

            else:
                print(f"WARNING: Collection '{effective_collection_name}' is empty. No results will be found.")

        except Exception as get_coll_err:
             # Handle case where collection doesn't exist (might be ValueError or similar)
             print(f"Error accessing collection '{effective_collection_name}': {get_coll_err}")
             print("Please ensure the collection exists and was populated correctly (run index and embed modes).")
             return # Stop processing if collection can't be accessed

    except Exception as client_err:
        print(f"Error creating ChromaDB client for verification at path '{db_path}': {client_err}")
        print("Cannot verify collection state.")
        # Decide whether to proceed or return
        # return

    print(f"--- End Collection Verification ---\n")
    # --- End Verification Step ---

    query_embed_ok = bool(llm_interface.gemini_client)
    subquery_ok = bool(llm_interface.gemini_client) if mode == "query" else True
    answer_ok = bool(llm_interface.gemini_client)

    if not query_embed_ok: print(f"Error: Client for query embedding model '{config.EMBEDDING_MODEL}' not available."); return
    if not subquery_ok and mode == "query": print(f"Error: Client for subquery model '{config.SUBQUERY_MODEL}' not available."); return
    if not answer_ok: print(f"Error: Client for answer model '{config.CHAT_MODEL}' not available."); return

    # Load BM25 for the correct collection (base or hype)
    bm25_instance, _ = bm25_manager.load_bm25_index(db_path, effective_collection_name)
    if not bm25_instance and bm25_manager.RANK_BM25_AVAILABLE:
        print("Warning: BM25 index not found or failed to load. Lexical search part of hybrid query will be skipped.")
    elif not bm25_manager.RANK_BM25_AVAILABLE:
         print("Info: BM25 library not installed. Lexical search part of hybrid query will be skipped.")

    if reranker_model and not querying.SENTENCE_TRANSFORMERS_AVAILABLE:
        print(f"Warning: Reranker model '{reranker_model}' specified, but sentence-transformers library not installed. Reranking will be disabled.")
        reranker_model = None

    final_answer = ""
    try:
        if mode == "query":
                final_answer = querying.iterative_rag_query(
                    initial_query=query,
                    db_path=db_path,
                    collection_name=collection_name, # Pass base name, function handles suffix
                    top_k=top_k,
                    subquery_model=config.SUBQUERY_MODEL_SIMPLE,
                    answer_model=config.CHAT_MODEL,
                    reranker_model=reranker_model,
                    rerank_candidate_count=rerank_candidates,
                    execution_mode=mode,
                    output_dir=output_dir,
                    query_index=query_index, # Pass query_index
                    use_hype = use_hype,
                    db_settings=db_settings # Pass db_settings
                )
        elif mode == "query_direct":
                final_answer = querying.query_index(
                    query=query,
                    db_path=db_path,
                    collection_name=collection_name, # Pass base name, function handles suffix
                    top_k=top_k,
                    answer_model=config.CHAT_MODEL,
                    reranker_model=reranker_model,
                    rerank_candidate_count=rerank_candidates,
                    output_dir=output_dir,
                    query_index=query_index, # Pass query_index
                    execution_mode=mode,
                    db_settings=db_settings # Pass db_settings
                )
        else:
             print(f"Error: Unknown query mode '{mode}'")
             return

        print("\n" + "="*20 + " Final Answer " + "="*20)
        print(final_answer)
        print("="*54 + "\n")

        try:
            output_filename = config_params.get("output_filename", "final_answer.txt")
            with open(output_filename, "w", encoding="utf-8") as f:
                f.write(final_answer)
            print(f"Final answer saved to {output_filename}")
        except Exception as e:
            print(f"Warning: Could not save final answer to file: {e}")

        return final_answer

    except Exception as query_err:
         print(f"\n!!! An error occurred during query execution: {query_err}")
         traceback.print_exc()
