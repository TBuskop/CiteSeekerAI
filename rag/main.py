#!/usr/bin/env python3
"""
main.py - Entry point for the Modular RAG System

Handles orchestration of calls to different modules based on a configuration dictionary.
"""

import os
import traceback
import sys
from typing import Dict, Any, Optional

# --- Ensure the 'rag' directory and its parent are in the Python path ---
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
     sys.path.insert(0, parent_dir)


# --- Local Imports (after path adjustment) ---
try:
    # Use absolute imports assuming script is run from parent directory
    import config # General configuration (assuming it's in the parent dir added to sys.path)
    from rag import llm_interface # API client initialization and calls
    from rag import chroma_manager # ChromaDB setup and interaction
    from rag import bm25_manager # BM25 setup and interaction
    from rag import indexing # Index mode logic
    from rag import embedding # Embed mode logic
    from rag import querying # Query modes logic
    # --- Add ChromaDB Client import for direct check ---
    import chromadb
except ImportError as e:
     print(f"Error importing local modules: {e}")
     print("Please ensure main.py is run correctly and the project structure is as expected.")
     print(f"Current sys.path: {sys.path}")
     sys.exit(1)


# --- Basic Validation Helper ---
def validate_config(config_params: Dict[str, Any]):
    """Performs basic validation on the configuration dictionary."""
    mode = config_params.get('mode')
    if not mode:
        raise ValueError("Configuration must include a 'mode' (index, embed, query, query_direct).")

    if mode == "index":
        if not config_params.get('document_path') and not config_params.get('folder_path'):
            raise ValueError("'document_path' or 'folder_path' is required for 'index' mode.")
    elif mode in ["query", "query_direct"]:
        query = config_params.get('query')
        if not query or not query.strip():
            raise ValueError("'query' cannot be empty for query modes.")
        if config_params.get('top_k', config.DEFAULT_TOP_K) <= 0:
            raise ValueError("'top_k' must be positive.")
        reranker = config_params.get('reranker', config.RERANKER_MODEL)
        rerank_candidates = config_params.get('rerank_candidates', config.DEFAULT_RERANK_CANDIDATE_COUNT)
        if reranker and rerank_candidates <= 0:
            raise ValueError("'rerank_candidates' must be positive when reranker is enabled.")
    elif mode == 'embed':
        if config_params.get('embed_batch_size', 100) <= 0:
            raise ValueError("'embed_batch_size' must be positive.")
        if config_params.get('embed_delay', 0.1) < 0:
            raise ValueError("'embed_delay' cannot be negative.")

# --- Mode Execution Logic (Modified to accept config_params dict) ---

def run_index_mode(config_params: Dict[str, Any]):
    """Handles the logic for the 'index' mode using a config dictionary."""
    print("--- Running Index Mode (Phase 1) ---")
    folder_path = config_params.get('folder_path')
    document_path = config_params.get('document_path')
    db_path = config_params.get('db_path', './rag_db')
    collection_name = config_params.get('collection_name', config.DEFAULT_CHROMA_COLLECTION_NAME)
    force_reindex = config_params.get('force_reindex', False)

    potential_files = indexing.find_files_to_index(folder_path, document_path)
    if not potential_files: return

    files_to_process, skipped_count = indexing.filter_files_for_processing(
        potential_files, db_path, collection_name, force_reindex
    )

    if not files_to_process and not force_reindex:
        print("No new files need processing.")
        return
    elif not files_to_process and force_reindex:
        print("No new files found, but force_reindex is enabled. Will proceed to rebuild BM25 index from existing ChromaDB data.")
        try:
            collection = chroma_manager.get_chroma_collection(db_path, collection_name, execution_mode="index")
            indexing.rebuild_bm25_index_from_chroma(collection, db_path, collection_name)
        except Exception as db_err:
             print(f"!!! Error accessing ChromaDB or rebuilding BM25 index: {db_err}")
             traceback.print_exc()
        print("\n--- Index Mode (Phase 1) Complete (BM25 Rebuild Only) ---")
        return

    all_phase1_chunks, _ = indexing.process_files_sequentially(files_to_process)

    if not all_phase1_chunks and not force_reindex:
        print("No valid new chunks were generated. Nothing to add to DB or index.")
        return
    elif not all_phase1_chunks and force_reindex:
        print("No valid new chunks generated, but force_reindex is enabled. Rebuilding BM25 from existing data.")

    try:
        collection = chroma_manager.get_chroma_collection(db_path, collection_name, execution_mode="index")
        if all_phase1_chunks:
             indexing.update_chromadb_raw_chunks(collection, all_phase1_chunks)
        else:
             print("Skipping ChromaDB upsert as no new chunks were generated.")
        indexing.rebuild_bm25_index_from_chroma(collection, db_path, collection_name)
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
    db_path = config_params.get('db_path', './rag_db')
    collection_name = config_params.get('collection_name', config.DEFAULT_CHROMA_COLLECTION_NAME)

    provider = "Unknown"
    client_ok = False
    model_name_lower = config.EMBEDDING_MODEL.lower()

    if any(m in model_name_lower for m in ["embedding-001", "text-embedding-004"]):
        provider = "Gemini"
        # Use the passed client for the check
        if llm_interface.GOOGLE_GENAI_AVAILABLE and client:
            client_ok = True

    if not client_ok:
        print(f"\nError: {provider} client needed for embedding model '{config.EMBEDDING_MODEL}' is not available or not initialized.")
        print("Please ensure the required API key is set and the client was initialized successfully.")
        return

    try:
        collection = chroma_manager.get_chroma_collection(db_path, collection_name, execution_mode="embed")
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
    query = config_params['query']
    top_k = config_params.get('top_k', config.DEFAULT_TOP_K)
    reranker_model = config_params.get('reranker', config.RERANKER_MODEL)
    if isinstance(reranker_model, str) and reranker_model.strip().lower() in ['none', '']:
        reranker_model = None
    rerank_candidates = config_params.get('rerank_candidates', config.DEFAULT_RERANK_CANDIDATE_COUNT)

    # --- Add Collection Verification Step ---
    print(f"\n--- Verifying Collection State ---")
    print(f"Checking collection '{collection_name}' at path '{db_path}'...")
    try:
        # Create a temporary client just for checking
        check_client = chromadb.PersistentClient(path=db_path)
        try:
            check_collection = check_client.get_collection(name=collection_name)
            collection_count = check_collection.count()
            print(f"Collection '{collection_name}' found. Total items: {collection_count}")

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
                # Optional: Query specifically for items with embeddings
                # embedded_count = check_collection.count(where={"has_embedding": True})
                # print(f"Count of items explicitly marked with has_embedding=True: {embedded_count}")

            else:
                print(f"WARNING: Collection '{collection_name}' is empty. No results will be found.")

        except Exception as get_coll_err:
             # Handle case where collection doesn't exist (might be ValueError or similar)
             print(f"Error accessing collection '{collection_name}': {get_coll_err}")
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

    bm25_instance, _ = bm25_manager.load_bm25_index(db_path, collection_name)
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
                    collection_name=collection_name,
                    top_k=top_k,
                    subquery_model=config.SUBQUERY_MODEL,
                    answer_model=config.CHAT_MODEL,
                    reranker_model=reranker_model,
                    rerank_candidate_count=rerank_candidates,
                    execution_mode=mode
                )
        elif mode == "query_direct":
                final_answer = querying.query_index(
                    query=query,
                    db_path=db_path,
                    collection_name=collection_name,
                    top_k=top_k,
                    answer_model=config.CHAT_MODEL,
                    reranker_model=reranker_model,
                    rerank_candidate_count=rerank_candidates,
                    execution_mode=mode
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

    except Exception as query_err:
         print(f"\n!!! An error occurred during query execution: {query_err}")
         traceback.print_exc()


# --- Main Orchestrator (Modified for direct calls) ---
def main(config_params: Optional[Dict[str, Any]] = None):
    """
    Main function to orchestrate the RAG script using a configuration dictionary.

    Args:
        config_params: A dictionary containing the execution parameters.
                       If None, uses a default example configuration.
    """
    if config_params is None:
        print("INFO: No configuration provided, using default example (query mode).")
        config_params = {
            "mode": "query",
            "folder_path": "cleaned_text/test_docs",
            "db_path": "chunk_database/chunks_db",
            "collection_name": "test_collection",
            "query": "what color are apples?",
            "top_k": 5,
            "reranker": config.RERANKER_MODEL,
            "rerank_candidates": config.DEFAULT_RERANK_CANDIDATE_COUNT,
        }

    try:
        print(f"Initializing API clients...")
        llm_interface.initialize_clients()
        # Store the initialized client locally
        initialized_client = llm_interface.gemini_client
        if not initialized_client:
             print("Warning: Gemini client initialization might have failed. Check API keys/config.")

        validate_config(config_params)
        print(f"Configuration validated for mode: {config_params['mode']}")

        mode = config_params['mode']
        if mode == "index":
            run_index_mode(config_params)
        elif mode == "embed":
            # Pass the initialized client to run_embed_mode
            run_embed_mode(config_params, initialized_client)
        elif mode in ["query", "query_direct"]:
            run_query_mode(config_params)
        else:
            print(f"Error: Unknown mode '{mode}' specified in configuration.")
            sys.exit(1)

    except ValueError as ve:
        print(f"\n!!! Configuration Error: {ve}")
        sys.exit(1)
    except RuntimeError as rte:
        print(f"\n!!! Runtime Error: {rte}")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n!!! An unexpected error occurred in main execution: {e}")
        print("Traceback:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
