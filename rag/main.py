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
    # --- Import the new workflows module ---
    from rag import workflows
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


# --- Main Orchestrator (Modified for direct calls) ---
def main(config_params: Optional[Dict[str, Any]] = None):
    """
    Main function to orchestrate the RAG script using a configuration dictionary.

    Args:
        config_params: A dictionary containing the execution parameters.
                       If None, uses a default example configuration.
    """
    # change run path to the current directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    if config_params is None:

        query = """
        what color are apples?
        """

        print("INFO: No configuration provided, using default example (query mode).")
        config_params = {
            "mode": "index_embed_query",
            "folder_path": "cleaned_text/test_docs",
            "db_path": "chunk_database/chunks_db",
            "collection_name": "test_collection",
            "query": query,
            "top_k": 8,
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
            # Call function from workflows module
            workflows.run_index_mode(config_params)
        elif mode == "embed":
            # Call function from workflows module, pass the initialized client
            workflows.run_embed_mode(config_params, initialized_client)
        elif mode == "index_embed":
            # Call function from workflows module
            # set config_params['mode'] to 'index' for index mode
            config_params['mode'] = "index"
            workflows.run_index_mode(config_params)
            config_params['mode'] = "embed"
            workflows.run_embed_mode(config_params, initialized_client)
        elif mode in ["query", "query_direct"]:
            # Call function from workflows module
            workflows.run_query_mode(config_params)
        elif mode == "index_embed_query":
            # Call function from workflows module
            config_params['mode'] = "index"
            workflows.run_index_mode(config_params)
            config_params['mode'] = "embed"
            workflows.run_embed_mode(config_params, initialized_client)
            config_params['mode'] = "query"
            workflows.run_query_mode(config_params)

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
