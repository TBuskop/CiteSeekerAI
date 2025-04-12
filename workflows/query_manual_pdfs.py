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
# Use absolute imports assuming script is run from parent directory
import config # General configuration (assuming it's in the parent dir added to sys.path)
from src.my_utils import llm_interface # API client initialization and calls
# --- Import the new workflows module ---
from src.rag import workflows
# --- Add ChromaDB Client import for direct check ---
import chromadb



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
    if config_params is None:

        query = """
        I am writing a discussion for an academic paper and need some papers to backup my discussion section. Especially integrating the reviewer comment: Somewhere in this section you should also address the review comments “what did we learn?” You should discuss the robustness of the patterns across the various scenarios in Figure 8, and the usefulness for this combined climate/socio-economic analysis for priority setting in policy decisions. It’s not the probability of things happening that is targeted here, but the probability of being helpful to design proper policy responses
        Here is the section I want to improve
        4.1 The consequences of uncertainties, the importance of exploration and reduction
        Results highlight the importance of using multiple climate storylines to robustly assess future discharge extremes. Flood patterns vary significantly across storylines; while some indicate 10% regional reductions in the 100 year return period discharge, others indicate increases up to 20%. The storylines also reveal spatial heterogeneity: in some storylines and regions flood preconditions such as snow melt are dampened, whereas other preconditions (such as higher soil moisture volume) are amplified.
        Projected socio-economic changes can further amplify these climate risks. As exposed asset values in the Latvian basin almost quadruple compared to the 2020 baseline, sectors are confronted with heightened flood-induced demand shocks related to reconstruction. Simultaneously, a trend of declining domestic manufacturing capacity doubles or triples most sectoral stresses relative to the 2020 baseline. In some scenarios, the demand shock exceeds absorptive capacity of sectors, amplifying and prolonging indirect impacts (Hallegatte et al., 2024). Furthermore, increased dependency on imports due to the offshoring of the manufacturing sectors may create new vulnerabilities to foreign supply disruptions (Ercin et al., 2021). Conversely, this may lead to enhanced regional resilience through import diversification (Willner et al., 2018).
        """
        absolute_pdf_folder_path = os.path.abspath("data/manual_pdf_upload/cleaned_text/robustness_uncertainty")
        absoluted_db_path = os.path.abspath("data/manual_pdf_upload/chunk_database/chunks_db")
        config_params = {
            "mode": "index_embed_query",
            "folder_path": absolute_pdf_folder_path,
            "db_path": absoluted_db_path,
            "collection_name": "robustness_uncertainty",
            "add_chunk_context": False, # Add this flag, set to False to disable context
            "query": query,
            "top_k": 100,
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
            config_params_index = config_params.copy() # Avoid modifying original dict if reused
            config_params_index['mode'] = "index"
            workflows.run_index_mode(config_params_index)
            config_params_embed = config_params.copy()
            config_params_embed['mode'] = "embed"
            workflows.run_embed_mode(config_params_embed, initialized_client)
        elif mode in ["query", "query_direct"]:
            # Call function from workflows module
            workflows.run_query_mode(config_params)
        elif mode == "index_embed_query":
            # Call function from workflows module
            config_params_index = config_params.copy()
            config_params_index['mode'] = "index"
            workflows.run_index_mode(config_params_index)
            config_params_embed = config_params.copy()
            config_params_embed['mode'] = "embed"
            workflows.run_embed_mode(config_params_embed, initialized_client)
            config_params_query = config_params.copy()
            config_params_query['mode'] = "query" # Ensure correct mode for query
            workflows.run_query_mode(config_params_query)

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
