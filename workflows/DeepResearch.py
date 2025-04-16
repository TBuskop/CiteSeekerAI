import os
import sys

# --- Add project root to sys.path ---
# This allows absolute imports from 'src' assuming the script is in 'workflows'
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# --- Updated Imports (with src prefix) ---
from src.scrape.search_scopus import run_scopus_search
from src.scrape.add_csv_to_chromadb import ingest_csv_to_chroma
from src.scrape.collect_relevant_abstracts import find_relevant_dois_from_abstracts
from src.scrape.download_papers import download_dois
from src.scrape.get_search_string import generate_scopus_search_string
from src.scrape.chunk_new_dois import process_folder_for_chunks
from src.scrape.build_relevant_papers_db import build_relevant_db
from src.my_utils import llm_interface
from src.rag import workflows as rag_workflows
from src.rag.querying import query_decomposition

import config


# --- Central Configuration ---
# General
# Assuming the script is run from the project root (e.g., python workflows/paper_collection_pipeline.py)
BASE_DATA_DIR = "../../data"
DOWNLOADS_DIR = os.path.join(BASE_DATA_DIR, "downloads")
FULL_TEXT_DIR = os.path.join(DOWNLOADS_DIR, "full_doi_texts")
CSV_DIR = os.path.join(DOWNLOADS_DIR, "csv")
# Abstract DB Config
ABSTRACT_DB_PATH = os.path.join(BASE_DATA_DIR, "databases", "abstract_chroma_db")
ABSTRACT_COLLECTION_NAME = "abstracts"
# --- Chunking DB Config ---
CHUNK_DB_PATH = os.path.join(BASE_DATA_DIR, "databases", "full_text_chunks_db") # Centralized path
CHUNK_COLLECTION_NAME = "paper_chunks_main"
CHUNK_SIZE = 1000 # Default chunk size from chunk_new_dois
CHUNK_OVERLAP = 150 # Default chunk overlap from chunk_new_dois
# Embedding config (defaults from chunk_new_dois will be used if not overridden)
# EMBED_BATCH_SIZE = 64
# EMBED_DELAY = 1.0
# --- Special Relevant Chunks DB Config ---
RELEVANT_CHUNKS_DB_PATH = os.path.join(BASE_DATA_DIR, "databases", "relevant_chunks_db") # New DB path
RELEVANT_CHUNKS_COLLECTION_NAME = "relevant_paper_chunks" # New collection name

# Search String Generation Configuration
INITIAL_RESEARCH_QUESTION = "Would the fast track approach to calculate the crop water footprint still work in a changing climate?" #"What are the effects of sea level rise on italy?"
# Fallback query if generation fails
# MANUAL_SCOPUS_QUERY =  "(\"virtual water\" OR \"water footprint\") AND trade AND (agriculture OR \"food security\")" # Example: Add your manual query here
MANUAL_SCOPUS_QUERY = """
"climate change" AND storylines
"""
DEFAULT_SCOPUS_QUERY = "climate OR 'climate change' OR 'climate variability' AND robustness AND uncertainty AND policy AND decision AND making"
SAVE_GENERATED_SEARCH_STRING = True # Whether to save the generated string to a file

# Scopus Search Configuration
SCOPUS_HEADLESS_MODE = False # Example: Add config for headless mode
SCOPUS_YEAR_FROM = None # Example: Add config for year filter start
SCOPUS_YEAR_TO = None   # Example: Add config for year filter end
# Credentials and Institution are expected to be in .env by search_scopus.py

# ChromaDB Ingestion Configuration
FORCE_REINDEX_CHROMA = False # Set to True to re-index existing documents

# Abstract Collection Configuration
TOP_K_ABSTRACTS = 5
USE_RERANK_ABSTRACTS = True
# Output filename - relative to where the script is run (project root assumed)
RELEVANT_ABSTRACTS_OUTPUT_FILENAME = os.path.join(BASE_DATA_DIR, "output", "relevant_abstracts.txt")

# --- Query Configuration (for final step) ---
QUERY_TOP_K = 5 # Example: Number of results for the final query
QUERY_RERANKER = config.RERANKER_MODEL # Use RAG config default
QUERY_RERANK_CANDIDATES = config.DEFAULT_RERANK_CANDIDATE_COUNT # Use RAG config default
QUERY_OUTPUT_FILENAME = os.path.join(BASE_DATA_DIR, "output", "final_answer.txt") # Output file for the final answer

# --- Ensure Directories Exist ---
os.makedirs(FULL_TEXT_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)
# Ensure the chunk DB directory exists
os.makedirs(os.path.dirname(CHUNK_DB_PATH), exist_ok=True)
# Ensure the relevant chunk DB directory exists
os.makedirs(os.path.dirname(RELEVANT_CHUNKS_DB_PATH), exist_ok=True) # Added for relevant chunks DB


# --- Step 1: Decompose query ---
print("\n--- Step 1: Decomposing Research Question ---")
decomposed_queries = query_decomposition(query=INITIAL_RESEARCH_QUESTION, number_of_sub_queries=3, model=config.SUBQUERY_MODEL)
if decomposed_queries:
    print("Decomposed queries:")
    for i, query in enumerate(decomposed_queries):
        print(f"Subquery {i+1}: {query}")

combined_answers_output_path = os.path.join(BASE_DATA_DIR, "output", "combined_answers.txt")
combined_answers = []

for i, query in enumerate(decomposed_queries):
    # loop over the decomposed queries and generate answers
    print("\n--- Step 3: Finding Relevant DOIs from Abstracts ---")
    # This step also implicitly relies on search_success being True
    relevant_doi_list = find_relevant_dois_from_abstracts(
        initial_query=query,
        db_path=ABSTRACT_DB_PATH,
        collection_name=ABSTRACT_COLLECTION_NAME,
        top_k=TOP_K_ABSTRACTS,
        use_rerank=USE_RERANK_ABSTRACTS,
        output_filename=RELEVANT_ABSTRACTS_OUTPUT_FILENAME
    )


    print("\n--- Step 4: Downloading Full Texts ---")
    if relevant_doi_list:
        download_dois(
            proposed_dois=relevant_doi_list,
            output_directory=FULL_TEXT_DIR
        )
    else:
        print("No relevant DOIs found or previous steps skipped, skipping download step.")

    print("\n--- Step 5: Chunking downloaded documents and adding to common database ---")
    # Call the main function from chunk_new_dois.py
    process_folder_for_chunks(
        folder_path=FULL_TEXT_DIR,         # Source folder for downloaded .txt files
        db_path=CHUNK_DB_PATH,             # Target database path for chunks
        collection_name=CHUNK_COLLECTION_NAME, # Target collection name for chunks
        chunk_size=CHUNK_SIZE,             # Use configured chunk size
        chunk_overlap=CHUNK_OVERLAP,       # Use configured chunk overlap
        # Optional: Pass embedding batch size and delay if you want to override defaults
        # embed_batch_size=EMBED_BATCH_SIZE,
        # embed_delay=EMBED_DELAY
    )

    print("\n--- Step 6: Collecting Relevant Chunks into Special Database ---")
    if relevant_doi_list:
        print(f"Found {len(relevant_doi_list)} relevant DOIs to collect chunks for.")
        # Call the function from the new module
        build_relevant_db(
            relevant_doi_list=relevant_doi_list,
            source_chunk_db_path=CHUNK_DB_PATH,
            source_chunk_collection_name=CHUNK_COLLECTION_NAME,
            abstract_db_path=ABSTRACT_DB_PATH,
            abstract_collection_name=ABSTRACT_COLLECTION_NAME,
            target_db_path=RELEVANT_CHUNKS_DB_PATH,
            target_collection_name=RELEVANT_CHUNKS_COLLECTION_NAME
        )
    else:
        print("No relevant DOIs identified in Step 3, skipping collection of relevant chunks.")


    print("\n--- Step 7: Initializing LLM Clients for Final Query ---")
    try:
        llm_interface.initialize_clients()
        if not llm_interface.gemini_client:
            print("Warning: Gemini client initialization might have failed. Check API keys/config.")
        else:
            print("LLM clients initialized successfully.")
    except Exception as init_err:
        print(f"Error initializing LLM clients: {init_err}")
        # Decide if you want to exit or continue without the final query
        # sys.exit(1) # Optional: Exit if client initialization fails

    print("\n--- Step 8: Querying Relevant Chunks Database ---")
    if relevant_doi_list and llm_interface.gemini_client: # Only run if DOIs were found and client is ready
        print(f"Querying the '{RELEVANT_CHUNKS_COLLECTION_NAME}' collection with the initial research question...")
        query_config = {
            "mode": "query", # Or "query_direct" if preferred
            "db_path": RELEVANT_CHUNKS_DB_PATH,
            "collection_name": RELEVANT_CHUNKS_COLLECTION_NAME,
            "query": query,
            "top_k": QUERY_TOP_K,
            "reranker": QUERY_RERANKER,
            "rerank_candidates": QUERY_RERANK_CANDIDATES,
            "output_filename": QUERY_OUTPUT_FILENAME # Add output filename
            # Add other necessary parameters for run_query_mode if needed
            # e.g., subquery_model, answer_model from rag_config if using "query" mode
        }
        try:
            # Ensure necessary models are available based on mode
            if query_config["mode"] == "query":
                query_config["subquery_model"] = config.SUBQUERY_MODEL
                query_config["answer_model"] = config.CHAT_MODEL
            elif query_config["mode"] == "query_direct":
                query_config["answer_model"] = config.CHAT_MODEL

            final_answer = rag_workflows.run_query_mode(query_config)
            combined_answers.append(final_answer)
            print(f"Final answer for subquery '{query}': {final_answer}")
            print(f"Query finished. Check the output or '{QUERY_OUTPUT_FILENAME}'.")
        except Exception as query_err:
            print(f"An error occurred during the final query step: {query_err}")
    elif not relevant_doi_list:
        print("Skipping final query step because no relevant DOIs were processed.")
    elif not llm_interface.gemini_client:
        print("Skipping final query step because LLM client initialization failed.")

    # write the combined answers to a file
    with open(combined_answers_output_path, "a", encoding="utf-8") as f:
        f.write(f"Subquery {i+1}: {query}\n")
        f.write(f"Final answer: {final_answer}\n\n")
        print(f"Combined answers written to {combined_answers_output_path}")

    print("\n--- Pipeline Finished ---")

