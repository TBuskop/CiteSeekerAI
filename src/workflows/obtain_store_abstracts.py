import os
import sys

# --- Add project root to sys.path ---
# This allows absolute imports from 'src' assuming the script is in 'workflows'
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, os.pardir, os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# --- Updated Imports (with src prefix) ---
from src.scrape.search_scopus import run_scopus_search
from src.scrape.add_csv_to_chromadb import ingest_csv_to_chroma
from src.scrape.get_search_string import generate_scopus_search_string
from src.my_utils import llm_interface
from src.rag.HyPE import run_hype_index  # Added import
import config

# --- Central Configuration ---
# General
# Assuming the script is run from the project root (e.g., python workflows/paper_collection_pipeline.py)
BASE_DATA_DIR = os.path.join(_PROJECT_ROOT, "data")
DOWNLOADS_DIR = os.path.join(BASE_DATA_DIR, "downloads")
FULL_TEXT_DIR = os.path.join(DOWNLOADS_DIR, "full_doi_texts")
CSV_DIR = os.path.join(DOWNLOADS_DIR, "csv")
# Abstract DB Config
ABSTRACT_DB_PATH = os.path.join(BASE_DATA_DIR, "databases", "abstract_chroma_db")
ABSTRACT_COLLECTION_NAME = "abstracts"
# --- Chunking DB Config ---
CHUNK_DB_PATH = os.path.join(
    BASE_DATA_DIR, "databases", "full_text_chunks_db"
)  # Centralized path
# --- Special Relevant Chunks DB Config ---
RELEVANT_CHUNKS_DB_PATH = os.path.join(
    BASE_DATA_DIR, "databases", "relevant_chunks_db"
)  # New DB path
# --- HyPE Config ---
HYPE_ABSTRACT_COLLECTION_NAME = config.HYPE_SOURCE_COLLECTION_NAME + config.HYPE_SUFFIX

# Search String Generation Configuration
INITIAL_RESEARCH_QUESTION = config.QUERY
# Fallback query if generation fails
MANUAL_SCOPUS_QUERY = config.SCOPUS_SEARCH_STRING  # Use from config.py
DEFAULT_SCOPUS_QUERY = "climate OR 'climate change' OR 'climate variability' AND robustness AND uncertainty AND policy AND decision AND making"
SAVE_GENERATED_SEARCH_STRING = True  # Whether to save the generated string to a file

# Scopus Search Configuration
SCOPUS_HEADLESS_MODE = False  # Example: Add config for headless mode
SCOPUS_YEAR_FROM = config.SCOPUS_START_YEAR  # Example: Add config for year filter start
SCOPUS_YEAR_TO = config.SCOPUS_END_YEAR  # Example: Add config for year filter end
# Credentials and Institution are expected to be in .env by search_scopus.py

# ChromaDB Ingestion Configuration
FORCE_REINDEX_CHROMA = False  # Set to True to re-index existing documents

# --- Ensure Directories Exist ---
os.makedirs(FULL_TEXT_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)
# Ensure the chunk DB directory exists
os.makedirs(os.path.dirname(CHUNK_DB_PATH), exist_ok=True)
# Ensure the relevant chunk DB directory exists
os.makedirs(
    os.path.dirname(RELEVANT_CHUNKS_DB_PATH), exist_ok=True
)  # Added for relevant chunks DB

# --- Pipeline Execution ---


def obtain_store_abstracts(
    search_query=None, scopus_search_scope=None, progress_callback=None
):  # Added scopus_search_scope
    # helper for UI progress and console
    def log_progress(msg):
        if progress_callback:
            progress_callback(msg)
        print(msg)

    log_progress("--- Initializing LLM Clients ---")
    llm_client = llm_interface.initialize_clients()  # Initialize LLM client

    # Use the passed question or fall back to config.QUERY
    # The search_query parameter is now the primary Scopus search string from the UI
    # search_query = search_query if search_query is not None else config.SCOPUS_SEARCH_STRING # This logic might change

    log_progress("--- Step 0: Generating Scopus Search String ---")
    # If a direct Scopus search query is provided from the UI, we use that.
    # The generate_scopus_search_string (which converts a research question to a Scopus query)
    # might not be needed if the user provides a direct Scopus query.
    # For now, we assume 'search_query' IS the Scopus query.
    if search_query:
        SCOPUS_QUERY = search_query
        print(f"Using provided Scopus query: {SCOPUS_QUERY}")
    elif (
        MANUAL_SCOPUS_QUERY
    ):  # Fallback to config if no direct query and MANUAL_SCOPUS_QUERY is set
        SCOPUS_QUERY = config.SCOPUS_SEARCH_STRING
        print(f"Using manual Scopus query from config: {SCOPUS_QUERY}")
    else:  # Fallback to generating from INITIAL_RESEARCH_QUESTION if no direct query and not manual
        success, generated_query = generate_scopus_search_string(
            query=INITIAL_RESEARCH_QUESTION,  # Generate based on the general research question from config
            save_to_file=SAVE_GENERATED_SEARCH_STRING,
        )
        if success and generated_query:
            SCOPUS_QUERY = generated_query
            print(f"Using generated Scopus query: {SCOPUS_QUERY}")
        else:
            SCOPUS_QUERY = DEFAULT_SCOPUS_QUERY  # Ultimate fallback
            print(
                f"Failed to generate search string. Using default Scopus query: {SCOPUS_QUERY}"
            )

    # Define Scopus output path based on the final query
    clean_query = (
        SCOPUS_QUERY.replace(" ", "_")
        .replace("'", "")
        .replace("(", "")
        .replace(")", "")
        .replace(":", "")
        .replace("*", "")
        .replace("?", "")
        .replace('"', "")
        .replace("\n", "")
    )[:50]

    SCOPUS_OUTPUT_CSV_FILENAME = f"scopus_{clean_query}.csv"
    SCOPUS_OUTPUT_CSV_PATH = os.path.join(CSV_DIR, SCOPUS_OUTPUT_CSV_FILENAME)

    log_progress("--- Step 1: Running Scopus Search ---")
    # Determine the search scope to use: passed argument or config default
    final_scopus_search_scope = (
        scopus_search_scope if scopus_search_scope else config.SCOPUS_SEARCH_SCOPE
    )

    # Run the Scopus search using the determined query and output path
    # Always run search if initiated from UI, even if file exists, to reflect current parameters.
    # Or, add more sophisticated logic to check if parameters match existing file.
    # For simplicity, we'll run it.
    search_success, actual_csv_path = run_scopus_search(
        query=SCOPUS_QUERY,
        output_csv_path=SCOPUS_OUTPUT_CSV_PATH,
        headless=SCOPUS_HEADLESS_MODE,
        year_from=SCOPUS_YEAR_FROM,
        year_to=SCOPUS_YEAR_TO,
        scopus_search_scope=final_scopus_search_scope,  # Pass the scope
    )
    # The original logic for MANUAL_SCOPUS_QUERY and checking file existence might need adjustment
    # if the goal is to always re-run with UI parameters.
    # The code below is simplified assuming a re-run.

    # Check if the search was successful and update the path variable
    if search_success and actual_csv_path:
        SCOPUS_OUTPUT_CSV_PATH = str(
            actual_csv_path
        )  # Update path to the actual one used/created
        log_progress(
            f"Scopus search successful. Results saved to: {SCOPUS_OUTPUT_CSV_PATH}"
        )
    else:
        print("Error: Scopus search failed. Check logs in search_scopus script.")
        # Decide how to proceed: exit or try to continue? Exiting is safer.
        exit()  # Exit if search failed

    log_progress("--- Step 2: Ingesting CSV to ChromaDB ---")
    # This step now implicitly relies on search_success being True from Step 1
    ingest_csv_to_chroma(
        csv_file_path=SCOPUS_OUTPUT_CSV_PATH,
        db_path=ABSTRACT_DB_PATH,  # Use abstract DB path
        collection_name=ABSTRACT_COLLECTION_NAME,  # Use abstract collection name
        force_reindex=FORCE_REINDEX_CHROMA,
    )

    log_progress("--- Step 2.5: Running HyPE Indexing on Abstracts ---")
    # This step generates hypothetical questions for abstracts and embeds them.
    # It uses the same DB as abstracts but a different collection.
    if config.HYPE:
        run_hype_index(
            db_path=ABSTRACT_DB_PATH,
            source_collection_name=ABSTRACT_COLLECTION_NAME,
            hype_collection_name=HYPE_ABSTRACT_COLLECTION_NAME,
            client=llm_client,  # Pass the initialized client
        )


if __name__ == "__main__":
    obtain_store_abstracts()
