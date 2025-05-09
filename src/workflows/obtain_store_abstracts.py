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
from src.rag.HyPE import run_hype_index # Added import
import config

# --- Central Configuration ---
# General
# Assuming the script is run from the project root (e.g., python workflows/paper_collection_pipeline.py)
BASE_DATA_DIR = os.path.join(_PROJECT_ROOT, 'data')
DOWNLOADS_DIR = os.path.join(BASE_DATA_DIR, "downloads")
FULL_TEXT_DIR = os.path.join(DOWNLOADS_DIR, "full_doi_texts")
CSV_DIR = os.path.join(DOWNLOADS_DIR, "csv")
# Abstract DB Config
ABSTRACT_DB_PATH = os.path.join(BASE_DATA_DIR, "databases", "abstract_chroma_db")
ABSTRACT_COLLECTION_NAME = "abstracts"
# --- Chunking DB Config ---
CHUNK_DB_PATH = os.path.join(BASE_DATA_DIR, "databases", "full_text_chunks_db") # Centralized path
# --- Special Relevant Chunks DB Config ---
RELEVANT_CHUNKS_DB_PATH = os.path.join(BASE_DATA_DIR, "databases", "relevant_chunks_db") # New DB path
# --- HyPE Config ---
HYPE_ABSTRACT_COLLECTION_NAME = config.HYPE_SOURCE_COLLECTION_NAME + config.HYPE_SUFFIX

# Search String Generation Configuration
INITIAL_RESEARCH_QUESTION = config.QUERY
# Fallback query if generation fails
MANUAL_SCOPUS_QUERY = config.SCOPUS_SEARCH_STRING # Use from config.py
DEFAULT_SCOPUS_QUERY = "climate OR 'climate change' OR 'climate variability' AND robustness AND uncertainty AND policy AND decision AND making"
SAVE_GENERATED_SEARCH_STRING = True # Whether to save the generated string to a file

# Scopus Search Configuration
SCOPUS_HEADLESS_MODE = False # Example: Add config for headless mode
SCOPUS_YEAR_FROM = None # Example: Add config for year filter start
SCOPUS_YEAR_TO = None   # Example: Add config for year filter end
# Credentials and Institution are expected to be in .env by search_scopus.py

# ChromaDB Ingestion Configuration
FORCE_REINDEX_CHROMA = False # Set to True to re-index existing documents

# --- Ensure Directories Exist ---
os.makedirs(FULL_TEXT_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)
# Ensure the chunk DB directory exists
os.makedirs(os.path.dirname(CHUNK_DB_PATH), exist_ok=True)
# Ensure the relevant chunk DB directory exists
os.makedirs(os.path.dirname(RELEVANT_CHUNKS_DB_PATH), exist_ok=True) # Added for relevant chunks DB

# --- Pipeline Execution ---

def obtain_store_abstracts():
    print("--- Initializing LLM Clients ---")
    llm_client = llm_interface.initialize_clients() # Initialize LLM client

    print("--- Step 0: Generating Scopus Search String ---")
    if MANUAL_SCOPUS_QUERY:
        SCOPUS_QUERY = MANUAL_SCOPUS_QUERY # Use manual query if provided
    else:
        success, generated_query = generate_scopus_search_string(
            query=INITIAL_RESEARCH_QUESTION,
            save_to_file=SAVE_GENERATED_SEARCH_STRING
        )
        if success and generated_query:
            SCOPUS_QUERY = generated_query
            print(f"Using generated Scopus query: {SCOPUS_QUERY}")
        else:
            SCOPUS_QUERY = DEFAULT_SCOPUS_QUERY
            print(f"Failed to generate search string. Using default Scopus query: {SCOPUS_QUERY}")

    # Define Scopus output path based on the final query
    clean_query = (SCOPUS_QUERY.replace(' ', '_')
                                 .replace("'", '')
                                 .replace('(', '')
                                 .replace(')', '')
                                 .replace(':', '')
                                 .replace('*', '')
                                 .replace('?', '')
                                 .replace("\"", '')
                                 .replace("\n", ''))[:50]

    SCOPUS_OUTPUT_CSV_FILENAME = f"scopus_{clean_query}.csv"
    SCOPUS_OUTPUT_CSV_PATH = os.path.join(CSV_DIR, SCOPUS_OUTPUT_CSV_FILENAME)

    print("\n--- Step 1: Running Scopus Search ---")
    # Run the Scopus search using the determined query and output path
    if MANUAL_SCOPUS_QUERY:
       # check if file exists before running the search
        if os.path.exists(SCOPUS_OUTPUT_CSV_PATH):
            print(f"File {SCOPUS_OUTPUT_CSV_PATH} already exists. Skipping search.")
            search_success = True
            actual_csv_path = SCOPUS_OUTPUT_CSV_PATH # Use existing file path
        else:
            search_success, actual_csv_path = run_scopus_search(
                query=SCOPUS_QUERY,
                output_csv_path=SCOPUS_OUTPUT_CSV_PATH, # Pass the desired full path
                headless=SCOPUS_HEADLESS_MODE,
                year_from=SCOPUS_YEAR_FROM,
                year_to=SCOPUS_YEAR_TO
            )
    else:
        search_success, actual_csv_path = run_scopus_search(
            query=SCOPUS_QUERY,
            output_csv_path=SCOPUS_OUTPUT_CSV_PATH, # Pass the desired full path
            headless=SCOPUS_HEADLESS_MODE,
            year_from=SCOPUS_YEAR_FROM,
            year_to=SCOPUS_YEAR_TO
            # Credentials and institution are handled by run_scopus_search loading .env
        )

    # Check if the search was successful and update the path variable
    if search_success and actual_csv_path:
        SCOPUS_OUTPUT_CSV_PATH = str(actual_csv_path) # Update path to the actual one used/created
        print(f"Scopus search successful. Results saved to: {SCOPUS_OUTPUT_CSV_PATH}")
    else:
        print(f"Error: Scopus search failed. Check logs in search_scopus script.")
        # Decide how to proceed: exit or try to continue? Exiting is safer.
        exit() # Exit if search failed

    print("\n--- Step 2: Ingesting CSV to ChromaDB which contains all abstracts ever collected ---")
    # This step now implicitly relies on search_success being True from Step 1
    ingest_csv_to_chroma(
        csv_file_path=SCOPUS_OUTPUT_CSV_PATH,
        db_path=ABSTRACT_DB_PATH, # Use abstract DB path
        collection_name=ABSTRACT_COLLECTION_NAME, # Use abstract collection name
        force_reindex=FORCE_REINDEX_CHROMA
    )

    print("\n--- Step 2.5: Running HyPE Indexing on Abstracts ---")
    # This step generates hypothetical questions for abstracts and embeds them.
    # It uses the same DB as abstracts but a different collection.
    if config.HYPE:
        run_hype_index(
            db_path=ABSTRACT_DB_PATH,
            source_collection_name=ABSTRACT_COLLECTION_NAME,
            hype_collection_name=HYPE_ABSTRACT_COLLECTION_NAME,
            client=llm_client # Pass the initialized client
        )

if __name__ == "__main__":
    obtain_store_abstracts()

