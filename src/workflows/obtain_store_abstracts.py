import os
import sys
import logging # Add logging import

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

logger = logging.getLogger(__name__) # Initialize logger

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
SCOPUS_YEAR_FROM = config.SCOPUS_START_YEAR  # Example: Add config for year filter start
SCOPUS_YEAR_TO = config.SCOPUS_END_YEAR  # Example: Add config for year filter end
MIN_CITATIONS = config.MIN_CITATIONS_STORE_ABSTRACT # Example: Add config for minimum citations
# Credentials and Institution are expected to be in .env by search_scopus.py

# ChromaDB Ingestion Configuration
FORCE_REINDEX_CHROMA = False # Set to True to re-index existing documents
# MIN_CITATIONS_FILTER = config.MIN_CITATIONS_STORE_ABSTRACT # This will be determined dynamically

# --- Ensure Directories Exist ---
os.makedirs(FULL_TEXT_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)
# Ensure the chunk DB directory exists
os.makedirs(os.path.dirname(CHUNK_DB_PATH), exist_ok=True)
# Ensure the relevant chunk DB directory exists
os.makedirs(os.path.dirname(RELEVANT_CHUNKS_DB_PATH), exist_ok=True) # Added for relevant chunks DB

# --- Pipeline Execution ---

def obtain_store_abstracts(search_query=None, scopus_search_scope=None, year_from=None, year_to=None, min_citations_param=None, progress_callback=None, force_continue_large_search=False): # Added force_continue_large_search parameter
    # helper for UI progress and console
    def log_progress(msg):
        if progress_callback:
            progress_callback(msg)
        logger.info(msg)
        print(msg) # Also print to console for backend visibility

    # --- Configuration & Setup ---
    log_progress(f"obtain_store_abstracts called with: search_query='{search_query}', "
                 f"scope='{scopus_search_scope}', year_from='{year_from}', year_to='{year_to}', "
                 f"min_citations='{min_citations_param}', force_continue='{force_continue_large_search}'")

    log_progress("--- Initializing LLM Clients ---")
    llm_client = None # Initialize to None
    llm_init_error = None # To store any error message from LLM initialization
    try:
        llm_client = llm_interface.initialize_clients() # Initialize LLM client
        if llm_client:
            log_progress("LLM Clients initialized successfully.")
        else:
            # This case handles if initialize_clients can return None without raising an exception
            log_progress("LLM Clients initialization returned no client, but no error was raised.")
            llm_init_error = "Initialization returned no client."
    except Exception as e:
        logger.error(f"Error initializing LLM clients: {e}", exc_info=True) # Log full traceback for backend
        log_progress(f"Error initializing LLM clients: {str(e)}. Proceeding; LLM-dependent features may be affected.")
        llm_init_error = str(e)


    # Determine final Scopus query
    log_progress("--- Step 0: Generating Scopus Search String ---")
    if search_query:
        final_scopus_query = search_query
        log_progress(f"Using provided Scopus query: {final_scopus_query}") # Changed print to log_progress
    elif MANUAL_SCOPUS_QUERY: # Fallback to config if no direct query and MANUAL_SCOPUS_QUERY is set
        final_scopus_query = config.SCOPUS_SEARCH_STRING
        log_progress(f"Using manual Scopus query from config: {final_scopus_query}") # Changed print to log_progress
    else: # Fallback to generating from INITIAL_RESEARCH_QUESTION if no direct query and not manual
        if llm_client: # Check if LLM client is available for generation
            success, generated_query = generate_scopus_search_string(
                query=INITIAL_RESEARCH_QUESTION, # Generate based on the general research question from config
                save_to_file=SAVE_GENERATED_SEARCH_STRING
                # Assuming generate_scopus_search_string uses the client initialized via llm_interface
            )
            if success and generated_query:
                final_scopus_query = generated_query
                log_progress(f"Using generated Scopus query: {final_scopus_query}") # Changed print to log_progress
            else:
                final_scopus_query = DEFAULT_SCOPUS_QUERY # Ultimate fallback
                log_progress(f"Failed to generate search string. Using default Scopus query: {final_scopus_query}") # Changed print to log_progress
        else: # LLM client not available
            final_scopus_query = DEFAULT_SCOPUS_QUERY
            log_progress(f"LLM client not available (Error: {llm_init_error}), cannot generate search string. Using default Scopus query: {final_scopus_query}")

    # Process other parameters
    # Process year filters - fall back to None for web interface
    final_year_from = year_from
    final_year_to = year_to
    log_progress(f"Year filter: {final_year_from or 'None'} to {final_year_to or 'None'}")
    
    # Process search scope - fall back to None for web interface
    final_scopus_search_scope = scopus_search_scope
    log_progress(f"Search scope: {final_scopus_search_scope or 'Default'}")
    
    # Process minimum citations - fall back to None for web interface
    final_min_citations = min_citations_param
    log_progress(f"Minimum citations filter: {final_min_citations or 'None'}")

    # Define Scopus output path based on the final query
    clean_query = (final_scopus_query.replace(' ', '_')
                                 .replace("'", '')
                                 .replace('(', '')
                                 .replace(')', '')
                                 .replace(':', '')
                                 .replace('*', '')
                                 .replace('?', '')
                                 .replace("\"", '')
                                 .replace("\n", ''))[:50]

    SCOPUS_OUTPUT_CSV_FILENAME = f"scopus_{clean_query}.csv"
    SCOPUS_OUTPUT_CSV_PATH = os.path.join(CSV_DIR, SCOPUS_OUTPUT_CSV_FILENAME)    # --- Step 1: Performing Scopus Search ---
    log_progress(f"--- Step 1: Performing Scopus Search ---")
    
    # If we're forcing continue on a large result set, add a special message
    if force_continue_large_search:
        log_progress(f"Processing a large result set by user request - this may take longer than usual")
        log_progress(f"Please be patient while we prepare and download the results...")
    
    search_status, actual_csv_path, results_count = run_scopus_search(
        query=final_scopus_query,
        headless=SCOPUS_HEADLESS_MODE,
        year_from=final_year_from,
        year_to=final_year_to,
        output_csv_path=SCOPUS_OUTPUT_CSV_PATH, # Use the centrally defined path
        scopus_search_scope=final_scopus_search_scope,
        force_continue_large_search=force_continue_large_search) # Pass the flag    )

    # Provide more detailed completion message
    if force_continue_large_search and search_status == "EXPORT_SUCCESS":
        log_progress(f"Successfully completed large result set search and export!")
        log_progress(f"Retrieved and exported {results_count} documents to {actual_csv_path}")
    else:
        log_progress(f"Scopus search completed. Status: {search_status}, Results: {results_count}, CSV Path: {actual_csv_path}")

    # Handle specific search outcomes
    if search_status == "SEARCH_WARNING_TOO_MANY_RESULTS":
        # This block is now only reached if force_continue_large_search was False during run_scopus_search call,
        # meaning user confirmation is genuinely needed.
        log_progress(f"Scopus search returned {results_count} results, which is a large number. Awaiting user confirmation.")
        return {
            "status": "AWAITING_USER_CONFIRMATION_LARGE_RESULTS",
            "message": f"Scopus search returned {results_count} results. Proceed anyway or cancel?",
            "results_count": results_count,
            "query_details": {
                "query": final_scopus_query,
                "scope": final_scopus_search_scope,
                "year_from": final_year_from,
                "year_to": final_year_to,
                "min_citations": final_min_citations # Ensure final_min_citations is defined in this scope
            }
        }    
    
    if search_status == "SEARCH_ERROR_LIMIT_EXCEEDED":
        log_progress(f"Scopus search error: Limit exceeded with {results_count} results.")
        return {"status": "ERROR_SCRAPE_LIMIT_EXCEEDED", "message": f"Search limit exceeded: {results_count} results. Please refine your query.", "results_count": results_count}
    
    # Check for CSV file creation after search attempt (successful or forced continue)
    if not actual_csv_path:
        log_progress(f"Scopus search/export did not produce a CSV file. Status: {search_status}, Results: {results_count}")
        if search_status.startswith("SEARCH_FAILURE"):
            if search_status == "SEARCH_FAILURE_EXCEPTION":
                # This is likely a browser closed or connection error
                return {"status": "ERROR_SCRAPE_SEARCH_FAILED", "message": "Search failed because the browser was closed or connection was lost. Please try again.", "results_count": results_count}
            else:
                return {"status": "ERROR_SCRAPE_SEARCH_FAILED", "message": f"Scopus search failed: {search_status}", "results_count": results_count}
    elif search_status == "EXPORT_FAILURE":
        return {"status": "ERROR_SCRAPE_EXPORT_FAILED", "message": "Failed to export Scopus results to CSV.", "results_count": results_count}    
    elif search_status == "EXPORT_SUCCESS": 
        log_progress(f"Search reported EXPORT_SUCCESS but no CSV path was returned. This is unexpected. Status: {search_status}")
        return {"status": "ERROR_SCRAPE_NO_CSV_UNEXPECTED", "message": "Scopus export reported success, but the CSV file was not found.", "results_count": results_count}
    else:
        # If we reach here, there's a CSV file but status isn't recognized - treat as unknown error
        return {"status": "ERROR_SCRAPE_UNKNOWN", "message": "An unknown error occurred during Scopus scraping.", "results_count": results_count}

    log_progress(f"Scopus CSV successfully created at: {actual_csv_path}")
    # Ensure SCOPUS_OUTPUT_CSV_PATH is updated if it's used later and might have changed.
    # If SCOPUS_OUTPUT_CSV_PATH is a global or module-level var, direct assignment might not be ideal.
    # Consider returning it or handling its update carefully.
    # For now, assuming it's handled or its update here is intended:
    # SCOPUS_OUTPUT_CSV_PATH = str(actual_csv_path) 

    # --- Step 2: Ingesting CSV into ChromaDB ---
    
    log_progress("--- Step 2: Ingesting CSV to ChromaDB ---")
    
    # Determine the minimum citations filter to use
    # Use the value from UI if provided and valid, otherwise fallback to config
    final_min_citations_filter = min_citations_param if min_citations_param is not None else config.MIN_CITATIONS_STORE_ABSTRACT
    if min_citations_param is not None:
        log_progress(f"Using minimum citations from UI: {min_citations_param}")
    else:
        log_progress(f"Using minimum citations from config: {config.MIN_CITATIONS_STORE_ABSTRACT}")    # This step now implicitly relies on search_success being True from Step 1
    ingest_csv_to_chroma(
        csv_file_path=SCOPUS_OUTPUT_CSV_PATH,
        db_path=ABSTRACT_DB_PATH, # Use abstract DB path
        collection_name=ABSTRACT_COLLECTION_NAME, # Use abstract collection name
        force_reindex=FORCE_REINDEX_CHROMA,
        min_citations=final_min_citations_filter # Pass the determined minimum citations filter
    )
    
    log_progress("--- Step 2.5: Running HyPE Indexing on Abstracts ---")
    # This step generates hypothetical questions for abstracts and embeds them.
    # It uses the same DB as abstracts but a different collection.
    if config.HYPE:
        if llm_client: # Proceed with HyPE only if LLM client was successfully initialized
            run_hype_index(
                db_path=ABSTRACT_DB_PATH,
                source_collection_name=ABSTRACT_COLLECTION_NAME,
                hype_collection_name=HYPE_ABSTRACT_COLLECTION_NAME,
                client=llm_client # Pass the initialized client
            )
        else:
            log_progress(f"Skipping HyPE indexing as LLM client initialization failed or client not available (Error: {llm_init_error}).")
      # Return success result
    return {
        "status": "SUCCESS", 
        "message": "Abstract collection completed successfully", 
        "file_path": SCOPUS_OUTPUT_CSV_PATH,
        "count": results_count
    }

if __name__ == "__main__":
    obtain_store_abstracts()

