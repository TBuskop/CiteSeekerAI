import os
from search_scopus import run_scopus_search # Assuming this function will be updated to accept config
from add_csv_to_chromadb import ingest_csv_to_chroma
from collect_relevant_abstracts import find_relevant_dois_from_abstracts
from download_papers import download_dois
from get_search_string import generate_scopus_search_string # Import the new function
# --- Import the chunking function ---
from chunk_new_dois import process_folder_for_chunks

# --- Central Configuration ---
# General
BASE_DATA_DIR = "./data"
DOWNLOADS_DIR = os.path.join(BASE_DATA_DIR, "downloads")
FULL_TEXT_DIR = os.path.join(DOWNLOADS_DIR, "full_texts")
CSV_DIR = os.path.join(DOWNLOADS_DIR, "csv")
# Abstract DB Config
ABSTRACT_DB_PATH = "./abstract_chroma_db"
ABSTRACT_COLLECTION_NAME = "abstracts"
# --- Chunking DB Config ---
CHUNK_DB_PATH = os.path.join(BASE_DATA_DIR, "chroma_dbs", "full_text_chunks_db") # Centralized path
CHUNK_COLLECTION_NAME = "paper_chunks_main"
CHUNK_SIZE = 1000 # Default chunk size from chunk_new_dois
CHUNK_OVERLAP = 150 # Default chunk overlap from chunk_new_dois
# Embedding config (defaults from chunk_new_dois will be used if not overridden)
# EMBED_BATCH_SIZE = 64
# EMBED_DELAY = 1.0

# Search String Generation Configuration
INITIAL_RESEARCH_QUESTION = "What are the effects of sea level rise on italy?"
# Fallback query if generation fails
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
TOP_K_ABSTRACTS = 10
USE_RERANK_ABSTRACTS = True
RELEVANT_ABSTRACTS_OUTPUT_FILENAME = "relevant_abstracts.txt"

# --- Ensure Directories Exist ---
os.makedirs(FULL_TEXT_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)
# Ensure the chunk DB directory exists
os.makedirs(os.path.dirname(CHUNK_DB_PATH), exist_ok=True)

# --- Pipeline Execution ---

print("--- Step 0: Generating Scopus Search String ---")
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
                             .replace('?', ''))[:50]

SCOPUS_OUTPUT_CSV_FILENAME = f"scopus_{clean_query}.csv"
SCOPUS_OUTPUT_CSV_PATH = os.path.join(CSV_DIR, SCOPUS_OUTPUT_CSV_FILENAME)


print("\n--- Step 1: Running Scopus Search ---")
# Run the Scopus search using the determined query and output path
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


print("\n--- Step 2: Ingesting CSV to ChromaDB ---")
# This step now implicitly relies on search_success being True from Step 1
ingest_csv_to_chroma(
    csv_file_path=SCOPUS_OUTPUT_CSV_PATH,
    db_path=ABSTRACT_DB_PATH, # Use abstract DB path
    collection_name=ABSTRACT_COLLECTION_NAME, # Use abstract collection name
    force_reindex=FORCE_REINDEX_CHROMA
)


print("\n--- Step 3: Finding Relevant DOIs from Abstracts ---")
# This step also implicitly relies on search_success being True
relevant_doi_list = find_relevant_dois_from_abstracts(
    initial_query=INITIAL_RESEARCH_QUESTION,
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

print("\n--- Pipeline Finished ---")