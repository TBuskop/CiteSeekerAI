import os
import sys
import concurrent.futures
import threading # Added for locking

# --- Add project root to sys.path ---
# This allows absolute imports from 'src' assuming the script is in 'workflows'
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# --- Updated Imports (with src prefix) ---
from src.scrape.collect_relevant_abstracts import find_relevant_dois_from_abstracts
from src.scrape.download_papers import download_dois
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
INITIAL_RESEARCH_QUESTION = "What is the influence of the climate on the evapotranspiration of crops and the overall crop water footprint (both green an blue footprint combined)? Give me a comparission of multiple studies" #"What are the effects of sea level rise on italy?"
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
TOP_K_ABSTRACTS = 25
USE_RERANK_ABSTRACTS = True
# Output filename - relative to where the script is run (project root assumed)
RELEVANT_ABSTRACTS_OUTPUT_FILENAME = os.path.join(BASE_DATA_DIR, "output", "relevant_abstracts.txt")

# --- Query Configuration (for final step) ---
QUERY_TOP_K = 10 # Example: Number of results for the final query
QUERY_RERANKER = config.RERANKER_MODEL # Use RAG config default
QUERY_RERANK_CANDIDATES = config.DEFAULT_RERANK_CANDIDATE_COUNT # Use RAG config default
COMBINED_ANSWERS_OUTPUT_FILENAME = os.path.join(BASE_DATA_DIR, "output", "combined_answers.txt") # Output file for combined final answers
QUERY_SPECIFIC_OUTPUT_DIR = os.path.join(BASE_DATA_DIR, "output", "query_specific") # Output directory for per-query files

# --- Ensure Directories Exist ---
os.makedirs(FULL_TEXT_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)
# Ensure the chunk DB directory exists
os.makedirs(os.path.dirname(CHUNK_DB_PATH), exist_ok=True)
# Ensure the relevant chunk DB directory exists
os.makedirs(os.path.dirname(RELEVANT_CHUNKS_DB_PATH), exist_ok=True) # Added for relevant chunks DB
os.makedirs(QUERY_SPECIFIC_OUTPUT_DIR, exist_ok=True) # Added for per-query output files


# --- Step 1: Decompose query ---
print("\n--- Step 1: Decomposing Research Question ---")
decomposed_queries, overall_goal = query_decomposition(query=INITIAL_RESEARCH_QUESTION, number_of_sub_queries=3, model=config.SUBQUERY_MODEL)
if decomposed_queries:
    print("Decomposed queries:")
    for i, query in enumerate(decomposed_queries):
        print(f"Subquery {i+1}: {query}")

combined_answers_output_path = COMBINED_ANSWERS_OUTPUT_FILENAME # Use configured path
results = [] # To store results from parallel tasks

# --- Initialize LLM Clients (Once before parallel execution) ---
print("\n--- Initializing LLM Clients ---")
try:
    llm_interface.initialize_clients()
    if not llm_interface.gemini_client:
        print("Warning: Gemini client initialization might have failed. Check API keys/config.")
    else:
        print("LLM clients initialized successfully.")
except Exception as init_err:
    print(f"Error initializing LLM clients: {init_err}. Final query step might fail.")
    # Decide if you want to exit or continue without the final query
    # sys.exit(1) # Optional: Exit if client initialization fails

# --- Define Worker Function for Parallel Processing ---
db_lock = threading.Lock() # Lock for synchronizing DB write access

def process_subquery(query, query_index):
    """Processes a single subquery through steps 3-8."""
    thread_id = threading.get_ident()
    print(f"[Thread-{thread_id}, Query {query_index+1}] Starting processing for: {query}")

    # --- Step 3: Finding Relevant DOIs from Abstracts ---
    print(f"[Thread-{thread_id}, Query {query_index+1}] Finding relevant DOIs...")
    # Use a unique output filename for each query's relevant abstracts list
    relevant_abstracts_output_filename_i = os.path.join(QUERY_SPECIFIC_OUTPUT_DIR, f"relevant_abstracts_{query_index+1}.txt")
    relevant_doi_list = find_relevant_dois_from_abstracts(
        initial_query=query,
        db_path=ABSTRACT_DB_PATH,
        collection_name=ABSTRACT_COLLECTION_NAME,
        top_k=TOP_K_ABSTRACTS,
        use_rerank=USE_RERANK_ABSTRACTS,
        output_filename=relevant_abstracts_output_filename_i # Use unique filename
    )
    print(f"[Thread-{thread_id}, Query {query_index+1}] Found {len(relevant_doi_list)} relevant DOIs.")

    # --- Step 4: Downloading Full Texts ---
    if relevant_doi_list:
        print(f"[Thread-{thread_id}, Query {query_index+1}] Downloading full texts...")
        download_dois(
            proposed_dois=relevant_doi_list,
            output_directory=FULL_TEXT_DIR # Shared dir, assumes unique filenames based on DOI
        )
        print(f"[Thread-{thread_id}, Query {query_index+1}] Download step completed.")
    else:
        print(f"[Thread-{thread_id}, Query {query_index+1}] No relevant DOIs found, skipping download.")

    # --- Step 5 & 6: Chunking and Building Relevant DB (Synchronized) ---
    if relevant_doi_list:
        # Acquire lock only if DB write operations are needed
        with db_lock:
            print(f"[Thread-{thread_id}, Query {query_index+1}] Acquiring lock for DB operations...")
            # Step 5: Chunking downloaded documents and adding to common database
            print(f"[Thread-{thread_id}, Query {query_index+1}] Chunking downloaded documents...")
            # Note: This chunks ALL files in FULL_TEXT_DIR. If run concurrently,
            # it might re-chunk files downloaded by other threads.
            # The lock prevents race conditions in DB writes, but work might be redundant.
            # A more refined approach would pass specific files to chunk.
            process_folder_for_chunks(
                folder_path=FULL_TEXT_DIR,
                db_path=CHUNK_DB_PATH,
                collection_name=CHUNK_COLLECTION_NAME,
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
            )
            print(f"[Thread-{thread_id}, Query {query_index+1}] Chunking complete.")

            # Step 6: Collecting Relevant Chunks into Special Database
            print(f"[Thread-{thread_id}, Query {query_index+1}] Collecting relevant chunks...")
            build_relevant_db(
                relevant_doi_list=relevant_doi_list, # Use DOIs specific to this query
                source_chunk_db_path=CHUNK_DB_PATH,
                source_chunk_collection_name=CHUNK_COLLECTION_NAME,
                abstract_db_path=ABSTRACT_DB_PATH,
                abstract_collection_name=ABSTRACT_COLLECTION_NAME,
                target_db_path=RELEVANT_CHUNKS_DB_PATH,
                target_collection_name=RELEVANT_CHUNKS_COLLECTION_NAME
            )
            print(f"[Thread-{thread_id}, Query {query_index+1}] Relevant chunk collection complete.")
            print(f"[Thread-{thread_id}, Query {query_index+1}] Releasing lock.")
    else:
         print(f"[Thread-{thread_id}, Query {query_index+1}] Skipping chunking and relevant DB build due to no relevant DOIs.")

    # --- Step 8: Querying Relevant Chunks Database ---
    final_answer = None
    if relevant_doi_list and llm_interface.gemini_client: # Only run if DOIs were found and client is ready
        print(f"[Thread-{thread_id}, Query {query_index+1}] Querying relevant chunks...")
        # Use a unique output filename for each query's final answer
        query_output_filename_i = os.path.join(QUERY_SPECIFIC_OUTPUT_DIR, f"final_answer_{query_index+1}.txt")
        query_config = {
            "mode": "query", # Or "query_direct" if preferred
            "db_path": RELEVANT_CHUNKS_DB_PATH, # Read access, should be safe concurrently
            "collection_name": RELEVANT_CHUNKS_COLLECTION_NAME,
            "query": query,
            "top_k": QUERY_TOP_K,
            "reranker": QUERY_RERANKER,
            "rerank_candidates": QUERY_RERANK_CANDIDATES,
            "output_filename": query_output_filename_i, # Use unique filename
            # Add other necessary parameters for run_query_mode if needed
            "subquery_model": config.SUBQUERY_MODEL,
            "answer_model": config.CHAT_MODEL
        }
        try:
            # Assuming run_query_mode writes to the unique file AND returns the answer string.
            # If it only writes to the file, you might need to read the file content here.
            final_answer = rag_workflows.run_query_mode(query_config)
            print(f"[Thread-{thread_id}, Query {query_index+1}] Query complete. Answer stored in {query_output_filename_i}.")
        except Exception as query_err:
            print(f"[Thread-{thread_id}, Query {query_index+1}] An error occurred during the final query step: {query_err}")
            final_answer = f"Error processing query: {query_err}" # Store error message
    elif not relevant_doi_list:
        print(f"[Thread-{thread_id}, Query {query_index+1}] Skipping final query step because no relevant DOIs were processed.")
        final_answer = "Skipped (no relevant DOIs found)."
    elif not llm_interface.gemini_client:
        print(f"[Thread-{thread_id}, Query {query_index+1}] Skipping final query step because LLM client initialization failed.")
        final_answer = "Skipped (LLM client initialization failed)."

    print(f"[Thread-{thread_id}, Query {query_index+1}] Finished processing.")
    # Return index, original query, and the final answer/status
    return query_index, query, final_answer


# --- Execute Queries in Parallel ---
print(f"\n--- Starting Parallel Processing for {len(decomposed_queries)} Subqueries ---")
with concurrent.futures.ThreadPoolExecutor() as executor:
    # Submit tasks
    future_to_query = {executor.submit(process_subquery, query, i): i for i, query in enumerate(decomposed_queries)}

    # Collect results as they complete
    for future in concurrent.futures.as_completed(future_to_query):
        query_index_completed = future_to_query[future]
        try:
            result_tuple = future.result() # result is (query_index, query, final_answer)
            results.append(result_tuple)
            print(f"--- Completed Task for Subquery {query_index_completed + 1} ---")
        except Exception as exc:
            print(f"--- Subquery {query_index_completed + 1} generated an exception: {exc} ---")
            # Store exception information if needed
            results.append((query_index_completed, decomposed_queries[query_index_completed], f"Failed with exception: {exc}"))

# Sort results based on the original query index to maintain order
results.sort(key=lambda x: x[0])

# --- Write Combined Results ---
print(f"\n--- Writing Combined Answers to {combined_answers_output_path} ---")
try:
    with open(combined_answers_output_path, "w", encoding="utf-8") as f:
        f.write("Original Research Question: " + INITIAL_RESEARCH_QUESTION + "\n")
        f.write(f"Refined Goal: {overall_goal}\n\n")
        f.write("--- Decomposed Queries and Answers ---\n\n")
        for i, query, final_answer in results:
            f.write(f"Subquery {i+1}: {query}\n")
            f.write(f"Final Answer:\n{final_answer}\n\n")
            f.write("-" * 20 + "\n\n")
    print(f"Combined answers successfully written.")
except IOError as e:
    print(f"Error writing combined answers file: {e}")

print("\n--- Pipeline Finished ---")

