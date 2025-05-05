import os
import sys
import time

# --- Add project root to sys.path ---
# Ensures that modules from the 'src' directory can be imported correctly.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# --- Project Imports ---
# Import necessary functions from different modules within the project.
from src.scrape.collect_relevant_abstracts import find_relevant_dois_from_abstracts
from src.scrape.download_papers import download_dois
from src.scrape.chunk_new_dois import process_folder_for_chunks
from src.scrape.build_relevant_papers_db import build_relevant_db
from src.my_utils import llm_interface
from src.rag import workflows as rag_workflows
from src.rag.querying import query_decomposition, follow_up_query_refinement

import config


# --- Central Configuration ---
# Define base directories for data, downloads, and databases.
BASE_DATA_DIR = os.path.join(_PROJECT_ROOT, "data")
DOWNLOADS_DIR = os.path.join(BASE_DATA_DIR, "downloads")
FULL_TEXT_DIR = os.path.join(DOWNLOADS_DIR, "full_doi_texts")
CSV_DIR = os.path.join(DOWNLOADS_DIR, "csv")

# Abstract Database Configuration
ABSTRACT_DB_PATH = os.path.join(BASE_DATA_DIR, "databases", "abstract_chroma_db")
ABSTRACT_COLLECTION_NAME = "abstracts"
ACTIVE_ABSTRACT_COLLECTION = f"{ABSTRACT_COLLECTION_NAME}{config.HYPE_SUFFIX if config.HYPE else ''}"

# Full-Text Chunk Database Configuration
CHUNK_DB_PATH = os.path.join(BASE_DATA_DIR, "databases", "full_text_chunks_db")
CHUNK_COLLECTION_NAME = "paper_chunks_main"
CHUNK_SIZE = config.DEFAULT_MAX_TOKENS
CHUNK_OVERLAP = config.DEFAULT_CHUNK_OVERLAP

# Relevant Chunks Database Configuration
RELEVANT_CHUNKS_DB_PATH = os.path.join(BASE_DATA_DIR, "databases", "relevant_chunks_db")
RELEVANT_CHUNKS_COLLECTION_NAME = "relevant_paper_chunks"

# --- Research Question & Search Configuration ---
INITIAL_RESEARCH_QUESTION = "what are the main drivers of crop water use for virtual trade?"

# Abstract Collection Configuration
TOP_K_ABSTRACTS = config.TOP_K_ABSTRACTS
USE_RERANK_ABSTRACTS = True

# --- Query & Output Configuration ---
QUERY_TOP_K = config.DEFAULT_TOP_K
QUERY_RERANKER = config.RERANKER_MODEL
QUERY_RERANK_CANDIDATES = config.DEFAULT_RERANK_CANDIDATE_COUNT

# Output file/directory setup
RUN_TIMESTAMP = time.strftime('%Y%m%d_%H%M%S')
BASE_OUTPUT_DIR = os.path.join(BASE_DATA_DIR, "output")
COMBINED_ANSWERS_OUTPUT_FILENAME = os.path.join(BASE_OUTPUT_DIR, f"combined_answers_{RUN_TIMESTAMP}.txt")
QUERY_SPECIFIC_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, "query_specific")
RUN_SPECIFIC_OUTPUT_DIR = os.path.join(QUERY_SPECIFIC_OUTPUT_DIR, RUN_TIMESTAMP)

# --- Ensure Directories Exist ---
os.makedirs(FULL_TEXT_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)
os.makedirs(os.path.dirname(CHUNK_DB_PATH), exist_ok=True)
os.makedirs(os.path.dirname(RELEVANT_CHUNKS_DB_PATH), exist_ok=True)
os.makedirs(QUERY_SPECIFIC_OUTPUT_DIR, exist_ok=True)
os.makedirs(RUN_SPECIFIC_OUTPUT_DIR, exist_ok=True)


# --- Step 1: Decompose Initial Research Question ---
print("\n--- Step 1: Decomposing Research Question ---")
decomposed_queries, overall_goal = query_decomposition(
    query=INITIAL_RESEARCH_QUESTION,
    number_of_sub_queries=2,
    model=config.SUBQUERY_MODEL
)
if decomposed_queries:
    print(f"Overall Goal: {overall_goal}")
    print("Decomposed queries:")
    for i, query in enumerate(decomposed_queries):
        print(f"  Subquery {i+1}: {query}")
else:
    print("Failed to decompose the query. Exiting.")
    sys.exit(1)

# --- Initialize LLM Clients ---
print("\n--- Initializing LLM Clients ---")
try:
    llm_interface.initialize_clients()
    if not llm_interface.gemini_client:
        print("Warning: Gemini client initialization might have failed. Check API keys/config.")
except Exception as init_err:
    print(f"Error initializing LLM clients: {init_err}. Final query step might fail.")


# --- Define Worker Function for Processing Each Subquery ---
def process_subquery(query: str, query_index: int):
    """
    Processes a single subquery through the pipeline steps:
    Find relevant DOIs -> Download papers -> Chunk papers -> Build relevant chunk DB -> Query relevant chunks.

    Args:
        query (str): The subquery string to process.
        query_index (int): The index of the subquery (0-based).

    Returns:
        tuple: (query_index, original_query, final_answer_or_status)
    """
    print(f"\n[Query {query_index+1}] Starting processing for: '{query}'")

    # --- Step 3: Finding Relevant DOIs from Abstracts ---
    print(f"[Query {query_index+1}] Finding relevant DOIs from abstracts...")
    relevant_abstracts_output_filename_i = os.path.join(RUN_SPECIFIC_OUTPUT_DIR, f"relevant_abstracts_{query_index+1}.txt")

    relevant_doi_list = find_relevant_dois_from_abstracts(
        initial_query=query,
        db_path=ABSTRACT_DB_PATH,
        collection_name=ACTIVE_ABSTRACT_COLLECTION,
        top_k=TOP_K_ABSTRACTS,
        use_rerank=USE_RERANK_ABSTRACTS,
        output_filename=relevant_abstracts_output_filename_i,
        use_hype=config.HYPE,
    )
    print(f"[Query {query_index+1}] Found {len(relevant_doi_list)} relevant DOIs.")

    # --- Step 4: Downloading Full Texts ---
    if relevant_doi_list:
        print(f"[Query {query_index+1}] Downloading full texts for {len(relevant_doi_list)} DOIs...")
        download_dois(
            proposed_dois=relevant_doi_list,
            output_directory=FULL_TEXT_DIR
        )
        print(f"[Query {query_index+1}] Download step completed.")
    else:
        print(f"[Query {query_index+1}] No relevant DOIs found, skipping download.")

    # --- Step 5 & 6: Chunking and Building Relevant DB ---
    if relevant_doi_list:
        print(f"[Query {query_index+1}] Chunking downloaded documents...")
        process_folder_for_chunks(
            folder_path=FULL_TEXT_DIR,
            db_path=CHUNK_DB_PATH,
            collection_name=CHUNK_COLLECTION_NAME,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        print(f"[Query {query_index+1}] Chunking complete.")

        print(f"[Query {query_index+1}] Building relevant chunks database...")
        build_relevant_db(
            relevant_doi_list=relevant_doi_list,
            source_chunk_db_path=CHUNK_DB_PATH,
            source_chunk_collection_name=CHUNK_COLLECTION_NAME,
            abstract_db_path=ABSTRACT_DB_PATH,
            abstract_collection_name=ABSTRACT_COLLECTION_NAME,
            target_db_path=RELEVANT_CHUNKS_DB_PATH,
            target_collection_name=RELEVANT_CHUNKS_COLLECTION_NAME
        )
        print(f"[Query {query_index+1}] Relevant chunk database build complete.")
    else:
         print(f"[Query {query_index+1}] Skipping chunking and relevant DB build (no relevant DOIs).")

    # --- Step 8: Querying Relevant Chunks Database ---
    final_answer = None
    query_status = "Processing..."

    if relevant_doi_list and llm_interface.gemini_client:
        print(f"[Query {query_index+1}] Querying the relevant chunks database...")
        query_output_filename_i = os.path.join(RUN_SPECIFIC_OUTPUT_DIR, f"final_answer_{query_index+1}.txt")

        query_config = {
            "mode": "query",
            "db_path": RELEVANT_CHUNKS_DB_PATH,
            "collection_name": RELEVANT_CHUNKS_COLLECTION_NAME,
            "query": query,
            "top_k": QUERY_TOP_K,
            "reranker": QUERY_RERANKER,
            "rerank_candidates": QUERY_RERANK_CANDIDATES,
            "output_filename": query_output_filename_i,
            "output_dir": RUN_SPECIFIC_OUTPUT_DIR,
            "query_index": query_index,
            "subquery_model": config.SUBQUERY_MODEL,
            "answer_model": config.CHAT_MODEL,
            "use_hype": False,
        }
        try:
            final_answer = rag_workflows.run_query_mode(query_config)
            query_status = f"Query complete. Answer stored in {query_output_filename_i}."
            print(f"[Query {query_index+1}] {query_status}")
        except Exception as query_err:
            error_message = f"An error occurred during the final query step: {query_err}"
            print(f"[Query {query_index+1}] {error_message}")
            final_answer = f"Error: {error_message}"
            query_status = "Query failed."
    elif not relevant_doi_list:
        query_status = "Skipped (no relevant DOIs found)."
        print(f"[Query {query_index+1}] {query_status}")
        final_answer = query_status
    elif not llm_interface.gemini_client:
        query_status = "Skipped (LLM client initialization failed)."
        print(f"[Query {query_index+1}] {query_status}")
        final_answer = query_status

    print(f"[Query {query_index+1}] Finished processing.")
    return query_index, query, final_answer


# --- Execute Subqueries Sequentially ---
print(f"\n--- Starting Sequential Processing for {len(decomposed_queries)} Subqueries ---")
results = []

for i, original_subquery in enumerate(decomposed_queries):
    print(f"\n--- Processing Subquery {i + 1} of {len(decomposed_queries)} ---")
    current_query = original_subquery

    try:
        if i > 0 and results:
            print(f"[Query {i+1}] Refining query based on previous results...")
            previous_queries = [res[1] for res in results]
            previous_answers = [res[2] for res in results]
            refined_query = follow_up_query_refinement(
                current_query=original_subquery,
                overall_goal=overall_goal,
                previous_queries=previous_queries,
                previous_answers=previous_answers,
                model=config.SUBQUERY_MODEL
            )
            if refined_query and refined_query != original_subquery:
                 print(f"[Query {i+1}] Refined query: '{refined_query}'")
                 current_query = refined_query
            else:
                 print(f"[Query {i+1}] Query refinement did not change the query or failed.")

        result_tuple = process_subquery(current_query, i)
        results.append(result_tuple)
        print(f"--- Completed Processing for Subquery {i + 1} ---")

    except Exception as exc:
        error_msg = f"Subquery {i + 1} ('{current_query}') generated an unhandled exception: {exc}"
        print(f"--- {error_msg} ---")
        results.append((i, current_query, f"Failed with exception: {exc}"))

results.sort(key=lambda x: x[0])

# --- Write Combined Results ---
print(f"\n--- Writing Combined Answers to {COMBINED_ANSWERS_OUTPUT_FILENAME} ---")
try:
    with open(COMBINED_ANSWERS_OUTPUT_FILENAME, "w", encoding="utf-8") as f:
        f.write(f"Original Research Question: {INITIAL_RESEARCH_QUESTION}\n")
        f.write(f"Refined Overall Goal: {overall_goal}\n\n")
        f.write("--- Decomposed Queries and Final Answers ---\n\n")
        for index, processed_query, final_answer in results:
            original_query_text = decomposed_queries[index]
            f.write(f"--- Subquery {index+1} ---\n")
            f.write(f"Original Subquery: {original_query_text}\n")
            if processed_query != original_query_text:
                 f.write(f"Refined Subquery: {processed_query}\n")
            f.write(f"Final Answer/Status:\n{final_answer}\n\n")
    print(f"Combined answers successfully written to {COMBINED_ANSWERS_OUTPUT_FILENAME}")
except IOError as e:
    print(f"Error writing combined answers file: {e}")

print("\n--- DeepResearch Sequential Pipeline Finished ---")

