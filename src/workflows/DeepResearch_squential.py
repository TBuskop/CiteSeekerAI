import os
import sys
import time

# --- Add project root to sys.path ---
# Ensures that modules from the 'src' directory can be imported correctly.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, os.pardir, os.pardir))
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
from chromadb.config import Settings # Import Settings


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
RELEVANT_CHUNKS_DB_SETTINGS = Settings(allow_reset=True) # Define settings for this DB

# --- Research Question & Search Configuration ---
INITIAL_RESEARCH_QUESTION = config.QUERY
QUERY_DECOMPOSITION_NR = config.QUERY_DECOMPOSITION_NR  # Number of sub-queries to generate from the main query 

# Abstract Collection Configuration
TOP_K_ABSTRACTS = config.TOP_K_ABSTRACTS
USE_RERANK_ABSTRACTS = True

# --- Query & Output Configuration ---
QUERY_TOP_K = config.DEFAULT_TOP_K
QUERY_RERANKER = config.RERANKER_MODEL
QUERY_RERANK_CANDIDATES = config.DEFAULT_RERANK_CANDIDATE_COUNT

# Output file/directory setup
# RUN_TIMESTAMP = time.strftime('%Y%m%d_%H%M%S')  # Default timestamp if no run_id provided # MODIFIED: Moved to function
BASE_OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "output")
# COMBINED_ANSWERS_OUTPUT_FILENAME = os.path.join(BASE_OUTPUT_DIR, f"combined_answers_{RUN_TIMESTAMP}.txt") # MODIFIED: Moved to function
QUERY_SPECIFIC_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, "query_specific")
# RUN_SPECIFIC_OUTPUT_DIR = os.path.join(QUERY_SPECIFIC_OUTPUT_DIR, RUN_TIMESTAMP) # MODIFIED: Moved to function

# --- Ensure Directories Exist ---
os.makedirs(FULL_TEXT_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)
os.makedirs(os.path.dirname(CHUNK_DB_PATH), exist_ok=True)
os.makedirs(os.path.dirname(RELEVANT_CHUNKS_DB_PATH), exist_ok=True)
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
os.makedirs(QUERY_SPECIFIC_OUTPUT_DIR, exist_ok=True)
# os.makedirs(RUN_SPECIFIC_OUTPUT_DIR, exist_ok=True) # MODIFIED: Moved to function


# --- Define Worker Function for Processing Each Subquery ---
def process_subquery(query: str, query_index: int, progress_callback=None, run_specific_output_dir=None, top_k_abstracts_val=None, top_k_chunks_val=None, min_citations_val=None): # MODIFIED: Added min_citations_val
    """
    Processes a single subquery through the pipeline steps:
    Find relevant DOIs -> Download papers -> Chunk papers -> Build relevant chunk DB -> Query relevant chunks.

    Args:
        query (str): The subquery string to process.
        query_index (int): The index of the subquery (0-based).
        progress_callback (function): Callback for progress updates.
        run_specific_output_dir (str): The output directory for this specific run.
        top_k_abstracts_val (int, optional): Override for TOP_K_ABSTRACTS.
        top_k_chunks_val (int, optional): Override for QUERY_TOP_K (for chunks).
        min_citations_val (int, optional): Override for MIN_CITATIONS_RELEVANT_PAPERS.

    Returns:
        tuple: (query_index, original_query, final_answer_or_status)
    """
    # helper to report subquery progress
    def log_progress_sub(msg, error_type=None):
        tag = f""
        payload = f"{tag} {msg}"
        if progress_callback:
            if error_type:
                # If this is an API error, send it with special type for UI handling
                progress_callback({"type": error_type, "message": payload})
            else:
                progress_callback(payload)
        print(f"{tag} {msg}")
    log_progress_sub(f"Starting processing for: '{query}'")

    # --- Step 3: Finding Relevant DOIs from Abstracts ---
    log_progress_sub("Finding relevant DOIs from abstracts...")
    relevant_abstracts_output_filename_i = os.path.join(run_specific_output_dir, f"relevant_abstracts_{query_index+1}.txt") # MODIFIED: Use passed run_specific_output_dir

    current_top_k_abstracts = top_k_abstracts_val if top_k_abstracts_val is not None else TOP_K_ABSTRACTS
    relevant_doi_list = find_relevant_dois_from_abstracts(
        initial_query=query,
        db_path=ABSTRACT_DB_PATH,
        collection_name=ACTIVE_ABSTRACT_COLLECTION,
        top_k=current_top_k_abstracts,
        use_rerank=USE_RERANK_ABSTRACTS,
        output_filename=relevant_abstracts_output_filename_i,
        use_hype=config.HYPE,
        min_citations_override=min_citations_val # Added
    )
    log_progress_sub(f"Found {len(relevant_doi_list)} relevant DOIs.")

    # --- Step 4: Downloading Full Texts ---
    if relevant_doi_list:
        # check the dois for which the full texts are already downloaded
        existing_dois = set()
        for filename in os.listdir(FULL_TEXT_DIR):
            if filename.endswith(".txt"):
                doi = filename[:-4]
                doi_original = doi.replace("doi:", "").replace("_", "/")
                # only add the doi if it is also in the relevant list
                if doi_original in relevant_doi_list:
                    existing_dois.add(doi_original)
        log_progress_sub(f"Downloading full texts not yet in databse for {len(relevant_doi_list) - len(existing_dois)}/{len(relevant_doi_list)} DOIs...")
        download_dois(
            proposed_dois=relevant_doi_list,
            output_directory=FULL_TEXT_DIR
        )
        log_progress_sub("Download step completed.")
    else:
        log_progress_sub("No relevant DOIs found, skipping download.")

    # --- Step 5 & 6: Chunking and Building Relevant DB ---
    if relevant_doi_list:
        log_progress_sub("Chunking downloaded documents...")
        process_folder_for_chunks(
            folder_path=FULL_TEXT_DIR,
            db_path=CHUNK_DB_PATH,
            collection_name=CHUNK_COLLECTION_NAME,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        log_progress_sub("Chunking complete.")

        log_progress_sub("Building relevant chunks database...")
        build_relevant_db(
            relevant_doi_list=relevant_doi_list,
            source_chunk_db_path=CHUNK_DB_PATH,
            source_chunk_collection_name=CHUNK_COLLECTION_NAME,
            abstract_db_path=ABSTRACT_DB_PATH,
            abstract_collection_name=ABSTRACT_COLLECTION_NAME,
            target_db_path=RELEVANT_CHUNKS_DB_PATH,
            target_collection_name=RELEVANT_CHUNKS_COLLECTION_NAME
        )
        log_progress_sub("Relevant chunk database build complete.")
    else:
         log_progress_sub("Skipping chunking and relevant DB build (no relevant DOIs).")

    # --- Step 8: Querying Relevant Chunks Database ---
    final_answer = None
    query_status = "Processing..."

    if relevant_doi_list and llm_interface.gemini_client:
        log_progress_sub("Querying the relevant chunks database...")
        query_output_filename_i = os.path.join(run_specific_output_dir, f"final_answer_{query_index+1}.txt") # MODIFIED: Use passed run_specific_output_dir

        current_query_top_k_chunks = top_k_chunks_val if top_k_chunks_val is not None else QUERY_TOP_K
        query_config = {
            "mode": "query",
            "db_path": RELEVANT_CHUNKS_DB_PATH,
            "collection_name": RELEVANT_CHUNKS_COLLECTION_NAME,
            "db_settings": RELEVANT_CHUNKS_DB_SETTINGS, # Pass the specific settings
            "query": query,
            "top_k": current_query_top_k_chunks,
            "reranker": QUERY_RERANKER,
            "rerank_candidates": QUERY_RERANK_CANDIDATES,
            "output_filename": query_output_filename_i,
            "output_dir": run_specific_output_dir, # MODIFIED: Use passed run_specific_output_dir
            "query_index": query_index,
            "subquery_model": config.SUBQUERY_MODEL,
            "answer_model": config.CHAT_MODEL,
            "use_hype": False,
        }
        try:
            final_answer = rag_workflows.run_query_mode(query_config)
            query_status = f"Query complete. Answer stored in {query_output_filename_i}."
            log_progress_sub(query_status)
        except Exception as query_err:
            error_message = f"An error occurred during the final query step: {query_err}"
            
            # Check if this is a Google API rate limit error
            error_str = str(query_err)
            if "Google API rate limit reached" in error_str or ("429" in error_str and "RESOURCE_EXHAUSTED" in error_str.upper()):
                # For rate limit errors, use the api_rate_limit error type to trigger special UI handling
                log_progress_sub(error_str, error_type="api_rate_limit")
            else:
                log_progress_sub(error_message)
                
            final_answer = f"Error: {error_message}"
            query_status = "Query failed."
    elif not relevant_doi_list:
        query_status = "Skipped (no relevant DOIs found)."
        log_progress_sub(query_status)
        final_answer = query_status
    elif not llm_interface.gemini_client:
        query_status = "Skipped (LLM client initialization failed)."
        log_progress_sub(query_status)
        final_answer = query_status

    log_progress_sub("Finished processing.")
    return query_index, query, final_answer


# --- Main Pipeline Function ---
def run_deep_research(question=None, query_numbers=None, progress_callback=None, run_id=None, top_k_abstracts_val=None, top_k_chunks_val=None, min_citations_val=None): # Added min_citations_val
    """
    Run the deep research pipeline.
    
    Args:
        question (str): The main research question to process
        query_numbers (int): Number of subqueries to generate
        progress_callback (function): Callback to report progress updates
        run_id (str): Optional ID to use for output files instead of generating a timestamp
        top_k_abstracts_val (int, optional): Number of abstracts to search.
        top_k_chunks_val (int, optional): Number of chunks to use for the answer.
        min_citations_val (int, optional): Minimum citations for papers to be considered relevant.
    """
    # --- Initialize Run-Specific Variables ---
    current_run_timestamp = run_id if run_id else time.strftime('%Y%m%d_%H%M%S')
    combined_answers_output_filename = os.path.join(BASE_OUTPUT_DIR, f"combined_answers_{current_run_timestamp}.txt")
    run_specific_output_dir = os.path.join(QUERY_SPECIFIC_OUTPUT_DIR, current_run_timestamp)

    # Ensure the run-specific directory exists
    os.makedirs(run_specific_output_dir, exist_ok=True)
    
    # Use the passed question or fall back to config.QUERY
    initial_research_question = question if question is not None else config.QUERY
    query_numbers = query_numbers if query_numbers is not None else config.QUERY_DECOMPOSITION_NR

    def log_progress_main(payload_type, data=None, message=None):
        if progress_callback:
            payload = {"type": payload_type}
            if data:
                payload["data"] = data
            if message:
                payload["message"] = message
            progress_callback(payload)
        if message: # also print simple messages
            print(message)

    # --- Step 1: Decompose Initial Research Question ---
    log_progress_main("status_update", message="\n--- Step 1: Decomposing Research Question ---")
    try:
        decomposed_queries, overall_goal, error_message = query_decomposition(
            query=initial_research_question,
            number_of_sub_queries=query_numbers,
            model=config.SUBQUERY_MODEL,
            output_dir=run_specific_output_dir
        )
    except Exception as exc:
        log_progress_main("status_update", message=f"Error during query decomposition. Exiting. Exception: {exc}")
        sys.exit(1)

    if error_message:
        # Check if the error_message is the specific detailed rate limit error
        if "Google API rate limit reached" in error_message and "Please ensure billing and credit card information" in error_message:
            # Reformat to the user-friendly version
            friendly_error_message = "Rate limit error: Google API rate limit reached. Please ensure billing and credit card information is configured for your Google Cloud project."
            log_progress_main("api_rate_limit", message=friendly_error_message)
        else:
            # Log other error messages as they are
            log_progress_main("api_rate_limit", message=error_message)
        return  # Stop further processing if rate limit error occurred

    if decomposed_queries:
        log_progress_main("initial_info", data={"overall_goal": overall_goal, "decomposed_queries": decomposed_queries, "initial_research_question": initial_research_question})
        print(f"Overall Goal: {overall_goal}")
        print("Decomposed queries:")
        for i, query in enumerate(decomposed_queries):
            print(f"  Subquery {i+1}: {query}")
    else:
        log_progress_main("status_update", message="Failed to decompose the query.")
        

    # --- Initialize LLM Clients ---
    log_progress_main("status_update", message="\n--- Initializing LLM Clients ---")
    try:
        llm_interface.initialize_clients()
        if not llm_interface.gemini_client:
            log_progress_main("status_update", message="Warning: Gemini client initialization might have failed. Check API keys/config.")
    except Exception as init_err:
        log_progress_main("status_update", message=f"Error initializing LLM clients: {init_err}. Final query step might fail.")

    # --- Execute Subqueries Sequentially ---
    log_progress_main("status_update", message=f"\n--- Starting Sequential Processing for {len(decomposed_queries)} Subqueries ---")
    results = []

    for i, original_subquery in enumerate(decomposed_queries):
        log_progress_main("status_update", message=f"\n--- Processing Subquery {i + 1} of {len(decomposed_queries)} ---")
        current_query = original_subquery

        try:
            if i > 0 and results:
                log_progress_main("status_update", message=f"Refining query based on previous results...")
                previous_queries = [res[1] for res in results]
                previous_answers = [res[2] for res in results]
                refined_query = follow_up_query_refinement(
                    intended_next_query=original_subquery,
                    overall_goal=overall_goal,
                    previous_queries=previous_queries,
                    previous_answers=previous_answers,
                    model=config.SUBQUERY_MODEL,
                    output_dir=run_specific_output_dir,
                    query_index=i
                )
                if refined_query and refined_query != original_subquery:
                     log_progress_main("status_update", message=f"Refined query: '{refined_query}'")
                     current_query = refined_query
                else:
                     log_progress_main("status_update", message=f"Query refinement did not change the query or failed.")
            
            # Pass the main log_progress_main to process_subquery if its internal logs should also go through the structured callback
            # For now, process_subquery uses its own print and a simpler progress_callback for its internal steps.
            # If process_subquery's callback needs to be structured, it would also need to adopt the dict payload.
            # For this change, we focus on reporting the *result* of process_subquery.
            
            # Simplified callback for process_subquery's internal steps, distinct from main progress
            def subquery_step_callback(msg):
                log_progress_main("status_update", message=f"[Subquery {i+1} Step] {msg}")

            result_tuple = process_subquery(
                current_query, 
                i, 
                progress_callback=subquery_step_callback, 
                run_specific_output_dir=run_specific_output_dir,
                top_k_abstracts_val=top_k_abstracts_val,
                top_k_chunks_val=top_k_chunks_val,
                min_citations_val=min_citations_val # Added
            )
            results.append(result_tuple)
            
            # Send structured update for this subquery's result
            log_progress_main("subquery_result", data={
                "index": i,
                "original_query": original_subquery,
                "processed_query": current_query,
                "answer": result_tuple[2]
            })
            log_progress_main("status_update", message=f"--- Completed Processing for Subquery {i + 1} ---")

        except Exception as exc:
            error_msg = f"Subquery {i + 1} ('{current_query}') generated an unhandled exception: {exc}"
            log_progress_main("status_update", message=f"--- {error_msg} ---")
            results.append((i, current_query, f"Failed with exception: {exc}"))
            # Optionally send this error as a subquery_result too, so UI knows it failed
            log_progress_main("subquery_result", data={
                "index": i,
                "original_query": original_subquery,
                "processed_query": current_query,
                "answer": f"Failed with exception: {exc}",
                "is_error": True
            })


    results.sort(key=lambda x: x[0])

    # --- Write Combined Results ---
    log_progress_main("status_update", message=f"\n--- Writing Combined Answers to {combined_answers_output_filename} ---")
    try:
        with open(combined_answers_output_filename, "w", encoding="utf-8") as f:
            f.write(f"## Original Research Question\n{initial_research_question}\n\n")
            f.write(f"## Refined Overall Goal\n{overall_goal}\n\n")
            f.write("### Decomposed Queries and Final Answers\n\n")
            for index, processed_query, final_answer in results:
                original_query_text = decomposed_queries[index]
                f.write(f"#### Subquery {index+1}\n\n")
                f.write(f"**Original Subquery:** {original_query_text}\n\n")
                if processed_query != original_query_text:
                     f.write(f"**Refined Subquery:** {processed_query}\n\n")
                f.write(f"**Final Answer:**\n{final_answer}\n\n")
                f.write("---\n\n")
        log_progress_main("status_update", message=f"Combined answers successfully written to {combined_answers_output_filename}")
    except IOError as e:
        log_progress_main("status_update", message=f"Error writing combined answers file: {e}")

    log_progress_main("status_update", message="\n--- DeepResearch Sequential Pipeline Finished ---")


if __name__ == "__main__":
    run_deep_research()

