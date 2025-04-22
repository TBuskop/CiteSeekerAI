import os
import re
import traceback
import json # <-- Add json import
from typing import List, Dict, Tuple, Optional

# --- Add google.genai import for type checking ---
try:
    from google import genai
    GENAI_AVAILABLE = True
except ImportError:
    genai = None # Define genai as None if import fails
    GENAI_AVAILABLE = False

import sys
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# --- Local Imports ---
from src.my_utils.utils import count_tokens, truncate_text
# Import the llm_interface module itself
from src.my_utils import llm_interface
from src.rag.retrieval import (
    retrieve_chunks_vector,
    retrieve_chunks_bm25,
    combine_results_rrf,
    rerank_chunks,
    SENTENCE_TRANSFORMERS_AVAILABLE, # Check reranker availability
)
from config import (
    SUBQUERY_MODEL,
    SUBQUERY_MODEL_SIMPLE,
    CHAT_MODEL,
    RERANKER_MODEL,
    DEFAULT_RERANK_CANDIDATE_COUNT,
    DEFAULT_TOP_K,
    EMBEDDING_MODEL, # Needed for client checks
    HYPE, HYPE_SUFFIX, HYPE_SOURCE_COLLECTION_NAME # Add HYPE_SUFFIX
)

from src.scrape.download_papers import remove_references_section

# --- Retrieval and Reranking Logic ---
def retrieve_and_rerank_chunks(
    query: str,
    db_path: str,
    collection_name: str,
    top_k: int = DEFAULT_TOP_K,
    reranker_model: Optional[str] = RERANKER_MODEL,
    rerank_candidate_count: int = DEFAULT_RERANK_CANDIDATE_COUNT,
    execution_mode: str = "retrieval_only", # Default mode for this function
    abstracts: Optional[bool] = True, # Flag to indicate if we are looking at abstracts of paper chunks
) -> List[Dict]:
    # Switch to Hype or Abstract collection based on config
    effective_collection_name = f"{collection_name}{HYPE_SUFFIX}" if HYPE else collection_name
    if HYPE:
        print(f"Info: retrieve_and_rerank_chunks using Hype collection '{effective_collection_name}'")
    """
    Performs hybrid retrieval (vector + BM25), RRF combination, and optional reranking.

    Args:
        query (str): The search query.
        db_path (str): Path to the ChromaDB database directory.
        collection_name (str): Name of the ChromaDB collection.
        top_k (int): The final number of chunks to return after ranking/reranking.
        reranker_model (Optional[str]): The model name for reranking, or None to skip.
        rerank_candidate_count (int): The number of candidates to fetch for reranking.
        execution_mode (str): Identifier for the execution context (e.g., for logging/tracking).

    Returns:
        List[Dict]: A list of the final ranked/reranked chunk dictionaries, sorted by relevance.
                     Returns an empty list if no chunks are found or an error occurs.
    """
    # Determine how many candidates to fetch initially
    fetch_k = rerank_candidate_count if reranker_model and SENTENCE_TRANSFORMERS_AVAILABLE else top_k
    fetch_k = max(fetch_k, top_k) # Ensure we fetch at least top_k

    print(f"\n--- Retrieving Top {fetch_k} Initial Candidates (Hybrid Vector + BM25) ---")
    print(f"  Query: \"{query[:100]}...\"")

    # 1. Retrieve Chunks (Hybrid)
    try:
        vector_results = retrieve_chunks_vector(query, db_path, effective_collection_name, fetch_k, execution_mode=execution_mode)
        bm25_results = retrieve_chunks_bm25(query, db_path, collection_name, fetch_k)
    except Exception as e:
        print(f"Error during initial retrieval: {e}")
        traceback.print_exc()
        return []

    # 2. Deduplicate and Combine using RRF (Handle potential empty results)
    print(f"\n--- Combining {len(vector_results)} Vector & {len(bm25_results)} BM25 results via RRF ---")

    # Deduplicate vector results (handle potential missing keys gracefully)
    deduped_vector_results_dict: Dict[str, Dict] = {}
    for chunk_meta in vector_results:
        chunk_id = chunk_meta.get('chunk_id')
        if not chunk_id: continue
        current_dist = chunk_meta.get('vector_distance', float('inf'))
        if chunk_id not in deduped_vector_results_dict or current_dist < deduped_vector_results_dict[chunk_id].get('vector_distance', float('inf')):
             deduped_vector_results_dict[chunk_id] = chunk_meta
    deduped_vector_results = list(deduped_vector_results_dict.values())

    # Deduplicate BM25 results
    deduped_bm25_results_dict: Dict[str, float] = {}
    for chunk_id, score in bm25_results:
         if chunk_id and (chunk_id not in deduped_bm25_results_dict or score > deduped_bm25_results_dict[chunk_id]):
              deduped_bm25_results_dict[chunk_id] = score
    deduped_bm25_results = list(deduped_bm25_results_dict.items())

    print(f"Unique Vector candidates: {len(deduped_vector_results)}, Unique BM25 candidates: {len(deduped_bm25_results)}")

    if not deduped_vector_results and not deduped_bm25_results:
        print("No candidates found from either vector or BM25 search.")
        return []

    try:
        combined_chunks_rrf = combine_results_rrf(
            deduped_vector_results, deduped_bm25_results, db_path, collection_name, execution_mode=execution_mode
            # Note: combine_results_rrf internally handles appending HYPE_SUFFIX based on config.HYPE
        )
        # Limit to fetch_k *before* reranking
        combined_chunks_rrf = combined_chunks_rrf[:fetch_k]
    except Exception as e:
        print(f"Error during RRF combination: {e}")
        traceback.print_exc()
        return []

    if not combined_chunks_rrf:
        print("No chunks remaining after RRF combination.")
        return []

    # 3. Optional Re-ranking Step
    if reranker_model and SENTENCE_TRANSFORMERS_AVAILABLE:
        print(f"\n--- Reranking Top {len(combined_chunks_rrf)} RRF Candidates using {reranker_model} ---")
        try:
            final_chunks_list = rerank_chunks(
                query,
                combined_chunks_rrf, # Pass the RRF combined list
                reranker_model,
                top_k, # Rerank and return the final top_k
                abstracts,
            )
        except Exception as e:
            print(f"Error during reranking: {e}")
            traceback.print_exc()
            # Fallback to RRF results if reranking fails
            print("Warning: Reranking failed. Falling back to RRF results.")
            final_chunks_list = combined_chunks_rrf[:top_k]
    else:
        final_chunks_list = combined_chunks_rrf[:top_k] # Take top_k from RRF results
        if reranker_model and not SENTENCE_TRANSFORMERS_AVAILABLE:
            print("\nInfo: Skipping re-ranking step (sentence-transformers not installed). Using RRF results.")
        elif not reranker_model:
            print("\nInfo: Skipping re-ranking step (no reranker model specified). Using RRF results.")

    if not final_chunks_list:
        print("No relevant chunks found after retrieval and potential re-ranking.")
        return []

    # 4. Sort final list (important even if not reranked, RRF provides scores)
    # Use 'ce_prob' from the reranker if available, otherwise fallback to 'rrf_score'
    final_chunks_list.sort(key=lambda c: c.get('ce_prob', c.get('rrf_score', -float('inf'))), reverse=True)


    print(f"\n--- Returning Top {len(final_chunks_list)} Final Chunks ---")
    return final_chunks_list


# --- Answer Generation ---
def generate_answer(query: str, combined_context: str, retrieved_chunks: List[dict],
                    model: str = CHAT_MODEL) -> str:
    """Generate answer using context and chunks, citing sources."""
    if not combined_context or not combined_context.strip():
         return "Could not generate an answer: no relevant context found."

    # Sort chunks for citation consistency (e.g., by final score: rerank > rrf)
    retrieved_chunks.sort(key=lambda c: c.get('rerank_score', c.get('rrf_score', -float('inf'))), reverse=True)

    # Create a unique list of top-level references (e.g., by final score: rerank > rrf)
    unique_references = []
    seen_files = set()
    for chunk in retrieved_chunks:
        if chunk and 'cited_by' in chunk and chunk['doi'] not in seen_files:
            # formated ref should contain authors, title, year, source_title, doi, cited_by
            authors = chunk.get('authors', 'N/A')
            title = chunk.get('title', 'N/A')
            year = chunk.get('year', 'N/A')
            source_title = chunk.get('source_title', 'N/A')
            doi = chunk.get('doi', 'N/A')
            cited_by = chunk.get('cited_by', 'N/A')

            formatted_ref = f"{authors}, {title}, {year}, {source_title}, https://www.doi.org/{doi}, Cited by: {cited_by}"
            unique_references.append(f"- {formatted_ref}")
            seen_files.add(doi) # Track unique references

        if chunk and 'file_name' in chunk and chunk['file_name'] not in seen_files:
            unique_references.append(f"- {chunk['file_name']}")
            seen_files.add(chunk['file_name'])

    unique_reference_list_str = "\n".join(sorted(unique_references)) # Sort alphabetically

    # Truncate context based on model limit
    # Define model context limits (adjust as needed)
    MODEL_CONTEXT_LIMITS = {
        "gemini-1.5-flash": 1000000, "gemini-1.5-pro": 1000000, # Use effective limits
        "gemini-pro": 30720, "gemini-1.0-pro": 30720,
        "gpt-4": 8000, "gpt-4o": 128000, "gpt-3.5-turbo": 16000,
        "gemini-2.5-pro-exp-03-25": 1000000,
        "gemini-2.5-pro-preview-03-25": 1000000,
        # Add other models used
    }
    # Clean model name for lookup
    clean_model_name = model.split('/')[-1].split(':')[0] # Handle prefixes/suffixes
    model_token_limit = MODEL_CONTEXT_LIMITS.get(clean_model_name, 128000) # Default to 8k
    print("DEBUG: Model token limit for", clean_model_name, "is", model_token_limit)

    # Calculate max context tokens, leaving room for prompt, query, citations, answer
    # Be conservative: leave ~20-30% for overhead and answer
    max_context_tokens = int(model_token_limit * 0.70)

    context_tokens = count_tokens(combined_context, model=model)
    if context_tokens > max_context_tokens:
        print(f"Warning: Combined context ({context_tokens} tokens) exceeds estimated limit ({max_context_tokens}) for {model}. Truncating.")
        combined_context = truncate_text(combined_context, max_context_tokens, model=model)
        context_tokens = count_tokens(combined_context, model=model) # Recalculate

    # --- Construct the Final Prompt ---
    # Load system prompt from file
    system_prompt = "You are a helpful AI assistant. Answer the user's question based on the provided context and cite the sources used." # Default prompt
    prompt_file = 'final_answer_system_prompt.txt'
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    # Ensure the path is relative to the script location or use an absolute path
    prompt_file_path = os.path.join(PROJECT_ROOT, '..','llm_prompts', prompt_file)
    if os.path.exists(prompt_file_path):
        try:
            with open(prompt_file_path, 'r', encoding='utf-8') as f:
                system_prompt = f.read().strip()
        except Exception as e:
            print(f"Warning: Could not read system prompt file '{prompt_file_path}': {e}")
    else:
        print(f"Warning: System prompt file not found at '{prompt_file_path}'. Using default prompt.")


    prompt = (
        f"{system_prompt}\n\n"
        f"## Context from Retrieved Documents:\n---\n{combined_context}\n---\n\n"
        f"## Sources Available (Referenced in Context):\n{unique_reference_list_str}\n\n"
        f"## User's Question:\n{query}\n\n"
        f"## Answer:\n"
        f"(Provide a comprehensive answer to the user's question based *only* on the provided context. "
        f"Synthesize information from multiple sources if applicable. "
        f"Explicitly cite the relevant source file names (e.g., [file_name.txt]) after the information derived from them. "
        f"If the context does not contain the answer, state that clearly.)"
    )

    prompt_to_print = (
        f"## User's Question:\n{query}\n\n"
        f"## Context from Retrieved Documents:\n---\n{combined_context}\n---\n\n"
        f"## Sources Available (Referenced in Context):\n{unique_reference_list_str}\n\n"
    )

    # Estimate prompt tokens to calculate max answer tokens
    prompt_template = prompt.replace(combined_context, "{context}").replace(query, "{query}")
    prompt_base_tokens = count_tokens(prompt_template, model=model)
    prompt_total_tokens = prompt_base_tokens + context_tokens + count_tokens(query, model=model)

    # Calculate max tokens for the answer, leave a buffer
    answer_max_tokens = max(150, model_token_limit - prompt_total_tokens - 200) # Min 150 tokens, buffer 200
    # Apply a reasonable hard cap to answer length if needed
    answer_max_tokens = max(answer_max_tokens, 4096) # E.g., max 4k tokens for the answer itself

    print(f"Generating final answer using {model} (context tokens: ~{context_tokens}, prompt tokens: ~{prompt_total_tokens}, max answer tokens: {answer_max_tokens})...")
    # save prompt, query and context to file for debugging
    # find the project root directory
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    save_path = os.path.join(PROJECT_ROOT, 'data', 'output', 'final_prompt.txt')
    try:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(prompt_to_print)
    except Exception as e:
        print(f"Warning: Could not write final_prompt.txt: {e}")

    # Generate the answer using the LLM interface
    final_answer_raw = llm_interface.generate_llm_response(prompt, max_tokens=answer_max_tokens, temperature=0.1, model=model) # Low temp for factual answers

    # Basic check if generation failed or was blocked
    if final_answer_raw.startswith("[Error") or final_answer_raw.startswith("[Blocked"):
         print(f"Warning: Final answer generation failed or was blocked.")
         # Return a more informative error message
         return f"Failed to generate the final answer. Details: {final_answer_raw}"

    # Append the reference list for clarity
    final_answer_formatted = f"{final_answer_raw}\n\n**References Used:**\n{unique_reference_list_str}"

    return final_answer_formatted


# --- Query Decomposition ---
def query_decomposition(query: str, number_of_sub_queries=5,model: str = SUBQUERY_MODEL) -> List[str]:
    """
    Decomposes a complex query into a list of simpler sub-questions using an LLM.

    Args:
        query: The original complex query string.
        model: The LLM model to use for decomposition.

    Returns:
        A list of simpler sub-question strings. Returns the original query in a list
        if decomposition fails.
    """
    print(f"\n--- Decomposing Query using {model} ---")
    print(f"Original Query: \"{query}\"")

    # 1. Load the decomposition prompt
    system_prompt = "You are an expert query decomposition assistant." # Default
    prompt_file = 'query_decomposition.txt'
    # Use _SCRIPT_DIR defined earlier if available, otherwise recalculate
    try:
        script_dir = _SCRIPT_DIR
    except NameError:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_file_path = os.path.join(script_dir, '..', 'llm_prompts', prompt_file)

    if os.path.exists(prompt_file_path):
        try:
            with open(prompt_file_path, 'r', encoding='utf-8') as f:
                system_prompt = f.read().strip()
        except Exception as e:
            print(f"Warning: Could not read query decomposition prompt file '{prompt_file_path}': {e}. Using default.")
    else:
        print(f"Warning: Query decomposition prompt file not found at '{prompt_file_path}'. Using default.")

    # 2. Construct the full prompt for the LLM
    full_prompt = f"{system_prompt}\n\nUser Query: \"{query}\"\n\nmax_subqueries: {number_of_sub_queries}\n\n" \

    # 3. Call the LLM
    try:
        # Ensure clients are ready
        llm_interface.initialize_clients()
        # Use a low temperature for structured output, adjust max_tokens as needed
        response = llm_interface.generate_llm_response(
            prompt=full_prompt,
            model=model,
            temperature=1,
            max_tokens=2000 # Adjust based on expected complexity
        )

        # 4. Parse the JSON response
        # Clean potential markdown code block fences
        response_cleaned = re.sub(r"```json\n?|\n?```", "", response).strip()

        # Attempt to parse the JSON
        parsed_response = json.loads(response_cleaned)

        subqueries, overall_goal = parsed_response.get("subqueries", []), parsed_response.get("overall_goal", None)

        if isinstance(subqueries, list) and all(isinstance(q, str) for q in subqueries) and subqueries:
            print(f"--- Generated {len(subqueries)} Subqueries ---")
            # for i, sq in enumerate(subqueries):
            #     print(f"  {i+1}. {sq}")
            return subqueries, overall_goal
        else:
            print("Warning: LLM response did not contain a valid 'subqueries' list. Falling back to original query.")
            return [query]

    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse JSON response from LLM for query decomposition: {e}")
        print(f"LLM Raw Response:\n{response}")
        return [query] # Fallback
    except Exception as e:
        print(f"Error during query decomposition LLM call: {e}")
        traceback.print_exc()
        return [query] # Fallback


# --- Iterative RAG Query ---
def iterative_rag_query(initial_query: str, db_path: str, collection_name: str,
                        top_k: int = DEFAULT_TOP_K,
                        subquery_model: str = SUBQUERY_MODEL,
                        answer_model: str = CHAT_MODEL,
                        reranker_model: Optional[str] = RERANKER_MODEL,
                        rerank_candidate_count: int = DEFAULT_RERANK_CANDIDATE_COUNT,
                        execution_mode: str = "query",
                        use_hype: bool = False) -> str:
    """Performs iterative RAG with subquery generation, separate hybrid retrieval, RRF, and optional reranking."""

    # Determine effective collection name
    
    if use_hype:
        effective_collection_name = f"{collection_name}{HYPE_SUFFIX}" if HYPE else collection_name
        print(f"Info: iterative_rag_query using Hype collection '{effective_collection_name}'")
    else:
        effective_collection_name = collection_name

    # --- Updated Client Checks using module namespace ---
    print(f"DEBUG (querying.iterative_rag_query): Checking client...")
    # Access gemini_client via the imported module
    llm_interface.initialize_clients() # Ensure clients are initialized
    print(f"DEBUG (querying.iterative_rag_query): llm_interface.gemini_client type = {type(llm_interface.gemini_client)}")
    print(f"DEBUG (querying.iterative_rag_query): llm_interface.gemini_client is None = {llm_interface.gemini_client is None}")

    if not GENAI_AVAILABLE:
        return "[Error: Google Generative AI library not installed or failed to import. Cannot proceed with query.]"
    # Check the client instance via the module namespace
    if not isinstance(llm_interface.gemini_client, genai.Client):
        print(f"DEBUG (querying.iterative_rag_query): isinstance check FAILED.") # Add failure log
        return "[Error: Gemini client is not initialized or invalid. Check API key and initialization logs.]"
    else:
        print(f"DEBUG (querying.iterative_rag_query): isinstance check PASSED.") # Add success log

    # 1. Generate Subqueries (if subquery model is available)
    if subquery_model:
        subquery_result = llm_interface.generate_subqueries(initial_query, model=subquery_model)
        bm25_queries = subquery_result.get("bm25_queries", [initial_query])
        vector_search_queries = subquery_result.get("vector_search_queries", [initial_query])
        print(f"--- Generated {len(bm25_queries)} BM25 and {len(vector_search_queries)} Vector subqueries ---")
    else:
        bm25_queries = [initial_query]
        vector_search_queries = [initial_query]
        print("--- Skipping subquery generation ---")

    # 2. Retrieve Chunks Separately for BM25 and Vector Queries
    all_vector_results = []
    processed_vector_ids = set()
    fetch_k_per_query = rerank_candidate_count if reranker_model and SENTENCE_TRANSFORMERS_AVAILABLE else top_k
    fetch_k_per_query = max(fetch_k_per_query, top_k)
    
    print(f"\n--- Retrieving Chunks for Vector Queries (fetch_k={fetch_k_per_query}) ---")
    for idx, query in enumerate(vector_search_queries):
        print(f"Vector Query {idx+1}/{len(vector_search_queries)}: \"{query[:100]}...\"")
        try:
            results = retrieve_chunks_vector(query, db_path, effective_collection_name, fetch_k_per_query,
                                             execution_mode=f"{execution_mode}_vector_{idx+1}")
            for chunk in results:
                cid = chunk.get("chunk_id")
                if cid and cid not in processed_vector_ids:
                    all_vector_results.append(chunk)
                    processed_vector_ids.add(cid)
        except Exception as e:
            print(f"Error during vector retrieval for query: {query[:50]}...: {e}")
            traceback.print_exc()
    
    all_bm25_results = []
    processed_bm25_ids = set()
    print(f"\n--- Retrieving Chunks for BM25 Queries (fetch_k={fetch_k_per_query}) ---")
    for idx, query in enumerate(bm25_queries):
        print(f"BM25 Query {idx+1}/{len(bm25_queries)}: \"{query[:100]}...\"")
        try:
            results = retrieve_chunks_bm25(query, db_path, collection_name, fetch_k_per_query)
            for cid, score in results:
                if cid and cid not in processed_bm25_ids:
                    all_bm25_results.append((cid, score))
                    processed_bm25_ids.add(cid)
        except Exception as e:
            print(f"Error during BM25 retrieval for query: {query[:50]}...: {e}")
            traceback.print_exc()
    
    print(f"\n--- Combining {len(all_vector_results)} Vector & {len(all_bm25_results)} BM25 results via RRF ---")
    try:
        # Pass the *base* collection name; combine_results_rrf handles the suffix internally
        combined_chunks = combine_results_rrf(all_vector_results, all_bm25_results, db_path, collection_name, 
                                              execution_mode=execution_mode)
    except Exception as e:
        print(f"Error during RRF combination: {e}")
        traceback.print_exc()
        return "An error occurred while combining search results."
    
    if not combined_chunks:
        print("No chunks remaining after RRF combination.")
        return "Could not find relevant information after combining search results."
    
    # 3. Optional Re-ranking Step
    if reranker_model and SENTENCE_TRANSFORMERS_AVAILABLE:
        print(f"\n--- Reranking Top {len(combined_chunks)} Candidates using {reranker_model} ---")
        try:
            final_chunks = rerank_chunks(initial_query, combined_chunks, reranker_model, DEFAULT_RERANK_CANDIDATE_COUNT, abstracts=False)
            final_chunks = final_chunks[:top_k] # Limit to top_k after reranking
            # only keep final chunks with ce_prob > 0.4
            pre_filter_len = len(final_chunks)
            relevance_score_filter = 0.4
            final_chunks = [chunk for chunk in final_chunks if chunk.get('ce_prob', 0) > relevance_score_filter]
            post_filter_len = len(final_chunks)
            print(f"discarded because chunk with relevance score < {relevance_score_filter}: {pre_filter_len-post_filter_len}")

        except Exception as e:
            print(f"Error during reranking: {e}")
            traceback.print_exc()
            final_chunks = combined_chunks[:top_k]
    else:
        final_chunks = combined_chunks[:top_k]
    
    if not final_chunks:
        return "No relevant chunks found after retrieval and potential re-ranking."
    
    # 4. Process Final Chunks & Generate Context String, then generate answer
    print(f"\n--- Processing Top {len(final_chunks)} Final Chunks for Context ---")
    context_parts = []
    for chunk in final_chunks:
        content = chunk.get('contextualized_text', '').strip() or chunk.get('text', '')
        if chunk.get('cited_by') is not None:
            context_parts.append(
                f"{chunk.get('authors', 'N/A').split(';')[0]} et al., {chunk.get('year', 'N/A')}, Chunk #{chunk.get('chunk_number', '?')}\n"
                f"Content:\n{content}"
            )
        else:
            context_parts.append(
                f"Source Document: {chunk.get('file_name', 'N/A')} [Chunk #{chunk.get('chunk_number', '?')}]\n"
                f"Content:\n{content}"
            )
    combined_context = "\n\n---\n\n".join(context_parts)
    final_answer = generate_answer(initial_query, combined_context, final_chunks, model=answer_model)
    return final_answer


# --- Direct Query Index ---
def query_index(query: str, db_path: str, collection_name: str,
                top_k: int = DEFAULT_TOP_K,
                answer_model: str = CHAT_MODEL,
                reranker_model: Optional[str] = RERANKER_MODEL,
                rerank_candidate_count: int = DEFAULT_RERANK_CANDIDATE_COUNT,
                execution_mode: str = "query_direct") -> str:
    """Performs direct hybrid query (no subqueries), RRF, optional reranking, and answer generation."""
    # Determine effective collection name
    effective_collection_name = f"{collection_name}{HYPE_SUFFIX}" if HYPE else collection_name
    if HYPE:
        print(f"Info: query_index using Hype collection '{effective_collection_name}'")

    print(f"--- Running Direct Hybrid Query ---")
    print(f"Query: {query}")

    # --- Updated Client Checks using module namespace ---
    print(f"DEBUG (querying.query_index): Checking client...")
    # Access gemini_client via the imported module
    print(f"DEBUG (querying.query_index): llm_interface.gemini_client type = {type(llm_interface.gemini_client)}")
    print(f"DEBUG (querying.query_index): llm_interface.gemini_client is None = {llm_interface.gemini_client is None}")

    if not GENAI_AVAILABLE:
        return "[Error: Google Generative AI library not installed or failed to import. Cannot proceed with query.]"
    # Check the client instance via the module namespace
    if not isinstance(llm_interface.gemini_client, genai.Client):
        print(f"DEBUG (querying.query_index): isinstance check FAILED.") # Add failure log
        return "[Error: Gemini client is not initialized or invalid. Check API key and initialization logs.]"
    else:
        print(f"DEBUG (querying.query_index): isinstance check PASSED.") # Add success log

    # If we reach here, the client library is available and the client object seems valid.
    query_embed_client_ok = True
    answer_client_ok = True
    # ---------------------

    # 1. Retrieve and Rank Chunks using the new function
    final_chunks_list = retrieve_and_rerank_chunks(
        query=query,
        db_path=db_path,
        collection_name=collection_name, # Pass base name, function handles suffix
        top_k=top_k,
        reranker_model=reranker_model,
        rerank_candidate_count=rerank_candidate_count,
        execution_mode=execution_mode
    )

    if not final_chunks_list:
        return "No relevant chunks found after retrieval and potential re-ranking."

    # 2. Process Final Chunks & Generate Context
    # Sorting is already done by retrieve_and_rerank_chunks
    print(f"\n--- Processing Top {len(final_chunks_list)} Final Chunks (Direct Query) ---")
    context_parts = []
    for chunk in final_chunks_list:
        content = chunk.get('contextualized_text', '').strip() or chunk.get('text', '')
        context_parts.append(
            f"Source Document: {chunk.get('file_name', 'N/A')} [Chunk #{chunk.get('chunk_number', '?')}]\n"
            f"Content:\n{content}"
        )
    combined_context = "\n\n---\n\n".join(context_parts)

    # 3. Generate Final Answer
    print("\n--- Generating Final Answer (Direct Hybrid Query) ---")
    answer = generate_answer(query, combined_context, final_chunks_list, model=answer_model)
    return answer

def follow_up_query_refinement(intended_next_query: str, overall_goal: str, previous_queries: str, previous_answers: str) -> str:
    """
    Generates a follow-up query based on the previous answer.
    This is a placeholder function and should be implemented as needed.
    """
    # open the system prompt file
    system_prompt_path = os.path.join(_PROJECT_ROOT, 'llm_prompts', 'follow_up_query_refinement.txt')
    with open(system_prompt_path, 'r', encoding='utf-8') as f:
        system_prompt = f.read().strip() 

    # Construct the full prompt for the LLM. Place each query and answer togheter
    full_prompt = (
        f"{system_prompt}\n\n"
        f"Overall Goal: \"{overall_goal}\"\n\n"
    )
    # Add the previous queries and answers to the prompt
    for i, query in enumerate(previous_queries):
        full_prompt += f"Query {i+1}: \"{query}\"\n"
        if i < len(previous_answers):
            previous_answer = previous_answers[i]
            # remove the reference list from the answer
            previous_answer = remove_references_section(previous_answer)
            full_prompt += f"Answer {i+1}: \"{previous_answer}\"\n"

    full_prompt += f"Intended Next Query: \"{intended_next_query}\"\n\n"
    
    response = llm_interface.generate_llm_response(
            prompt=full_prompt,
            model=SUBQUERY_MODEL,
            temperature=1,
            max_tokens=100000 # Adjust based on expected complexity
        )
    
    new_subquery = response.strip()

    return new_subquery