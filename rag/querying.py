import os
import re
import traceback
from typing import List, Dict, Tuple, Optional

# --- Add google.genai import for type checking ---
try:
    from google import genai
    GENAI_AVAILABLE = True
except ImportError:
    genai = None # Define genai as None if import fails
    GENAI_AVAILABLE = False

# --- Local Imports ---
from rag.utils import count_tokens, truncate_text
# Import the llm_interface module itself
from rag import llm_interface
from rag.retrieval import (
    retrieve_chunks_vector,
    retrieve_chunks_bm25,
    combine_results_rrf,
    rerank_chunks,
    SENTENCE_TRANSFORMERS_AVAILABLE # Check reranker availability
)
from rag.config import (
    SUBQUERY_MODEL,
    CHAT_MODEL,
    RERANKER_MODEL,
    DEFAULT_RERANK_CANDIDATE_COUNT,
    DEFAULT_TOP_K,
    EMBEDDING_MODEL # Needed for client checks
)

# --- Retrieval and Reranking Logic ---
def retrieve_and_rerank_chunks(
    query: str,
    db_path: str,
    collection_name: str,
    top_k: int = DEFAULT_TOP_K,
    reranker_model: Optional[str] = RERANKER_MODEL,
    rerank_candidate_count: int = DEFAULT_RERANK_CANDIDATE_COUNT,
    execution_mode: str = "retrieval_only" # Default mode for this function
) -> List[Dict]:
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
        vector_results = retrieve_chunks_vector(query, db_path, collection_name, fetch_k, execution_mode=execution_mode)
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
                top_k # Rerank and return the final top_k
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
    final_chunks_list.sort(key=lambda c: c.get('rerank_score', c.get('rrf_score', -float('inf'))), reverse=True)

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
    # Ensure the path is relative to the script location or use an absolute path
    prompt_file_path = os.path.join(os.path.dirname(__file__), prompt_file)
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
    try:
        with open('final_prompt.txt', 'w', encoding='utf-8') as f:
            f.write(prompt)
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


# --- Iterative RAG Query ---
def iterative_rag_query(initial_query: str, db_path: str, collection_name: str,
                        top_k: int = DEFAULT_TOP_K,
                        subquery_model: str = SUBQUERY_MODEL,
                        answer_model: str = CHAT_MODEL,
                        reranker_model: Optional[str] = RERANKER_MODEL,
                        rerank_candidate_count: int = DEFAULT_RERANK_CANDIDATE_COUNT,
                        execution_mode: str = "query") -> str:
    """Performs iterative RAG with subquery generation, hybrid retrieval, RRF, and optional reranking."""

    # --- Updated Client Checks using module namespace ---
    print(f"DEBUG (querying.iterative_rag_query): Checking client...")
    # Access gemini_client via the imported module
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


    # If we reach here, the client library is available and the client object seems valid.
    query_embed_client_ok = True
    subquery_client_ok = True # Assume ok, specific model errors handled later
    answer_client_ok = True   # Assume ok, specific model errors handled later

    # Check subquery model only if it's actually configured and needed
    if subquery_model and not subquery_client_ok:
        # This check might be redundant now but kept for clarity
        return f"[Error: Client for subquery model '{subquery_model}' not available or invalid.]"
    # ---------------------

    # 1. Generate Subqueries (if subquery model is available)
    all_queries = [initial_query]
    if subquery_model: # Check if a model is configured
        # Use generate_subqueries from llm_interface
        generated_subqueries = llm_interface.generate_subqueries(initial_query, model=subquery_model)
        if generated_subqueries and generated_subqueries != [initial_query]:
            print("--- Generated Subqueries ---")
            for idx, subq in enumerate(generated_subqueries): print(f"  {idx+1}. {subq}")
            all_queries.extend(generated_subqueries)
        else:
            print("--- Using only the initial query (no distinct subqueries generated or subquery model unavailable) ---")
    else:
        print("--- Skipping subquery generation (no subquery model specified) ---")


    # 2. Retrieve and Rank Chunks for ALL queries combined
    all_final_chunks = []
    processed_chunk_ids = set() # Keep track of processed chunks to avoid duplicates from subqueries

    # Determine fetch_k for retrieval based on reranking needs
    fetch_k_retrieval = rerank_candidate_count if reranker_model and SENTENCE_TRANSFORMERS_AVAILABLE else top_k
    fetch_k_retrieval = max(fetch_k_retrieval, top_k)

    print(f"\n--- Retrieving & Ranking for {len(all_queries)} queries (Initial fetch_k={fetch_k_retrieval}) ---")
    for q_idx, current_query in enumerate(all_queries):
        print(f"\nProcessing Query {q_idx+1}/{len(all_queries)}: \"{current_query[:100]}...\"")
        # Use the new function for retrieval and ranking for *each* query
        # Note: We retrieve potentially more (fetch_k_retrieval) and rerank down to top_k *per query* here.
        # We might want to adjust this logic later (e.g., combine all results *before* final reranking)
        # but for now, this keeps it simpler.
        query_chunks = retrieve_and_rerank_chunks(
            query=current_query,
            db_path=db_path,
            collection_name=collection_name,
            top_k=top_k, # Get top_k *per query* after potential reranking
            reranker_model=reranker_model,
            rerank_candidate_count=fetch_k_retrieval, # Fetch enough candidates for reranking
            execution_mode=f"{execution_mode}_subquery_{q_idx+1}"
        )
        # Add unique chunks to the overall list
        for chunk in query_chunks:
            chunk_id = chunk.get('chunk_id')
            if chunk_id and chunk_id not in processed_chunk_ids:
                all_final_chunks.append(chunk)
                processed_chunk_ids.add(chunk_id)

    if not all_final_chunks:
        return "No relevant chunks found across all queries after retrieval and potential re-ranking."

    # Re-sort the combined list based on the best score (rerank or rrf)
    all_final_chunks.sort(key=lambda c: c.get('rerank_score', c.get('rrf_score', -float('inf'))), reverse=True)
    # Optionally limit the total number of chunks used for the final answer
    final_chunks_for_answer = all_final_chunks[:top_k]

    # 3. Process Final Chunks & Generate Context String
    print(f"\n--- Processing Top {len(final_chunks_for_answer)} Combined Chunks for Context ---")
    context_parts = []
    # Use the combined and re-sorted list
    for chunk in final_chunks_for_answer:
        content = chunk.get('contextualized_text', '').strip() or chunk.get('text', '')
        context_parts.append(
            f"Source Document: {chunk.get('file_name', 'N/A')} [Chunk #{chunk.get('chunk_number', '?')}]\n"
            f"Content:\n{content}"
        )
    combined_context = "\n\n---\n\n".join(context_parts)

    # 4. Generate Final Answer
    print("\n--- Generating Final Answer (Iterative Hybrid Retrieval) ---")
    # Pass the combined list for citation generation
    final_answer = generate_answer(initial_query, combined_context, final_chunks_for_answer, model=answer_model)
    return final_answer


# --- Direct Query Index ---
def query_index(query: str, db_path: str, collection_name: str,
                top_k: int = DEFAULT_TOP_K,
                answer_model: str = CHAT_MODEL,
                reranker_model: Optional[str] = RERANKER_MODEL,
                rerank_candidate_count: int = DEFAULT_RERANK_CANDIDATE_COUNT,
                execution_mode: str = "query_direct") -> str:
    """Performs direct hybrid query (no subqueries), RRF, optional reranking, and answer generation."""
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
        collection_name=collection_name,
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
