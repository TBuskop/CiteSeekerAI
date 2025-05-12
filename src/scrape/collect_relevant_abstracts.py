import os
import sys
import traceback
from typing import List, Dict, Tuple, Optional

# --- Add necessary imports ---
# Adjust path to import from the parent directory (rag)
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

# Updated imports from querying and retrieval
from rag.querying import SENTENCE_TRANSFORMERS_AVAILABLE
from rag.retrieval import (
    retrieve_chunks_vector,
    retrieve_chunks_bm25,
    combine_results_rrf,
    rerank_chunks,
)
from config import (
    RERANKER_MODEL,
    DEFAULT_RERANK_CANDIDATE_COUNT,
    SUBQUERY_MODEL,
    SUBQUERY_MODEL_SIMPLE,
    HYPE_SUFFIX,
)
from rag.chroma_manager import get_chroma_collection
from my_utils.llm_interface import initialize_clients, GOOGLE_GENAI_AVAILABLE, generate_subqueries

# --- Helper Function ---
def extract_metadata(chunks: List[Dict]) -> List[Tuple[str, str, str, str, float, str]]:
    """Extracts Authors, Year, DOI, Title, Score, and Score Type from chunk metadata, ensuring uniqueness."""
    extracted_info: Dict[str, Tuple[str, str, str, str, float, str]] = {}
    print(f"DEBUG: Extracting metadata from {len(chunks)} chunks.")
    for i, chunk in enumerate(chunks):
        authors = chunk.get('authors', 'N/A')
        year = str(chunk.get('year', 'N/A'))
        doi = chunk.get('doi', 'N/A')
        if doi == "N/A":
            doi = chunk.get('original_chunk_id', 'N/A')
        title = chunk.get('title', 'N/A')

        # Get the best available score and its type
        score = chunk.get('ce_prob', chunk.get('rerank_score', chunk.get('rrf_score')))
        score_val = score if score is not None and score != -float('inf') else -float('inf') # Use -inf for sorting if no score

        # Determine score type based on which key was found and valid
        if 'ce_prob' in chunk and score_val != -float('inf'):
            score_type = "CE Prob"
        elif 'rerank_score' in chunk and score_val != -float('inf'):
            score_type = "Rerank"
        elif 'rrf_score' in chunk and score_val != -float('inf'):
            score_type = "RRF"
        else:
            score_type = "N/A"

        unique_key_doi = doi if doi and doi != 'N/A' else None

        if unique_key_doi:
            # Store the best score found for this DOI
            current_best_score = extracted_info.get(unique_key_doi, ('', '', '', '', -float('inf'), ''))[4]
            if score_val > current_best_score:
                extracted_info[unique_key_doi] = (authors, year, doi, title, score_val, score_type)

    # Convert dict values to list
    info_list = list(extracted_info.values())

    # Sort by score descending, then year descending, then authors ascending
    def sort_key(item):
        score_val = item[4]
        authors_str = item[0]
        year_str = item[1]
        try:
            year_int = int(year_str)
        except (ValueError, TypeError):
            year_int = -1 # Treat non-integer years as very old
        # Primary sort: score descending. Secondary: year descending. Tertiary: authors ascending.
        return (-score_val, -year_int, authors_str.lower())

    return sorted(info_list, key=sort_key)


# --- Refactored Functions ---

def generate_search_queries(initial_query: str) -> Tuple[List[str], List[str]]:
    """Generates BM25 and Vector Search queries, including subqueries if configured."""
    bm25_queries = [initial_query]
    vector_search_queries = [initial_query]

    if SUBQUERY_MODEL_SIMPLE and GOOGLE_GENAI_AVAILABLE:
        print(f"\n--- Generating Subqueries using {SUBQUERY_MODEL_SIMPLE} ---")
        try:
            subquery_result = generate_subqueries(initial_query, model=SUBQUERY_MODEL_SIMPLE)
            generated_bm25 = subquery_result.get('bm25_queries', [])
            generated_vector = subquery_result.get('vector_search_queries', [])

            if generated_bm25 and (len(generated_bm25) > 1 or generated_bm25[0] != initial_query):
                 bm25_queries = generated_bm25
            if generated_vector and (len(generated_vector) > 1 or generated_vector[0] != initial_query):
                 vector_search_queries = generated_vector

            if bm25_queries == [initial_query] and vector_search_queries == [initial_query]:
                 print("Using only the initial query (subquery generation failed or returned no distinct queries).")
            else:
                 print(f"Generated {len(bm25_queries)} BM25 queries and {len(vector_search_queries)} Vector queries.")

        except Exception as e:
            print(f"Warning: Failed to generate subqueries: {e}. Proceeding with initial query only.")
            traceback.print_exc()
            bm25_queries = [initial_query]
            vector_search_queries = [initial_query]
    else:
        print("\n--- Skipping subquery generation (no subquery model specified or client unavailable) ---")

    return bm25_queries, vector_search_queries

def configure_reranker(use_rerank: bool) -> Optional[str]:
    """Determines the reranker model to use based on configuration and availability."""
    reranker_model_to_use = None
    if not use_rerank:
        print("Reranking is disabled.")
    elif not SENTENCE_TRANSFORMERS_AVAILABLE:
        print("Warning: Reranker model specified but sentence-transformers library not found. Reranking will be skipped.")
    elif RERANKER_MODEL:
        reranker_model_to_use = RERANKER_MODEL
        print(f"Reranking enabled using model: {reranker_model_to_use}")
    else:
        print("No reranker model configured. Reranking will be skipped.")
    return reranker_model_to_use

def debug_inspect_collection(db_path: str, collection_name: str):
    """Prints debugging information about the Chroma collection."""
    print(f"\n--- Debugging: Inspecting Collection '{collection_name}' ---")
    try:
        debug_collection = get_chroma_collection(
            db_path=db_path,
            collection_name=collection_name,
            execution_mode="query_debug" # Unique mode for debugging
        )
        collection_count = debug_collection.count()
        print(f"Collection count: {collection_count}")
        if collection_count > 0:
            print("Peeking at first few items (metadata):")
            peek_result = debug_collection.peek(limit=5)
            if peek_result and peek_result.get('metadatas'):
                for i, meta in enumerate(peek_result['metadatas']):
                    print(f"  Item {i+1} ID: {peek_result['ids'][i]}")
                    print(f"    Metadata: {meta}")
            else:
                print("  Could not peek into the collection or it's empty.")
        else:
            print("Collection appears to be empty.")
    except Exception as e:
        print(f"Error inspecting collection: {e}")
        traceback.print_exc()
    print("--- End Debugging ---")

def retrieve_initial_chunks(
    vector_queries: List[str],
    bm25_queries: List[str],
    db_path: str,
    collection_name: str,
    fetch_k: int,
    execution_mode: str,
    use_hype: bool = False,
) -> Tuple[List[Dict], List[Tuple[str, float]]]:
    """Retrieves chunks using both vector search and BM25 for the given queries."""
    all_vector_results_raw: List[Dict] = []
    all_bm25_results_raw: List[Tuple[str, float]] = []
    processed_vector_chunk_ids = set()
    processed_bm25_chunk_ids = set()

    print(f"\n--- Retrieving Chunks (fetch_k_per_query={fetch_k}) ---")

    print(f"--- Vector Search Queries ({len(vector_queries)}) ---")
    for q_idx, query in enumerate(vector_queries):
        print(f"  Query {q_idx+1}/{len(vector_queries)}: \"{query[:100]}...\"")
        try:
            vector_results = retrieve_chunks_vector(
                query, db_path, collection_name, fetch_k,
                execution_mode=f"{execution_mode}_vector_subquery_{q_idx+1}"
            )
            for chunk_meta in vector_results:
                 unique_id = chunk_meta.get('doi') or chunk_meta.get('chunk_id')
                 if unique_id and unique_id != 'N/A' and unique_id not in processed_vector_chunk_ids:
                      all_vector_results_raw.append(chunk_meta)
                      processed_vector_chunk_ids.add(unique_id)
        except Exception as e:
            print(f"Error during vector retrieval for query '{query[:50]}...': {e}")
            traceback.print_exc()

    print(f"--- BM25 Queries ({len(bm25_queries)}) ---")
    for q_idx, query in enumerate(bm25_queries):
        print(f"  Query {q_idx+1}/{len(bm25_queries)}: \"{query[:100]}...\"")
        try:
            if collection_name.endswith(HYPE_SUFFIX):
                # remove from collection_name to avoid confusion
                collection_name_bm25 = collection_name.replace(HYPE_SUFFIX, '')
            else:
                collection_name_bm25 = collection_name
                
            bm25_results = retrieve_chunks_bm25(query, db_path, collection_name_bm25, fetch_k)
            for chunk_id, score in bm25_results:
                if chunk_id and chunk_id not in processed_bm25_chunk_ids:
                    all_bm25_results_raw.append((chunk_id, score))
                    processed_bm25_chunk_ids.add(chunk_id)
        except Exception as e:
            print(f"Error during BM25 retrieval for query '{query[:50]}...': {e}")
            traceback.print_exc()

    print(f"Retrieved {len(all_vector_results_raw)} unique vector results and {len(all_bm25_results_raw)} unique BM25 results.")
    return all_vector_results_raw, all_bm25_results_raw

def combine_and_rerank_results(
    vector_results: List[Dict],
    bm25_results: List[Tuple[str, float]],
    initial_query: str,
    db_path: str,
    collection_name: str,
    reranker_model: Optional[str],
    rerank_candidates_count: int,
    execution_mode: str
) -> List[Dict]:
    """Combines results using RRF and optionally reranks them."""
    if not vector_results and not bm25_results:
        print("\nNo results from either vector or BM25 retrieval to combine.")
        return []

    print(f"\n--- Combining {len(vector_results)} Vector & {len(bm25_results)} BM25 results via RRF ---")
    try:
        combined_chunks_rrf = combine_results_rrf(
            vector_results, bm25_results, db_path, collection_name,
            execution_mode=f"{execution_mode}_rrf"
        )
    except Exception as e:
        print(f"Error during RRF combination: {e}")
        traceback.print_exc()
        combined_chunks_rrf = []

    if not combined_chunks_rrf:
        print("RRF combination resulted in an empty list.")
        return []

    if reranker_model:
        print(f"\n--- Reranking Top {min(len(combined_chunks_rrf), rerank_candidates_count)} RRF Candidates using {reranker_model} ---")
        try:
            candidates_to_rerank = combined_chunks_rrf
            final_ranked_chunks = rerank_chunks(
                initial_query,
                candidates_to_rerank,
                reranker_model,
                top_n=rerank_candidates_count,
                abstracts=True
            )
            if len(final_ranked_chunks) < len(candidates_to_rerank):
                 print(f"Warning: Reranker returned {len(final_ranked_chunks)} items, expected up to {len(candidates_to_rerank)}.")
                 processed_reranked_ids = {c.get('chunk_id') for c in final_ranked_chunks}
                 remaining_chunks = [
                     chunk for chunk in combined_chunks_rrf
                     if chunk.get('chunk_id') not in processed_reranked_ids
                 ]
                 for chunk in remaining_chunks:
                     chunk['rerank_score'] = -float('inf')
                 final_ranked_chunks.extend(remaining_chunks)
                 final_ranked_chunks.sort(key=lambda c: c.get('rerank_score', -float('inf')), reverse=True)

        except Exception as e:
            print(f"Error during reranking: {e}")
            traceback.print_exc()
            print("Warning: Reranking failed. Falling back to RRF results sorted by RRF score.")
            final_ranked_chunks = sorted(combined_chunks_rrf, key=lambda c: c.get('rrf_score', 0.0), reverse=True)
    else:
         final_ranked_chunks = sorted(combined_chunks_rrf, key=lambda c: c.get('rrf_score', 0.0), reverse=True)

    return final_ranked_chunks


def deduplicate_and_finalize(ranked_chunks: List[Dict], top_k: int) -> List[Dict]:
    """Deduplicates chunks by DOI and selects the top K."""
    if not ranked_chunks:
        return []

    final_unique_chunks_by_doi = []
    processed_dois = set()
    print(f"\n--- Deduplicating final {len(ranked_chunks)} ranked chunks by DOI ---")
    for chunk in ranked_chunks:
        doi = chunk.get('doi')
        if doi == None:
            doi = chunk.get('original_chunk_id')
        if doi and doi != 'N/A' and doi not in processed_dois:
            final_unique_chunks_by_doi.append(chunk)
            processed_dois.add(doi)

    final_unique_chunks = final_unique_chunks_by_doi[:top_k]
    print(f"Selected Top {len(final_unique_chunks)} unique documents based on DOI.")
    return final_unique_chunks


def display_results(final_chunks: List[Dict]) -> List[str]:
    """Extracts metadata, prints it in a formatted table sorted by score, and returns DOIs."""
    if not final_chunks:
        print("\nNo final documents to display.")
        return []

    print(f"\n--- Extracting Metadata from Final {len(final_chunks)} Chunks ---")
    # Assuming extract_metadata now returns (authors, year, doi, title, score, score_type) tuples
    extracted_metadata = extract_metadata(final_chunks)

    if not extracted_metadata:
        print("No metadata could be extracted from the final retrieved chunks.")
        return []

    # Adjust header and separator width to include Score
    header_format = "{:<50} {:<6} {:<35} {:<60} {:<10} {:<8}"
    separator_width = 50 + 6 + 35 + 60 + 10 + 8 + (5 * 1) # Widths + spaces
    print("\n" + header_format.format("Authors", "Year", "DOI", "Title", "Score", "Type"))
    print("-" * separator_width)

    list_of_dois = []
    for authors, year, doi, title, score, score_type in extracted_metadata:
        display_authors = authors if len(authors) <= 48 else authors[:45] + "..."
        display_title = title if len(title) <= 58 else title[:55] + "..."
        score_str = f"{score:.3f}" if score != -float('inf') else "N/A"
        # Adjust print format to match header
        print(header_format.format(display_authors, year, doi, display_title, score_str, score_type))
        list_of_dois.append(doi)

    return list_of_dois


def save_abstracts_to_file(query,final_chunks: List[Dict], filename: str):
    """Saves the title, metadata, and abstract of the final chunks to a file."""
    if not final_chunks:
        print("\nNo abstracts to save.")
        return

    print(f"\nSaving abstracts to {filename}...")
    try:
        with open(filename, "w", encoding="utf-8") as outfile:
            # Write the query at the top of the file
            outfile.write(f"Query: {query}\n\n")
            for i, chunk in enumerate(final_chunks):
                title = chunk.get('title', 'N/A')
                authors = chunk.get('authors', 'N/A')
                year = str(chunk.get('year', 'N/A'))
                cited_by = chunk.get('cited_by', 'N/A')
                source_title = chunk.get('source_title', 'N/A')
                doi = chunk.get('doi', 'N/A')
                doc_text = chunk.get('text', '')

                # Get the best available score (CE Prob > Rerank Logit > RRF)
                score = chunk.get('ce_prob', chunk.get('rerank_score', chunk.get('rrf_score')))
                score_str = f"{score:.3f}" if score is not None and score != -float('inf') else "N/A" # Handle -inf score

                # Determine score type based on which key was found and valid
                if 'ce_prob' in chunk and score != -float('inf'):
                    score_type = "CE Prob" # Use CE Prob as the label
                elif 'rerank_score' in chunk and score != -float('inf'):
                     score_type = "Rerank" # Fallback label if old key exists
                else:
                    score_type = "RRF" # Default to RRF if no valid rerank score

                abstract = 'N/A'
                if doc_text and '\n' in doc_text:
                    try:
                        # Assuming the first line is title/metadata and the rest is abstract
                        abstract_parts = doc_text.split('\n', 1)
                        if len(abstract_parts) > 1:
                            abstract = abstract_parts[1].strip()
                            if not abstract:
                                abstract = 'N/A (Empty after title)'
                        else:
                             abstract = doc_text.strip() # If only one line, treat it as abstract? Or N/A?
                    except Exception as split_err:
                        print(f"Warning: Could not split text for abstract extraction in chunk {i}: {split_err}")
                        abstract = doc_text.strip() # Fallback to full text
                elif doc_text:
                    abstract = doc_text.strip() # Use full text if no newline

                outfile.write(f"--- Document {i+1} ({score_type} Score: {score_str}) ---\n") # Updated score type label
                outfile.write(f"Title: {title}\n")
                outfile.write(f"Authors: {authors}\n")
                outfile.write(f"Year: {year}\n")
                outfile.write(f"Cited by: {cited_by}\n")
                outfile.write(f"Source Title: {source_title}\n")
                outfile.write(f"DOI: https://www.doi.org/{doi}\n")
                outfile.write(f"Abstract: {abstract}\n\n")

        print(f"Relevant abstracts saved to {filename}.")
    except IOError as e:
        print(f"Error writing abstracts to file '{filename}': {e}")
        traceback.print_exc()
    except Exception as e:
         print(f"An unexpected error occurred during file writing: {e}")
         traceback.print_exc()

def find_relevant_dois_from_abstracts(
    initial_query: str,
    db_path: str,
    collection_name: str,
    top_k: int,
    use_rerank: bool,
    output_filename: str,
    rerank_candidates: int = DEFAULT_RERANK_CANDIDATE_COUNT,
    use_hype: bool = False
) -> List[str]:
    """Main function to find relevant DOIs based on the initial query."""
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print("Initializing API clients...")
    initialize_clients()

    config = {
        "initial_query": initial_query,
        "db_path": db_path,
        "collection_name": collection_name,
        "top_k": top_k,
        "use_rerank": use_rerank,
        "output_filename": output_filename,
        "rerank_candidates": rerank_candidates,
        "execution_mode": "collect_abstracts"
    }
    

    bm25_queries, vector_search_queries = generate_search_queries(config["initial_query"])

    reranker_model_to_use = configure_reranker(config["use_rerank"])

    debug_inspect_collection(config["db_path"], config["collection_name"])

    fetch_k_per_query = config["rerank_candidates"] if reranker_model_to_use else config["top_k"]
    fetch_k_per_query = max(fetch_k_per_query * 2, config["top_k"] * 2)

    vector_results, bm25_results = retrieve_initial_chunks(
        vector_search_queries,
        bm25_queries,
        config["db_path"],
        config["collection_name"],
        fetch_k=fetch_k_per_query,
        execution_mode=config["execution_mode"],
        use_hype=use_hype
    )

    if not vector_results and not bm25_results:
        print("\nNo relevant documents found after initial retrieval.")
        return []

    ranked_chunks = combine_and_rerank_results(
        vector_results,
        bm25_results,
        config["initial_query"],
        config["db_path"],
        config["collection_name"],
        reranker_model_to_use,
        config["rerank_candidates"],
        config["execution_mode"]
    )

    if not ranked_chunks:
         print("\nNo relevant documents found after combining/ranking retrieval results.")
         return []

    final_unique_chunks = deduplicate_and_finalize(ranked_chunks, config["top_k"])

    if not final_unique_chunks:
        print("\nNo relevant documents with unique DOIs found after final processing.")
        return []

    relevant_dois = display_results(final_unique_chunks)
    print("\n--- Final DOIs ---")
    print(relevant_dois)
    save_abstracts_to_file(initial_query, final_unique_chunks, config["output_filename"])
    return relevant_dois