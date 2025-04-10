import os
import sys
import traceback  # Added for error logging
from typing import List, Dict, Set, Tuple

# --- Add necessary imports ---
# Adjust path to import from the parent directory (rag)
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

# Updated imports from querying and retrieval
from rag.querying import SENTENCE_TRANSFORMERS_AVAILABLE  # Keep this check
from rag.retrieval import (  # Import specific retrieval functions
    retrieve_chunks_vector,
    retrieve_chunks_bm25,
    combine_results_rrf,
    rerank_chunks
)
from rag.config import (
    RERANKER_MODEL,
    DEFAULT_TOP_K,
    DEFAULT_RERANK_CANDIDATE_COUNT,
    SUBQUERY_MODEL,
)
from rag.chroma_manager import get_chroma_collection

# Import client initialization function and subquery generation
from rag.llm_interface import initialize_clients, GOOGLE_GENAI_AVAILABLE, generate_subqueries

DB_PATH = "./abstract_chroma_db"
COLLECTION_NAME = 'abstracts'


def extract_metadata(chunks: List[Dict]) -> List[Tuple[str, str, str]]:
    """Extracts Authors, Year, and DOI from chunk metadata, ensuring uniqueness."""
    extracted_info: Set[Tuple[str, str, str]] = set()
    print(f"DEBUG: Extracting metadata from {len(chunks)} chunks.")  # Add debug log
    for i, chunk in enumerate(chunks):
        if i < 3:
            print(f"  DEBUG: Chunk {i} keys: {list(chunk.keys())}")

        authors = chunk.get('authors', 'N/A')
        year = str(chunk.get('year', 'N/A'))
        doi = chunk.get('doi', 'N/A')

        unique_key_doi = doi if doi and doi != 'N/A' else None

        if unique_key_doi:
            info_tuple = (authors, year, doi)
            if info_tuple not in extracted_info:
                extracted_info.add(info_tuple)

    def sort_key(item):
        authors_str = item[0]
        year_str = item[1]
        try:
            year_int = int(year_str)
        except (ValueError, TypeError):
            year_int = -1
        return (-year_int, authors_str.lower())

    return sorted(list(extracted_info), key=sort_key)


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    initial_query = "Academic papers discussing how to assess the robustness of climate information and how to design effective policy and prioritise in uncertainty."
    top_k = 10  # Final number of unique documents to aim for
    rerank_candidates = DEFAULT_RERANK_CANDIDATE_COUNT  # Candidates for reranking *after* RRF
    db_path = DB_PATH
    collection_name = COLLECTION_NAME
    use_rerank = True
    execution_mode = "collect_abstracts"  # Base execution mode identifier

    print("Initializing API clients...")
    initialize_clients()

    # --- Subquery Generation ---
    bm25_queries = [initial_query]
    vector_search_queries = [initial_query]
    if SUBQUERY_MODEL and GOOGLE_GENAI_AVAILABLE:
        print(f"\n--- Generating Subqueries using {SUBQUERY_MODEL} ---")
        try:
            # generate_subqueries now returns a dict
            subquery_result = generate_subqueries(initial_query, model=SUBQUERY_MODEL)
            bm25_queries = subquery_result.get('bm25_queries', [initial_query])
            vector_search_queries = subquery_result.get('vector_search_queries', [initial_query])
            # Check if we got more than just the initial query back
            if len(bm25_queries) == 1 and bm25_queries[0] == initial_query and \
               len(vector_search_queries) == 1 and vector_search_queries[0] == initial_query:
                print("Using only the initial query (subquery generation failed or returned no distinct queries).")
            else:
                pass
        except Exception as e:
            print(f"Warning: Failed to generate subqueries: {e}. Proceeding with initial query only.")
            traceback.print_exc()  # Add traceback
            bm25_queries = [initial_query]
            vector_search_queries = [initial_query]
    else:
        print("\n--- Skipping subquery generation (no subquery model specified or client unavailable) ---")
    # --- End Subquery Generation ---

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

    print(f"\n--- Debugging: Inspecting Collection '{collection_name}' ---")
    try:
        debug_collection = get_chroma_collection(
            db_path=db_path,
            collection_name=collection_name,
            execution_mode="query"
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
                    print(f"    Has Embedding Flag: {meta.get('has_embedding', 'Not Set')}")
            else:
                print("  Could not peek into the collection or it's empty.")
        else:
            print("Collection appears to be empty.")
    except Exception as e:
        print(f"Error inspecting collection: {e}")
    print("--- End Debugging ---")

    # --- Retrieve Chunks Separately for BM25 and Vector Queries ---
    all_vector_results_raw: List[Dict] = []
    all_bm25_results_raw: List[Tuple[str, float]] = []
    processed_vector_chunk_ids = set()
    processed_bm25_chunk_ids = set()

    fetch_k_per_query = rerank_candidates if reranker_model_to_use else top_k
    fetch_k_per_query = max(fetch_k_per_query, top_k) * 2

    print(f"\n--- Retrieving Chunks (fetch_k_per_query={fetch_k_per_query}) ---")

    # Vector Retrieval Loop
    print(f"--- Vector Search Queries ({len(vector_search_queries)}) ---")
    for q_idx, query in enumerate(vector_search_queries):
        print(f"  Query {q_idx+1}/{len(vector_search_queries)}: \"{query[:100]}...\"")
        try:
            vector_results = retrieve_chunks_vector(
                query, db_path, collection_name, fetch_k_per_query,
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

    # BM25 Retrieval Loop
    print(f"--- BM25 Queries ({len(bm25_queries)}) ---")
    for q_idx, query in enumerate(bm25_queries):
        print(f"  Query {q_idx+1}/{len(bm25_queries)}: \"{query[:100]}...\"")
        try:
            bm25_results = retrieve_chunks_bm25(query, db_path, collection_name, fetch_k_per_query)
            for chunk_id, score in bm25_results:
                if chunk_id and chunk_id not in processed_bm25_chunk_ids:
                    all_bm25_results_raw.append((chunk_id, score))
                    processed_bm25_chunk_ids.add(chunk_id)
        except Exception as e:
            print(f"Error during BM25 retrieval for query '{query[:50]}...': {e}")
            traceback.print_exc()

    print(f"\n--- Combining {len(all_vector_results_raw)} Vector & {len(all_bm25_results_raw)} BM25 results via RRF ---")
    try:
        combined_chunks_rrf = combine_results_rrf(
            all_vector_results_raw, all_bm25_results_raw, db_path, collection_name,
            execution_mode=execution_mode
        )
    except Exception as e:
        print(f"Error during RRF combination: {e}")
        traceback.print_exc()
        combined_chunks_rrf = []

    if not combined_chunks_rrf:
        print("\nNo relevant documents found after combining retrieval results.")
        return

    if reranker_model_to_use:
        print(f"\n--- Reranking Top {len(combined_chunks_rrf)} RRF Candidates using {reranker_model_to_use} ---")
        try:
            candidates_to_rerank = combined_chunks_rrf[:rerank_candidates]
            final_ranked_chunks = rerank_chunks(
                initial_query,
                candidates_to_rerank,
                reranker_model_to_use,
                top_n=len(candidates_to_rerank)
            )
            if len(final_ranked_chunks) < len(combined_chunks_rrf):
                remaining_chunks = combined_chunks_rrf[len(final_ranked_chunks):]
                for chunk in remaining_chunks:
                    if 'rerank_score' not in chunk:
                        chunk['rerank_score'] = -float('inf')
                final_ranked_chunks.extend(remaining_chunks)
                final_ranked_chunks.sort(key=lambda c: c.get('rerank_score', -float('inf')), reverse=True)
        except Exception as e:
            print(f"Error during reranking: {e}")
            traceback.print_exc()
            print("Warning: Reranking failed. Falling back to RRF results.")
            final_ranked_chunks = sorted(combined_chunks_rrf, key=lambda c: c.get('rrf_score', 0.0), reverse=True)
    else:
        final_ranked_chunks = sorted(combined_chunks_rrf, key=lambda c: c.get('rrf_score', 0.0), reverse=True)

    final_unique_chunks_by_doi = []
    processed_dois = set()
    print(f"\n--- Deduplicating final {len(final_ranked_chunks)} ranked chunks by DOI ---")
    for chunk in final_ranked_chunks:
        doi = chunk.get('doi')
        if doi and doi != 'N/A' and doi not in processed_dois:
            final_unique_chunks_by_doi.append(chunk)
            processed_dois.add(doi)

    final_unique_chunks = final_unique_chunks_by_doi[:top_k]

    if not final_unique_chunks:
        print("\nNo relevant documents with unique DOIs found after final processing.")
        return

    print(f"\n--- Selected Top {len(final_unique_chunks)} unique documents based on DOI ---")

    print(f"\n--- Extracting Metadata from Final {len(final_unique_chunks)} Chunks ---")
    extracted_metadata = extract_metadata(final_unique_chunks)

    if not extracted_metadata:
        print("No metadata could be extracted from the final retrieved chunks.")
        return

    print("\n{:<60} {:<6} {:<40}".format("Authors", "Year", "DOI"))
    print("-" * 110)

    for authors, year, doi in extracted_metadata:
        display_authors = authors if len(authors) <= 58 else authors[:55] + "..."
        print("{:<60} {:<6} {:<40}".format(display_authors, year, doi))

    try:
        print("\nSaving abstracts to relevant_abstracts.txt...")

        with open("relevant_abstracts.txt", "w", encoding="utf-8") as outfile:
            for i, chunk in enumerate(final_unique_chunks):
                title = chunk.get('title', 'N/A')
                authors = chunk.get('authors', 'N/A')
                year = str(chunk.get('year', 'N/A'))
                cited_by = chunk.get('cited_by', 'N/A')
                source_title = chunk.get('source_title', 'N/A')
                doi = chunk.get('doi', 'N/A')

                doc_text = chunk.get('text', '')

                abstract = 'N/A'
                if doc_text and '\n' in doc_text:
                    try:
                        abstract_parts = doc_text.split('\n', 1)
                        if len(abstract_parts) > 1:
                            abstract = abstract_parts[1].strip()
                            if not abstract:
                                abstract = 'N/A (Empty after title)'
                        else:
                            abstract = 'N/A (Only title found in text field)'
                    except IndexError:
                        abstract = 'N/A (Split failed)'
                elif doc_text:
                    abstract = doc_text.strip()
                else:
                    abstract = 'N/A (Text field empty or missing)'

                outfile.write(f"Title: {title}\nAuthors: {authors}\nYear: {year}\nCited by: {cited_by}\nSource Title: {source_title}\nDOI: https://www.doi.org/{doi}\nAbstract: {abstract}\n\n")

        print("\nRelevant abstracts saved to relevant_abstracts.txt.")
    except Exception as e:
        print(f"Error writing abstracts to file: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
