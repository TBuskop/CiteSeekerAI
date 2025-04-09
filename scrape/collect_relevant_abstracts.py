import os
import sys
from typing import List, Dict, Set, Tuple

# --- Add necessary imports ---
# Adjust path to import from the parent directory (rag)
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

from rag.querying import retrieve_and_rerank_chunks, SENTENCE_TRANSFORMERS_AVAILABLE
from rag.config import (
    RERANKER_MODEL,
    DEFAULT_TOP_K,
    DEFAULT_RERANK_CANDIDATE_COUNT,
    SUBQUERY_MODEL,  # Import SUBQUERY_MODEL
)
from rag.chroma_manager import get_chroma_collection

DB_PATH = "./abstract_chroma_db"
COLLECTION_NAME = 'abstracts'

# Import client initialization function and subquery generation
from rag.llm_interface import initialize_clients, GOOGLE_GENAI_AVAILABLE, generate_subqueries  # Import generate_subqueries

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
    rerank_candidates = DEFAULT_RERANK_CANDIDATE_COUNT
    db_path = DB_PATH
    collection_name = COLLECTION_NAME
    use_rerank = True

    print("Initializing API clients...")
    clients = initialize_clients()  # Clients are initialized but not explicitly passed down currently

    # --- Subquery Generation ---
    all_queries = [initial_query]
    if SUBQUERY_MODEL and GOOGLE_GENAI_AVAILABLE:  # Check if model is configured and client available
        print(f"\n--- Generating Subqueries using {SUBQUERY_MODEL} ---")
        try:
            # Pass the model name explicitly
            generated_subqueries = generate_subqueries(initial_query, model=SUBQUERY_MODEL)
            if generated_subqueries and generated_subqueries != [initial_query]:
                print("Generated Subqueries:")
                for idx, subq in enumerate(generated_subqueries): print(f"  {idx+1}. {subq}")
                all_queries.extend(generated_subqueries)
            else:
                print("Using only the initial query (no distinct subqueries generated).")
        except Exception as e:
            print(f"Warning: Failed to generate subqueries: {e}. Proceeding with initial query only.")
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

    # --- Retrieve Chunks for All Queries ---
    all_retrieved_chunks = []
    processed_chunk_ids = set()  # To deduplicate chunks across queries

    # Determine fetch_k for retrieval based on reranking needs
    fetch_k_per_query = rerank_candidates if reranker_model_to_use else top_k
    fetch_k_per_query = max(fetch_k_per_query, top_k) * 2  # Fetch more candidates per query

    print(f"\n--- Retrieving & Ranking for {len(all_queries)} queries (Fetch k per query={fetch_k_per_query}) ---")
    for q_idx, current_query in enumerate(all_queries):
        print(f"\nProcessing Query {q_idx+1}/{len(all_queries)}: \"{current_query[:100]}...\"")
        retrieved_chunks_for_query = retrieve_and_rerank_chunks(
            query=current_query,
            db_path=db_path,
            collection_name=collection_name,
            top_k=fetch_k_per_query,
            reranker_model=reranker_model_to_use,
            rerank_candidate_count=rerank_candidates,
            execution_mode=f"collect_abstracts_subquery_{q_idx+1}"
        )

        # Add unique chunks to the overall list
        for chunk in retrieved_chunks_for_query:
            unique_id = chunk.get('doi') or chunk.get('chunk_id')
            if unique_id and unique_id != 'N/A' and unique_id not in processed_chunk_ids:
                all_retrieved_chunks.append(chunk)
                processed_chunk_ids.add(unique_id)

    if not all_retrieved_chunks:
        print("\nNo relevant documents found across all queries.")
        return

    # --- Combine and Re-sort/Limit Results ---
    all_retrieved_chunks.sort(key=lambda c: c.get('rerank_score', c.get('rrf_score', -float('inf'))), reverse=True)
    final_unique_chunks = all_retrieved_chunks[:top_k]
    print(f"\n--- Combined and selected Top {len(final_unique_chunks)} unique chunks across all queries ---")

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
                        abstract = doc_text.split('\n', 1)[1].strip()
                        if not abstract:
                            abstract = 'N/A (Empty after title)'
                    except IndexError:
                        abstract = 'N/A (Split failed)'
                elif doc_text:
                    abstract = 'N/A (No newline found in text field)'
                else:
                    abstract = 'N/A (Text field empty or missing)'

                outfile.write(f"Title: {title}\nAuthors: {authors}\nYear: {year}\nCited by: {cited_by}\nSource Title: {source_title}\nDOI: https://www.doi.org/{doi}\nAbstract: {abstract}\n\n")

        print("\nRelevant abstracts saved to relevant_abstracts.txt.")
    except Exception as e:
        print(f"Error writing abstracts to file: {e}")

if __name__ == "__main__":
    main()
