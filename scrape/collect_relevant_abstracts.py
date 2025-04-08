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
    # We might need EMBEDDING_MODEL if query embedding happens implicitly
    # within retrieve_chunks_vector, but client initialization handles API keys.
)
# --- Add import for get_chroma_collection ---
from rag.chroma_manager import get_chroma_collection

DB_PATH = "./abstract_chroma_db"
COLLECTION_NAME = 'abstracts'

# Import client initialization function
from rag.llm_interface import initialize_clients, GOOGLE_GENAI_AVAILABLE

def extract_metadata(chunks: List[Dict]) -> List[Tuple[str, str, str]]:
    """Extracts Authors, Year, and DOI from chunk metadata, ensuring uniqueness."""
    extracted_info: Set[Tuple[str, str, str]] = set()
    print(f"DEBUG: Extracting metadata from {len(chunks)} chunks.") # Add debug log
    for i, chunk in enumerate(chunks):
        # DEBUG: Print the keys of the first few chunk dictionaries
        if i < 3:
            print(f"  DEBUG: Chunk {i} keys: {list(chunk.keys())}")

        # Access fields directly from the chunk dictionary
        # Remove the nested 'metadata' access layer
        authors = chunk.get('authors', 'N/A') # Use 'authors' directly from chunk
        year = str(chunk.get('year', 'N/A')) # Use 'year' directly from chunk, ensure string
        doi = chunk.get('doi', 'N/A') # Use 'doi' directly from chunk

        # Use DOI as the primary key for uniqueness
        unique_key_doi = doi if doi and doi != 'N/A' else None

        # Only add if we have a DOI
        if unique_key_doi:
            # Store the tuple (authors, year, doi)
            info_tuple = (authors, year, doi)
            if info_tuple not in extracted_info:
                 extracted_info.add(info_tuple)
                 # DEBUG: Log successful extraction
                 # print(f"  DEBUG: Extracted: {info_tuple}")
            # else: print(f"  DEBUG: Duplicate DOI found: {doi}") # Optional debug
        # else:
            # print(f"Warning: Skipping chunk {chunk.get('chunk_id', f'index_{i}')} due to missing DOI.")


    # Sort the results, e.g., by year (descending) then authors
    # Handle potential 'N/A' or non-integer years during sorting
    def sort_key(item):
        authors_str = item[0]
        year_str = item[1]
        try:
            year_int = int(year_str)
        except (ValueError, TypeError):
            year_int = -1 # Place items with invalid years at the beginning/end
        return (-year_int, authors_str.lower()) # Sort descending year, then ascending author

    return sorted(list(extracted_info), key=sort_key)

def main():
    # change run path to current
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # --- Define Parameters Directly ---
    query = "Storylines and its impacts" # Replace with the desired query
    top_k = 20
    rerank_candidates = DEFAULT_RERANK_CANDIDATE_COUNT
    db_path = DB_PATH
    collection_name = COLLECTION_NAME # Use the constant
    use_rerank = True # Set to False to disable reranking

    # --- Initialize API Clients ---
    # Necessary for embedding the query implicitly via retrieve_chunks_vector
    print("Initializing API clients...")
    clients = initialize_clients()
   

    # Determine reranker model based on parameters
    reranker_model_to_use = None
    if not use_rerank:
        print("Reranking is disabled.")
    elif not SENTENCE_TRANSFORMERS_AVAILABLE:
        print("Warning: Reranker model specified but sentence-transformers library not found. Reranking will be skipped.")
    elif RERANKER_MODEL:
        reranker_model_to_use = RERANKER_MODEL
    else:
         print("No reranker model configured. Reranking will be skipped.")


    # --- Debugging: Inspect Collection Before Search ---
    print(f"\n--- Debugging: Inspecting Collection '{collection_name}' ---")
    try:
        # Get collection in 'query' mode (or 'embed' mode, shouldn't matter for peek/count)
        debug_collection = get_chroma_collection(
            db_path=db_path,
            collection_name=collection_name,
            execution_mode="query" # Or "embed"
        )
        collection_count = debug_collection.count()
        print(f"Collection count: {collection_count}")
        if collection_count > 0:
            print("Peeking at first few items (metadata):")
            peek_result = debug_collection.peek(limit=5) # Get up to 5 items
            if peek_result and peek_result.get('metadatas'):
                 for i, meta in enumerate(peek_result['metadatas']):
                      print(f"  Item {i+1} ID: {peek_result['ids'][i]}")
                      print(f"    Metadata: {meta}")
                      # Specifically check for has_embedding flag
                      print(f"    Has Embedding Flag: {meta.get('has_embedding', 'Not Set')}")
            else:
                 print("  Could not peek into the collection or it's empty.")
        else:
            print("Collection appears to be empty.")
    except Exception as e:
        print(f"Error inspecting collection: {e}")
    print("--- End Debugging ---")
    # --- End Debugging ---


    # --- Retrieve and Rank Chunks ---
    print(f"\nSearching for documents related to: '{query}'")
    retrieved_chunks = retrieve_and_rerank_chunks(
        query=query,
        db_path=db_path,
        collection_name=collection_name, # Use variable
        top_k=top_k,
        reranker_model=reranker_model_to_use,
        rerank_candidate_count=rerank_candidates,
        execution_mode="collect_abstracts"
    )

    if not retrieved_chunks:
        print("\nNo relevant documents found.")
        return

    # --- Extract and Print Metadata ---
    print(f"\n--- Relevant Documents (Top {len(retrieved_chunks)}) ---")
    extracted_metadata = extract_metadata(retrieved_chunks)

    if not extracted_metadata:
        print("No metadata could be extracted from the retrieved chunks.")
        return

    # Print header - Use the keys from the CSV/metadata
    print("\n{:<60} {:<6} {:<40}".format("Authors", "Year", "DOI"))
    print("-" * 110) # Separator line

    # Print extracted info
    for authors, year, doi in extracted_metadata:
        # Truncate long author lists for display if needed
        display_authors = authors if len(authors) <= 58 else authors[:55] + "..."
        print("{:<60} {:<6} {:<40}".format(display_authors, year, doi))

if __name__ == "__main__":
    main()
