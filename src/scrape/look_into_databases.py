# peek in the scrape/abstract_chroma_db database and the scrape/data/chroma_dbs/full_text_chunks_db
# to see how many unique dois are in the database
import os
import chromadb
from typing import Set, Optional, Tuple, Dict # Added Dict
from tqdm import tqdm
import pprint # Added pprint

def count_unique_dois_chromadb_client(db_path: str, collection_name: str) -> Optional[Tuple[int, Set[str], Optional[Dict]]]:
    """
    Counts the number of unique DOIs, identifies metadata field names,
    and retrieves a sample metadata entry from a ChromaDB collection using the client.

    Args:
        db_path: Path to the ChromaDB database directory.
        collection_name: Name of the collection within the database.

    Returns:
        A tuple containing:
          - The count of unique DOIs (int).
          - A set of unique metadata field names found (Set[str]).
          - A sample metadata dictionary (Dict) or None if unavailable.
        Returns None if a critical error occurs. Returns (0, set(), None) if collection is empty or not found.
    """
    print(f"\nInspecting DB: '{db_path}', Collection: '{collection_name}'")
    if not os.path.isdir(db_path):
        print(f"Error: Database directory not found: {db_path}")
        return None

    unique_dois: Set[str] = set()
    metadata_fields: Set[str] = set() # To store unique metadata keys
    sample_metadata: Optional[Dict] = None # To store a sample metadata entry

    try:
        client = chromadb.PersistentClient(path=db_path)
        print("Client created.")

        try:
            collection = client.get_collection(name=collection_name)
            print(f"Collection '{collection_name}' retrieved.")
        except ValueError as e:
             if "Could not find collection" in str(e) or "doesn't exist" in str(e):
                 print(f"Collection '{collection_name}' not found in database '{db_path}'.")
                 return 0, set(), None # Return 0 count, empty set, and None sample
             else:
                 print(f"Error getting collection '{collection_name}': {e}")
                 return None # Other error getting collection
        except Exception as e:
             print(f"Unexpected error getting collection '{collection_name}': {e}")
             return None

        estimated_count = collection.count()
        if estimated_count == 0:
            print("Collection is empty.")
            return 0, set(), None # Return 0 count, empty set, and None sample

        # --- Get Sample Metadata for Field Names and Example ---
        try:
            sample = collection.peek(limit=1) # Fetch 1 item to inspect metadata structure
            if sample and sample.get('metadatas') and sample['metadatas']:
                first_metadata = sample['metadatas'][0]
                if first_metadata: # Check if metadata dictionary is not None
                    metadata_fields.update(first_metadata.keys())
                    sample_metadata = first_metadata # Store the sample
            print(f"Sample metadata fields found: {metadata_fields or 'None (or empty metadata in sample)'}")
            if sample_metadata:
                print("Sample metadata entry retrieved.")
            else:
                print("Could not retrieve a sample metadata entry.")
        except Exception as peek_err:
            print(f"Warning: Could not peek collection to get sample metadata: {peek_err}")
            print("Will attempt to discover fields during full metadata fetch.")
        # --- End Sample Metadata ---

        print(f"Fetching metadata for {estimated_count} items...")
        # Fetch metadata in batches to handle potentially large collections
        fetch_limit = 10000 # Adjust batch size as needed
        offset = 0
        fetched_count = 0
        with tqdm(total=estimated_count, desc="Fetching metadata", unit="chunks") as pbar:
            while True:
                try:
                    results = collection.get(limit=fetch_limit, offset=offset, include=['metadatas'])
                    if not results or not results.get('ids'):
                        break # No more results

                    metadatas = results.get('metadatas', [])
                    for meta in metadatas:
                        if meta: # Check if metadata dictionary is not None
                            # Update DOI set
                            doi = meta.get('doi')
                            if doi:
                                # Replace all '/' with '_' in the DOI
                                doi = doi.replace('/', '_')
                                unique_dois.add(doi)
                            # Update metadata fields set (discover fields missed by peek)
                            metadata_fields.update(meta.keys())

                    fetched_count += len(results['ids'])
                    pbar.update(len(results['ids']))
                    offset += len(results['ids'])

                    if len(results['ids']) < fetch_limit:
                        break # Reached the end
                except Exception as get_err:
                    print(f"\nError fetching batch starting at offset {offset}: {get_err}")
                    # Decide whether to continue or stop on batch error
                    break # Stop processing this collection on error

        print(f"Finished fetching metadata.")
        print(f"  Found {len(unique_dois)} unique DOIs.")
        print(f"  Found {len(metadata_fields)} unique metadata fields.")
        return len(unique_dois), metadata_fields, sample_metadata # Return sample metadata

    except Exception as e:
        print(f"An error occurred while processing database '{db_path}': {e}")
        import traceback
        traceback.print_exc()
        return None

# --- Define Database Paths and Collection Names ---
# Assuming this script is run from the 'scrape' directory
ABSTRACT_DB_PATH = "./scrape/data/chroma_dbs/abstract_chroma_db"
FULL_TEXT_DB_PATH = "./scrape/data/chroma_dbs/full_text_chunks_db" # Updated path based on chunk_new_dois.py example
RELEVANR_DB_PATH = "./scrape/data/chroma_dbs/relevant_chunks_db" # From build_relevant_papers_db.py


ABSTRACT_COLLECTION_NAME = "abstracts" # From paper_collection_pipeline.py
FULL_TEXT_COLLECTION_NAME = "paper_chunks_main" # From chunk_new_dois.py example usage
RELEVANT_CHUNKS_COLLECTION_NAME = "relevant_paper_chunks" # New collection name


# --- Run the Counts ---
abstract_result = count_unique_dois_chromadb_client(ABSTRACT_DB_PATH, ABSTRACT_COLLECTION_NAME)
full_text_result = count_unique_dois_chromadb_client(FULL_TEXT_DB_PATH, FULL_TEXT_COLLECTION_NAME)
relevant_result = count_unique_dois_chromadb_client(RELEVANR_DB_PATH, RELEVANT_CHUNKS_COLLECTION_NAME) # Uncomment if needed

# --- Print Results ---
print("\n--- Summary ---")
if abstract_result is not None:
    abstract_doi_count, abstract_fields, abstract_sample = abstract_result # Unpack sample
    print(f"Abstracts DB ('{ABSTRACT_COLLECTION_NAME}'):")
    print(f"  Unique DOIs: {abstract_doi_count}")
    print(f"  Metadata Fields: {sorted(list(abstract_fields)) if abstract_fields else 'None Found'}")
    if abstract_sample:
        print("  Sample Metadata Entry:")
        pprint.pprint(abstract_sample, indent=4, width=100)
    else:
        print("  (Could not retrieve a sample metadata entry)")
else:
    print(f"Could not determine details for Abstracts DB ('{ABSTRACT_COLLECTION_NAME}').")

if full_text_result is not None:
    full_text_doi_count, full_text_fields, full_text_sample = full_text_result # Unpack sample
    print(f"\nFull Text DB ('{FULL_TEXT_COLLECTION_NAME}'):")
    print(f"  Unique DOIs: {full_text_doi_count}")
    print(f"  Metadata Fields: {sorted(list(full_text_fields)) if full_text_fields else 'None Found'}")
    if full_text_sample:
        print("  Sample Metadata Entry:")
        pprint.pprint(full_text_sample, indent=4, width=100)
    else:
        print("  (Could not retrieve a sample metadata entry)")

if relevant_result is not None:
    relevant_doi_count, relevant_fields, relevant_sample = relevant_result # Unpack sample
    print(f"\nFull Text DB ('{FULL_TEXT_COLLECTION_NAME}'):")
    print(f"  Unique DOIs: {relevant_doi_count}")
    print(f"  Metadata Fields: {sorted(list(relevant_fields)) if relevant_fields else 'None Found'}")
    if relevant_sample:
        print("  Sample Metadata Entry:")
        pprint.pprint(relevant_sample, indent=4, width=100)
    else:
        print("  (Could not retrieve a sample metadata entry)")


else:
    print(f"\nCould not determine details for Full Text DB ('{FULL_TEXT_COLLECTION_NAME}').")


