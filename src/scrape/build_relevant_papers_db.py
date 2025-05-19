import chromadb
import traceback
from typing import List
# from chromadb.errors import NotFoundError # Not directly used if relying on reset()
from chromadb.config import Settings # Import Settings
# --- Add BM25 Import ---
from src.rag.bm25_manager import build_and_save_bm25_index
# --- Custom Embedding Function Import ---
from src.rag.chroma_manager import ConfigurableEmbeddingFunction
import os # For path checking and makedirs
import gc # For garbage collection
import time # Import time for sleep

def build_relevant_db(
    relevant_doi_list: List[str],
    source_chunk_db_path: str,
    source_chunk_collection_name: str,
    abstract_db_path: str,
    abstract_collection_name: str,
    target_db_path: str,
    target_collection_name: str
):
    """
    Collects chunks corresponding to relevant DOIs from a source chunk database,
    enriches them with metadata from an abstract database, and stores them
    in a target database.

    Args:
        relevant_doi_list: List of DOIs identified as relevant.
        source_chunk_db_path: Path to the source ChromaDB containing all chunks.
        source_chunk_collection_name: Name of the collection in the source chunk DB.
        abstract_db_path: Path to the ChromaDB containing abstracts and metadata.
        abstract_collection_name: Name of the collection in the abstract DB.
        target_db_path: Path to the target ChromaDB for relevant chunks.
        target_collection_name: Name of the collection in the target relevant chunk DB.
    """
    print("--- Starting: Building Relevant Papers Database ---")
    if not relevant_doi_list:
        print("No relevant DOIs provided. Skipping build process.")
        return

    try:
        # --- Connect to Source and Abstract Databases ---
        print(f"Connecting to source chunk DB: {source_chunk_db_path}")
        source_client = chromadb.PersistentClient(path=source_chunk_db_path) # Assuming default settings are fine here
        source_collection = source_client.get_collection(name=source_chunk_collection_name)
        print(f"Source collection '{source_chunk_collection_name}' retrieved.")

        print(f"Connecting to abstract DB: {abstract_db_path}")
        abstract_client = chromadb.PersistentClient(path=abstract_db_path) # Assuming default settings are fine here
        abstract_collection = abstract_client.get_collection(name=abstract_collection_name)
        print(f"Abstract collection '{abstract_collection_name}' retrieved.")

        # --- Define ChromaDB settings to allow reset ---
        chroma_settings = Settings(allow_reset=True)

        # --- Reset existing target database to ensure freshness ---
        # Ensure the directory for the database exists before trying to connect/reset
        os.makedirs(target_db_path, exist_ok=True)
        
        print(f"Attempting to reset target database at: {target_db_path}")
        try:
            # Connect to the existing DB with reset enabled and reset it
            # This client is specifically for resetting
            temp_reset_client = chromadb.PersistentClient(path=target_db_path, settings=chroma_settings)
            temp_reset_client.reset() # This deletes all collections and data
            print(f"Successfully reset database at: {target_db_path}")
            del temp_reset_client # Ensure client is deleted
            gc.collect() # Trigger garbage collection to help release resources
            time.sleep(1) # Add a 1-second delay to allow OS to release file locks
        except Exception as reset_err:
            print(f"Error resetting database at {target_db_path}: {reset_err}")
            traceback.print_exc()
            # If reset fails, subsequent operations might also fail or use stale data.
            # Consider raising an error here to stop the process if a clean state is critical.
            print("Proceeding with client creation, but the database might not be completely fresh.")
            # raise RuntimeError(f"Failed to reset database at {target_db_path}. Cannot guarantee a fresh state.") from reset_err


        # --- Initialize Target Client (after potential reset) ---
        print(f"Initializing target relevant chunk DB at: {target_db_path}")
        # The client used for actual operations. Reset should also be allowed for this instance
        # if any further reset-like operations were intended, though typically not needed right after a reset.
        target_client = chromadb.PersistentClient(path=target_db_path, settings=chroma_settings)
        emb_func = ConfigurableEmbeddingFunction(current_mode="index")

        # Create the target collection
        # After a successful reset, the database should be empty, so create_collection should not find duplicates.
        try:
            print(f"Creating target collection '{target_collection_name}'...")
            target_collection = target_client.create_collection(
                name=target_collection_name,
                embedding_function=emb_func,
                metadata={"hnsw:space": "cosine"}
            )
            print(f"Target collection '{target_collection_name}' created successfully.")
        except chromadb.errors.InternalError as ie: # Catch InternalError
            # This might happen if the reset above failed or was incomplete.
            if "already exists" in str(ie).lower() or "duplicate" in str(ie).lower():
                error_msg = (
                    f"Critical Error: Failed to create collection '{target_collection_name}' because it already exists, "
                    f"even after attempting to reset the database. This indicates an unexpected state. Error: {ie}"
                )
                print(error_msg)
                traceback.print_exc()
                raise RuntimeError(error_msg) from ie # Stop execution
            else:
                # Re-raise if it's a different InternalError
                print(f"An unexpected InternalError occurred while creating collection '{target_collection_name}': {ie}")
                traceback.print_exc()
                raise
        except Exception as create_err: # Catch other potential errors during creation
            print(f"Error creating target collection '{target_collection_name}': {create_err}")
            traceback.print_exc()
            raise

        # --- Fetch Abstract Metadata for all relevant DOIs ---
        print(f"\nFetching metadata from abstract DB for {len(relevant_doi_list)} DOIs...")
        abstract_metadata_map = {}
        try:
            valid_dois_for_query = [doi for doi in relevant_doi_list if doi and isinstance(doi, str)]
            if valid_dois_for_query:
                abstract_results = abstract_collection.get(
                    ids=valid_dois_for_query,
                    include=['metadatas']
                )
                if abstract_results and abstract_results.get('ids'):
                    for i, fetched_id in enumerate(abstract_results['ids']):
                        abstract_metadata_map[fetched_id] = abstract_results['metadatas'][i]
                    print(f"Successfully fetched abstract metadata for {len(abstract_metadata_map)} DOIs.")
                else:
                    print("Warning: No metadata found in abstract DB for the provided DOIs.")
            else:
                print("Warning: No valid DOIs provided to fetch abstract metadata.")
        except Exception as abstract_fetch_err:
            print(f"Error fetching metadata from abstract DB: {abstract_fetch_err}")

        # --- Prepare lists for final upsert ---
        all_relevant_ids = []
        all_relevant_documents = []
        all_relevant_metadatas = []
        all_relevant_embeddings = []

        # --- Iterate through relevant DOIs, fetch chunks, merge metadata ---
        print("\nProcessing DOIs and fetching/merging chunks:")
        for doi in relevant_doi_list:
            if not doi or not isinstance(doi, str):
                print(f"  Skipping invalid DOI entry: {doi}")
                continue

            print(f"  Processing DOI: {doi}")
            specific_abstract_meta = abstract_metadata_map.get(doi)
            if not specific_abstract_meta:
                print(f"    Warning: No abstract metadata found for DOI {doi}. Chunks will not be enriched.")
                abstract_data_to_merge = {
                    'authors': None, 'title': None, 'year': None,
                    'source_title': None, 'cited_by': None
                }
            else:
                abstract_data_to_merge = {
                    'authors': specific_abstract_meta.get('authors'),
                    'title': specific_abstract_meta.get('title'),
                    'year': specific_abstract_meta.get('year'),
                    'source_title': specific_abstract_meta.get('source_title'),
                    'cited_by': specific_abstract_meta.get('cited_by')
                }

            try:
                chunk_results = source_collection.get(
                    where={"doi": doi},
                    include=['documents', 'metadatas', 'embeddings']
                )

                if chunk_results and chunk_results.get('ids'):
                    count = len(chunk_results['ids'])
                    print(f"    Found {count} chunks for DOI {doi}. Merging metadata...")

                    for i in range(count):
                        chunk_id = chunk_results['ids'][i]
                        chunk_doc = chunk_results['documents'][i]
                        chunk_embedding = chunk_results['embeddings'][i]
                        chunk_meta = chunk_results['metadatas'][i]

                        enriched_meta = chunk_meta.copy()
                        enriched_meta.update(abstract_data_to_merge)
                        enriched_meta['doi'] = chunk_meta.get('doi', doi)

                        all_relevant_ids.append(chunk_id)
                        all_relevant_documents.append(chunk_doc)
                        all_relevant_embeddings.append(chunk_embedding)
                        all_relevant_metadatas.append(enriched_meta)
                else:
                    print(f"    No chunks found for DOI {doi} in the source collection.")

            except Exception as chunk_fetch_err:
                print(f"    Error fetching chunks for DOI {doi}: {chunk_fetch_err}")
                traceback.print_exc()

        # --- Add/update the collected chunks with enriched metadata in the target collection ---
        if all_relevant_ids:
            print(f"\nAdding/updating {len(all_relevant_ids)} relevant chunks with enriched metadata to '{target_collection_name}'...")
            try:
                target_collection.upsert(
                    ids=all_relevant_ids,
                    documents=all_relevant_documents,
                    metadatas=all_relevant_metadatas,
                    embeddings=all_relevant_embeddings
                )
                print("Relevant chunks added/updated successfully.")

                # --- Build and Save BM25 Index for the new collection ---
                print(f"\nBuilding BM25 index for '{target_collection_name}'...")
                build_and_save_bm25_index(
                    chunk_ids=all_relevant_ids,
                    chunk_texts=all_relevant_documents,
                    db_path=target_db_path,
                    collection_name=target_collection_name
                )
                # --- End BM25 Index Building ---

            except Exception as upsert_err:
                 print(f"Error upserting enriched chunks to target DB: {upsert_err}")
                 traceback.print_exc()
        else:
            print("No relevant chunks found across all DOIs to add to the special database.")

    except Exception as e:
        print(f"Error during building relevant papers DB: {e}")
        traceback.print_exc()

    print("--- Finished: Building Relevant Papers Database ---")