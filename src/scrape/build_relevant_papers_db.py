import chromadb
import traceback
from typing import List
from chromadb.errors import NotFoundError

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
        # --- Connect to Databases ---
        print(f"Connecting to source chunk DB: {source_chunk_db_path}")
        source_client = chromadb.PersistentClient(path=source_chunk_db_path)
        source_collection = source_client.get_collection(name=source_chunk_collection_name)
        print(f"Source collection '{source_chunk_collection_name}' retrieved.")

        print(f"Connecting to abstract DB: {abstract_db_path}")
        abstract_client = chromadb.PersistentClient(path=abstract_db_path)
        abstract_collection = abstract_client.get_collection(name=abstract_collection_name)
        print(f"Abstract collection '{abstract_collection_name}' retrieved.")

        print(f"Connecting to target relevant chunk DB: {target_db_path}")
        target_client = chromadb.PersistentClient(path=target_db_path)

        # --- Delete existing target collection to ensure it's empty ---
        try:
            print(f"Attempting to delete existing target collection '{target_collection_name}'...")
            target_client.delete_collection(name=target_collection_name)
            print(f"Collection '{target_collection_name}' deleted successfully.")
        except NotFoundError:
            print(f"Collection '{target_collection_name}' does not exist, no need to delete.")
        except Exception as delete_err:
            print(f"Warning: An error occurred while trying to delete collection '{target_collection_name}': {delete_err}")
            print("Proceeding to create the collection.")

        # --- Create the target collection (now guaranteed to be new or recreated) ---
        target_collection = target_client.get_or_create_collection(name=target_collection_name)
        print(f"Target collection '{target_collection_name}' created/retrieved.")

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
            except Exception as upsert_err:
                 print(f"Error upserting enriched chunks to target DB: {upsert_err}")
                 traceback.print_exc()
        else:
            print("No relevant chunks found across all DOIs to add to the special database.")

    except Exception as e:
        print(f"Error during building relevant papers DB: {e}")
        traceback.print_exc()

    print("--- Finished: Building Relevant Papers Database ---")