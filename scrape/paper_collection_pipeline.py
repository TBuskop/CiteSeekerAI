import os
from search_scopus import run_scopus_search # Assuming this function will be updated to accept config
from add_csv_to_chromadb import ingest_csv_to_chroma
from collect_relevant_abstracts import find_relevant_dois_from_abstracts
from download_papers import download_dois
from get_search_string import generate_scopus_search_string # Import the new function
# --- Import the chunking function ---
from chunk_new_dois import process_folder_for_chunks
# --- Import ChromaDB client ---
import chromadb # Added import
import traceback # Added for detailed error printing

# --- Central Configuration ---
# General
BASE_DATA_DIR = "./data"
DOWNLOADS_DIR = os.path.join(BASE_DATA_DIR, "downloads")
FULL_TEXT_DIR = os.path.join(DOWNLOADS_DIR, "full_texts")
CSV_DIR = os.path.join(DOWNLOADS_DIR, "csv")
# Abstract DB Config
ABSTRACT_DB_PATH = "./abstract_chroma_db"
ABSTRACT_COLLECTION_NAME = "abstracts"
# --- Chunking DB Config ---
CHUNK_DB_PATH = os.path.join(BASE_DATA_DIR, "chroma_dbs", "full_text_chunks_db") # Centralized path
CHUNK_COLLECTION_NAME = "paper_chunks_main"
CHUNK_SIZE = 1000 # Default chunk size from chunk_new_dois
CHUNK_OVERLAP = 150 # Default chunk overlap from chunk_new_dois
# Embedding config (defaults from chunk_new_dois will be used if not overridden)
# EMBED_BATCH_SIZE = 64
# EMBED_DELAY = 1.0
# --- Special Relevant Chunks DB Config ---
RELEVANT_CHUNKS_DB_PATH = os.path.join(BASE_DATA_DIR, "chroma_dbs", "relevant_chunks_db") # New DB path
RELEVANT_CHUNKS_COLLECTION_NAME = "relevant_paper_chunks" # New collection name

# Search String Generation Configuration
INITIAL_RESEARCH_QUESTION = "What are the effects of sea level rise on italy?"
# Fallback query if generation fails
DEFAULT_SCOPUS_QUERY = "climate OR 'climate change' OR 'climate variability' AND robustness AND uncertainty AND policy AND decision AND making"
SAVE_GENERATED_SEARCH_STRING = True # Whether to save the generated string to a file

# Scopus Search Configuration
SCOPUS_HEADLESS_MODE = False # Example: Add config for headless mode
SCOPUS_YEAR_FROM = None # Example: Add config for year filter start
SCOPUS_YEAR_TO = None   # Example: Add config for year filter end
# Credentials and Institution are expected to be in .env by search_scopus.py

# ChromaDB Ingestion Configuration
FORCE_REINDEX_CHROMA = False # Set to True to re-index existing documents

# Abstract Collection Configuration
TOP_K_ABSTRACTS = 10
USE_RERANK_ABSTRACTS = True
RELEVANT_ABSTRACTS_OUTPUT_FILENAME = "relevant_abstracts.txt"

# --- Ensure Directories Exist ---
os.makedirs(FULL_TEXT_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)
# Ensure the chunk DB directory exists
os.makedirs(os.path.dirname(CHUNK_DB_PATH), exist_ok=True)
# Ensure the relevant chunk DB directory exists
os.makedirs(os.path.dirname(RELEVANT_CHUNKS_DB_PATH), exist_ok=True) # Added for relevant chunks DB

# --- Pipeline Execution ---

print("--- Step 0: Generating Scopus Search String ---")
success, generated_query = generate_scopus_search_string(
    query=INITIAL_RESEARCH_QUESTION,
    save_to_file=SAVE_GENERATED_SEARCH_STRING
)

if success and generated_query:
    SCOPUS_QUERY = generated_query
    print(f"Using generated Scopus query: {SCOPUS_QUERY}")
else:
    SCOPUS_QUERY = DEFAULT_SCOPUS_QUERY
    print(f"Failed to generate search string. Using default Scopus query: {SCOPUS_QUERY}")

# Define Scopus output path based on the final query
clean_query = (SCOPUS_QUERY.replace(' ', '_')
                             .replace("'", '')
                             .replace('(', '')
                             .replace(')', '')
                             .replace(':', '')
                             .replace('*', '')
                             .replace('?', ''))[:50]

SCOPUS_OUTPUT_CSV_FILENAME = f"scopus_{clean_query}.csv"
SCOPUS_OUTPUT_CSV_PATH = os.path.join(CSV_DIR, SCOPUS_OUTPUT_CSV_FILENAME)


print("\n--- Step 1: Running Scopus Search ---")
# Run the Scopus search using the determined query and output path
search_success, actual_csv_path = run_scopus_search(
    query=SCOPUS_QUERY,
    output_csv_path=SCOPUS_OUTPUT_CSV_PATH, # Pass the desired full path
    headless=SCOPUS_HEADLESS_MODE,
    year_from=SCOPUS_YEAR_FROM,
    year_to=SCOPUS_YEAR_TO
    # Credentials and institution are handled by run_scopus_search loading .env
)

# Check if the search was successful and update the path variable
if search_success and actual_csv_path:
    SCOPUS_OUTPUT_CSV_PATH = str(actual_csv_path) # Update path to the actual one used/created
    print(f"Scopus search successful. Results saved to: {SCOPUS_OUTPUT_CSV_PATH}")
else:
    print(f"Error: Scopus search failed. Check logs in search_scopus script.")
    # Decide how to proceed: exit or try to continue? Exiting is safer.
    exit() # Exit if search failed


print("\n--- Step 2: Ingesting CSV to ChromaDB ---")
# This step now implicitly relies on search_success being True from Step 1
ingest_csv_to_chroma(
    csv_file_path=SCOPUS_OUTPUT_CSV_PATH,
    db_path=ABSTRACT_DB_PATH, # Use abstract DB path
    collection_name=ABSTRACT_COLLECTION_NAME, # Use abstract collection name
    force_reindex=FORCE_REINDEX_CHROMA
)


print("\n--- Step 3: Finding Relevant DOIs from Abstracts ---")
# This step also implicitly relies on search_success being True
relevant_doi_list = find_relevant_dois_from_abstracts(
    initial_query=INITIAL_RESEARCH_QUESTION,
    db_path=ABSTRACT_DB_PATH,
    collection_name=ABSTRACT_COLLECTION_NAME,
    top_k=TOP_K_ABSTRACTS,
    use_rerank=USE_RERANK_ABSTRACTS,
    output_filename=RELEVANT_ABSTRACTS_OUTPUT_FILENAME
)


print("\n--- Step 4: Downloading Full Texts ---")
if relevant_doi_list:
    download_dois(
        proposed_dois=relevant_doi_list,
        output_directory=FULL_TEXT_DIR
    )
else:
    print("No relevant DOIs found or previous steps skipped, skipping download step.")

print("\n--- Step 5: Chunking downloaded documents and adding to common database ---")
# Call the main function from chunk_new_dois.py
process_folder_for_chunks(
    folder_path=FULL_TEXT_DIR,         # Source folder for downloaded .txt files
    db_path=CHUNK_DB_PATH,             # Target database path for chunks
    collection_name=CHUNK_COLLECTION_NAME, # Target collection name for chunks
    chunk_size=CHUNK_SIZE,             # Use configured chunk size
    chunk_overlap=CHUNK_OVERLAP,       # Use configured chunk overlap
    # Optional: Pass embedding batch size and delay if you want to override defaults
    # embed_batch_size=EMBED_BATCH_SIZE,
    # embed_delay=EMBED_DELAY
)


print("\n--- Step 6: Collecting Relevant Chunks into Special Database ---")
if relevant_doi_list:
    print(f"Found {len(relevant_doi_list)} relevant DOIs to collect chunks for.")
    try:
        # --- Connect to Databases ---
        print(f"Connecting to source chunk DB: {CHUNK_DB_PATH}")
        source_client = chromadb.PersistentClient(path=CHUNK_DB_PATH)
        source_collection = source_client.get_collection(name=CHUNK_COLLECTION_NAME)
        print(f"Source collection '{CHUNK_COLLECTION_NAME}' retrieved.")

        print(f"Connecting to abstract DB: {ABSTRACT_DB_PATH}")
        abstract_client = chromadb.PersistentClient(path=ABSTRACT_DB_PATH)
        abstract_collection = abstract_client.get_collection(name=ABSTRACT_COLLECTION_NAME)
        print(f"Abstract collection '{ABSTRACT_COLLECTION_NAME}' retrieved.")

        print(f"Connecting to target relevant chunk DB: {RELEVANT_CHUNKS_DB_PATH}")
        target_client = chromadb.PersistentClient(path=RELEVANT_CHUNKS_DB_PATH)
        target_collection = target_client.get_or_create_collection(name=RELEVANT_CHUNKS_COLLECTION_NAME)
        print(f"Target collection '{RELEVANT_CHUNKS_COLLECTION_NAME}' retrieved/created.")

        # --- Fetch Abstract Metadata for all relevant DOIs ---
        print(f"\nFetching metadata from abstract DB for {len(relevant_doi_list)} DOIs...")
        # Assuming the ID in the abstract collection is the DOI itself
        # Handle potential errors if some DOIs are not found
        abstract_metadata_map = {}
        try:
            # Fetch using the DOIs as IDs. Filter out None/empty DOIs just in case.
            valid_dois_for_query = [doi for doi in relevant_doi_list if doi and isinstance(doi, str)]
            if valid_dois_for_query:
                abstract_results = abstract_collection.get(
                    ids=valid_dois_for_query,
                    include=['metadatas']
                )
                if abstract_results and abstract_results.get('ids'):
                    for i, fetched_id in enumerate(abstract_results['ids']):
                        # Store metadata keyed by DOI (which is the ID here)
                        abstract_metadata_map[fetched_id] = abstract_results['metadatas'][i]
                    print(f"Successfully fetched abstract metadata for {len(abstract_metadata_map)} DOIs.")
                else:
                    print("Warning: No metadata found in abstract DB for the provided DOIs.")
            else:
                print("Warning: No valid DOIs provided to fetch abstract metadata.")
        except Exception as abstract_fetch_err:
            print(f"Error fetching metadata from abstract DB: {abstract_fetch_err}")
            # Continue without abstract metadata enrichment if fetching fails

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
            # Retrieve pre-fetched abstract metadata
            specific_abstract_meta = abstract_metadata_map.get(doi)
            if not specific_abstract_meta:
                print(f"    Warning: No abstract metadata found for DOI {doi}. Chunks will not be enriched.")
                # Define the keys we want even if metadata is missing, setting them to None
                abstract_data_to_merge = {
                    'authors': None, 'title': None, 'year': None,
                    'source_title': None, 'cited_by': None
                }
            else:
                # Extract desired fields from abstract metadata
                abstract_data_to_merge = {
                    'authors': specific_abstract_meta.get('authors'),
                    'title': specific_abstract_meta.get('title'),
                    'year': specific_abstract_meta.get('year'),
                    'source_title': specific_abstract_meta.get('source_title'),
                    'cited_by': specific_abstract_meta.get('cited_by')
                    # Keep DOI from chunk metadata later to ensure consistency
                }

            # Fetch chunks for the current DOI from the source chunk DB
            try:
                chunk_results = source_collection.get(
                    where={"doi": doi}, # Query by DOI metadata field in chunk DB
                    include=['documents', 'metadatas', 'embeddings']
                )

                if chunk_results and chunk_results.get('ids'):
                    count = len(chunk_results['ids'])
                    print(f"    Found {count} chunks for DOI {doi}. Merging metadata...")

                    # Iterate through the chunks found for this DOI
                    for i in range(count):
                        chunk_id = chunk_results['ids'][i]
                        chunk_doc = chunk_results['documents'][i]
                        chunk_embedding = chunk_results['embeddings'][i]
                        chunk_meta = chunk_results['metadatas'][i]

                        # Create enriched metadata: start with chunk meta, update with abstract meta
                        enriched_meta = chunk_meta.copy() # Start with existing chunk metadata
                        enriched_meta.update(abstract_data_to_merge) # Add/overwrite with abstract data
                        # Ensure DOI consistency (preferring the one from the chunk if different, though unlikely)
                        enriched_meta['doi'] = chunk_meta.get('doi', doi)

                        # Add to lists for final upsert
                        all_relevant_ids.append(chunk_id)
                        all_relevant_documents.append(chunk_doc)
                        all_relevant_embeddings.append(chunk_embedding)
                        all_relevant_metadatas.append(enriched_meta) # Add the merged metadata
                else:
                    print(f"    No chunks found for DOI {doi} in the source collection.")

            except Exception as chunk_fetch_err:
                print(f"    Error fetching chunks for DOI {doi}: {chunk_fetch_err}")
                traceback.print_exc() # Print detailed error for this DOI

        # --- Add/update the collected chunks with enriched metadata in the target collection ---
        if all_relevant_ids:
            print(f"\nAdding/updating {len(all_relevant_ids)} relevant chunks with enriched metadata to '{RELEVANT_CHUNKS_COLLECTION_NAME}'...")
            try:
                target_collection.upsert(
                    ids=all_relevant_ids,
                    documents=all_relevant_documents,
                    metadatas=all_relevant_metadatas, # Use the enriched metadata
                    embeddings=all_relevant_embeddings
                )
                print("Relevant chunks added/updated successfully.")
            except Exception as upsert_err:
                 print(f"Error upserting enriched chunks to target DB: {upsert_err}")
                 traceback.print_exc()
        else:
            print("No relevant chunks found across all DOIs to add to the special database.")

    except Exception as e:
        print(f"Error during Step 6 (Collecting Relevant Chunks): {e}")
        traceback.print_exc() # Print detailed error for the whole step
else:
    print("No relevant DOIs identified in Step 3, skipping collection of relevant chunks.")


print("\n--- Pipeline Finished ---")