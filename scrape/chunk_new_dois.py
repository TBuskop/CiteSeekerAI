import os
import glob
import re
import sys
import traceback
import time
from typing import List, Dict, Optional, Tuple, Set, Any
from tqdm import tqdm

# --- Add project root to sys.path if needed ---
# This allows importing from the 'rag' package when running this script directly
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
# --- End Path Addition ---

try:
    import chromadb
    from langchain.text_splitter import RecursiveCharacterTextSplitter # Example splitter
    # Import necessary components from the rag package
    from rag.chroma_manager import get_chroma_collection
    from rag.embedding import find_chunks_to_embed, generate_embeddings_in_batches
    from rag import llm_interface # Import the module to call initialize_clients and access client
    from rag.config import (
        DEFAULT_EMBED_BATCH_SIZE,
        DEFAULT_EMBED_DELAY,
        EMBEDDING_MODEL, # Needed for embedding function config
        OUTPUT_EMBEDDING_DIMENSION # Needed for embedding function config
    )
    print("Successfully imported components from 'rag' package.")
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Ensure 'langchain', 'chromadb', 'tqdm', and the 'rag' package are installed and accessible.")
    print(f"Project Root: {_PROJECT_ROOT}")
    print(f"Sys Path includes Project Root: {_PROJECT_ROOT in sys.path}")
    traceback.print_exc()
    sys.exit(1)

# --- Constants ---
DEFAULT_CHUNK_DB_PATH = "./all_doi_chunks_db"
DEFAULT_CHUNK_COLLECTION_NAME = "all_doi_chunks"
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 150
INDEX_BATCH_SIZE = 100 # How many raw chunks to upsert to ChromaDB at once

# --- Helper Functions ---

def extract_doi_from_filename(filename: str) -> Optional[str]:
    """
    Extracts DOI from filename assuming format 'DOI_with_underscores.txt'.
    Replaces underscores with slashes. Handles potential errors.
    """
    try:
        # Remove .txt extension
        doi_part = filename[:-4]
        # Replace underscores back to slashes
        # This simple replacement assumes DOIs don't contain underscores themselves.
        # A more robust method might be needed if they do.
        doi = doi_part.replace("_", "/")
        return doi
    except Exception as e:
        print(f"Error extracting DOI from filename '{filename}': {e}")
        return None

def find_new_text_files(folder_path: str, db_path: str, collection_name: str) -> List[str]:
    """
    Scans a folder for .txt files and returns a list of paths for files
    whose corresponding DOI is not found in the ChromaDB collection metadata.
    """
    print(f"Scanning folder '{folder_path}' for new .txt files...")
    if not os.path.isdir(folder_path):
        print(f"Error: Folder not found: {folder_path}")
        return []

    all_txt_files = glob.glob(os.path.join(folder_path, "*.txt"))
    if not all_txt_files:
        print("No .txt files found in the folder.")
        return []

    print(f"Found {len(all_txt_files)} total .txt files. Checking against database...")
    existing_dois: Set[str] = set()
    try:
        # Connect in 'index' mode to check existing data without triggering embedding calls
        collection = get_chroma_collection(db_path, collection_name, execution_mode="index")
        estimated_count = collection.count()
        if estimated_count > 0:
            print(f"Fetching existing DOIs from metadata in '{collection_name}'...")
            # Fetch metadata in batches to avoid memory issues for large collections
            fetch_limit = 10000
            offset = 0
            fetched_count = 0
            with tqdm(total=estimated_count, desc="Fetching metadata", unit="chunks") as pbar:
                 while True:
                     results = collection.get(limit=fetch_limit, offset=offset, include=['metadatas'])
                     if not results or not results.get('ids'): break
                     metadatas = results.get('metadatas', [])
                     batch_dois = {meta.get('doi') for meta in metadatas if meta and meta.get('doi')}
                     existing_dois.update(batch_dois)
                     fetched_count += len(results['ids'])
                     pbar.update(len(results['ids']))
                     offset += len(results['ids'])
                     if len(results['ids']) < fetch_limit: break # Reached the end
            print(f"Found {len(existing_dois)} unique existing DOIs in the database.")
        else:
            print("Collection is empty or does not exist yet.")

    except Exception as e:
        print(f"Warning: Could not check existing DOIs in ChromaDB: {e}. Assuming all files are new.")
        # Proceed as if all files are new if DB check fails

    new_files = []
    skipped_count = 0
    for file_path in all_txt_files:
        filename = os.path.basename(file_path)
        doi = extract_doi_from_filename(filename)
        if doi and doi in existing_dois:
            skipped_count += 1
        elif doi: # DOI extracted but not in existing set
            new_files.append(file_path)
        else: # DOI extraction failed
            print(f"Skipping file '{filename}' due to DOI extraction failure.")
            skipped_count += 1

    print(f"Found {len(new_files)} new files to process.")
    if skipped_count > 0:
        print(f"Skipped {skipped_count} files (already processed or DOI extraction failed).")
    return new_files

def process_and_index_files(
    new_file_paths: List[str],
    db_path: str,
    collection_name: str,
    chunk_size: int,
    chunk_overlap: int
):
    """
    Reads new files, chunks them, and adds raw chunks (without embeddings) to ChromaDB.
    """
    if not new_file_paths:
        print("No new files provided for processing and indexing.")
        return

    print(f"\n--- Processing and Indexing {len(new_file_paths)} New Files ---")
    try:
        collection = get_chroma_collection(db_path, collection_name, execution_mode="index")
    except Exception as e:
        print(f"Error connecting to ChromaDB for indexing: {e}")
        return

    # Initialize text splitter (using Langchain's RecursiveCharacterTextSplitter as an example)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )

    all_ids_to_add: List[str] = []
    all_documents_to_add: List[str] = []
    all_metadatas_to_add: List[Dict[str, Any]] = []
    processed_files_count = 0
    failed_files_count = 0

    for file_path in tqdm(new_file_paths, desc="Chunking & Preparing Files", unit="file"):
        filename = os.path.basename(file_path)
        doi = extract_doi_from_filename(filename)
        if not doi:
            print(f"\nSkipping file {filename} again due to DOI extraction error.")
            failed_files_count += 1
            continue

        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                text = f.read()

            if not text or not text.strip():
                print(f"\nSkipping empty file: {filename}")
                failed_files_count += 1
                continue

            # Chunk the document
            chunks = text_splitter.split_text(text)
            if not chunks:
                print(f"\nNo chunks generated for file: {filename}")
                failed_files_count += 1
                continue

            # Prepare data for ChromaDB
            for i, chunk_text in enumerate(chunks):
                if not chunk_text or not chunk_text.strip(): continue
                chunk_id = f"{doi}_{i}" # Simple ID based on DOI and chunk index
                metadata = {
                    "doi": doi,
                    "source_filename": filename,
                    "chunk_number": i,
                    "has_embedding": False # Mark for embedding later
                }
                all_ids_to_add.append(chunk_id)
                all_documents_to_add.append(chunk_text)
                all_metadatas_to_add.append(metadata)

            processed_files_count += 1

        except Exception as e:
            print(f"\nError processing file {filename}: {e}")
            traceback.print_exc()
            failed_files_count += 1

    # Upsert collected data to ChromaDB in batches
    if all_ids_to_add:
        print(f"\nUpserting {len(all_ids_to_add)} raw chunks to ChromaDB...")
        num_batches = (len(all_ids_to_add) + INDEX_BATCH_SIZE - 1) // INDEX_BATCH_SIZE
        for i in tqdm(range(num_batches), desc="Upserting Raw Chunks"):
            start_idx = i * INDEX_BATCH_SIZE
            end_idx = start_idx + INDEX_BATCH_SIZE
            batch_ids = all_ids_to_add[start_idx:end_idx]
            batch_docs = all_documents_to_add[start_idx:end_idx]
            batch_metas = all_metadatas_to_add[start_idx:end_idx]

            if not batch_ids: continue
            try:
                collection.upsert(
                    ids=batch_ids,
                    documents=batch_docs,
                    metadatas=batch_metas
                    # No embeddings provided here
                )
            except Exception as upsert_err:
                print(f"\n!!! Error upserting batch {i+1}/{num_batches} to ChromaDB: {upsert_err}")
                # Consider logging failed IDs or retrying
    else:
        print("\nNo new chunks were generated to add to the database.")

    print(f"\nIndexing Phase Summary:")
    print(f"  Successfully processed and prepared chunks for: {processed_files_count} files.")
    print(f"  Failed to process: {failed_files_count} files.")
    print(f"  Total raw chunks potentially added/updated: {len(all_ids_to_add)}")

def embed_new_chunks(
    db_path: str,
    collection_name: str,
    batch_size: int,
    delay: float
):
    """
    Finds chunks marked with 'has_embedding': False and generates embeddings for them.
    """
    print("\n--- Embedding Phase: Generating Embeddings for New Chunks ---")

    # Initialize LLM Client (needed for embedding generation)
    print("Initializing LLM clients...")
    try:
        # Call initialize_clients via the imported module
        llm_interface.initialize_clients()
        # Check if the required client (e.g., Gemini) is available
        # Access gemini_client via the module namespace AFTER initialization
        if llm_interface.gemini_client is None and "embedding" in EMBEDDING_MODEL:
             print("Warning: Google GenAI client (Gemini) not available, embedding might fail if using a Gemini model.")
             # Allow proceeding, generate_embeddings_in_batches might handle fallback or error
        print("LLM clients initialized.")
    except Exception as client_err:
        print(f"Error initializing LLM clients: {client_err}")
        print("Cannot proceed with embedding generation.")
        return

    print("Connecting to ChromaDB in 'embed' mode...")
    try:
        # Get collection in 'embed' mode (uses actual embedding function)
        collection = get_chroma_collection(
            db_path=db_path,
            collection_name=collection_name,
            execution_mode="embed" # Crucial for embedding generation
        )
        print("Connected.")
    except Exception as e:
        print(f"Error initializing ChromaDB for embedding: {e}")
        traceback.print_exc()
        return # Stop processing

    print("Finding chunks that need embeddings (has_embedding: False)...")
    try:
        # Use the function from rag.embedding
        ids_to_embed, metadatas_to_embed, documents_to_embed = find_chunks_to_embed(collection)
    except Exception as e:
        print(f"Error finding items to embed: {e}")
        traceback.print_exc()
        return

    if not ids_to_embed:
        print("No chunks found requiring embedding.")
    else:
        print(f"Found {len(ids_to_embed)} chunks to embed.")
        print(f"Starting embedding generation with batch size {batch_size} and delay {delay}s...")

        # Use the function from rag.embedding
        # Pass the specific client if needed by generate_embeddings_in_batches implementation
        # Assuming generate_embeddings_in_batches uses the client from llm_interface if needed
        failed_embed_ids = generate_embeddings_in_batches(
            collection=collection,
            ids_to_embed=ids_to_embed,
            metadatas_to_embed=metadatas_to_embed,
            documents_to_embed=documents_to_embed,
            batch_size=batch_size,
            delay=delay,
            client=llm_interface.gemini_client # Pass the initialized client
        )

        # Report final failures
        if failed_embed_ids:
            print(f"\nEmbedding Phase Summary: Failed to process/embed {len(failed_embed_ids)} unique chunks.")
            print("These chunks were not updated and remain marked as 'has_embedding': False.")
            # Consider logging failed_embed_ids
        else:
            print("\nEmbedding Phase Summary: All found chunks requiring embeddings were processed successfully.")

    print("\n--- Embedding Phase Finished ---")


# --- Main Orchestrator Function ---
def process_folder_for_chunks(
    folder_path: str,
    db_path: str = DEFAULT_CHUNK_DB_PATH,
    collection_name: str = DEFAULT_CHUNK_COLLECTION_NAME,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
    embed_delay: float = DEFAULT_EMBED_DELAY
):
    """
    Main function to orchestrate the process:
    1. Find new text files in the folder.
    2. Process and index the raw chunks of new files into ChromaDB.
    3. Find and embed any chunks in the DB missing embeddings.
    """
    print(f"--- Starting Chunk Processing Workflow ---")
    print(f"Source Folder: {folder_path}")
    print(f"Database Path: {db_path}")
    print(f"Collection Name: {collection_name}")
    print(f"Chunk Size: {chunk_size}, Overlap: {chunk_overlap}")
    print(f"Embedding Batch Size: {embed_batch_size}, Delay: {embed_delay}s")
    print("-" * 40)

    # 1. Find new files
    new_files = find_new_text_files(folder_path, db_path, collection_name)

    # 2. Process and index new files if any found
    if new_files:
        process_and_index_files(new_files, db_path, collection_name, chunk_size, chunk_overlap)
    else:
        print("\nNo new files found to index.")

    # 3. Embed any chunks missing embeddings (including newly added ones)
    embed_new_chunks(db_path, collection_name, embed_batch_size, embed_delay)

    print(f"\n--- Chunk Processing Workflow Finished ---")


# --- Example Usage (Optional) ---
if __name__ == "__main__":
    # This block runs only when the script is executed directly
    print("Running chunk_new_dois.py directly...")

    # --- Configuration for Direct Run ---
    # !!! Adjust these paths and parameters as needed !!!
    SOURCE_TEXT_FOLDER = "scrape/data/downloads/full_texts" # Example path relative to project root
    TARGET_DB_PATH = "scrape/data/chroma_dbs/full_text_chunks_db" # Example DB path
    TARGET_COLLECTION_NAME = "paper_chunks_main"

    # Ensure the source folder exists relative to the project root
    abs_source_folder = os.path.join(_PROJECT_ROOT, SOURCE_TEXT_FOLDER)
    abs_target_db_path = os.path.join(_PROJECT_ROOT, TARGET_DB_PATH)

    if not os.path.isdir(abs_source_folder):
         print(f"Error: Source folder '{abs_source_folder}' not found.")
         print("Please create the folder or update the SOURCE_TEXT_FOLDER variable.")
    else:
        # Ensure target DB directory exists
        os.makedirs(os.path.dirname(abs_target_db_path), exist_ok=True)

        # Run the main workflow
        process_folder_for_chunks(
            folder_path=abs_source_folder,
            db_path=abs_target_db_path,
            collection_name=TARGET_COLLECTION_NAME,
            # Optional: override defaults if needed
            # chunk_size=1200,
            # chunk_overlap=200,
            # embed_batch_size=64,
            # embed_delay=1.0
        )
