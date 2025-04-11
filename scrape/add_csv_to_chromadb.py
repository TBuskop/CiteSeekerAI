# filepath: c:\Users\buskop\OneDrive - Stichting Deltares\Documents\GitHub\academic_lit_llm_2\scrape\add_csv_to_chromadb.py
import pandas as pd
import uuid
import os
import sys
import traceback
import math
from typing import Optional, Any, Dict, List

# --- Add project root to sys.path ---
# Ensure the 'rag' package can be found when running from the 'scrape' directory
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
# --- End Path Addition ---

# --- Local Imports ---
# Use standard absolute imports now that the package structure is fixed
try:
    from rag.chroma_manager import get_chroma_collection
    # Import the module itself or specific functions/variables as needed
    from rag import llm_interface # Import the module
    from rag.embedding import find_chunks_to_embed, generate_embeddings_in_batches
    # --- Import BM25 building function ---
    from rag.bm25_manager import build_and_save_bm25_index
    # --- End BM25 import ---
    from rag.config import (
        DEFAULT_EMBED_BATCH_SIZE,
        DEFAULT_EMBED_DELAY,
        EMBEDDING_MODEL,
        OUTPUT_EMBEDDING_DIMENSION
    )
    print("Successfully imported components from 'rag' package.")
except ImportError as e:
    print(f"Error importing from 'rag' package: {e}")
    print(f"Project Root: {_PROJECT_ROOT}")
    print(f"Sys Path includes Project Root: {_PROJECT_ROOT in sys.path}")
    print(f"Check if 'rag/__init__.py' exists: {os.path.exists(os.path.join(_PROJECT_ROOT, 'rag', '__init__.py'))}")
    print("Ensure all modules within 'rag' use absolute imports (e.g., 'from rag.config import ...').")
    traceback.print_exc()
    sys.exit(1)

# --- Constants ---
DEFAULT_DB_PATH = "./abstract_chroma_db"
DEFAULT_COLLECTION_NAME = "abstracts"
INDEX_BATCH_SIZE = 500 # How many items to upsert to ChromaDB at once during indexing

# --- Helper Functions ---
def _clean_numeric(value: Any) -> Optional[int]:
    """Safely convert value to integer, handling potential errors and non-numeric types."""
    if pd.isna(value):
        return None # Strategy: Use None for missing numeric values
    try:
        # Attempt conversion to float first to handle "10.0" then to int
        return int(float(value))
    except (ValueError, TypeError):
        return None # Strategy: Use None if conversion fails

def _generate_id(doi: Optional[str], index: int, seen_ids: set) -> str:
    """
    Generate a unique ID using DOI if available, otherwise fallback to row index.
    Ensures the returned ID is unique by checking against seen_ids.
    
    Args:
        doi: The DOI string or None
        index: The row index in the CSV
        seen_ids: Set of IDs already generated in this session
    
    Returns:
        A unique ID string
    """
    if doi and isinstance(doi, str) and doi.strip():
        base_id = doi.strip()
    else:
        # Fallback strategy: Use a prefix and the row index
        base_id = f"row_{index}"
    
    # Ensure uniqueness
    if base_id in seen_ids:
        # If already seen, append a suffix to make unique
        suffix = 1
        while f"{base_id}_{suffix}" in seen_ids:
            suffix += 1
        unique_id = f"{base_id}_{suffix}"
    else:
        unique_id = base_id
    
    # Add to seen IDs set
    seen_ids.add(unique_id)
    return unique_id

# --- Main Ingestion Function ---
def ingest_csv_to_chroma(
    csv_file_path: str,
    db_path: str,
    collection_name: str,
    force_reindex: bool = False
):
    """
    Reads a CSV, adds publication data to ChromaDB without embeddings,
    then generates and adds embeddings for the abstracts.

    Args:
        csv_file_path: Path to the input CSV file.
        db_path: Path to the ChromaDB persistence directory.
        collection_name: Name of the ChromaDB collection.
        force_reindex: If True, reindex documents even if they exist in DB
    """
    print(f"--- Starting CSV Ingestion ---")
    print(f"CSV Source: {csv_file_path}")
    print(f"ChromaDB Path: {db_path}")
    print(f"Collection Name: {collection_name}")
    print(f"Force Reindex: {force_reindex}")

    # == Phase 1: Indexing (Add data without embeddings) ==
    print("\n--- Phase 1: Indexing Data ---")
    try:
        # Get collection in 'index' mode (uses placeholder embeddings)
        print("Connecting to ChromaDB in 'index' mode...")
        index_collection = get_chroma_collection(
            db_path=db_path,
            collection_name=collection_name,
            execution_mode="index"
        )
        print("Connected.")
    except Exception as e:
        print(f"Error initializing ChromaDB for indexing: {e}")
        traceback.print_exc()
        return # Stop processing

    try:
        print(f"Reading CSV file: {csv_file_path}...")
        # Strategy: Keep missing string values as NaN initially, handle during iteration
        df = pd.read_csv(csv_file_path, keep_default_na=True)
        
        # Remove duplicate DOIs before processing if they exist
        if 'DOI' in df.columns:
            original_count = len(df)
            # Remove rows with duplicate DOIs (keeping first occurrence)
            df = df.drop_duplicates(subset=['DOI'], keep='first')
            # Print info about duplicates
            dup_count = original_count - len(df)
            if dup_count > 0:
                print(f"Removed {dup_count} rows with duplicate DOIs from CSV.")
        
        print(f"Processing {len(df)} rows from CSV.")
    except FileNotFoundError:
        print(f"Error: CSV file not found at '{csv_file_path}'")
        return
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        traceback.print_exc()
        return

    ids_to_add: List[str] = []
    documents_to_add: List[str] = []
    metadatas_to_add: List[Dict[str, Any]] = []
    # --- Lists to accumulate for BM25 ---
    all_processed_ids: List[str] = []
    all_processed_documents: List[str] = []
    # --- End BM25 lists ---
    skipped_rows = 0
    processed_rows = 0

    # Track IDs we've seen to avoid duplicates
    seen_ids = set()

    print("Processing CSV rows for indexing...")
    for index, row in df.iterrows():
        try:
            # Use exact column names from CSV (case-sensitive)
            abstract = row.get('Abstract')
            # Strategy: Skip rows with missing or empty abstracts, as they cannot be embedded.
            if pd.isna(abstract) or not str(abstract).strip():
                skipped_rows += 1
                continue

            # Extract and clean data using exact column names
            authors = str(row.get('Authors', '')) # Strategy: Default empty string for missing text
            title = str(row.get('Title', ''))
            source_title = str(row.get('Source title', '')) # This one was already correct
            doi = str(row.get('DOI', '')) # Keep as string for ID generation
            year = _clean_numeric(row.get('Year'))
            cited_by = _clean_numeric(row.get('Cited by'))

            # Generate unique ID using updated function with seen_ids tracking
            doc_id = _generate_id(doi, index, seen_ids)
            
            # --- Skip row if already in DB via ID and not force_reindex ---
            if not force_reindex:
                existing = index_collection.get(ids=[doc_id])
                if existing.get('ids'):
                    print(f"Skipping row {index} (doc_id: {doc_id}) already in database.")
                    continue
            # --- End Skip ---

            # --- Changed: Combine title with abstract ---
            merged_text = f"{title.strip()}\n{str(abstract).strip()}"
            # --- End Change ---

            # Prepare metadata
            metadata = {
                'authors': authors,
                'title': title,
                'year': year,
                'source_title': source_title,
                'cited_by': cited_by,
                'doi': doi.strip() if doi else None, # Store cleaned DOI or None
                'original_csv_row': index,
                'has_embedding': False # Mark for embedding later
            }
            # Filter out None values from metadata for cleaner storage
            metadata = {k: v for k, v in metadata.items() if v is not None}

            ids_to_add.append(doc_id)
            documents_to_add.append(merged_text) # Store the merged text for Chroma upsert
            metadatas_to_add.append(metadata)

            # --- Accumulate for BM25 ---
            all_processed_ids.append(doc_id)
            all_processed_documents.append(merged_text)
            # --- End Accumulate ---
            processed_rows += 1

            # Upsert in batches
            if len(ids_to_add) >= INDEX_BATCH_SIZE:
                print(f"  Upserting batch of {len(ids_to_add)} items to ChromaDB...")
                index_collection.upsert(
                    ids=ids_to_add,
                    documents=documents_to_add,
                    metadatas=metadatas_to_add
                )
                ids_to_add, documents_to_add, metadatas_to_add = [], [], [] # Reset lists

        except Exception as e:
            print(f"Error processing row {index}: {e}")
            traceback.print_exc()
            skipped_rows += 1

    # Upsert any remaining items
    if ids_to_add:
        print(f"  Upserting final batch of {len(ids_to_add)} items to ChromaDB...")
        index_collection.upsert(
            ids=ids_to_add,
            documents=documents_to_add,
            metadatas=metadatas_to_add
        )

    print(f"Indexing Phase Complete.")
    print(f"  Processed {processed_rows} rows for indexing.")
    print(f"  Skipped {skipped_rows} rows (e.g., missing abstract or processing error).")
    print(f"  Total items potentially in collection '{collection_name}': {index_collection.count()}")

    # --- Build BM25 Index ---
    if all_processed_ids:
        print("\n--- Building BM25 Index ---")
        try:
            build_and_save_bm25_index(
                chunk_ids=all_processed_ids,
                chunk_texts=all_processed_documents,
                db_path=db_path,
                collection_name=collection_name
            )
        except Exception as bm25_err:
            print(f"!!! Error building/saving BM25 index: {bm25_err}")
            traceback.print_exc()
    else:
        print("\nSkipping BM25 index build: No documents were successfully processed.")
    # --- End Build BM25 Index ---


    # == Phase 2: Embedding Generation ==
    print("\n--- Phase 2: Generating Embeddings ---")

    # Initialize LLM Client (needed for embedding generation)
    print("Initializing LLM clients...")
    llm_interface.initialize_clients() # Call initialize_clients via the module

    # Check if Gemini client is available (required for embedding)
    # Access gemini_client via the module namespace AFTER initialization
    if llm_interface.gemini_client is None:
        print("Error: Google GenAI client (Gemini) failed to initialize properly.")
        print("Cannot proceed with embedding generation.")
        return

    print("Reconnecting to ChromaDB in 'embed' mode...")
    try:
        # Get collection in 'embed' mode (uses actual embedding function)
        embed_collection = get_chroma_collection(
            db_path=db_path,
            collection_name=collection_name,
            execution_mode="embed"
        )
        print("Connected.")
    except Exception as e:
        print(f"Error initializing ChromaDB for embedding: {e}")
        traceback.print_exc()
        return # Stop processing

    print("Finding items that need embedding (has_embedding: False)...")
    try:
        # Update to receive documents as well
        ids_to_embed, metadatas_to_embed, documents_to_embed = find_chunks_to_embed(embed_collection)
    except Exception as e:
        print(f"Error finding items to embed: {e}")
        traceback.print_exc()
        return

    if not ids_to_embed:
        print("No items found requiring embedding.")
    else:
        print(f"Found {len(ids_to_embed)} items to embed.")
        # Prepare config for embedding function (can be simplified if defaults are okay)
        config_params = {
            'embed_batch_size': DEFAULT_EMBED_BATCH_SIZE,
            'embed_delay': DEFAULT_EMBED_DELAY,
            # These are implicitly used by get_embedding via config, but good to be aware
            'embedding_model': EMBEDDING_MODEL,
            'output_dimension': OUTPUT_EMBEDDING_DIMENSION
        }

        print(f"Starting embedding generation with batch size {config_params['embed_batch_size']} and delay {config_params['embed_delay']}s...")
        # Generate embeddings and update DB incrementally
        failed_embed_ids = generate_embeddings_in_batches(
            collection=embed_collection,
            ids_to_embed=ids_to_embed,
            metadatas_to_embed=metadatas_to_embed,
            documents_to_embed=documents_to_embed, # Pass the retrieved documents
            batch_size=config_params['embed_batch_size'],
            delay=config_params['embed_delay'],
            client=llm_interface.gemini_client # Pass the client accessed via the module
        )

        # Report final failures
        if failed_embed_ids:
            print(f"\nEmbedding Phase Summary: Failed to process/embed {len(failed_embed_ids)} unique items.")
            print("These items were not updated and remain marked as 'has_embedding': False.")
            # Consider logging failed_embed_ids if needed
        else:
            print("\nEmbedding Phase Summary: All found items requiring embeddings were processed successfully.")

    print("\n--- CSV Ingestion Process Finished ---")