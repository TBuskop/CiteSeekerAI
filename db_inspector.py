#!/usr/bin/env python3
"""
db_inspector.py

A utility script to inspect a ChromaDB database created by rag_chroma.py.
It displays:
- Basic collection information (total item count, metadata).
- Count and list of unique source filenames found in metadata.
- A sample of chunk IDs, their associated metadata, and the chunk text.

Usage:
  python db_inspector.py --db_path ./hybrid_db --collection_name my_hybrid_docs [--limit 5]

Arguments:
  --db_path          (Required) Path to the ChromaDB persistent storage directory.
  --collection_name  (Required) Name of the collection within the database to inspect.
  --limit            (Optional) Number of sample chunks to display (default: 5).
"""

import argparse
import os
import sys
import chromadb
from chromadb.errors import NotFoundError
import pprint # For pretty printing metadata dictionaries
from tqdm import tqdm # Add tqdm for progress on large metadata fetch

# --- Configuration ---
DEFAULT_SAMPLE_LIMIT = 5
MAX_TEXT_DISPLAY_LEN = 300 # Max characters of chunk text to display
METADATA_FETCH_BATCH_SIZE = 5000 # Batch size for fetching all metadata if needed

def display_truncated_text(text: str, max_len: int = MAX_TEXT_DISPLAY_LEN) -> str:
    """Truncates text for cleaner display."""
    if not isinstance(text, str):
        return str(text) # Return string representation if not a string
    if len(text) > max_len:
        return text[:max_len // 2] + " ... [truncated] ... " + text[-(max_len // 2):]
    return text

def inspect_chromadb(db_path: str, collection_name: str, limit: int):
    """Connects to ChromaDB, retrieves info, and prints samples."""

    print(f"--- ChromaDB Inspector ---")
    print(f"Database Path: {os.path.abspath(db_path)}")
    print(f"Collection Name: {collection_name}")
    print(f"Sample Limit: {limit}")
    print("-" * 26)

    if not os.path.isdir(db_path):
        print(f"Error: Database path '{db_path}' not found or is not a directory.", file=sys.stderr)
        sys.exit(1)

    try:
        print("Connecting to ChromaDB...")
        client = chromadb.PersistentClient(path=db_path)
    except Exception as e:
        print(f"Error: Failed to connect to ChromaDB at '{db_path}': {e}", file=sys.stderr)
        sys.exit(1)

    try:
        print(f"Accessing collection '{collection_name}'...")
        collection = client.get_collection(name=collection_name)
        print("Collection accessed successfully.")
    except NotFoundError:
        print(f"\nError: Collection '{collection_name}' not found. Database path: '{db_path}'.", file=sys.stderr)
        print("\nAvailable collections:")
        try:
            collections = client.list_collections()
            if collections:
                for coll in collections:
                    print(f"- {coll.name}")
            else:
                print("(No collections found in this database)")
        except Exception as list_e:
            print(f"(Error listing collections: {list_e})")
        sys.exit(1)
    except Exception as e:
        print(f"Error: Failed to get collection '{collection_name}' due to unexpected exception: {type(e).__name__}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

    try:
        # --- Get Collection Information ---
        print("\n--- Collection Information ---")
        item_count = collection.count()
        print(f"Total Items (Chunks): {item_count}")

        collection_metadata = collection.metadata
        if collection_metadata:
            print("Collection Metadata:")
            pprint.pprint(collection_metadata, indent=2)
        else:
            print("Collection Metadata: (None specified)")

        if item_count == 0:
            print("\nCollection is empty. Cannot analyze filenames or sample chunks.")
            return # Exit the function cleanly

        # --- Get Unique Filenames ---
        print("\n--- Analyzing Unique Source Filenames ---")
        print(f"Fetching metadata for all {item_count} items (this may take a moment)...")
        unique_filenames = set()
        all_metadatas = []
        
        # Fetch metadata in batches to handle potentially large collections
        try:
            num_batches = (item_count + METADATA_FETCH_BATCH_SIZE - 1) // METADATA_FETCH_BATCH_SIZE
            offset = 0
            for _ in tqdm(range(num_batches), desc="Fetching metadata batches", unit="batch"):
                 batch_data = collection.get(
                     limit=METADATA_FETCH_BATCH_SIZE,
                     offset=offset,
                     include=['metadatas']
                 )
                 batch_metadatas = batch_data.get('metadatas')
                 if batch_metadatas:
                     all_metadatas.extend(batch_metadatas)
                 else:
                     # Stop if a batch returns no metadata (shouldn't happen if offset logic is correct)
                     print(f"\nWarning: Batch starting at offset {offset} returned no metadata. Stopping fetch.")
                     break
                 offset += METADATA_FETCH_BATCH_SIZE
                 if offset >= item_count: # Break if we've theoretically fetched everything
                     break
                     
            print(f"Finished fetching metadata. Found {len(all_metadatas)} metadata entries.")

        except Exception as meta_fetch_err:
            print(f"\nError fetching all metadata: {meta_fetch_err}. Filename analysis may be incomplete.", file=sys.stderr)


        if not all_metadatas:
            print("Could not retrieve metadata to analyze filenames.")
        else:
            # Process the fetched metadata
            processed_count = 0
            skipped_meta_count = 0
            for meta in all_metadatas:
                if meta and isinstance(meta, dict) and 'file_name' in meta:
                    filename = meta['file_name']
                    if filename and isinstance(filename, str) and filename.strip():
                        unique_filenames.add(filename.strip())
                        processed_count += 1
                    else:
                        # Count entries with 'file_name' key but empty/invalid value
                         skipped_meta_count +=1
                else:
                    # Count entries missing metadata or 'file_name' key
                    skipped_meta_count += 1
            
            if skipped_meta_count > 0:
                print(f"Note: Skipped {skipped_meta_count} entries during filename analysis due to missing/invalid metadata or empty 'file_name'.")

            unique_count = len(unique_filenames)
            print(f"\nTotal Unique Source Filenames Found: {unique_count}")

            if unique_count > 0:
                print("List of Unique Filenames:")
                # Sort for consistent output
                sorted_filenames = sorted(list(unique_filenames))
                for fname in sorted_filenames:
                    print(f"- {fname}")

        # --- Get Sample Chunk Data (Existing Logic) ---
        # Ensure limit is not greater than the total items
        actual_limit = min(limit, item_count)
        if actual_limit < limit:
            print(f"\nNote: Requested {limit} samples, but only {item_count} items exist.")
        elif actual_limit == 0:
             print("\nNo items available to sample.")
             return # Skip sampling if limit somehow became 0

        print(f"\n--- Sample Chunks (Showing {actual_limit}) ---")

        # Get sample data using limit
        samples = collection.get(
            limit=actual_limit,
            include=['metadatas', 'documents'] # Explicitly ask for docs and metadata
        )

        if not samples or not samples.get('ids'):
            print("\nWarning: Could not retrieve samples from the collection (maybe empty or error?).")
            return

        ids = samples.get('ids', [])
        documents = samples.get('documents', [])
        metadatas_sample = samples.get('metadatas', []) # Use a different variable name

        # Ensure all retrieved lists have the same length for zipping
        num_retrieved = len(ids)
        if not (num_retrieved == len(documents) == len(metadatas_sample)):
             print(f"\nWarning: Mismatch in retrieved sample data lengths (ids: {len(ids)}, docs: {len(documents)}, meta: {len(metadatas_sample)}). Displaying minimum common items.")
             num_retrieved = min(len(ids), len(documents), len(metadatas_sample))
             ids = ids[:num_retrieved]
             documents = documents[:num_retrieved]
             metadatas_sample = metadatas_sample[:num_retrieved]

        if num_retrieved == 0:
            print("\nNo sample data retrieved despite non-zero count. Check DB integrity.")
            return

        # Display Samples
        for i in range(num_retrieved):
            chunk_id = ids[i]
            # Use the metadata from the sample fetch
            chunk_meta = metadatas_sample[i] if metadatas_sample and i < len(metadatas_sample) else {}
            chunk_doc = documents[i] if documents and i < len(documents) else "[Document Text Missing]"

            print(f"\n--- Sample {i+1}/{actual_limit} ---")
            print(f"Chunk ID: {chunk_id}")

            print("Metadata:")
            if chunk_meta and isinstance(chunk_meta, dict): # Check if it's a dict
                 pprint.pprint(chunk_meta, indent=2, width=100)
            elif chunk_meta:
                 print(f"  (Unexpected metadata format: {type(chunk_meta)})")
                 print(f"  Raw: {chunk_meta}")
            else:
                 print("  (No metadata found for this sample chunk)")


            print("\nChunk Text (Document):")
            print(display_truncated_text(chunk_doc))
            print("-" * 20) # Separator for the next chunk

    except Exception as e:
        print(f"\nError during data retrieval or display: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(
        description="Inspect a ChromaDB collection created by rag_chroma.py.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--db_path", required=True, type=str,
                        help="Path to the persistent ChromaDB directory.")
    parser.add_argument("--collection_name", required=True, type=str,
                        help="Name of the ChromaDB collection to inspect.")
    parser.add_argument("--limit", type=int, default=DEFAULT_SAMPLE_LIMIT,
                        help="Number of sample chunks to display.")

    args = parser.parse_args()

    if args.limit < 0:
        print("Warning: --limit must be a positive integer. Using default.")
        args.limit = DEFAULT_SAMPLE_LIMIT

    inspect_chromadb(args.db_path, args.collection_name, args.limit)

if __name__ == "__main__":
    main()
    # os.system("python db_inspector.py --db_path chunk_database/chunks_db --collection_name test_collection --limit 0")