import os
import datetime
import traceback
from typing import List, Dict, Tuple
from tqdm import tqdm

# --- Local Imports ---
from rag.utils import compute_file_hash, chunk_document_tokens, count_tokens
from rag.llm_interface import generate_chunk_context
from rag.chroma_manager import get_chroma_collection
from rag.bm25_manager import build_and_save_bm25_index, tokenize_text_bm25 # Import tokenizer if needed here
from .config import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_OVERLAP,
    DEFAULT_TOTAL_CONTEXT_WINDOW,
    CHUNK_CONTEXT_MODEL
)

# --- File Discovery ---
def find_files_to_index(folder_path: str, document_path: str) -> List[str]:
    """Identifies potential .txt files to index from folder or single path."""
    potential_files = []
    if folder_path:
        if not os.path.isdir(folder_path):
            raise ValueError(f"Folder not found: {folder_path}")
        print(f"Scanning folder: {folder_path}")
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(".txt"):
                    potential_files.append(os.path.join(root, file))
    elif document_path:
        if not os.path.isfile(document_path):
            raise ValueError(f"File not found: {document_path}")
        if document_path.lower().endswith(".txt"):
            potential_files.append(document_path)
        else:
            print(f"Warning: Skipping non-txt file: {document_path}")

    if not potential_files:
        print("No source .txt files found to index.")
    else:
        print(f"Found {len(potential_files)} potential source file(s).")
    return potential_files

# --- File Filtering (Check against DB) ---
def filter_files_for_processing(potential_files: List[str], db_path: str, collection_name: str, force_reindex: bool) -> Tuple[List[str], int]:
    """Checks ChromaDB for existing file hashes and filters the list of files."""
    files_to_process = []
    skipped_files_count = 0
    existing_hashes = set()

    if not force_reindex:
        print("Checking ChromaDB for already indexed files (based on hash)...")
        try:
            # Use 'index' mode for get_collection to ensure correct embedding func behavior if called
            collection = get_chroma_collection(db_path, collection_name, execution_mode="index")
            # Fetch only metadata, specifically looking for 'file_hash'
            # Use a reasonable limit initially, potentially fetching all if needed,
            # but getting all metadata can be slow for large collections.
            # A more robust approach might involve querying distinct file_hashes if Chroma supports it well.
            # For simplicity, let's fetch metadata in batches if the collection is large.
            estimated_count = collection.count()
            fetch_limit = 10000 # Adjust as needed
            if estimated_count > 0:
                 print(f"Fetching metadata for up to {min(estimated_count, fetch_limit*5)} existing chunks to check hashes...") # Limit check scope
                 # Fetching all metadata can be very slow, get only needed field if possible
                 # ChromaDB `get` might not support selecting specific metadata fields easily.
                 # Fetch in batches to avoid memory issues.
                 offset = 0
                 while offset < estimated_count and offset < fetch_limit*5: # Limit total checked
                     results = collection.get(limit=fetch_limit, offset=offset, include=['metadatas'])
                     if not results or not results.get('ids'): break # No more results
                     metadatas = results.get('metadatas', [])
                     for meta in metadatas:
                         if meta and 'file_hash' in meta:
                             existing_hashes.add(meta['file_hash'])
                     offset += len(results['ids'])
                     if len(results['ids']) < fetch_limit: break # Reached the end

                 print(f"Found {len(existing_hashes)} unique file hashes from checked chunks in ChromaDB.")
            else:
                 print("Collection is empty, no existing hashes found.")

        except Exception as e:
            # Handle case where collection doesn't exist yet gracefully
            if "does not exist" in str(e).lower() or "not found" in str(e).lower():
                 print("Collection not found or empty, assuming no files are indexed yet.")
            else:
                 print(f"Warning: Could not check existing files in ChromaDB: {e}. Processing all files.")
                 # traceback.print_exc() # Optional: for debugging DB connection issues
    else:
        print("Force re-index enabled, processing all found files.")

    print("Filtering files to process...")
    skipped_files_info = [] # Store names of skipped files
    for file_path in tqdm(potential_files, desc="Checking files", unit="file"):
        try:
            file_hash = compute_file_hash(file_path)
            if not force_reindex and file_hash in existing_hashes:
                skipped_files_info.append(os.path.basename(file_path))
                skipped_files_count += 1
            else:
                files_to_process.append(file_path)
        except FileNotFoundError:
            print(f"\nWarning: File not found during check: {file_path}. Skipping.")
            skipped_files_count += 1
        except Exception as e:
            print(f"\nError hashing file {os.path.basename(file_path)}, skipping: {e}")
            skipped_files_count += 1

    if skipped_files_count > 0:
        print(f"Skipping {skipped_files_count} file(s) already indexed or with errors.")
        # if skipped_files_info: print(f"  Skipped files: {', '.join(skipped_files_info[:10])}{'...' if len(skipped_files_info) > 10 else ''}")

    return files_to_process, skipped_files_count


# --- Document Processing (Chunking & Context) ---
def index_document_phase1(document_path: str,
                          max_tokens: int = DEFAULT_MAX_TOKENS,
                          overlap: int = DEFAULT_OVERLAP,
                          context_total_window: int = DEFAULT_TOTAL_CONTEXT_WINDOW
                         ) -> List[Dict]:
    """
    Processes a single document: reads, chunks, gets context.
    Returns list of dicts {id, metadata, document_text}. Metadata includes 'has_embedding': False.
    """
    processed_chunk_data = []
    try:
        # print(f"Processing document: {os.path.basename(document_path)}") # Debug
        with open(document_path, "r", encoding="utf-8", errors='replace') as f:
            document_text = f.read()
    except Exception as e:
        print(f"Error reading {os.path.basename(document_path)}: {e}")
        return [] # Return empty list on read error

    if not document_text or not document_text.strip():
        # print(f"Skipping empty document: {os.path.basename(document_path)}")
        return []

    # Compute hash once per document
    try:
        file_hash = compute_file_hash(document_path)
    except Exception as e:
        print(f"Error computing hash for {os.path.basename(document_path)}: {e}. Skipping document.")
        return []

    processing_date = datetime.datetime.now().isoformat()

    # Chunk the document
    raw_chunks_with_indices = chunk_document_tokens(document_text, max_tokens=max_tokens, overlap=overlap)
    if not raw_chunks_with_indices:
        # print(f"No chunks generated for {os.path.basename(document_path)}.")
        return []

    # print(f"Generated {len(raw_chunks_with_indices)} raw chunks for {os.path.basename(document_path)}.") # Debug

    # Process each chunk
    for idx, (raw_chunk, start_tok, end_tok) in enumerate(raw_chunks_with_indices):
        if not raw_chunk or not raw_chunk.strip(): continue

        # Generate context for the chunk
        chunk_context = generate_chunk_context(
            full_document_text=document_text,
            chunk_text=raw_chunk,
            start_token_idx=start_tok,
            end_token_idx=end_tok,
            total_window_tokens=context_total_window,
            context_model=CHUNK_CONTEXT_MODEL
        )

        # Handle context generation failure (use placeholder)
        if "Error generating context" in chunk_context or "Context could not be generated" in chunk_context:
             # print(f"Warning: Using placeholder context for chunk {idx} in {os.path.basename(document_path)}.")
             chunk_context = "Context generation failed or unavailable." # Consistent placeholder

        # Combine context and raw chunk for potential use in embedding/retrieval
        contextualized_text = f"Context: {chunk_context}\n\nText:\n{raw_chunk}"
        tokens_count = count_tokens(raw_chunk) # Count tokens of the raw chunk

        chunk_id = f"{file_hash}_{idx}"
        metadata = {
            "file_hash": file_hash,
            "file_name": os.path.basename(document_path),
            "processing_date": processing_date,
            "chunk_number": idx,
            "start_token": start_tok,
            "end_token": end_tok,
            "text": raw_chunk, # Store the raw chunk text
            "context": chunk_context, # Store the generated context
            "contextualized_text": contextualized_text, # Store combined text
            "tokens": tokens_count,
            "has_embedding": False # Crucial flag for Phase 2 (embedding)
        }

        processed_chunk_data.append({
            "id": chunk_id,
            "metadata": metadata,
            "document": raw_chunk # The 'document' for ChromaDB should be the text to be indexed/searched
                                  # Usually the raw chunk, maybe contextualized text if preferred. Let's use raw.
        })

    return processed_chunk_data

# --- Sequential File Processing ---
def process_files_sequentially(files_to_process: List[str]) -> Tuple[List[Dict], List[Tuple[str, str]]]:
    """Processes a list of files sequentially for Phase 1 (chunking)."""
    all_phase1_chunks = []
    failed_files_info = [] # List of (filename, error_message)

    print("--- Starting Sequential Chunk Processing (Phase 1) ---")
    for file_path in tqdm(files_to_process, desc="Processing Files (Phase 1)", unit="file"):
        try:
            # Call the single document processing function
            processed_data = index_document_phase1(
                document_path=file_path,
                max_tokens=DEFAULT_MAX_TOKENS, # Use config defaults
                overlap=DEFAULT_OVERLAP,
                context_total_window=DEFAULT_TOTAL_CONTEXT_WINDOW
            )
            if processed_data:
                all_phase1_chunks.extend(processed_data)
            # else: print(f"No chunks generated or processed for {os.path.basename(file_path)}") # Debug
        except Exception as e:
            err_msg = f"CRITICAL Error processing {os.path.basename(file_path)} (Phase 1): {e}"
            print(f"\n{err_msg}")
            # traceback.print_exc() # Optionally print full traceback
            failed_files_info.append((os.path.basename(file_path), str(e)))

    successful_files = len(files_to_process) - len(failed_files_info)
    print(f"\n--- Phase 1 Processing Summary ---")
    print(f"Attempted to process: {len(files_to_process)} files")
    print(f"Successfully processed (yielding chunks): {successful_files}") # Note: Success here means no critical error, might still yield 0 chunks.
    if failed_files_info:
        print(f"Failed processing attempts (critical errors): {len(failed_files_info)}")
        for fname, err in failed_files_info[:5]: print(f"  - Example Failure: {fname}: {err}")
        if len(failed_files_info) > 5: print("  ...")

    return all_phase1_chunks, failed_files_info


# --- Update ChromaDB with Raw Chunks ---
def update_chromadb_raw_chunks(collection, all_phase1_chunks: List[Dict]):
    """Adds/Updates raw chunk data (Phase 1) to ChromaDB, omitting embeddings."""
    if not all_phase1_chunks:
        print("No new raw chunks to add/update in ChromaDB.")
        return 0 # Return count of updated chunks

    print(f"Adding/Updating {len(all_phase1_chunks)} raw chunks in ChromaDB (metadata and documents)...")
    chroma_ids = [chunk['id'] for chunk in all_phase1_chunks]
    chroma_metadatas = [chunk['metadata'] for chunk in all_phase1_chunks]
    # The 'document' stored in Chroma should be the text used for retrieval/embedding later.
    # Typically the raw chunk text, but could be contextualized text if preferred.
    # Let's stick to raw text ('text' field in metadata) for consistency with BM25.
    chroma_documents = [chunk['metadata'].get('text', '') for chunk in all_phase1_chunks]

    updated_count = 0
    batch_size = 140 # Keep batch size reasonable
    num_batches = (len(chroma_ids) + batch_size - 1) // batch_size

    for i in tqdm(range(num_batches), desc="Upserting Raw Chunks to ChromaDB"):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        batch_ids = chroma_ids[start_idx:end_idx]
        batch_metadatas = chroma_metadatas[start_idx:end_idx]
        batch_documents = chroma_documents[start_idx:end_idx]

        if not batch_ids: continue
        try:
            # Use upsert: adds new chunks and updates existing ones based on ID.
            # Crucially, DO NOT provide the 'embeddings' parameter here.
            # The collection's embedding function (ConfigurableEmbeddingFunction in 'index' mode)
            # should handle returning zero vectors if Chroma calls it implicitly during upsert.
            # print(f"DEBUG: Calling upsert for batch {i+1} (IDs: {batch_ids[:3]}...)") # Debug
            collection.upsert(
                ids=batch_ids,
                metadatas=batch_metadatas,
                documents=batch_documents # Provide the text content
                # NO 'embeddings=' parameter here!
            )
            # print(f"DEBUG: Upsert for batch {i+1} completed.") # Debug
            updated_count += len(batch_ids)
        except Exception as upsert_err:
            print(f"\n!!! Error upserting batch {i+1}/{num_batches} to ChromaDB: {upsert_err}")
            print(f"!!! Failed IDs in this batch: {batch_ids}")
            traceback.print_exc() # Provide full traceback for upsert errors
            # Decide on error handling: continue? stop? retry? For now, continue.

    print(f"Finished upserting raw chunks. {updated_count} chunks processed in upsert calls.")
    return updated_count


# --- Rebuild BM25 Index ---
def rebuild_bm25_index_from_chroma(collection, db_path: str, collection_name: str):
    """Rebuilds and saves the BM25 index using all current data in ChromaDB."""
    print("Rebuilding BM25 index using all data currently in ChromaDB...")
    all_chunk_ids = []
    all_chunk_texts = []
    try:
        # Fetch all data (IDs and metadata containing text) from ChromaDB
        # This can be slow for very large collections. Fetch in batches.
        estimated_count = collection.count()
        if estimated_count == 0:
             print("Collection is empty. Skipping BM25 rebuild.")
             return
        print(f"Fetching data for {estimated_count} chunks from ChromaDB for BM25 rebuild...")
        fetch_limit = 5000 # Adjust batch size for fetching
        offset = 0
        with tqdm(total=estimated_count, desc="Fetching ChromaDB data") as pbar:
            while True:
                results = collection.get(
                    limit=fetch_limit,
                    offset=offset,
                    include=['metadatas'] # Only need metadata containing 'text'
                )
                if not results or not results.get('ids'): break

                ids_batch = results['ids']
                metadatas_batch = results.get('metadatas', [])

                if len(ids_batch) != len(metadatas_batch):
                     print(f"Warning: Mismatch between IDs ({len(ids_batch)}) and Metadatas ({len(metadatas_batch)}) in fetched batch. Skipping batch.")
                     # Potentially log problematic IDs/offset
                else:
                    for chunk_id, meta in zip(ids_batch, metadatas_batch):
                        # Ensure metadata exists and contains the 'text' field
                        if meta and 'text' in meta and meta['text'] and meta['text'].strip():
                            all_chunk_ids.append(chunk_id)
                            all_chunk_texts.append(meta['text'])
                        # else: print(f"Warning: Skipping chunk {chunk_id} due to missing/empty text in metadata.") # Too verbose

                pbar.update(len(ids_batch))
                offset += len(ids_batch)
                if len(ids_batch) < fetch_limit: break # Reached the end

        if not all_chunk_ids:
            print("No valid chunk text found in ChromaDB metadata. Skipping BM25 build.")
            return

        print(f"Found {len(all_chunk_ids)} valid chunks with text in ChromaDB.")

        # Build and save the index using the dedicated bm25_manager function
        build_and_save_bm25_index(all_chunk_ids, all_chunk_texts, db_path, collection_name)

    except Exception as bm25_err:
        print(f"!!! Error during BM25 rebuild phase (fetching or building): {bm25_err}")
        traceback.print_exc()
