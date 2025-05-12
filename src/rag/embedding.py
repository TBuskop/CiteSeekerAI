import time
import datetime
import traceback
from typing import List, Dict, Tuple, Any, Optional
from tqdm import tqdm

# --- Local Imports ---
from rag.chroma_manager import _update_db_batch # Use the helper from chroma_manager
from my_utils.llm_interface import get_embedding
from config import EMBEDDING_MODEL, OUTPUT_EMBEDDING_DIMENSION
# --- Add default values from config for fallback ---
from config import DEFAULT_EMBED_BATCH_SIZE, DEFAULT_EMBED_DELAY

# --- Google API Exceptions & Types ---
try:
    from google.api_core import exceptions as google_exceptions
    from google import genai # Import genai directly
    from google.genai.types import EmbedContentConfig # Import EmbedContentConfig
    EMBEDDING_IMPORTS_OK = True # Flag indicating successful import here
except ImportError:
    google_exceptions = None
    genai = None # Set to None on failure
    EmbedContentConfig = None # Set to None on failure
    EMBEDDING_IMPORTS_OK = False # Flag import failure


# --- Find Chunks Needing Embedding ---
def find_chunks_to_embed(collection) -> Tuple[List[str], List[Dict], List[str]]: # Add List[str] for documents
    """Finds chunks in the collection marked with 'has_embedding': False."""
    print(f"Checking collection '{collection.name}' for chunks needing embedding...")
    ids_to_embed = []
    metadatas_to_embed = []
    documents_to_embed = [] # Initialize list for documents
    try:
        # Use get() with the where filter to retrieve the actual items
        results = collection.get(
            where={"has_embedding": False},
            include=['metadatas', 'documents'] # Include 'documents'
        )
        ids_to_embed = results.get('ids', [])
        metadatas_to_embed = results.get('metadatas', [])
        documents_to_embed = results.get('documents', []) # Get the documents

        if not ids_to_embed:
            print(f"No chunks found needing embedding in collection '{collection.name}'.")
        else:
            # The actual count is simply the length of the retrieved IDs
            print(f"Found {len(ids_to_embed)} chunks to embed.")
            # Add a check for mismatched lengths (shouldn't happen ideally)
            if len(ids_to_embed) != len(documents_to_embed):
                print(f"Warning: Mismatch between number of IDs ({len(ids_to_embed)}) and documents ({len(documents_to_embed)}) retrieved.")
                # Handle this case? For now, proceed but log warning.

    except TypeError as te:
        # Catch if .get() also doesn't support where in some version (though it should)
        print(f"Error querying ChromaDB for chunks to embed (potentially unsupported 'where' filter in get?): {te}")
        print("Traceback:", traceback.format_exc())
    except Exception as e:
        print(f"Error querying ChromaDB for chunks to embed: {e}")
        print("Traceback:", traceback.format_exc())

    return ids_to_embed, metadatas_to_embed, documents_to_embed # Return documents


# --- Generate Embeddings in Batches ---
def generate_embeddings_in_batches(
    collection, # Pass the collection object directly
    ids_to_embed: List[str],
    metadatas_to_embed: List[Dict],
    documents_to_embed: List[str], # Add documents parameter
    batch_size: int,
    delay: float,
    client: Optional[Any] # Accept the client object
) -> List[str]: # Return only the list of unique failed IDs
    """
    Generates embeddings and updates ChromaDB incrementally after each successful batch.
    Handles rate limits and other API errors. Stops processing if a rate limit error (429) is encountered.

    Args:
        collection: The ChromaDB collection object to update.
        ids_to_embed: List of chunk IDs needing embedding.
        metadatas_to_embed: Corresponding metadata dictionaries.
        documents_to_embed: Corresponding document texts.
        batch_size: Number of chunks per API batch call.
        delay: Delay in seconds between batch API calls.
        client: The initialized API client object (e.g., Gemini client).

    Returns:
        A list of unique chunk IDs that failed during the process (due to errors or rate limits).
    """
    all_failed_ids = set() # Track unique failed IDs across all batches
    total_processed_count = 0
    total_updated_in_db = 0

    num_batches = (len(ids_to_embed) + batch_size - 1) // batch_size
    print(f"Processing {len(ids_to_embed)} chunks in {num_batches} batches of size {batch_size}...")


    # --- Process Batches ---
    for i in tqdm(range(num_batches), desc="Generating Embeddings & Updating DB", unit="batch"):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, len(ids_to_embed)) # Ensure end_idx doesn't exceed list length
        current_batch_ids = ids_to_embed[start_idx:end_idx]
        current_batch_metadatas = metadatas_to_embed[start_idx:end_idx]
        current_batch_documents = documents_to_embed[start_idx:end_idx] # Slice documents too

        if not current_batch_ids: continue # Skip empty batch slice

        batch_texts_to_embed = []
        batch_ids_prepared = []
        batch_metadatas_prepared = []

        # --- Prepare Batch Data (Extract Text) ---
        # Iterate using index to access corresponding document and metadata
        for idx_in_batch in range(len(current_batch_ids)):
            chunk_id = current_batch_ids[idx_in_batch]
            meta = current_batch_metadatas[idx_in_batch]
            doc_text = current_batch_documents[idx_in_batch] # Get text from documents list

            if not meta:
                # print(f"Warning: Missing metadata for chunk {chunk_id} in batch {i+1}. Skipping.")
                all_failed_ids.add(chunk_id)
                continue

            # Use the retrieved document text directly
            text_to_embed = doc_text
            if not text_to_embed or not text_to_embed.strip():
                # print(f"Warning: Empty text for chunk {chunk_id} in batch {i+1}. Skipping.")
                all_failed_ids.add(chunk_id)
                continue

            text_to_embed = text_to_embed.replace("\n", " ").strip() # Preprocess
            if not text_to_embed:
                # print(f"Warning: Text became empty after preprocessing for chunk {chunk_id}. Skipping.")
                all_failed_ids.add(chunk_id)
                continue

            # Store prepared data
            batch_texts_to_embed.append(text_to_embed)
            batch_ids_prepared.append(chunk_id)
            batch_metadatas_prepared.append(meta)


        if not batch_texts_to_embed:
            # print(f"Skipping batch {i+1} as it contains no valid texts to embed.")
            if delay > 0 and i < num_batches - 1: time.sleep(delay) # Apply delay anyway
            continue # Skip to the next batch

        # --- Make Batch Embedding API Call ---
        batch_embeddings_ok: List[List[float]] = []
        batch_ids_ok: List[str] = []
        batch_metadatas_ok: List[Dict] = []
        rate_limit_hit = False

        try:
            # Check conditions for using Gemini batch embedding more robustly
            # Use EMBEDDING_IMPORTS_OK to ensure genai and EmbedContentConfig were imported in *this* module
            is_gemini_batch_possible = False # Default to false
            # Check imports succeeded before attempting isinstance
            if EMBEDDING_IMPORTS_OK and "embedding" in EMBEDDING_MODEL and isinstance(client, genai.Client):
                 is_gemini_batch_possible = True

            if is_gemini_batch_possible:
                 api_model_name = EMBEDDING_MODEL if EMBEDDING_MODEL.startswith("models/") else f"models/{EMBEDDING_MODEL}"
                 task_type_api = "RETRIEVAL_DOCUMENT" # Default for embedding stored chunks
                 embed_config_args = {'task_type': task_type_api}
                 if OUTPUT_EMBEDDING_DIMENSION is not None and "text-embedding-004" in api_model_name:
                     embed_config_args['output_dimensionality'] = OUTPUT_EMBEDDING_DIMENSION

                 # print(f"DEBUG: Calling client.models.embed_content (batch) with {len(batch_texts_to_embed)} texts...")
                 # Use the passed client object
                 response = client.models.embed_content(
                     model=api_model_name,
                     contents=batch_texts_to_embed, # Pass the list of texts to 'contents'
                     config=EmbedContentConfig(**embed_config_args) if embed_config_args else None
                 )

                 # Process batch response - updated for EmbedContentResponse
                 if hasattr(response, 'embeddings') and response.embeddings and len(response.embeddings) == len(batch_texts_to_embed):
                     for idx, content_embedding in enumerate(response.embeddings):
                         chunk_id = batch_ids_prepared[idx]
                         original_meta = batch_metadatas_prepared[idx]

                         if hasattr(content_embedding, 'values') and isinstance(content_embedding.values, list) and len(content_embedding.values) > 0:
                             batch_embeddings_ok.append(content_embedding.values)
                             updated_meta = original_meta.copy()
                             updated_meta['has_embedding'] = True
                             batch_metadatas_ok.append(updated_meta)
                             batch_ids_ok.append(chunk_id)
                         else:
                             print(f"\nWarning: Failed to get valid embedding vector (ContentEmbedding.values) for chunk {chunk_id} in batch {i+1}.")
                             all_failed_ids.add(chunk_id)
                 else:
                     print(f"\nError: Invalid or mismatched API response (EmbedContentResponse) for batch {i+1}. Failing entire batch.")
                     for chunk_id in batch_ids_prepared: all_failed_ids.add(chunk_id)

            else:
                 # Fallback to iterating: Provide a more specific warning
                 warning_reason = ""
                 if "embedding" not in EMBEDDING_MODEL:
                     warning_reason = "Not a Gemini embedding model configured."
                 elif not EMBEDDING_IMPORTS_OK:
                     warning_reason = "Required Google GenAI libraries (genai, types) failed to import in embedding module."
                 # Check client type again, adding check for genai being None
                 elif not genai or not isinstance(client, genai.Client):
                      warning_reason = f"Passed client object is not a valid genai.Client instance (genai loaded: {genai is not None}, client type: {type(client)})."
                 else:
                      warning_reason = "Unknown reason." # Fallback

                 print(f"Warning: Gemini batch embedding conditions not met ({warning_reason}). Using iterative embedding calls.")

                 for idx, text in enumerate(batch_texts_to_embed):
                     chunk_id = batch_ids_prepared[idx]
                     original_meta = batch_metadatas_prepared[idx]
                     # Call the single embedding function from llm_interface
                     # Pass the client object to get_embedding
                     embedding_np = get_embedding(
                         text=text,
                         model=EMBEDDING_MODEL,
                         task_type="retrieval_document",
                         embedding_dimension=OUTPUT_EMBEDDING_DIMENSION,
                         client=client # Pass the client received by this function
                     )
                     if embedding_np is not None and embedding_np.size > 0:
                         batch_embeddings_ok.append(embedding_np.tolist())
                         updated_meta = original_meta.copy()
                         updated_meta['has_embedding'] = True
                         batch_metadatas_ok.append(updated_meta)
                         batch_ids_ok.append(chunk_id)
                     else:
                         # print(f"\nWarning: Failed to embed chunk {chunk_id} in batch {i+1} (iterative).")
                         all_failed_ids.add(chunk_id)
                     # Apply intra-batch delay if needed for iterative calls
                     # if delay > 0.01 and idx < len(batch_texts_to_embed) - 1: time.sleep(max(0.01, delay / len(batch_texts_to_embed)))


        except google_exceptions.ResourceExhausted as rate_limit_error:
            rate_limit_hit = True
            print(f"\n!!! Rate Limit Error (429) encountered during batch {i+1}: {rate_limit_error}")
            print("Stopping further embedding in this run.")
            # Mark all items prepared for *this specific batch* as failed
            for chunk_id in batch_ids_prepared: all_failed_ids.add(chunk_id)

        except Exception as embed_err:
             print(f"\n!!! Critical Error during API call for batch {i+1}: {embed_err}")
             traceback.print_exc()
             # Mark all items prepared for this batch as failed
             for chunk_id in batch_ids_prepared: all_failed_ids.add(chunk_id)

        # --- Update DB with successful results from THIS batch ---
        if batch_ids_ok:
            try:
                # Use the helper function imported from chroma_manager
                _update_db_batch(collection, batch_ids_ok, batch_embeddings_ok, batch_metadatas_ok)
                total_updated_in_db += len(batch_ids_ok)
            except Exception:
                # _update_db_batch already prints error, mark these as failed too
                print(f"!!! Marking {len(batch_ids_ok)} chunks from batch {i+1} as failed due to DB update error.")
                for chunk_id in batch_ids_ok: all_failed_ids.add(chunk_id)

        total_processed_count += len(batch_ids_prepared) # Count attempts in this batch

        # --- Check if we need to stop due to rate limit ---
        if rate_limit_hit:
            break # Exit the loop immediately

        # --- Apply Inter-Batch Delay ---
        if delay > 0 and i < num_batches - 1:
            # print(f"Waiting {delay}s before next batch...")
            time.sleep(delay)

    # --- End of Loop ---
    print(f"\nEmbedding generation loop finished.")
    print(f"Attempted to process {total_processed_count} chunks.")
    print(f"Successfully updated {total_updated_in_db} chunks in the database during this run.")
    unique_failed_ids = list(all_failed_ids)
    if unique_failed_ids:
        print(f"Encountered {len(unique_failed_ids)} unique failures (check logs). These chunks were not embedded or updated.")

    return unique_failed_ids # Return list of all unique failed IDs


# --- Main Embed Mode Runner ---
def run_embed_mode_logic(config_params: Dict[str, Any], collection: 'chromadb.Collection', client: Optional[Any]):
    """
    Handles the core logic for finding, embedding, and updating chunks.
    Accepts a configuration dictionary and the initialized client object.
    """
    print("Finding chunks that need embeddings...")
    ids_to_embed, metadatas_to_embed, documents_to_embed = find_chunks_to_embed(collection)

    if not ids_to_embed:
        print("No chunks found needing embedding in this collection.")
        return # Nothing more to do

    print(f"Found {len(ids_to_embed)} chunks to process.")

    # --- Get parameters from config_params dictionary ---
    batch_size = config_params.get('embed_batch_size', DEFAULT_EMBED_BATCH_SIZE)
    delay = config_params.get('embed_delay', DEFAULT_EMBED_DELAY)
    # --- End parameter extraction ---

    # Generate embeddings and update DB incrementally
    failed_embed_ids = generate_embeddings_in_batches(
        collection=collection,
        ids_to_embed=ids_to_embed,
        metadatas_to_embed=metadatas_to_embed,
        documents_to_embed=documents_to_embed, # Pass documents
        batch_size=batch_size, # Use extracted value
        delay=delay,           # Use extracted value
        client=client          # Pass the client object
    )

    # Report final failures
    if failed_embed_ids:
        print(f"\nSummary: Failed to process/embed {len(failed_embed_ids)} unique chunks during this run.")
        print("These chunks were not updated and likely remain marked as 'has_embedding': False.")
        print("You can re-run '--mode embed' to retry processing them.")

        # Optional: Log failed_ids to a file
        log_filename = "embedding_failures.log"
        try:
            with open(log_filename, "a", encoding="utf-8") as f:
                timestamp = datetime.datetime.now().isoformat()
                f.write(f"--- Run at {timestamp} ---\n")
                f.write(f"Failed IDs ({len(failed_embed_ids)}):\n")
                for fid in failed_embed_ids:
                     f.write(f"{fid}\n")
                f.write("-" * 20 + "\n")
            print(f"List of failed chunk IDs appended to '{log_filename}'.")
        except Exception as log_err:
            print(f"Warning: Could not write failed IDs to log file '{log_filename}': {log_err}")
    else:
        print("\nSummary: All found chunks requiring embeddings were processed successfully.")
