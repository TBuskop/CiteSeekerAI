import os
import traceback
from typing import List, Dict, Optional

import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings

# --- Local Imports ---
from rag.config import EMBEDDING_MODEL, OUTPUT_EMBEDDING_DIMENSION
# Import get_embedding carefully to avoid circular dependency if possible,
# or pass embedding function instance/callable.
# For ConfigurableEmbeddingFunction, we need get_embedding.
from rag.llm_interface import get_embedding


# --- Custom Embedding Function for ChromaDB ---
class ConfigurableEmbeddingFunction(EmbeddingFunction):
    """
    A ChromaDB EmbeddingFunction that can return zero vectors during indexing
    and uses the configured embedding model/client during embedding/querying.
    """
    def __init__(self,
                 model_name: Optional[str] = None,
                 output_dimension_override: Optional[int] = None,
                 task_type: str = "retrieval_document",
                 current_mode: str = "unknown"): # Mode determines behavior
        self._model_name = model_name if model_name else EMBEDDING_MODEL
        if not self._model_name:
             raise ValueError("Embedding model name must be provided.")

        self._mode = current_mode
        self._task_type = task_type # Task type for actual embedding calls

        # Determine dimension: Override > Config > Default (None)
        determined_dimension = output_dimension_override if output_dimension_override is not None else OUTPUT_EMBEDDING_DIMENSION
        self.dimension = determined_dimension

        # Dimension is crucial unless in 'index' mode (where zeros are returned)
        if self.dimension is None and self._mode != "index":
             raise ValueError(f"Embedding dimension must be set (config/override) for mode '{self._mode}'.")
        elif self.dimension is None and self._mode == "index":
             # This is risky - Chroma might fail if it needs dimension info during collection creation/upsert
             # even if we return zeros later. Best practice is to always define OUTPUT_EMBEDDING_DIMENSION.
             print("WARNING (EmbeddingFunction Init): Dimension not set in 'index' mode. ChromaDB operations might fail if dimension is required upfront.")
             # We cannot generate zero vectors of unknown size later.

    def __call__(self, input_texts: Documents) -> Embeddings:
        """
        Generates embeddings or zero vectors based on the mode.
        """
        # print(f"DEBUG: ConfigurableEmbeddingFunction.__call__ invoked in mode '{self._mode}' for {len(input_texts)} texts.")

        # --- MODE: index ---
        # Return zero vectors. Requires self.dimension to be known.
        if self._mode == "index":
            if self.dimension is None:
                # This should not happen if the warning in __init__ was heeded, but is a fatal error here.
                print("ERROR (EmbeddingFunction): Cannot generate zero vectors in 'index' mode without a known dimension!")
                raise ValueError("Dimension required for placeholder vectors in index mode.")
            # print(f"INFO (EmbeddingFunction): In 'index' mode, returning zero vectors ({self.dimension}d).")
            zero_vector = [0.0] * self.dimension
            return [zero_vector for _ in input_texts]

        # --- MODES: embed, query, etc. ---
        # Generate actual embeddings using the llm_interface.get_embedding function.
        all_embeddings: Embeddings = []
        if self.dimension is None: # Should have been caught in init, but double-check
             raise ValueError("Dimension required for generating actual embeddings.")

        for text in input_texts:
            embedding = None
            try:
                # Call the actual embedding generation function
                embedding_np = get_embedding(
                    text=text,
                    model=self._model_name,
                    task_type=self._task_type, # Use task type set during init
                    embedding_dimension=self.dimension # Pass dimension if needed by model
                )

                if embedding_np is not None and embedding_np.size > 0:
                    # Optional dimension check (already done in get_embedding)
                    all_embeddings.append(embedding_np.tolist())
                else:
                    # Failed to get a valid embedding (API error handled in get_embedding, or empty result)
                    print(f"ERROR (EmbeddingFunction): Failed to get valid embedding for text: '{text[:50]}...'. Appending zero vector.")
                    all_embeddings.append([0.0] * self.dimension) # Append ZEROS

            except Exception as e:
                # Catch errors from get_embedding itself (like rate limits propagated)
                print(f"ERROR (EmbeddingFunction): Exception during embedding generation for text '{text[:50]}...': {e}. Appending zero vector.")
                # traceback.print_exc() # Optionally print traceback here too
                all_embeddings.append([0.0] * self.dimension) # Append ZEROS

        return all_embeddings

# --- ChromaDB Client and Collection Setup ---
def get_chroma_collection(db_path: str, collection_name: str, execution_mode: str) -> chromadb.Collection:
    """
    Gets or creates a ChromaDB collection with the appropriate embedding function behavior.

    Args:
        db_path: Path to the persistent database directory.
        collection_name: Name of the ChromaDB collection.
        execution_mode: The current script mode ('index', 'embed', 'query', etc.).

    Returns:
        A chromadb.Collection instance.
    """
    try:
        # Instantiate the custom embedding function, passing the mode
        emb_func = ConfigurableEmbeddingFunction(
            model_name=EMBEDDING_MODEL,
            current_mode=execution_mode
            # Dimension is determined internally using OUTPUT_EMBEDDING_DIMENSION
        )

        # Log the dimension the function instance holds (important for debugging)
        # print(f"DEBUG: Instantiated ConfigurableEmbeddingFunction for mode '{execution_mode}' with dimension: {emb_func.dimension}")

        # Ensure the database directory exists
        os.makedirs(db_path, exist_ok=True)

        # Create the persistent client
        chroma_client = chromadb.PersistentClient(path=db_path)

        # print(f"DEBUG: Getting/Creating collection '{collection_name}' with EmbeddingFunction instance.")

        # Get or create the collection.
        # Pass the embedding function instance. ChromaDB uses this instance.
        # If creating, ChromaDB *should* infer the dimension from emb_func.dimension if needed by the index.
        # If the dimension is None and mode is 'index', this *might* still work if Chroma doesn't strictly
        # need the dimension upfront, but it's safer to always define OUTPUT_EMBEDDING_DIMENSION.
        collection = chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=emb_func, # Pass the mode-aware instance
            metadata={"hnsw:space": "cosine"} # Specify distance metric
        )

        # print(f"DEBUG: Successfully retrieved/created collection '{collection_name}'.")
        return collection

    except ValueError as ve:
        # Catch potential dimension errors from embedding function init
        print(f"!!! Configuration Error for collection '{collection_name}': {ve}")
        raise
    except Exception as e:
        # Catch ChromaDB client/collection errors
        print(f"!!! Error connecting/creating ChromaDB collection '{collection_name}' at path '{db_path}': {e}")
        traceback.print_exc()
        raise

# --- Batch Update Helper ---
def _update_db_batch(collection: chromadb.Collection, ids: List[str], embeddings: List[List[float]], metadatas: List[Dict]):
    """Internal helper to update a single batch in ChromaDB (used by embedding mode)."""
    if not ids:
        return # Nothing to update
    try:
        # print(f"DEBUG: Updating {len(ids)} successfully embedded chunks in DB.")
        collection.update(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas # Ensure metadata includes 'has_embedding: True'
        )
    except Exception as db_update_err:
        print(f"\n!!! Error updating batch in ChromaDB (IDs: {ids[:5]}...): {db_update_err}")
        # Consider how critical this is - should we add these IDs back to a failed list?
        # For now, just log the error. The metadata won't be marked 'has_embedding: True'.
        # The calling function (generate_embeddings_in_batches) should handle tracking failures.
        raise # Re-raise the error so the caller knows the batch failed
