import numpy as np
import traceback
from typing import List, Dict, Tuple, Optional, Any

# --- Local Imports ---
from rag.chroma_manager import get_chroma_collection
from rag.bm25_manager import load_bm25_index, tokenize_text_bm25, RANK_BM25_AVAILABLE, BM25Okapi
from my_utils.llm_interface import get_embedding
from config import EMBEDDING_MODEL, RERANKER_MODEL, DEFAULT_RERANK_CANDIDATE_COUNT

# --- Sentence Transformers Import ---
try:
    from sentence_transformers import CrossEncoder
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    # print("Warning: sentence-transformers not installed (`pip install sentence-transformers`). Re-ranking will be disabled.")
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    CrossEncoder = None # Define as None to avoid NameError

# Global cache for the reranker model
reranker_model_cache: Dict[str, Any] = {}


# --- Vector Retrieval ---
def retrieve_chunks_vector(query: str, db_path: str, collection_name: str,
                           top_k: int, execution_mode: str = "query") -> List[dict]:
    """Retrieve top-K chunks from ChromaDB using vector similarity."""
    if top_k <= 0: top_k = 1
    retrieved_chunks = []
    print(f"DEBUG (retrieve_chunks_vector): Retrieving top {top_k} for query: '{query[:50]}...'") # Add log
    try:
        # Get collection with appropriate embedding function behavior for query mode
        collection = get_chroma_collection(db_path, collection_name, execution_mode=execution_mode)
        print(f"DEBUG (retrieve_chunks_vector): Collection '{collection_name}' obtained.") # Add log

        # Embed the query using the llm_interface
        query_vec = get_embedding(query, model=EMBEDDING_MODEL, task_type="retrieval_query")
        if query_vec is None:
            print(f"Error: Failed to embed query '{query[:50]}...'")
            return []
        else:
            print(f"DEBUG (retrieve_chunks_vector): Query embedded successfully. Vector shape: {query_vec.shape}")
            # print(f"DEBUG (retrieve_chunks_vector): Query vector (first 5 dims): {query_vec[:5]}") # Optional: print part of vector
        if not isinstance(query_vec, np.ndarray) or query_vec.ndim != 1:
             print(f"Error: Query embedding has unexpected shape {type(query_vec)}.")
             return []

        # Query ChromaDB
        print(f"DEBUG (retrieve_chunks_vector): Querying ChromaDB collection '{collection_name}'...") # Add log
        # It implicitly searches only chunks that have embeddings.
        results = collection.query(
            query_embeddings=[query_vec.tolist()],
            n_results=top_k,
            include=['metadatas', 'distances', 'documents'] # Include 'documents'
            # Optional: Filter for 'has_embedding: True' if paranoid, but query should only match embedded items
            # where={"has_embedding": True} # This might slow down query if not needed
        )

        # Process results
        if results and results.get('ids') and results['ids'][0]:
             metadatas = results.get('metadatas', [[]])[0]
             distances = results.get('distances', [[]])[0]
             ids = results.get('ids', [[]])[0] # For debug or consistency checks
             documents = results.get('documents', [[]])[0] # Get documents
             print(f"DEBUG (retrieve_chunks_vector): Found {len(ids)} vector results.") # Add log
             for i, meta in enumerate(metadatas):
                 if meta is None: 
                     print(f"Warning: Vector result {i} has None metadata. Skipping.")
                     continue # Skip if metadata is missing for some reason

                 # Ensure necessary fields exist before adding
                 # Use chunk_id from results['ids'] for consistency
                 chunk_id = ids[i] if i < len(ids) else None
                 # REMOVED: Check for 'file_hash' and 'chunk_number' as they are not relevant for abstract collection
                 if chunk_id: # Check only if chunk_id exists
                     dist = distances[i] if i < len(distances) else None
                     doc_text = documents[i] if i < len(documents) else None # Get corresponding document text
                     # Add retrieval info to metadata
                     meta['vector_distance'] = dist
                     # Calculate similarity (cosine: 1 - distance)
                     meta['vector_similarity'] = (1.0 - dist) if dist is not None and dist <= 2.0 else 0.0 # Cosine distance is 0-2
                     meta['retrieved_by'] = meta.get('retrieved_by', []) + ['vector'] # Track retrieval method
                     meta['chunk_id'] = chunk_id # Ensure chunk_id is in metadata
                     meta['text'] = doc_text # Add the document text under the 'text' key
                     retrieved_chunks.append(meta)
                 else:
                      # This case should be less likely now, primarily if ids[i] itself is None/empty
                      print(f"Warning: Vector result missing chunk_id (Index: {i}). Skipping.")

        else:
             print(f"DEBUG (retrieve_chunks_vector): No vector results found in ChromaDB response for query '{query[:50]}...'") # Add log

        print(f"DEBUG (retrieve_chunks_vector): Returning {len(retrieved_chunks)} processed vector chunks.") # Add log
        return retrieved_chunks
    except Exception as e:
        print(f"!!! Error querying ChromaDB (vector search) '{collection_name}': {e}")
        traceback.print_exc()
        return []

# --- BM25 Retrieval ---
def retrieve_chunks_bm25(query: str, db_path: str, collection_name: str, top_k: int) -> List[Tuple[str, float]]:
    """Retrieve top-K chunk IDs and scores using BM25."""
    if not RANK_BM25_AVAILABLE: return []
    if top_k <= 0: top_k = 1

    # Load BM25 index and ID mapping
    bm25_instance, bm25_chunk_ids_ordered = load_bm25_index(db_path, collection_name)

    if bm25_instance is None or bm25_chunk_ids_ordered is None:
        # print("BM25 index is not available for querying.") # Already printed by load_bm25_index
        return []

    # Tokenize the query using the same method as indexing
    tokenized_query = tokenize_text_bm25(query)
    if not tokenized_query:
        # print("Warning: BM25 query empty after tokenization.")
        return []

    try:
        # Get scores for the tokenized query
        scores = bm25_instance.get_scores(tokenized_query)

        # Ensure scores and IDs align (crucial check)
        if len(scores) != len(bm25_chunk_ids_ordered):
             print(f"!!! Error: BM25 score count ({len(scores)}) mismatch with ID map count ({len(bm25_chunk_ids_ordered)}). Index may be corrupt.")
             return []

        # Combine chunk IDs with their scores
        scored_results = list(zip(bm25_chunk_ids_ordered, scores))

        # Sort by score descending
        sorted_results = sorted(scored_results, key=lambda item: item[1], reverse=True)

        # Get top K results with scores > 0 (BM25 scores can be negative, often irrelevant)
        # We might want to keep scores >= 0 depending on interpretation. Let's stick to > 0.
        results = [(chunk_id, score) for chunk_id, score in sorted_results if score > 0][:top_k]
        return results

    except Exception as e:
        print(f"!!! Error during BM25 scoring/ranking: {e}")
        traceback.print_exc()
        return []

# --- RRF Combination ---
def combine_results_rrf(vector_results: List[Dict], bm25_results: List[Tuple[str, float]],
                        db_path: str, collection_name: str, execution_mode: str = "query", k_rrf: int = 60) -> List[Dict]:
    """Combines vector and BM25 results using Reciprocal Rank Fusion (RRF)."""
    combined_scores: Dict[str, float] = {}
    chunk_metadata_cache: Dict[str, Dict] = {} # Cache metadata keyed by chunk_id

    # --- Process Vector Results ---
    # Vector results already contain metadata. Sort by distance (lower is better).
    vector_results.sort(key=lambda c: c.get('vector_distance', float('inf')))
    for rank, chunk_meta in enumerate(vector_results):
        chunk_id = chunk_meta.get('chunk_id') # Assumes chunk_id was added in retrieve_chunks_vector
        if not chunk_id: continue # Skip if ID missing

        # Calculate RRF score contribution
        score = 1.0 / (k_rrf + rank)
        combined_scores[chunk_id] = combined_scores.get(chunk_id, 0.0) + score

        # Store metadata if not already cached
        if chunk_id not in chunk_metadata_cache:
             # Store a copy to avoid modifying the original list results
             chunk_metadata_cache[chunk_id] = chunk_meta.copy()
        else:
             # If already exists (e.g., from BM25 processing below), update retrieval method
             chunk_metadata_cache[chunk_id]['retrieved_by'] = list(set(chunk_metadata_cache[chunk_id].get('retrieved_by', []) + ['vector']))


    # --- Process BM25 Results ---
    # BM25 results are (chunk_id, score). We only need rank for RRF.
    bm25_ids_to_fetch_metadata = []
    for rank, (chunk_id, bm25_score) in enumerate(bm25_results):
        if not chunk_id: continue # Skip empty IDs

        # Calculate RRF score contribution
        score = 1.0 / (k_rrf + rank)
        combined_scores[chunk_id] = combined_scores.get(chunk_id, 0.0) + score

        # Add BM25 score and retrieval method to cached metadata if it exists,
        # otherwise mark for fetching.
        if chunk_id in chunk_metadata_cache:
            chunk_metadata_cache[chunk_id]['bm25_score'] = bm25_score
            chunk_metadata_cache[chunk_id]['retrieved_by'] = list(set(chunk_metadata_cache[chunk_id].get('retrieved_by', []) + ['bm25']))
        else:
            # Need to fetch metadata for this BM25-only result
            if chunk_id not in bm25_ids_to_fetch_metadata:
                bm25_ids_to_fetch_metadata.append(chunk_id)

    # --- Fetch Metadata for BM25-only Results ---
    if bm25_ids_to_fetch_metadata:
        # print(f"Fetching metadata for {len(bm25_ids_to_fetch_metadata)} BM25-specific chunks...") # Debug
        try:
            collection = get_chroma_collection(db_path, collection_name, execution_mode=execution_mode)
            # Fetch in batches for potentially large number of IDs
            batch_size_fetch = 200 # Adjust as needed
            num_batches_fetch = (len(bm25_ids_to_fetch_metadata) + batch_size_fetch - 1) // batch_size_fetch

            for i in range(num_batches_fetch):
                 batch_ids = bm25_ids_to_fetch_metadata[i*batch_size_fetch : (i+1)*batch_size_fetch]
                 if not batch_ids: continue
                 try:
                     # Include 'documents' when fetching for BM25 results
                     fetched_data = collection.get(ids=batch_ids, include=['metadatas', 'documents'])
                     if fetched_data and fetched_data.get('ids'):
                          fetched_metadatas = fetched_data.get('metadatas', [])
                          fetched_documents = fetched_data.get('documents', []) # Get documents
                          for j, fetched_id in enumerate(fetched_data['ids']):
                              # Check if ID is relevant and metadata exists
                              if fetched_id in combined_scores and j < len(fetched_metadatas):
                                  meta = fetched_metadatas[j]
                                  doc_text = fetched_documents[j] if j < len(fetched_documents) else None # Get corresponding text
                                  if meta and fetched_id not in chunk_metadata_cache: # Ensure metadata is not None and not already cached
                                      # Find the original BM25 score for this ID
                                      original_bm25_score = next((s for id, s in bm25_results if id == fetched_id), None)
                                      meta['bm25_score'] = original_bm25_score
                                      # Add placeholder vector scores for consistency if needed later
                                      meta['vector_distance'] = None
                                      meta['vector_similarity'] = None
                                      meta['retrieved_by'] = ['bm25'] # Mark as retrieved by BM25
                                      meta['chunk_id'] = fetched_id # Ensure chunk_id is present
                                      meta['text'] = doc_text # Add the document text under the 'text' key
                                      chunk_metadata_cache[fetched_id] = meta
                                  # else: print(f"Warning: Null or already cached metadata fetched for BM25 chunk ID {fetched_id}") # Debug
                 except Exception as batch_fetch_err:
                      print(f"!!! Error fetching metadata batch for BM25 results (IDs: {batch_ids[:5]}...): {batch_fetch_err}")

        except Exception as fetch_err:
            print(f"!!! Error accessing collection for fetching BM25 metadata: {fetch_err}")
            traceback.print_exc()

    # --- Combine and Sort ---
    # Add RRF score to each metadata entry in the cache
    final_results_list = []
    for chunk_id, meta in chunk_metadata_cache.items():
        if chunk_id in combined_scores:
            meta['rrf_score'] = combined_scores[chunk_id]
            final_results_list.append(meta)
        # else: This shouldn't happen if logic is correct

    # Sort the final list by RRF score descending
    final_results_list.sort(key=lambda c: c.get('rrf_score', 0.0), reverse=True)

    return final_results_list


# --- Re-ranking ---
def rerank_chunks(query: str, chunks: List[Dict],
                  model_name: Optional[str] = RERANKER_MODEL,
                  top_n: int = DEFAULT_RERANK_CANDIDATE_COUNT) -> List[Dict]:
    """
    Re-ranks chunks using a CrossEncoder model based on relevance to the query.
    """
    # Check if reranking is possible and requested
    if not model_name or not SENTENCE_TRANSFORMERS_AVAILABLE or not CrossEncoder or not chunks:
        if model_name and not SENTENCE_TRANSFORMERS_AVAILABLE:
             print("Info: Re-ranking skipped (sentence-transformers not installed).")
        elif not model_name:
             # print("Info: No reranker model specified, skipping re-ranking.") # Less verbose
             pass
        # Return top_n based on original order (e.g., RRF score) if reranking is skipped
        # Ensure chunks have 'rrf_score' or handle missing key gracefully
        return sorted(chunks, key=lambda c: c.get('rrf_score', 0.0), reverse=True)[:top_n]

    print(f"--- Re-ranking top {len(chunks)} candidates using {model_name} ---")

    try:
        # Load model from cache or instantiate
        if model_name not in reranker_model_cache:
            print(f"Loading re-ranker model: {model_name}...")
            # Consider adding model_args={'device': 'cuda'} if GPU is available and desired
            reranker_model_cache[model_name] = CrossEncoder(model_name) #, max_length=512) # Optional: set max_length
            print("Re-ranker model loaded.")
        model = reranker_model_cache[model_name]

        # Prepare query-chunk pairs for the model
        query_chunk_pairs = []
        valid_chunks_for_reranking = [] # Store chunks that have valid text for indexing
        chunk_indices_map = {} # Map pair index back to original chunk list index

        for i, chunk in enumerate(chunks):
            # Prioritize contextualized text, fall back to raw text
            text_to_rank = chunk.get('contextualized_text', chunk.get('text', ''))
            if text_to_rank and isinstance(text_to_rank, str) and text_to_rank.strip():
                query_chunk_pairs.append([query, text_to_rank.strip()])
                valid_chunks_for_reranking.append(chunk)
                chunk_indices_map[len(query_chunk_pairs) - 1] = i # Store original index
            else:
                print(f"Warning: Skipping chunk {chunk.get('chunk_id', f'index_{i}')} for re-ranking due to missing/empty text.")
                # Mark skipped chunks with a very low score so they end up last
                chunk['rerank_score'] = -float('inf')


        if not valid_chunks_for_reranking:
            print("Warning: No valid chunks found to re-rank.")
            # Return original chunks sorted by RRF score (or original order if no score)
            return sorted(chunks, key=lambda c: c.get('rrf_score', 0.0), reverse=True)[:top_n]

        # Get scores from the cross-encoder
        # print(f"Predicting scores for {len(query_chunk_pairs)} pairs...") # Debug
        scores = model.predict(query_chunk_pairs, show_progress_bar=False) # Set show_progress_bar=True for large batches

        # Add scores back to the corresponding chunks in the original list
        for pair_idx, score in enumerate(scores):
            original_chunk_idx = chunk_indices_map[pair_idx]
            # Ensure the key exists before assigning, though it should have been added if skipped
            if original_chunk_idx < len(chunks):
                 chunks[original_chunk_idx]['rerank_score'] = float(score) # Ensure it's a float
            # else: This case should not happen if map is correct

        # Sort the original chunk list by the new rerank_score (higher is better)
        # Chunks skipped earlier will have -inf score and end up last.
        reranked_chunks = sorted(chunks, key=lambda c: c.get('rerank_score', -float('inf')), reverse=True)

        # Correct the print statement to use top_n
        print(f"Re-ranking complete. Returning top {top_n} based on cross-encoder scores.")
        return reranked_chunks[:top_n]

    except Exception as e:
        print(f"!!! Error during re-ranking with {model_name}: {e}")
        print(traceback.format_exc())
        print("Warning: Falling back to original chunk order (before re-ranking).")
        # Fallback: return top_n based on the order they came in (likely RRF sorted)
        # Ensure fallback also respects top_n
        return sorted(chunks, key=lambda c: c.get('rrf_score', 0.0), reverse=True)[:top_n]
