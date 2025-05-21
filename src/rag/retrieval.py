import numpy as np
import traceback
from typing import List, Dict, Tuple, Optional, Any
from chromadb.config import Settings # Import for type hinting

# --- Local Imports ---
from rag.chroma_manager import get_chroma_collection
from rag.bm25_manager import load_bm25_index, tokenize_text_bm25, RANK_BM25_AVAILABLE
from my_utils.llm_interface import get_embedding
from config import EMBEDDING_MODEL, RERANKER_MODEL, DEFAULT_RERANK_CANDIDATE_COUNT, HYPE, HYPE_SUFFIX

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
                           top_k: int, execution_mode: str = "query",
                           settings: Optional[Settings] = None) -> List[dict]: # Add settings parameter
    """Retrieve top-K chunks from ChromaDB using vector similarity."""
    
    if top_k <= 0: top_k = 1
    retrieved_chunks = []
    print(f"DEBUG (retrieve_chunks_vector): Retrieving top {top_k} from '{collection_name}' for query: '{query[:50]}...'")
    try:
        # Get collection with appropriate embedding function behavior for query mode
        collection = get_chroma_collection(db_path, collection_name, execution_mode=execution_mode, settings=settings) # Pass settings
        print(f"DEBUG (retrieve_chunks_vector): Collection '{collection_name}' obtained.")

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
        print(f"DEBUG (retrieve_chunks_vector): Querying ChromaDB collection '{collection_name}'...")
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
                     meta['cited_by'] = meta.get('cited_by', 0) # Ensure 'cited_by' key is present, default to 0
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
    # Append suffix if HYPE mode is enabled
    effective_collection_name = f"{collection_name}{HYPE_SUFFIX}" if HYPE else collection_name
    if HYPE:
        print(f"Info: retrieve_chunks_bm25 using Hype collection '{collection_name}'")

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
                        db_path: str, collection_name: str, execution_mode: str = "query", k_rrf: int = 50, weight_vector_bm25=[0.7, 0.3],
                        db_settings: Optional[Settings] = None) -> List[Dict]: # Add db_settings parameter
    """Combines vector and BM25 results using Reciprocal Rank Fusion (RRF)."""
    # Determine effective collection name based on HYPE flag
    effective_collection_name = f"{collection_name}{HYPE_SUFFIX}" if HYPE else collection_name

    print(f"DEBUG (combine_results_rrf): Combining {len(vector_results)} vector results and {len(bm25_results)} BM25 results.") # Add log

    combined_scores: Dict[str, float] = {}
    chunk_metadata_cache: Dict[str, Dict] = {} # Cache metadata keyed by chunk_id

    # --- Process Vector Results ---
    # Vector results already contain metadata. Sort by distance (lower is better).
    vector_results.sort(key=lambda c: c.get('vector_distance', float('inf')))
    for rank, chunk_meta in enumerate(vector_results):
        chunk_id = chunk_meta.get('chunk_id') # Assumes chunk_id was added in retrieve_chunks_vector
        if not chunk_id: continue # Skip if ID missing

        # Calculate RRF score contribution
        score = 1.0 / (k_rrf + rank)*weight_vector_bm25[0]
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
    bm25_processed_count = 0 # Debug counter
    bm25_added_to_cache_count = 0 # Debug counter
    for rank, (chunk_id, bm25_score) in enumerate(bm25_results):
        if not chunk_id: continue # Skip empty IDs
        bm25_processed_count += 1 # Increment counter

        # Calculate RRF score contribution
        score = 1.0 / (k_rrf + rank) * weight_vector_bm25[1]
        combined_scores[chunk_id] = combined_scores.get(chunk_id, 0.0) + score

        # Add BM25 score and retrieval method to cached metadata if it exists,
        # otherwise mark for fetching.
        if chunk_id in chunk_metadata_cache:
            chunk_metadata_cache[chunk_id]['bm25_score'] = bm25_score
            chunk_metadata_cache[chunk_id]['retrieved_by'] = list(set(chunk_metadata_cache[chunk_id].get('retrieved_by', []) + ['bm25']))
            bm25_added_to_cache_count += 1 # Increment counter
        else:
            # Need to fetch metadata for this BM25-only result
            if chunk_id not in bm25_ids_to_fetch_metadata:
                bm25_ids_to_fetch_metadata.append(chunk_id)

    print(f"DEBUG (combine_rrf): Processed {bm25_processed_count} raw BM25 results. Added score directly to {bm25_added_to_cache_count} existing cache entries.") # Add summary log
    print(f"DEBUG (combine_rrf): Marked {len(bm25_ids_to_fetch_metadata)} BM25-specific chunk IDs for metadata fetching.") # Add log

    # --- Fetch Metadata for BM25-only Results ---
    if bm25_ids_to_fetch_metadata:
        bm25_metadata_fetched_count = 0 # Debug counter
        bm25_score_added_during_fetch = 0 # Debug counter
        try:
            # Ensure we fetch metadata from the BASE collection where BM25 IDs exist
            base_collection_name = collection_name.replace(HYPE_SUFFIX, "") if HYPE else collection_name
            print(f"DEBUG (combine_rrf fetch): Fetching metadata from BASE collection: '{base_collection_name}'") # Add log
            collection = get_chroma_collection(db_path, base_collection_name, execution_mode=execution_mode, settings=db_settings) # Pass db_settings

            # Fetch in batches for potentially large number of IDs
            batch_size_fetch = 200 # Adjust as needed
            num_batches_fetch = (len(bm25_ids_to_fetch_metadata) + batch_size_fetch - 1) // batch_size_fetch

            for i in range(num_batches_fetch):
                 batch_ids = bm25_ids_to_fetch_metadata[i*batch_size_fetch : (i+1)*batch_size_fetch]
                 if not batch_ids: continue
                 try:
                     # Include 'documents' when fetching for BM25 results
                     print(f"DEBUG (combine_rrf fetch): Calling collection.get() for {len(batch_ids)} IDs (e.g., {batch_ids[:5]})" ) # Add log
                     fetched_data = collection.get(ids=batch_ids, include=['metadatas', 'documents'])
                     print(f"DEBUG (combine_rrf fetch): collection.get() returned {len(fetched_data.get('ids', []))} IDs") # Add log

                     if fetched_data and fetched_data.get('ids'):
                          fetched_metadatas = fetched_data.get('metadatas', [])
                          fetched_documents = fetched_data.get('documents', []) # Get documents
                          for j, fetched_id in enumerate(fetched_data['ids']):
                              bm25_metadata_fetched_count += 1 # Increment counter
                              # Check if ID is relevant and metadata exists
                              if fetched_id in combined_scores and j < len(fetched_metadatas):
                                  meta = fetched_metadatas[j]
                                  doc_text = fetched_documents[j] if j < len(fetched_documents) else None # Get corresponding text
                                  if meta and fetched_id not in chunk_metadata_cache: # Ensure metadata is not None and not already cached
                                      # Find the original BM25 score for this ID
                                      original_bm25_score = next((s for id, s in bm25_results if id == fetched_id), None)
                                      # Ensure bm25_score is a float, default to 0.0 if None
                                      meta['bm25_score'] = float(original_bm25_score) if original_bm25_score is not None else 0.0
                                      if original_bm25_score is not None:
                                          bm25_score_added_during_fetch += 1 # Increment counter
                                      # else: print(f"DEBUG (combine_rrf fetch): Could not find original BM25 score for fetched ID {fetched_id}") # Optional detailed log

                                      # Add placeholder vector scores for consistency if needed later
                                      meta['vector_distance'] = None
                                      meta['vector_similarity'] = None
                                      meta['retrieved_by'] = ['bm25'] # Mark as retrieved by BM25
                                      meta['chunk_id'] = fetched_id # Ensure chunk_id is present
                                      meta['text'] = doc_text # Add the document text under the 'text' key
                                      meta['cited_by'] = meta.get('cited_by', 0) # Ensure 'cited_by' key is present, default to 0
                                      chunk_metadata_cache[fetched_id] = meta
                                  # else: print(f"DEBUG (combine_rrf fetch): Condition not met for fetched ID {fetched_id} (in combined_scores: {fetched_id in combined_scores}, meta ok: {meta is not None}, not in cache: {fetched_id not in chunk_metadata_cache})", flush=True) # Detailed condition log
                 except Exception as batch_fetch_err:
                      print(f"!!! Error fetching metadata batch for BM25 results (IDs: {batch_ids[:5]}...): {batch_fetch_err}")

        except Exception as fetch_err:
            print(f"!!! Error accessing collection for fetching BM25 metadata: {fetch_err}")
            traceback.print_exc()
        print(f"DEBUG (combine_rrf): Fetched metadata for {bm25_metadata_fetched_count} BM25-specific IDs. Added BM25 score to {bm25_score_added_during_fetch} new cache entries.") # Add summary log

    # --- Combine and Sort ---
    # Add RRF score to each metadata entry in the cache
    final_results_list = []
    for chunk_id, meta in chunk_metadata_cache.items():
        if chunk_id in combined_scores:
            meta['rrf_score'] = combined_scores[chunk_id]
            final_results_list.append(meta)

    # Sort the final list by RRF score descending
    final_results_list.sort(key=lambda c: c.get('rrf_score', 0.0), reverse=True)
    

    # Enrich metadata when working on the HyPE embeddings collection
    # Check if the *effective* name used ends with the HYPE suffix
    if effective_collection_name.endswith(HYPE_SUFFIX):
        collection_name_metadata = effective_collection_name.replace(HYPE_SUFFIX, "") # Remove suffix for metadata
        try:
            source_col = get_chroma_collection(db_path, collection_name_metadata, execution_mode=execution_mode, settings=db_settings) # Pass db_settings
            doi_list = [m.get('original_chunk_id') for m in final_results_list if m.get('original_chunk_id')]
            doi_list += [m.get('doi') for m in final_results_list if m.get('doi')]
            doi_list = np.unique(doi_list).tolist() # Remove duplicates

            if doi_list:
                fetched = source_col.get(ids=doi_list, include=['metadatas'])
                id_to_meta = dict(zip(fetched.get('ids', []), fetched.get('metadatas', [])))
                for m in final_results_list:
                    doi = m.get('doi')
                    if doi == None:
                        doi = m.get('original_chunk_id')
                    src = id_to_meta.get(doi)
                    if src:
                        for field in ['authors', 'year', 'title', 'source_title', 'cited_by']:
                            if field in src:
                                m[field] = src[field]
        except Exception as e:
            print(f"Warning: Could not enrich metadata for HyPE entries: {e}")

    # Add debug print before returning
    count_with_bm25_score = sum(1 for chunk in final_results_list if 'bm25_score' in chunk and chunk['bm25_score'] is not None and chunk['bm25_score'] > 0) # Be more specific
    print(f"DEBUG (combine_results_rrf): Returning {len(final_results_list)} combined chunks. {count_with_bm25_score} have a positive 'bm25_score'.") # Updated log

    # print top 10 final result_list names and year for debug top score first
    # print(f"DEBUG (combine_results_rrf): Top 10 final results: {[{'name': m.get('title'), 'year': m.get('year')} for m in final_results_list[:10]]}") # Optional detailed log
    # show top ten based on bm25 score
    # handle potential None values in bm25_score during sorting for debug print
    # print(f"DEBUG (combine_results_rrf): Top 10 final results by BM25 score: {[{'name': m.get('title'), 'bm25_score': m.get('bm25_score')} for m in sorted(final_results_list, key=lambda c: c.get('bm25_score', 0.0), reverse=True)[:10]]}") # Optional detailed log
    # show top ten based on vector similarity from vector_results
    # Handle potential None values in vector_similarity during sorting for debug print
    # print(f"DEBUG (combine_results_rrf): Top 10 final results by vector similarity: {[{'name': m.get('title'), 'vector_similarity': m.get('vector_similarity')} for m in sorted(final_results_list, key=lambda c: c.get('vector_similarity') if c.get('vector_similarity') is not None else -1.0, reverse=True)[:10]]}") # Optional detailed log

    return final_results_list


# --- Re-ranking ---
def rerank_chunks(query: str, chunks: List[Dict],
                  model_name: Optional[str] = RERANKER_MODEL,
                  top_n: int = DEFAULT_RERANK_CANDIDATE_COUNT,
                  abstracts: bool = False) -> List[Dict]:
    """
    Re-ranks chunks using a CrossEncoder model based on relevance to the query,
    converting raw logits to calibrated probabilities, with smart pruning:
      - Filters to keep only the best RRF-scoring chunk per DOI.
      - Early-discard passages with low vector and BM25 scores.
    """
    # Fallback if reranker unavailable
    if not model_name or not SENTENCE_TRANSFORMERS_AVAILABLE or not CrossEncoder or not chunks:
        print(f"DEBUG (rerank_chunks): Skipping reranking (unavailable/no chunks). Returning top {top_n} by RRF.")
        return sorted(chunks, key=lambda c: c.get('rrf_score', 0.0), reverse=True)[:top_n]

    # Compute average BM25 score for pruning
    bm25_scores = [c.get('bm25_score', 0.0) for c in chunks if 'bm25_score' in c]
    avg_bm25    = sum(bm25_scores) / len(bm25_scores) if bm25_scores else 0.0

    # Add debug print
    print(f"DEBUG (rerank_chunks): Input chunks: {len(chunks)}. Found {len(bm25_scores)} BM25 scores. Avg BM25: {avg_bm25:.4f}")

    # Prepare model
    if model_name not in reranker_model_cache:
        print(f"DEBUG (rerank_chunks): Loading CrossEncoder model: {model_name}")
        reranker_model_cache[model_name] = CrossEncoder(model_name)
    model = reranker_model_cache[model_name]

    import torch

    # Pre-rank by RRF score
    base_sorted = sorted(chunks, key=lambda c: c.get('rrf_score', 0.0), reverse=True)

    # --- Filter: Keep only the best RRF-scoring chunk per DOI  if HyPE ---
    base_sorted_filtered = []
    if abstracts:
        seen_dois = set()
        for chunk in base_sorted:
            # Prioritize 'doi', fallback to 'original_chunk_id' if available
            doi = chunk.get('doi') or chunk.get('original_chunk_id')
            if doi: # Only process if a DOI identifier is found
                if doi not in seen_dois:
                    base_sorted_filtered.append(chunk)
                    seen_dois.add(doi)
            else:
                # If no DOI, keep the chunk (might be from a source without DOI)
                base_sorted_filtered.append(chunk)
        print(f"DEBUG (rerank_chunks): Filtered to {len(base_sorted_filtered)} chunks (best RRF per DOI).")

    else:
        # If not HyPE, keep all chunks (or apply other filtering logic if needed)
        base_sorted_filtered = base_sorted


    base_sorted_filtered = base_sorted_filtered[:top_n] # Limit to top_n after filtering
    print(f"DEBUG (rerank_chunks): Limited to top {len(base_sorted_filtered)} for reranking.")
    # --- End Filter ---

    # --- Rerank all filtered chunks ---
    pairs, idx_map = [], []

    # Prepare pairs for CrossEncoder, applying early discard
    discard_count = 0
    for i, chunk in enumerate(base_sorted_filtered):
        # Use .get with defaults, and ensure comparison handles potential None
        vec_sim = chunk.get('vector_similarity', 0.0)
        bm25_sc = chunk.get('bm25_score', 0.0)
        # Add explicit checks for None before comparison
        vec_sim_check = vec_sim if vec_sim is not None else 1.0 # Treat None as high similarity
        bm25_sc_check = bm25_sc if bm25_sc is not None else avg_bm25 # Treat None as average score

        # Early discard condition
        # Note: avg_bm25 can be 0 if no bm25 scores were found initially.
        # The condition vec_sim_check < 0.2 might still trigger.
        if vec_sim_check < 0.2 and bm25_sc_check < avg_bm25*0.5:
            # Assign low scores directly if discarded
            chunk['ce_logit'] = -float('inf')
            chunk['ce_prob']  = 0.0
            discard_count += 1
            print(f"DEBUG (rerank_chunks): Early discard for chunk {i} (vec_sim: {vec_sim_check:.4f}, bm25: {bm25_sc_check:.4f}).")
        else:
            text = chunk.get('contextualized_text', chunk.get('text', '')).strip()
            
            if text:
                pairs.append([query, text])
                idx_map.append(i) # Store the original index in base_sorted_filtered
            else:
                # Assign low scores if text is empty
                chunk['ce_logit'] = -float('inf')
                chunk['ce_prob']  = 0.0
                discard_count += 1 # Count as discarded if no text
                print(f"DEBUG (rerank_chunks): Empty text for chunk {i}. Discarded.") # Optional debug log

    print(f"DEBUG (rerank_chunks): Prepared {len(pairs)} pairs for CrossEncoder. Discarded {discard_count} chunks early.")

    # Predict scores if there are valid pairs
    if pairs:
        # Predict logits and convert to probabilities
        print(f"DEBUG (rerank_chunks): Running CrossEncoder prediction...")
        logits_tensor = torch.tensor(model.predict(pairs, show_progress_bar=False))
        probs_tensor  = torch.sigmoid(logits_tensor)
        logits, probs = logits_tensor.tolist(), probs_tensor.tolist()
        print(f"DEBUG (rerank_chunks): CrossEncoder prediction complete.")

        # Attach CE scores back to the corresponding chunks in base_sorted_filtered
        for j, original_idx in enumerate(idx_map):
            base_sorted_filtered[original_idx]['ce_logit'] = float(logits[j])
            base_sorted_filtered[original_idx]['ce_prob']  = float(probs[j])

    # Sort the entire filtered list by CE probability (descending)
    # Chunks discarded or without text will have low scores and end up at the bottom
    result = sorted(base_sorted_filtered, key=lambda c: c.get('ce_prob', 0.0), reverse=True)

    print(f"DEBUG (rerank_chunks): Reranking complete. Returning top {min(top_n, len(result))} chunks.")
    return result
