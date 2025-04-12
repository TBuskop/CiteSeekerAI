import os
import pickle
import re
from typing import List, Tuple, Optional, Dict, Any
from tqdm import tqdm

# --- NLTK Imports ---
try:
    import nltk
    from nltk.corpus import stopwords
    try:
        # Attempt to load stopwords
        stop_words_english = set(stopwords.words('english'))
    except LookupError:
        print("NLTK stopwords not found. Downloading...")
        try:
            # Download necessary NLTK data (punkt for tokenization, stopwords)
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            stop_words_english = set(stopwords.words('english'))
            print("NLTK data downloaded.")
        except Exception as dl_err:
            print(f"Warning: Failed to download NLTK data: {dl_err}. Proceeding without stopwords.")
            stop_words_english = set() # Use empty set as fallback
except ImportError:
    print("Warning: NLTK not installed (`pip install nltk`). Using basic tokenization without stopwords.")
    stop_words_english = set() # Use empty set if NLTK is not installed

# --- BM25 Imports ---
try:
    from rank_bm25 import BM25Okapi
    RANK_BM25_AVAILABLE = True
except ImportError:
    print("Warning: rank_bm25 not installed (`pip install rank_bm25`). BM25 functionality will be disabled.")
    RANK_BM25_AVAILABLE = False
    BM25Okapi = None # Define as None to avoid NameError

# --- BM25 Configuration and Caching ---
bm25_index_cache: Dict[Tuple[str, str], Any] = {} # Cache for loaded BM25 index objects
bm25_ids_cache: Dict[Tuple[str, str], List[str]] = {} # Cache for the ordered list of chunk IDs

def get_bm25_paths(db_path: str, collection_name: str) -> Tuple[str, str]:
    """Constructs the file paths for the BM25 index and ID mapping."""
    base_path = os.path.join(db_path, f"{collection_name}_bm25")
    index_path = f"{base_path}_index.pkl"
    map_path = f"{base_path}_ids.pkl"
    return index_path, map_path

# --- BM25 Tokenization ---
def tokenize_text_bm25(text: str) -> List[str]:
    """
    Tokenizes text for BM25: lowercase, remove punctuation, split, remove stopwords.
    """
    if not isinstance(text, str): return []
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text) # Remove punctuation
    tokens = text.split()
    # Remove stopwords and empty tokens
    tokens = [word for word in tokens if word and word not in stop_words_english]
    return tokens

# --- BM25 Index Loading ---
def load_bm25_index(db_path: str, collection_name: str) -> Tuple[Optional[BM25Okapi], Optional[List[str]]]:
    """
    Loads the BM25 index and corresponding chunk ID list from files.
    Uses a cache to avoid redundant loading.

    Returns:
        A tuple containing the loaded BM25Okapi instance (or None) and
        the list of chunk IDs (or None).
    """
    if not RANK_BM25_AVAILABLE:
        # print("BM25 is unavailable (rank_bm25 not installed).") # Already warned at import
        return None, None

    cache_key = (db_path, collection_name)
    if cache_key in bm25_index_cache:
        # print(f"Using cached BM25 index for {collection_name}.") # Debug
        return bm25_index_cache[cache_key], bm25_ids_cache[cache_key]

    index_path, map_path = get_bm25_paths(db_path, collection_name)

    if os.path.exists(index_path) and os.path.exists(map_path):
        try:
            # print(f"Loading BM25 index from: {index_path}") # Verbose
            with open(index_path, 'rb') as f_idx:
                bm25_instance = pickle.load(f_idx)
            with open(map_path, 'rb') as f_map:
                bm25_chunk_ids_ordered = pickle.load(f_map)

            # Validate loaded data types (basic check)
            if BM25Okapi and isinstance(bm25_instance, BM25Okapi) and isinstance(bm25_chunk_ids_ordered, list):
                bm25_index_cache[cache_key] = bm25_instance
                bm25_ids_cache[cache_key] = bm25_chunk_ids_ordered
                print(f"BM25 index loaded successfully ({len(bm25_chunk_ids_ordered)} documents).")
                return bm25_instance, bm25_chunk_ids_ordered
            else:
                print(f"!!! Error: Loaded BM25 data has unexpected type. Index: {type(bm25_instance)}, IDs: {type(bm25_chunk_ids_ordered)}")
                return None, None
        except (pickle.UnpicklingError, EOFError, AttributeError, ImportError, IndexError) as load_err:
            print(f"!!! Error loading or unpickling BM25 index files: {load_err}")
            # Clear potentially corrupted cache entries if they were partially set
            if cache_key in bm25_index_cache: del bm25_index_cache[cache_key]
            if cache_key in bm25_ids_cache: del bm25_ids_cache[cache_key]
            return None, None
        except Exception as e:
            print(f"!!! Unexpected error loading BM25 index files: {e}")
            if cache_key in bm25_index_cache: del bm25_index_cache[cache_key]
            if cache_key in bm25_ids_cache: del bm25_ids_cache[cache_key]
            return None, None
    else:
        # print(f"BM25 index files not found at {index_path}/{map_path}. Will be built if indexing.") # Informative
        return None, None

# --- BM25 Index Building and Saving ---
def build_and_save_bm25_index(
    chunk_ids: List[str],
    chunk_texts: List[str],
    db_path: str,
    collection_name: str
):
    """
    Builds a BM25 index from provided texts and saves it along with the ID mapping.
    """
    if not RANK_BM25_AVAILABLE:
        print("Cannot build BM25 index: rank_bm25 library not installed.")
        return

    if not chunk_ids or not chunk_texts or len(chunk_ids) != len(chunk_texts):
        print("Warning: Invalid input for BM25 build (empty lists or mismatched lengths). Skipping build.")
        return

    print(f"Tokenizing {len(chunk_texts)} chunks for BM25 index...")
    tokenized_corpus = [
        tokenize_text_bm25(text)
        for text in tqdm(chunk_texts, desc="Tokenizing for BM25")
    ]

    # Check if tokenization resulted in a usable corpus
    if not any(tokenized_corpus):
        print("Warning: Tokenization resulted in an empty corpus. Skipping BM25 build.")
        return

    print(f"Building BM25 index...")
    try:
        bm25_index = BM25Okapi(tokenized_corpus)
    except Exception as build_err:
        print(f"!!! Error building BM25 index: {build_err}")
        return # Don't proceed to save if build fails

    bm25_index_path, bm25_mapping_path = get_bm25_paths(db_path, collection_name)

    try:
        print(f"Saving BM25 index ({len(chunk_ids)} docs) to: {bm25_index_path}")
        os.makedirs(os.path.dirname(bm25_index_path), exist_ok=True)
        with open(bm25_index_path, 'wb') as f_idx:
            pickle.dump(bm25_index, f_idx)

        print(f"Saving BM25 ID mapping to: {bm25_mapping_path}")
        with open(bm25_mapping_path, 'wb') as f_map:
            pickle.dump(chunk_ids, f_map)

        print("BM25 index saved successfully.")

        # Clear cache for this collection as it's been updated
        cache_key = (db_path, collection_name)
        if cache_key in bm25_index_cache: del bm25_index_cache[cache_key]
        if cache_key in bm25_ids_cache: del bm25_ids_cache[cache_key]

    except Exception as save_err:
        print(f"!!! Error saving BM25 index or mapping files: {save_err}")
        # Attempt to clean up potentially partially written files
        if os.path.exists(bm25_index_path): os.remove(bm25_index_path)
        if os.path.exists(bm25_mapping_path): os.remove(bm25_mapping_path)
