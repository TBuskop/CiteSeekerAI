import hashlib
import tiktoken
from rag.config import EMBEDDING_MODEL, DEFAULT_MAX_TOKENS, DEFAULT_OVERLAP, CHAT_MODEL

# --- File Hashing ---
def compute_file_hash(file_path: str) -> str:
    """Computes the SHA256 hash of a file."""
    hasher = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            while True:
                chunk = f.read(4096)
                if not chunk: break
                hasher.update(chunk)
        return hasher.hexdigest()
    except FileNotFoundError: raise
    except Exception as e: raise RuntimeError(f"Error hashing file {file_path}: {e}") from e

# --- Token Counting ---
def count_tokens(text: str, model: str = EMBEDDING_MODEL) -> int:
    """Counts tokens in a string using tiktoken."""
    try:
        # Cache encoding objects for efficiency
        if not hasattr(count_tokens, "_encoding_cache"): count_tokens._encoding_cache = {}
        if model not in count_tokens._encoding_cache:
             try: count_tokens._encoding_cache[model] = tiktoken.encoding_for_model(model)
             except Exception:
                 try: count_tokens._encoding_cache[model] = tiktoken.get_encoding("cl100k_base")
                 except Exception: count_tokens._encoding_cache[model] = None # Fallback if all else fails

        encoding = count_tokens._encoding_cache.get(model)
        # Use encoding if available, otherwise fallback to simple split
        return len(encoding.encode(text)) if encoding else len(text.split())
    except Exception:
        # Fallback on any unexpected error during tokenization
        return len(text.split())

# --- Text Chunking ---
def chunk_document_tokens(document: str, max_tokens: int = DEFAULT_MAX_TOKENS, overlap: int = DEFAULT_OVERLAP) -> list[tuple[str, int, int]]:
    """Splits a document into chunks based on token count with overlap."""
    if max_tokens <= 0: raise ValueError("max_tokens must be positive.")
    if overlap < 0: raise ValueError("overlap cannot be negative.")
    if overlap >= max_tokens:
        print(f"Warning: Overlap ({overlap}) >= max_tokens ({max_tokens}). Adjusting overlap to {max_tokens // 2}.")
        overlap = max_tokens // 2 # Ensure overlap is less than max_tokens

    try:
        # Prioritize the embedding model's tokenizer, fallback to cl100k_base
        encoding = tiktoken.encoding_for_model(EMBEDDING_MODEL)
    except Exception:
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            print("Warning: Using fallback 'cl100k_base' encoding for chunking.")
        except Exception as e:
            raise ValueError(f"Could not get tiktoken encoding for chunking: {e}")

    try:
        tokens = encoding.encode(document)
    except Exception as e:
        print(f"Error encoding document for chunking: {e}. Returning empty list.")
        return []

    if not tokens: return []

    chunks = []
    start_token_idx = 0
    total_tokens = len(tokens)

    while start_token_idx < total_tokens:
        end_token_idx = min(start_token_idx + max_tokens, total_tokens)
        chunk_tokens = tokens[start_token_idx:end_token_idx]

        if not chunk_tokens: break # Should not happen if logic is correct, but safeguard

        try:
            # Decode the chunk tokens back to text
            chunk_text = encoding.decode(chunk_tokens, errors='replace').strip()
        except Exception as e:
            print(f"Warning: Error decoding tokens {start_token_idx}-{end_token_idx}: {e}. Skipping this potential chunk position.")
            # Advance start index carefully to avoid infinite loops on persistent errors
            next_start = start_token_idx + max_tokens - overlap
            start_token_idx = next_start if next_start > start_token_idx else start_token_idx + 1
            continue

        if chunk_text: # Only add non-empty chunks
            chunks.append((chunk_text, start_token_idx, end_token_idx))

        # Calculate the start of the next chunk
        next_start = start_token_idx + max_tokens - overlap

        # Ensure progress is made, especially if max_tokens is small or overlap is large
        if next_start <= start_token_idx:
            start_token_idx += 1
        else:
            start_token_idx = next_start

    return chunks


# --- Text Truncation ---
def truncate_text(text: str, token_limit: int, model: str = CHAT_MODEL) -> str:
    """Truncates text to a specified token limit."""
    if token_limit <= 0: return ""
    try:
        # Use the specified model's tokenizer, fallback to cl100k_base
        encoding = tiktoken.encoding_for_model(model)
    except Exception:
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            # print(f"Warning: Using fallback 'cl100k_base' encoding for truncation (model: {model}).")
        except Exception as e:
            print(f"Error getting encoding for truncation: {e}. Returning original text.")
            return text
    try:
        tokens = encoding.encode(text)
        if len(tokens) > token_limit:
            tokens = tokens[:token_limit]
            # Decode truncated tokens, replacing errors
            text = encoding.decode(tokens, errors='replace')
        return text
    except Exception as e:
        print(f"Error truncating text: {e}. Returning original text.")
        return text
