"""
Configuration settings for the RAG application.
This file handles environment variables and other configuration settings.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv(override=True)  # Force override of existing environment variables

# --- Query ---
QUERY = """
    How can probabbilities be integrated into climate impact storylines?
"""
QUERY_DECOMPOSITION_NR = 3  # Number of sub-queries to generate from the main query

# --- API Keys ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- Core Model Configuration ---
# These are the primary models you'll interact with.
EMBEDDING_MODEL = "text-embedding-004"  # Model for creating text embeddings
CHAT_MODEL = "gemini-2.5-flash-preview-04-17" #"gemini-2.0-flash-lite", "gemini-2.5-pro-exp-03-25"  # Main model for chat and generation
SUBQUERY_MODEL = "gemini-2.5-pro-exp-03-25" # Model for generating sub-queries (more powerful is better and relevant questions)

# --- Query Configuration ---
# How many results to retrieve at different stages.
TOP_K_ABSTRACTS = 2  # Number of papers to retrieve based on abstract similarity
DEFAULT_TOP_K = 4    # Number of chunks to retrieve from the selected papers for context

# --- Chunking Parameters ---
# Settings for how documents are split into smaller pieces.
DEFAULT_MAX_TOKENS = 1000        # Maximum tokens per chunk
DEFAULT_CHUNK_OVERLAP = 150      # Number of tokens to overlap between chunks
DEFAULT_CONTEXT_LENGTH = 120     # Desired length of context extracted from chunks (e.g., for summaries)
DEFAULT_TOTAL_CONTEXT_WINDOW = 2000 # Total context window size for models (informational)

# --- Embedding Configuration ---
OUTPUT_EMBEDDING_DIMENSION = 768 # Expected dimension of the embeddings

# --- Advanced Model Configuration ---
# Models for more specific or internal tasks.
CHUNK_CONTEXT_MODEL = "gemini-1.5-flash-8b"   # Model for processing/summarizing chunks
SUBQUERY_MODEL_SIMPLE = "gemini-2.0-flash-lite" # Simpler/faster model for sub-queries

# --- Hypothetical Document Embeddings (HyPE) Configuration ---
HYPE = True  # Enable or disable HyPE for query generation
# Suffix to append for HyPE collections in the vector store
HYPE_SUFFIX = "_hype"
# Source abstracts collection for enriching HyPE metadata
HYPE_SOURCE_COLLECTION_NAME = os.getenv("HYPE_SOURCE_COLLECTION_NAME", "abstracts")

# --- Re-ranker Configuration ---
# For refining search results after initial retrieval.
RERANKER_MODEL = 'cross-encoder/ms-marco-MiniLM-L-12-v2' # Model for re-ranking
DEFAULT_RERANK_CANDIDATE_COUNT = 100 # Number of candidates to re-rank

# --- Other Defaults & ChromaDB Settings ---
DEFAULT_EMBED_BATCH_SIZE = 100  # Max batch size for embedding (API limits may apply)
DEFAULT_EMBED_DELAY = 0         # Delay between embedding batches (if needed for rate limiting)
DEFAULT_CHROMA_COLLECTION_NAME = "rag_chunks_hybrid_default" # Default collection name in ChromaDB

# --- Validate Essential Config ---
if not GEMINI_API_KEY:
    print("Warning: GEMINI_API_KEY environment variable not set.")
    # Depending on usage, you might want to raise an error here if Gemini is essential
    # raise ValueError("GEMINI_API_KEY is required.")
