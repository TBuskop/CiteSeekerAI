"""
Configuration settings for the RAG application.
This file handles environment variables and other configuration settings.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv(override=True)  # Force override of existing environment variables

# API Keys (Load from environment variables)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") # Example, keep if you might use OpenAI
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Model Configuration (now hardcoded, not from env)
EMBEDDING_MODEL = "text-embedding-004"
CHUNK_CONTEXT_MODEL = "gemini-1.5-flash-8b"
SUBQUERY_MODEL = "gemini-2.5-pro-exp-03-25" # gemini-2.5-pro-exp-03-25 # gemini-2.5-pro-preview-03-25 (this is a paid one but the action is small so still chip)
SUBQUERY_MODEL_SIMPLE = "gemini-2.0-flash-lite" # gemini-2.5-pro-exp-03-25 # gemini-2.5-pro-preview-03-25 (this is a paid one but the action is small so still chip)
CHAT_MODEL = "gemini-2.5-pro-exp-03-25" #"gemini-2.0-flash-lite"  # gemini-2.5-pro-exp-03-25 (using free but good one for large requests)

# Chunking Parameters
DEFAULT_MAX_TOKENS = 1000
DEFAULT_CHUNK_OVERLAP = 150
DEFAULT_CONTEXT_LENGTH = 120
DEFAULT_TOTAL_CONTEXT_WINDOW = 2000

# Embedding Configuration
OUTPUT_EMBEDDING_DIMENSION = 768

# Query configuration
DEFAULT_TOP_K = 50

# --- Re-ranker Configuration ---
RERANKER_MODEL = 'cross-encoder/ms-marco-MiniLM-L-12-v2'
DEFAULT_RERANK_CANDIDATE_COUNT = 50

# Other defaults (if needed)
DEFAULT_EMBED_BATCH_SIZE = 50
DEFAULT_EMBED_DELAY = 25
DEFAULT_CHROMA_COLLECTION_NAME = "rag_chunks_hybrid_default"

# --- Validate Essential Config ---
if not GEMINI_API_KEY:
    print("Warning: GEMINI_API_KEY environment variable not set.")
    # Depending on usage, you might want to raise an error here if Gemini is essential
    # raise ValueError("GEMINI_API_KEY is required.")
