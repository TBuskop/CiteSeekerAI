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

# Model Configuration
# Make sure EMBEDDING_MODEL is defined, e.g., "models/text-embedding-004" or "models/embedding-001"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/text-embedding-004")
CHUNK_CONTEXT_MODEL = os.getenv("CHUNK_CONTEXT_MODEL", "models/gemini-1.5-flash")
SUBQUERY_MODEL = os.getenv("SUBQUERY_MODEL", "models/gemini-1.5-flash")
CHAT_MODEL = os.getenv("CHAT_MODEL", "models/gemini-1.5-flash") # Or your preferred chat model
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2") # Optional

# Chunking Parameters
DEFAULT_MAX_TOKENS = int(os.getenv("DEFAULT_MAX_TOKENS", 500))
DEFAULT_OVERLAP = int(os.getenv("DEFAULT_OVERLAP", 50))
DEFAULT_TOTAL_CONTEXT_WINDOW = int(os.getenv("DEFAULT_TOTAL_CONTEXT_WINDOW", 1024)) # Window for context generation

# Retrieval Parameters
DEFAULT_CONTEXT_LENGTH = int(os.getenv("DEFAULT_CONTEXT_LENGTH", 64)) # Max tokens for generated context summary
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", 5))
DEFAULT_RERANK_CANDIDATE_COUNT = int(os.getenv("DEFAULT_RERANK_CANDIDATE_COUNT", 50)) # How many candidates to feed reranker

# Embedding Configuration
# Set this based on your chosen EMBEDDING_MODEL's output dimension if not truncating
# e.g., 768 for embedding-001, 768 for text-embedding-004 (default), 1536 for text-embedding-ada-002
# Set to None if you don't want to explicitly override or request truncation.
OUTPUT_EMBEDDING_DIMENSION = int(os.getenv("OUTPUT_EMBEDDING_DIMENSION", 768)) if os.getenv("OUTPUT_EMBEDDING_DIMENSION") else None

# ChromaDB Configuration
DEFAULT_CHROMA_COLLECTION_NAME = "rag_chunks_hybrid_default"

# --- Validate Essential Config ---
if not GEMINI_API_KEY:
    print("Warning: GEMINI_API_KEY environment variable not set.")
    # Depending on usage, you might want to raise an error here if Gemini is essential
    # raise ValueError("GEMINI_API_KEY is required.")
