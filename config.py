"""
Configuration settings for the RAG application.
This file handles environment variables and other configuration settings.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv(override=True)  # Force override of existing environment variables

# OpenAI API configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
CHUNK_CONTEXT_MODEL = os.getenv("CHUNK_CONTEXT_MODEL")
SUBQUERY_MODEL = os.getenv("SUBQUERY_MODEL")
CHAT_MODEL = os.getenv("CHAT_MODEL")

# Chunking configuration - explicitly force the correct default value if not found
DEFAULT_MAX_TOKENS = int(os.environ.get("DEFAULT_MAX_TOKENS", "300"))
DEFAULT_OVERLAP = int(os.environ.get("DEFAULT_OVERLAP", "20"))
DEFAULT_CONTEXT_LENGTH = int(os.environ.get("DEFAULT_CONTEXT_LENGTH", "20"))

#Embedding configuration
OUTPUT_EMBEDDING_DIMENSION= int(os.getenv("OUTPUT_EMBEDDING_DIMENSION", "768"))

# Query configuration
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "5"))

# --- Re-ranker Configuration ---
# Model examples: 'cross-encoder/ms-marco-MiniLM-L-6-v2', 'cross-encoder/ms-marco-TinyBERT-L-2-v2'
# Or set to None to disable re-ranking
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
# Number of candidates to fetch *before* re-ranking
DEFAULT_RERANK_CANDIDATE_COUNT = int(os.getenv("DEFAULT_RERANK_CANDIDATE_COUNT", "100"))