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
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-3.5-turbo")

# Chunking configuration - explicitly force the correct default value if not found
DEFAULT_MAX_TOKENS = int(os.environ.get("DEFAULT_MAX_TOKENS", "300"))
DEFAULT_OVERLAP = int(os.environ.get("DEFAULT_OVERLAP", "20"))
DEFAULT_CONTEXT_LENGTH = int(os.environ.get("DEFAULT_CONTEXT_LENGTH", "20"))

# Query configuration
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "5"))
