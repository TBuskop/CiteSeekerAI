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
    find academic peer reviewed literature to support statements in the following text. Give me a list the supported statements and which papers are relevant for it and why

We are approaching a world where society is faced with tough choices due to the effects of climate change. Where we thought that we could adapt to 1.5 degrees warming we find our selves in a situation where global leaders have not taken enough action to limit emissions, meaning we will overshoot the goal set in Paris 2015. At the same time, our adaptation plans to keep society function as it does now reach short of that goal. We reach limits to what we can do to keep things the same. As a consequence we need to find the trade-offs in future actions. What climate do we adapt to, what do we prioritise and what gets left behind?  

One example we detail here is food production in the European Union. The EU has for a long time subsidised farmers within its border in order to guarantee nutritious food production at low consumer prices and secure the livelihoods of those producing this food. Currently a third of the annual EU budget goes towards agricultural policies. One of the biggest agricultural producers and receiver of funds is Spain. Many countries within the EU depend on Spain to produce part of their food supply. 

Now as, the climate changes, Spain is projected to be one of the most water scarce regions in Europe. At the moment many regions in Spain are already water stressed and water use limits are in place. The future could exacerbate the situation leading to competition of water resources among purposes. If no action is taken, this not only threatens local production capacity and livelihoods, but also the food supply to other EU member states. 

Now the question is what do we do? How much less water is available for crop production? What measures and policies can be implemented? What are the trade-ofss between them? Who should fund these

Potential policies could be:
- increase irrigation from groundwater
	- Long term depletion of strategic reserves
	- Many smallholders in Spain probably unable to invest
- scale down other water users
	- reduce tourism, industry, domestic use
- Increase water storage capacity
	- 
- Moving production elsewhere
	- more secure food supply for the European community
	- Local jobs are lost in a place already affected harshly by climate change
"""
QUERY_DECOMPOSITION_NR = 5  # Number of sub-queries to generate from the main query

# --- Scopus Search string ---
SCOPUS_SEARCH_STRING = """
    ("impact attribution" AND "climate")
"""

USE_SCIHUB = True  # Use SciHub for collecting papers


# --- API Keys ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- Core Model Configuration ---
# These are the primary models you'll interact with.
CHAT_MODEL = "gemini-2.5-flash-preview-04-17" #"gemini-2.0-flash-lite", "gemini-2.5-pro-exp-03-25"  # Main model for chat and generation
SUBQUERY_MODEL = "gemini-2.5-pro-exp-03-25" # Model for generating sub-queries (more powerful is better and relevant questions)

# --- Query Configuration ---
# How many results to retrieve at different stages.
TOP_K_ABSTRACTS = 50  # Number of papers to retrieve based on abstract similarity
DEFAULT_TOP_K = 20    # Number of chunks to retrieve from the selected papers for context

# --- Chunking Parameters ---
# Settings for how documents are split into smaller pieces.
DEFAULT_MAX_TOKENS = 1000        # Maximum tokens per chunk
DEFAULT_CHUNK_OVERLAP = 150      # Number of tokens to overlap between chunks
DEFAULT_CONTEXT_LENGTH = 120     # Desired length of context extracted from chunks (e.g., for summaries)
DEFAULT_TOTAL_CONTEXT_WINDOW = 2000 # Total context window size for models (informational)

# --- Embedding Configuration ---
EMBEDDING_MODEL = "text-embedding-004"  # Model for creating text embeddings
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
HYPE_SOURCE_COLLECTION_NAME = "abstracts"
HYPE_MODEL = "gemini-2.0-flash-lite" # cheap model for HyPE question generation


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
