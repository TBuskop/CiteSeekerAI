# Academic Literature RAG Pipeline

## Overview

This project implements a Retrieval-Augmented Generation (RAG) pipeline specifically designed for academic literature research. It aims to answer complex research questions by:
1.  Collect abstracts of all papers resulting from a SCOPUS search query 
2.  Decomposing the main question of the user into logical sub-queries.
3.  Retrieving relevant academic papers based on abstracts using vector search (ChromaDB).
3.  Downloading the full text of relevant papers.
4.  Chunking the full text and storing embeddings in a vector database.
5.  Building a query-specific database of relevant chunks.
6.  Using a RAG approach (retrieval, re-ranking, generation) to answer each sub-query based on the relevant text chunks.
7.  LLM based refinement of initially generated sub-queries based on previous answers.
8.  Combining the answers to provide a comprehensive response to the initial research question.

## Features

*   **Query Decomposition:** Breaks down complex questions using LLMs.
*   **Vector Search:** Uses ChromaDB for efficient retrieval of abstracts and text chunks.
*   **Full-Text Download:** Automates downloading papers based on DOIs. This requires being on a network that has access to the journals of interest. One can be on eduroam for example or use a university VPN when outside the campus area.
*   **Text Chunking & Embedding:** Processes PDFs/XMLs into manageable chunks and generates embeddings.
*   **RAG Implementation:** Leverages retrieval, re-ranking, and LLM generation for answering queries based on source material.
*   **Sequential Processing:** Handles sub-queries one after another, allowing for potential refinement based on prior results.
*   **Configurable:** Uses a central `config.py` for models, paths, and parameters.
*   **HyPE Support:** Option to use Hypothetical Prompt Embeddings for improved retrieval.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd academic_lit_llm_2
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: A `requirements.txt` file should be created based on the project's imports. You might need to generate it using `pip freeze > requirements.txt` after installing necessary packages like `chromadb`, `google-generativeai`, `python-dotenv`, `sentence-transformers`, `requests`, `beautifulsoup4`, `PyPDF2`, etc.)*

4.  **Environment Variables:**
    Create a `.env` file in the project root directory by replacing the .example extension and add the gemini API key.
    

## Configuration

Key parameters, model names, and paths are configured in `config.py`. Review and adjust settings like:

*   Model names (`EMBEDDING_MODEL`, `SUBQUERY_MODEL`, `CHAT_MODEL`, etc.)
*   Chunking parameters (`DEFAULT_MAX_TOKENS`, `DEFAULT_CHUNK_OVERLAP`)
*   Retrieval parameters (`TOP_K_ABSTRACTS`, `DEFAULT_TOP_K`)
*   `HYPE` setting (True/False)
*   Re-ranker settings

Database paths and output directories are configured in the workflow script but rely on a base `data` directory structure within the project root.

## Running the Workflow

The main sequential workflow is executed via:

```bash
python workflows/DeepResearch_squential.py
```