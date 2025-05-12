import os
import subprocess  # Import the subprocess module
# from rag_bm25 import initialize_clients # Keep this if needed elsewhere, but not strictly required for running subprocess

def main():
    # Initialize clients if rag_bm25.py doesn't do it internally on startup
    # print("Initializing API clients...")
    # initialize_clients() # Usually the target script handles its own initialization
    # change run path to rag folder
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    db_path = "chunk_database/chunks_db"
    collection_name = "test_collection"
    folder_path = "cleaned_text/paper_2_intro"

    # --- Define the query (strip leading/trailing whitespace) ---
    query = """
    What are plausibilistic climate storylines and how are they different from other climate storylines? Write me a short literature review on this topic. 
    """.strip() # Use strip() to remove leading/trailing newlines/spaces

    # query = """
    #     I'm preparing a proposal and I need about half an a4 of text in literature review, with relevant references to support the following points:
    #         A link to EUCRA, IPCC, world drought atlas, or other authoritative sources showing that climate change is making Europe drier.
    #         That increasing drought has negative effects on agriculture.
    #         That these effects are especially severe in Mediterranean countries — ideally with an example from a paper (e.g., crop failure, severe water shortages).
    #         If easily available, a map or figure showing drought risk in Europe or in a Mediterranean country.
    #     """.strip() # Use strip() to remove leading/trailing newlines/spaces

    # # --- Example 1: Indexing (using subprocess) ---
    # print("EXAMPLE 1: Indexing all documents in folder")
    # index_command_list = [
    #     'python',
    #     'rag_bm25.py',
    #     '--mode', 'index',
    #     '--folder_path', folder_path,
    #     '--db_path', db_path,
    #     '--collection_name', collection_name
    #     # Add --force_reindex if needed: '--force_reindex'
    # ]
    # print(f"Running: {' '.join(index_command_list)}") # Print a readable version
    # # Use check=True to raise an error if the command fails
    # subprocess.run(index_command_list, check=True)

    # # --- Example 2: Embedding (using subprocess) ---
    # print("\nEXAMPLE 2: Embed all chunks in database")
    # embed_command_list = [
    #     'python',
    #     'rag_bm25.py',
    #     '--mode', 'embed',
    #     '--db_path', db_path,
    #     '--collection_name', collection_name,
    #     '--embed_batch_size', '50', # Pass numbers as strings
    #     '--embed_delay', '30'       # Pass numbers as strings
    # ]
    # print(f"Running: {' '.join(embed_command_list)}")
    # subprocess.run(embed_command_list, check=True)

    # # --- Example 3: Querying (using subprocess) ---
    print("\nEXAMPLE 3: Query with iterative RAG")
    print("="*50)
    query_command_list = [
        'python',
        'rag_bm25.py',
        '--mode', 'query',
        '--query', query,             # Pass the query string directly
        '--db_path', db_path,
        '--collection_name', collection_name
        # Add --top_k if needed: '--top_k', '5'
    ]
    print(f"Running: {' '.join(query_command_list[:5])} --query \"<QUERY CONTENT>\" ...") # Print truncated command
    subprocess.run(query_command_list, check=True)

if __name__ == "__main__":
    main()