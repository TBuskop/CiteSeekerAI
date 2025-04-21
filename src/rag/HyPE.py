import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, os.pardir, os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

print(f"{_PROJECT_ROOT} added to sys.path")

import argparse
import json
import concurrent.futures
from typing import List
from src.rag.chroma_manager import get_chroma_collection
from src.my_utils.llm_interface import initialize_clients, generate_llm_response  # Keep others
from config import EMBEDDING_MODEL, OUTPUT_EMBEDDING_DIMENSION, DEFAULT_EMBED_BATCH_SIZE, DEFAULT_EMBED_DELAY
from src.rag.embedding import run_embed_mode_logic, EMBEDDING_IMPORTS_OK, genai  # batch embedding helper

SYSTEM_PROMPT = """
Role: You are the HyPE Question Generator, designed to simulate researcher search queries based on scientific abstracts.
Core Task: You will receive a JSON array containing multiple scientific abstracts, each tagged with a unique ID. For each abstract object in the input array, your task is to:
Read the abstract text.
Generate exactly 6 to 12 concise questions that represent realistic search queries a researcher might use to find a paper addressing the topics, methods, and findings presented in that specific abstract.
Adhere strictly to all constraints and guidelines below for the questions generated for each abstract.
Input Format:
You will be provided with the input abstracts after this system prompt, formatted as a JSON array. Each element in the array will be a JSON object with the following structure:
[
  {
    "id": "unique_abstract_id_1",
    "abstract": "Text of the first abstract..."
  },
  {
    "id": "unique_abstract_id_2",
    "abstract": "Text of the second abstract..."
  },
  ...
]

Required Output Format:
Your entire response MUST be a single, valid JSON array. Each element in the output array must be a JSON object corresponding to one of the input abstracts, containing:
The exact same id as the corresponding input abstract.
A key named questions whose value is a JSON list of strings. Each string in this list must be one generated question for that abstract.
The output structure MUST look like this:
[
  {
    "id": "unique_abstract_id_1",
    "questions": [
      "Question 1 for abstract 1",
      "Question 2 for abstract 1",
      "...",
      "Question N for abstract 1"
    ]
  },
  {
    "id": "unique_abstract_id_2",
    "questions": [
      "Question 1 for abstract 2",
      "Question 2 for abstract 2",
      "...",
      "Question M for abstract 2"
    ]
  },
  ...
]

CRITICAL: Output only the valid JSON array. Do not include any introductory text, explanations, apologies, or any other text before or after the JSON array structure. Ensure the JSON is well-formed.
Constraints & Guidelines (Apply to question generation for EACH abstract individually):
Quantity Per Abstract: Generate exactly 6 to 12 questions for each abstract.
Question Format: Within the questions list for each abstract, each question string should be on its own conceptually (as an element in the JSON list). Do NOT include numbering or bullet points within the question strings themselves.
Crucial Constraint: No Self-Reference:
Absolutely Forbidden Phrasing: The generated question strings MUST NOT contain any words or phrases referring directly to the specific abstract or paper being processed (e.g., avoid "this study", "the paper", "the authors", "results reported here", "according to the abstract").
Perspective Shift: Frame questions as if you are a researcher searching a database before finding this specific paper, using its concepts as search terms. Questions must be about the subject matter, not about a document discussing the subject matter.
Example: If an abstract discusses Method X applied to Y: Incorrect: "How did this study apply Method X to Y?" Correct: "How is Method X applied to Y?" or "What are applications of Method X for Y?"
Preserve Specifics: Strictly preserve any specific numbers, dates, locations, percentages, proper nouns (e.g., model names, dataset names, specific chemicals), or named entities found in the abstract. Copy them exactly into relevant question strings.
Target Diverse Facets: Ensure questions cover various aspects potentially mentioned in the abstract: motivation, methods, datasets, metrics, key findings, quantitative results, limitations, scope (geographic/temporal), implications.
Concise & Specific (Search Intent): Aim for approximately 10-18 words per question string. Make them specific enough to be useful search terms.
Plain Language: Use clear, accessible language suitable for search queries. Avoid excessive jargon unless it's a specific named entity.
Include Variations/Synonyms (Optional): Where helpful for search, incorporate synonyms or alternative phrasings for key concepts (e.g., "precipitation change" vs "rainfall variability").
"""

model = "gemini-2.0-flash-lite" # cheap model for HyPE question generation

def run_hype_index(db_path: str, source_collection_name: str, hype_collection_name: str, client=None):
    # Initialization of clients should happen outside, not in parallel contexts
    config_params = {'embed_batch_size': DEFAULT_EMBED_BATCH_SIZE, 'embed_delay': DEFAULT_EMBED_DELAY}

    source_col = get_chroma_collection(db_path, source_collection_name, execution_mode="query")
    hype_col = get_chroma_collection(db_path, hype_collection_name, execution_mode="index")
    # determine which abstracts already processed
    existing = hype_col.get(include=['metadatas'])
    processed_chunks = {md.get('original_chunk_id') for md in existing.get('metadatas', [])}
    # load all source abstracts
    results = source_col.get(include=['documents'])  # ids are returned by default
    all_ids, all_texts = results['ids'], results['documents']
    # filter out already processed
    filtered = [(cid, txt) for cid, txt in zip(all_ids, all_texts) if cid not in processed_chunks]
    if not filtered:
        print("HyPE: No new abstracts to HyPE")
        print("\nHyPE: Starting embedding phase...")
        is_valid_client = EMBEDDING_IMPORTS_OK and genai is not None and isinstance(client, genai.Client)
        if is_valid_client:
            print(f"HyPE: Calling embedding logic with valid client type: {type(client)}")
            run_embed_mode_logic(config_params, hype_col, client)
        else:
            print(f"HyPE: Skipping embedding logic due to invalid or missing client (Type: {type(client)}, Imports OK: {EMBEDDING_IMPORTS_OK}, GenAI Module Loaded: {genai is not None})")
        print("HyPE: Indexing and embedding completed.")
        return

    chunk_ids, chunk_texts = zip(*filtered)

    # batch parameters
    BATCH_SIZE = 10

    #print number of batches
    num_batches = len(chunk_ids) // BATCH_SIZE + (1 if len(chunk_ids) % BATCH_SIZE > 0 else 0)
    print(f"HyPE: Processing {len(chunk_ids)} abstracts in {num_batches} batches of size {BATCH_SIZE}.")

    # Prepare batch list for parallel question generation
    batches = [(chunk_ids[i:i+BATCH_SIZE], chunk_texts[i:i+BATCH_SIZE], i//BATCH_SIZE+1)
               for i in range(0, len(chunk_ids), BATCH_SIZE)]

    def process_batch(batch_ids, batch_texts, batch_number):
        print(f"HyPE: Processing batch {batch_number}/{num_batches}...")
        abstracts_list = [{'id': aid, 'abstract': txt} for aid, txt in zip(batch_ids, batch_texts)]
        prompt = SYSTEM_PROMPT + "\n\nProvide output as JSON array where each element has 'id' and 'questions' list.\n"
        prompt += f"Input abstracts: {json.dumps(abstracts_list)}"
        response = generate_llm_response(prompt, max_tokens=2048, temperature=0, model=model).strip()
        # remove markdown wrappers
        for wrapper in ("```json\n", "\n```"):
            if response.startswith(wrapper): response = response[len(wrapper):]
            if response.endswith(wrapper): response = response[:-len(wrapper)]
        try:
            batch_output = json.loads(response)
        except Exception:
            print(f"HyPE: Failed to parse JSON for batch {batch_number}.")
            return
        ids_, mds, docs = [], [], []
        for item in batch_output:
            aid = item.get('id')
            qs = item.get('questions', [])
            doc_result = source_col.get(ids=[aid], include=['documents'])
            docs_list = doc_result.get('documents', [])
            if not docs_list:
                print(f"HyPE: Warning: No document found for id {aid} in batch {batch_number}. Skipping.")
                continue
            text = docs_list[0]
            for idx_q, question in enumerate(qs):
                ids_.append(f"{aid}_hq{idx_q}")
                mds.append({'original_chunk_id': aid, 'question': question, 'has_embedding': False})
                docs.append(text)
        if ids_:
            # detect duplicate question IDs
            unique_ids = set(ids_)
            if len(ids_) != len(unique_ids):
                # find duplicates
                dup_ids = sorted({i for i in ids_ if ids_.count(i) > 1})
                print(f"HyPE: Error: Expected IDs to be unique but found {len(dup_ids)} duplicate IDs: {', '.join(dup_ids)} in batch {batch_number}/{num_batches}. Skipping upsert.")
                return
            hype_col.upsert(ids=ids_, metadatas=mds, documents=docs)
            print(f"HyPE: Upserted {len(ids_)} questions for batch {batch_number}/{num_batches}.")
        else:
            print(f"HyPE: No new questions for batch {batch_number}/{num_batches}.")

    # generate questions in parallel per batch
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(process_batch, b_ids, b_txt, b_num) for b_ids, b_txt, b_num in batches]
        for fut in concurrent.futures.as_completed(futures):
            try:
                fut.result()
            except Exception as e:
                print(f"HyPE: Error in batch processing: {e}")

    # after all batches, embed newly added entries
    print("\nHyPE: Starting embedding phase...")
    is_valid_client = EMBEDDING_IMPORTS_OK and genai is not None and isinstance(client, genai.Client)
    if is_valid_client:
        print(f"HyPE: Calling embedding logic with valid client type: {type(client)}")
        run_embed_mode_logic(config_params, hype_col, client)
    else:
        print(f"HyPE: Skipping embedding logic due to invalid or missing client (Type: {type(client)}, Imports OK: {EMBEDDING_IMPORTS_OK}, GenAI Module Loaded: {genai is not None})")

    print("HyPE: Indexing and embedding completed.")


if __name__ == '__main__':
    import sys
    if len(sys.argv) == 1:
        db_path = f'{_PROJECT_ROOT}/data/databases/abstract_chroma_db'
        source_collection = 'abstracts'
        hype_collection = 'abstracts_hype'
        print(f"Running HyPE for {source_collection} -> {hype_collection} in {db_path}")
        client_to_pass = initialize_clients()
        print(f"HyPE __main__: Client object after init: {type(client_to_pass)}")
        run_hype_index(db_path, source_collection, hype_collection, client=client_to_pass)
    else:
        parser = argparse.ArgumentParser(description="HyPE indexing: generate hypothetical prompt embeddings for chunks")
        parser.add_argument('--db_path', required=True, help='Path to ChromaDB directory')
        parser.add_argument('--source_collection', required=True, help='Name of the source chunks collection')
        parser.add_argument('--hype_collection', required=True, help='Name of the HyPE collection to create/update')
        parser.add_argument('--questions_per_chunk', type=int, default=6, help='Number of hypothetical questions per chunk')
        args = parser.parse_args()
        client_to_pass = initialize_clients()
        print(f"HyPE __main__ (args): Client object after init: {type(client_to_pass)}")
        run_hype_index(args.db_path, args.source_collection, args.hype_collection, client=client_to_pass)
