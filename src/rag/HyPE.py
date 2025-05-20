import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, os.pardir, os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import argparse
import json
import concurrent.futures
from typing import List
from src.rag.chroma_manager import get_chroma_collection
from src.my_utils.llm_interface import initialize_clients, generate_llm_response  # Keep others
from config import DEFAULT_EMBED_BATCH_SIZE, DEFAULT_EMBED_DELAY, HYPE_MODEL
from src.rag.embedding import run_embed_mode_logic, EMBEDDING_IMPORTS_OK, genai  # batch embedding helper

# read system prompt from file
SYSTEM_PROMPT = ""
with open(os.path.join(_PROJECT_ROOT, 'src', 'llm_prompts', 'HyPE.txt'), 'r') as f:
    SYSTEM_PROMPT = f.read().strip()
# read model name from config

model = HYPE_MODEL # cheap model for HyPE question generation

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
        
        response_text = generate_llm_response(prompt, max_tokens=4096, temperature=0, model=model).strip() # Increased max_tokens
        
        # More robust markdown wrapper removal
        cleaned_response = response_text
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[len("```json"):]

        elif cleaned_response.startswith("```"): # General case if ```json was not specific enough
            cleaned_response = cleaned_response[len("```"):]

        if cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[:-len("```")]
        cleaned_response = cleaned_response.strip()

        try:
            raw_batch_output_from_llm = json.loads(cleaned_response)
        except json.JSONDecodeError as e:
            print(f"HyPE: Failed to parse JSON for batch {batch_number}. Error: {e}. Response snippet: '{cleaned_response[:500]}...'")
            return

        if not isinstance(raw_batch_output_from_llm, list):
            print(f"HyPE: LLM output for batch {batch_number} is not a list as expected. Type: {type(raw_batch_output_from_llm)}. Output snippet: {str(raw_batch_output_from_llm)[:500]}")
            return

        valid_items_for_processing = []
        processed_input_aids_in_llm_response = set()

        for item_from_llm in raw_batch_output_from_llm:
            if not isinstance(item_from_llm, dict):
                print(f"HyPE: Warning (Batch {batch_number}): Item in LLM response is not a dictionary. Item: '{str(item_from_llm)[:200]}'. Skipping this item.")
                continue

            aid = item_from_llm.get('id')
            questions = item_from_llm.get('questions')

            if not aid:
                print(f"HyPE: Warning (Batch {batch_number}): Item in LLM response is missing 'id'. Item: '{str(item_from_llm)[:200]}'. Skipping this item.")
                continue
            
            if aid not in batch_ids: 
                print(f"HyPE: Warning (Batch {batch_number}): LLM returned id '{aid}' which was not in the original input batch_ids. Skipping this item.")
                continue

            if not isinstance(questions, list):
                print(f"HyPE: Warning (Batch {batch_number}): 'questions' for id '{aid}' is not a list or is missing. Type: {type(questions)}. Item: '{str(item_from_llm)[:200]}'. Skipping this item.")
                continue
            
            if not all(isinstance(q, str) for q in questions):
                print(f"HyPE: Warning (Batch {batch_number}): Not all questions for id '{aid}' are strings. Questions: {questions}. Skipping this item.")
                continue

            if aid in processed_input_aids_in_llm_response:
                print(f"HyPE: Warning (Batch {batch_number}): Duplicate aid '{aid}' encountered in LLM's response for this batch. Processing first instance only.")
                continue 
            
            processed_input_aids_in_llm_response.add(aid)
            valid_items_for_processing.append(item_from_llm)
        
        if len(processed_input_aids_in_llm_response) < len(batch_ids):
            missing_ids_from_response = set(batch_ids) - processed_input_aids_in_llm_response
            print(f"HyPE: Info (Batch {batch_number}): LLM response did not cover all {len(batch_ids)} input IDs. Missing {len(missing_ids_from_response)} IDs: {list(missing_ids_from_response)[:5]}" + 
                  (f"... and {len(missing_ids_from_response) - 5} more" if len(missing_ids_from_response) > 5 else ""))

        ids_to_upsert, metadatas_to_upsert, documents_to_upsert = [], [], []
        for item in valid_items_for_processing:
            aid = item['id'] 
            qs = item['questions'] 

            doc_result = source_col.get(ids=[aid], include=['documents'])
            docs_list = doc_result.get('documents', [])
            if not docs_list:
                print(f"HyPE: Warning (Batch {batch_number}): No source document found for id '{aid}' in source collection. Skipping questions for this id.")
                continue
            original_text = docs_list[0]

            for idx_q, question_text in enumerate(qs):
                generated_hype_id = f"{aid}_hq{idx_q}"
                ids_to_upsert.append(generated_hype_id)
                metadatas_to_upsert.append({'original_chunk_id': aid, 'question': question_text, 'has_embedding': False})
                documents_to_upsert.append(original_text)

        if ids_to_upsert:
            unique_generated_ids = set(ids_to_upsert)
            if len(ids_to_upsert) != len(unique_generated_ids):
                from collections import Counter 
                id_counts = Counter(ids_to_upsert)
                actual_dup_ids = [item_id for item_id, count in id_counts.items() if count > 1]
                print(f"HyPE: Error (Batch {batch_number}): Internal error - generated duplicate HyPE IDs before upsert despite filtering. Duplicates: {', '.join(actual_dup_ids[:10])}" + 
                      (f"... and {len(actual_dup_ids) - 10} more" if len(actual_dup_ids) > 10 else "") + ". Skipping upsert.")
                return 
            
            hype_col.upsert(ids=ids_to_upsert, metadatas=metadatas_to_upsert, documents=documents_to_upsert)
            print(f"HyPE: Upserted {len(ids_to_upsert)} questions for batch {batch_number}/{num_batches}.")
        else:
            print(f"HyPE: No new valid questions to upsert for batch {batch_number}/{num_batches}.")

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

def check_hype_db(path: str, dois: List[str]) -> List[dict]:
    """
    Check if the HyPE database contains the specified DOIs. And return the generated questions for those DOIs.
    """
    hype_col = get_chroma_collection(path, 'abstracts_hype', execution_mode="query")
    results = hype_col.get(include=['metadatas'])
    metadata_list = results.get('metadatas', [])
    found_ids = {md.get('original_chunk_id') for md in metadata_list}
    
    if not found_ids:
        print("HyPE: No DOIs found in the HyPE database.")
        return None
    # filter for requested DOIs
    filtered_metadata = [md for md in metadata_list if md.get('original_chunk_id') in dois]
    if not filtered_metadata:
        print("HyPE: No matching DOIs found in the HyPE database.")
        return None
    
    # return a dictionary with DOI as key and questions as value
    doi_questions_dict = {}
    for md in filtered_metadata:
        doi = md.get('original_chunk_id')
        question = md.get('question', [])
        if doi not in doi_questions_dict:
            doi_questions_dict[doi] = []
        doi_questions_dict[doi].append(question)
    
    return doi_questions_dict


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
        questions_dois = check_hype_db(db_path, ['10.1088/1748-9326/aa6b09', '10.5194/hess-26-923-2022'])
        # nicely print the questions_dois dict
        if questions_dois:
            print("HyPE: Found the following questions for the specified DOIs:")
            for doi, questions in questions_dois.items():
                print(f"DOI: {doi}")
                for question in questions:
                    print(f"  - {question}")
        else:
            print("HyPE: No questions found for the specified DOIs.")

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
