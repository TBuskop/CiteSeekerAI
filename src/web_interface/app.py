print("Loading packages and modules...")

from flask import Flask, render_template, request, redirect, url_for, jsonify, send_from_directory, session
import os
import sys
import time
import threading
import json
import glob
from datetime import datetime
from collections import OrderedDict
from markupsafe import Markup
import re # Import re for sanitization
import importlib # For reloading config
import shutil # For file operations like backup


# --- Ensure project root is in sys.path ---
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, os.pardir, os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
    

# --- Global variable for API key validation status ---
API_KEY_VALIDATION_MESSAGE = None
ENV_FILE_PATH = os.path.join(_PROJECT_ROOT, ".env") # Path to .env file
ENV_EXAMPLE_FILE_PATH = os.path.join(_PROJECT_ROOT, ".env.example") # Path to .env.example file
# CONFIG_FILE_PATH is no longer directly written to by save_api_key, but config.py itself will use _PROJECT_ROOT

# create .env file if it does not exist. copy content from .env.example
if not os.path.exists(ENV_FILE_PATH):
    if os.path.exists(ENV_EXAMPLE_FILE_PATH):
        shutil.copy2(ENV_EXAMPLE_FILE_PATH, ENV_FILE_PATH)
        print(f"Created {ENV_FILE_PATH} from {ENV_EXAMPLE_FILE_PATH}.")

# Import project modules
from src.workflows.DeepResearch_squential import run_deep_research
from src.workflows.obtain_store_abstracts import obtain_store_abstracts
from src.rag.chroma_manager import get_chroma_collection
from src.scrape.download_papers import download_dois # Import download_dois
from src.my_utils import llm_interface # Import the llm_interface module

import config

app = Flask(__name__, static_folder=os.path.join(_PROJECT_ROOT, "src", "web_interface", "static"), 
           template_folder=os.path.join(_PROJECT_ROOT, "src", "web_interface", "templates"))
app.secret_key = "citeseekai_secret_key"

# --- Register custom Jinja2 filter for basename ---
@app.template_filter('basename')
def basename_filter(path):
    return os.path.basename(path)

# Global variables
processing_jobs = {}  # Store active processing jobs
chat_history = OrderedDict()  # Store chat history
OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "output")
PAPER_DOWNLOAD_DIR = os.path.join(_PROJECT_ROOT, "data", "downloads", "full_doi_texts") # Path to downloaded PDFs

# --- Function to check API key validity on startup ---
def check_api_key_on_startup():
    """
    Checks the validity of the Gemini API key on application startup.
    Sets a global message (API_KEY_VALIDATION_MESSAGE) based on the outcome.
    """
    global API_KEY_VALIDATION_MESSAGE
    API_KEY_VALIDATION_MESSAGE = None # Reset before check
    print("Performing API key validation check...")

    try:
        llm_interface.initialize_clients()

        if llm_interface.gemini_client is None:
            API_KEY_VALIDATION_MESSAGE = (
                "WARNING: Gemini API client could not be initialized. "
                "Check `GEMINI_API_KEY` in `.env` and 'google-generativeai' installation."
            )
            print(f"API Key Check Result: {API_KEY_VALIDATION_MESSAGE}")
            return

        test_prompt = "Briefly say hello."
        print(f"Attempting test call to LLM model '{config.CHAT_MODEL}'")
        test_response = llm_interface.generate_llm_response(
            prompt=test_prompt,
            max_tokens=5,
            temperature=0.1,
            model='gemini-1.5-flash'
        )
        
        print(f"Test call response: {test_response}")

        if not test_response:
            API_KEY_VALIDATION_MESSAGE = (
                f"WARNING: Test call to model '{config.CHAT_MODEL}' yielded no response. "
                "Unable to confirm API key validity."
            )
        else:
            response_lower = test_response.lower()
            is_invalid_key_error = "clienterror" in response_lower and \
                                   ("api key not valid" in response_lower or \
                                    ("invalid_argument" in response_lower and "api key" in response_lower))
            is_auth_error = "permissiondenied" in response_lower or "authentication failed" in response_lower
            is_rate_limit_error = "rate limit" in response_lower or "429" in test_response or "resource_exhausted" in response_lower
            is_blocked_error = test_response.startswith(("[Error", "[Blocked"))
            is_empty_response = not test_response.strip()

            if is_invalid_key_error or is_auth_error:
                API_KEY_VALIDATION_MESSAGE = "ERROR: API key not valid. Please provide a valid API key."
            elif is_rate_limit_error:
                API_KEY_VALIDATION_MESSAGE = (
                    "WARNING: Gemini API rate limit may have been reached or billing is not configured. "
                    "Check Google Cloud project quotas and billing."
                )
            elif is_blocked_error:
                API_KEY_VALIDATION_MESSAGE = (
                    f"WARNING: Test call to model '{config.CHAT_MODEL}' failed or was blocked. "
                    f"Response: {test_response[:200]}... Check model access and safety settings."
                )
            elif is_empty_response:
                API_KEY_VALIDATION_MESSAGE = (
                    f"WARNING: Test call to model '{config.CHAT_MODEL}' returned an empty response. "
                    "This might indicate an issue."
                )
            else:
                 # If response is not an error and not empty, assume key is likely okay
                 print("API key validation test call was successful or did not indicate a critical key/auth issue.")
                 API_KEY_VALIDATION_MESSAGE = None # Explicitly set to None for success

    except Exception as e:
        API_KEY_VALIDATION_MESSAGE = (
            f"CRITICAL ERROR during API key validation: {str(e)}. "
            "AI features likely non-functional."
        )
        print(f"Exception during API key check: {e}")
        import traceback
        traceback.print_exc()
    
    if API_KEY_VALIDATION_MESSAGE:
        print(f"API Key Check Result: {API_KEY_VALIDATION_MESSAGE}")
    else:
        print("API Key Check: No critical issues detected with the API key or basic LLM communication.")


@app.route('/save_api_key', methods=['POST'])
def save_api_key():
    global API_KEY_VALIDATION_MESSAGE
    new_api_key = request.form.get('api_key', '').strip()

    if not new_api_key:
        return jsonify({
            "status": "error",
            "message": "API key cannot be empty.",
            "api_key_message": API_KEY_VALIDATION_MESSAGE 
        }), 400

    try:
        
        # 2. Read .env content (or prepare for new file)
        env_lines = []
        if os.path.exists(ENV_FILE_PATH): 
            with open(ENV_FILE_PATH, 'r', encoding='utf-8') as f_read:
                env_lines = f_read.readlines()

        # 3. Replace or add GEMINI_API_KEY in .env content
        new_env_lines = []
        key_found_and_updated = False
        env_key_pattern = re.compile(r"^\s*GEMINI_API_KEY\s*=\s*.*", re.IGNORECASE)
        for line in env_lines:
            if env_key_pattern.match(line):
                new_env_lines.append(f'GEMINI_API_KEY={new_api_key}\n') 
                key_found_and_updated = True
            else:
                new_env_lines.append(line)
        
        if not key_found_and_updated:
            new_env_lines.append(f'GEMINI_API_KEY={new_api_key}\n')
            print(f"GEMINI_API_KEY not found in {ENV_FILE_PATH}, appending new key.")

        # 4. Write new content to .env
        with open(ENV_FILE_PATH, 'w', encoding='utf-8') as f_write:
            f_write.writelines(new_env_lines)
        print(f"Updated GEMINI_API_KEY in {ENV_FILE_PATH}")

        # --- config.py will load from .env upon reload ---

        # 5. Reload config module (for runtime changes in the current process)
        # This will cause config.py to re-execute, loading the new key from .env
        importlib.reload(sys.modules['config']) 
        print("Reloaded config module.")
        
        # 6. Re-initialize LLM clients
        llm_interface.initialize_clients() 
        print("Re-initialized LLM clients.")
        
        # 7. Re-check API key validity
        check_api_key_on_startup() 
        print(f"Re-checked API key. New status: {API_KEY_VALIDATION_MESSAGE}")
        
        return jsonify({
            "status": "success",
            "message": "API Key updated in .env and re-validated.",
            "api_key_message": API_KEY_VALIDATION_MESSAGE
        })

    except Exception as e:
        print(f"ERROR saving API key or reloading: {str(e)}")
        import traceback
        traceback.print_exc()
        
        try:
            importlib.reload(sys.modules['config'])
            llm_interface.initialize_clients()
            check_api_key_on_startup()
        except Exception as e_reload_final:
            print(f"Error during final reload/recheck after failed save: {e_reload_final}")

        return jsonify({
            "status": "error",
            "message": f"Failed to save API key: {str(e)}",
            "api_key_message": API_KEY_VALIDATION_MESSAGE 
        }), 500


def get_timestamp():
    """Generate a timestamp for job ID"""
    return datetime.now().strftime('%Y%m%d_%H%M%S')

def process_research_question(job_id, question, subquestions_count=3, top_k_abstracts=None, top_k_chunks=None, min_citations=None):
    """Process a research question using the CiteSeekerAI pipeline"""
    try:
        job_data = processing_jobs[job_id]
        job_data.update({
            "status": "Processing",
            "progress": "Initializing deep research...",
            "progress_message": "Initializing deep research...",
            "initial_research_question": question,
            "overall_goal": None,
            "decomposed_queries": [],
            "subquery_results_stream": []
        })
        
        def update_web_progress(payload):
            if job_id not in processing_jobs:
                return

            current_job_data = processing_jobs[job_id]
            if isinstance(payload, dict):
                payload_type = payload.get("type")
                data = payload.get("data", {})

                if payload_type == "initial_info":
                    current_job_data.update({
                        "overall_goal": data.get("overall_goal"),
                        "decomposed_queries": data.get("decomposed_queries", []),
                        "initial_research_question": data.get("initial_research_question", question),
                        "progress": "Query decomposed. Starting subquery processing.",
                        "progress_message": "Query decomposed. Starting subquery processing."
                    })
                elif payload_type == "subquery_result":
                    current_job_data["subquery_results_stream"].append(data)
                    progress_msg = f"Subquery {data.get('index', -1) + 1} completed."
                    current_job_data.update({
                        "progress": progress_msg,
                        "progress_message": progress_msg
                    })
                elif payload_type == "api_rate_limit":
                    message = payload.get("message", "API rate limit reached")
                    current_job_data.update({
                        "progress": message,
                        "progress_message": message,
                        "api_error": "rate_limit",
                        "error_details": message
                    })
                elif payload_type == "status_update":
                    message = payload.get("message", "Processing...")
                    current_job_data.update({
                        "progress": message,
                        "progress_message": message
                    })
            elif isinstance(payload, str): 
                current_job_data.update({
                    "progress": payload,
                    "progress_message": payload
                })
        
        run_deep_research(
            question=question, 
            query_numbers=subquestions_count, 
            progress_callback=update_web_progress, 
            run_id=job_id,
            top_k_abstracts_val=top_k_abstracts,
            top_k_chunks_val=top_k_chunks,
            min_citations_val=min_citations
        )

        output_file = os.path.join(OUTPUT_DIR, f"combined_answers_{job_id}.txt")
        subqueries_file = os.path.join(OUTPUT_DIR, "query_specific", job_id, "subqueries.json")
        
        subqueries_list = []
        if os.path.exists(subqueries_file):
            try:
                with open(subqueries_file, 'r', encoding='utf-8') as f_sub:
                    subqueries_data = json.load(f_sub)
                    subqueries_list = subqueries_data.get('subqueries', subqueries_data.get('original_subqueries', []))
            except Exception as e_sub:
                print(f"Error reading subqueries.json for job {job_id}: {e_sub}")

        if os.path.exists(output_file):
            with open(output_file, 'r', encoding='utf-8') as f:
                answer = f.read()
            
            chat_history[job_id] = {
                "question": question,
                "answer": answer,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "file_path": output_file,
                "subqueries": subqueries_list
            }
            processing_jobs[job_id].update({"status": "Completed", "output_file": output_file})
        else:
            print(f"Error: Expected output file {output_file} not found for job {job_id}.")
            processing_jobs[job_id].update({
                "status": "Error", 
                "error": f"Output file combined_answers_{job_id}.txt not found."
            })
    
    except Exception as e:
        processing_jobs[job_id].update({"status": "Error", "error": str(e)})
        print(f"Error processing question for job {job_id}: {str(e)}")
        import traceback
        traceback.print_exc()

def find_latest_output_file():
    """Find the latest combined answers output file"""
    pattern = os.path.join(OUTPUT_DIR, "combined_answers_*.txt")
    files = glob.glob(pattern)
    if files:
        return max(files, key=os.path.getctime)
    return None

def load_chat_history():
    """Load chat history from output directory"""
    global chat_history
    chat_history = OrderedDict() 
    pattern = os.path.join(OUTPUT_DIR, "combined_answers_*.txt")
    files = glob.glob(pattern)
    
    for file_path in sorted(files, key=os.path.getctime):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            question = "Unknown Question" 
            
            match_new = re.search(r"## Original Research Question\s*\n(.*?)\n", content, re.DOTALL)
            if match_new and match_new.group(1).strip():
                question = match_new.group(1).strip()
            else:
                match_old = re.search(r"Original Research Question:\s*(.*?)\n", content)
                if match_old and match_old.group(1).strip():
                    question = match_old.group(1).strip()
            
            job_id_from_file = os.path.basename(file_path).replace("combined_answers_", "").replace(".txt", "")
            
            subqueries_list = []
            subqueries_file_path = os.path.join(OUTPUT_DIR, "query_specific", job_id_from_file, "subqueries.json")
            if os.path.exists(subqueries_file_path):
                try:
                    with open(subqueries_file_path, 'r', encoding='utf-8') as f_sub:
                        subqueries_data = json.load(f_sub)
                        subqueries_list = subqueries_data.get('subqueries', subqueries_data.get('original_subqueries', []))
                except Exception as e_sub_load:
                    print(f"Error loading subqueries for {job_id_from_file} from {subqueries_file_path}: {e_sub_load}")

            chat_history[job_id_from_file] = {
                "question": question,
                "answer": content,
                "timestamp": datetime.fromtimestamp(os.path.getctime(file_path)).strftime("%Y-%m-%d %H:%M:%S"),
                "file_path": file_path,
                "subqueries": subqueries_list
            }
        except Exception as e:
            print(f"Error loading chat history from {file_path}: {str(e)}")

@app.route('/')
def index():
    """Home page with chat interface"""
    # Sort chat_history by timestamp descending (newest first)
    sorted_history = OrderedDict(
        sorted(
            chat_history.items(),
            key=lambda item: item[1]["timestamp"],
            reverse=True
        )
    )
    return render_template('index.html', chat_history=sorted_history, api_key_message=API_KEY_VALIDATION_MESSAGE)

@app.route('/history_list')
def history_list():
    """Serves an HTML snippet of the chat history list."""
    sorted_history = OrderedDict(
        sorted(
            chat_history.items(),
            key=lambda item: item[1]["timestamp"],
            reverse=True
        )
    )
    return render_template('_history_list.html', chat_history=sorted_history)

@app.route('/ask', methods=['POST'])
def ask_question():
    """Handle new research questions"""
    question = request.form.get('question', '').strip()
    if not question:
        return jsonify({"status": "error", "message": "Question cannot be empty"}), 400

    try:
        subquestions_count = int(request.form.get('subquestions_count', '3'))
        if not (0 <= subquestions_count <= 10): 
            subquestions_count = 3 
    except ValueError:
        subquestions_count = 3

    try:
        top_k_abstracts = int(request.form.get('top_k_abstracts', config.TOP_K_ABSTRACTS))
    except (ValueError, TypeError):
        top_k_abstracts = config.TOP_K_ABSTRACTS
    
    try:
        top_k_chunks = int(request.form.get('top_k_chunks', config.DEFAULT_TOP_K))
    except (ValueError, TypeError):
        top_k_chunks = config.DEFAULT_TOP_K
    
    try: 
        min_citations = int(request.form.get('min_citations', config.MIN_CITATIONS_RELEVANT_PAPERS))
    except (ValueError, TypeError): 
        min_citations = config.MIN_CITATIONS_RELEVANT_PAPERS 
    
    job_id = get_timestamp()
    
    processing_jobs[job_id] = {
        "status": "Starting", 
        "progress": "Initializing...",
        "progress_message": "Initializing...",
        "initial_research_question": question, 
    }
    
    threading.Thread(target=process_research_question, 
                     args=(job_id, question, subquestions_count, top_k_abstracts, top_k_chunks, min_citations)).start() 
    
    return jsonify({
        "status": "success",
        "job_id": job_id,
        "message": "Your question is being processed. This may take several minutes."
    })

@app.route('/status/<job_id>')
def job_status(job_id):
    """Check the status of a processing job"""
    job_info = processing_jobs.get(job_id)
    if not job_info:
        return jsonify({"status": "not_found", "progress_message": "Job not found."}), 404
    
    response_data = {
        "status": job_info.get("status", "Unknown"),
        "progress": job_info.get("progress", ""), 
        "progress_message": job_info.get("progress_message", "Waiting for update..."),
        "initial_research_question": job_info.get("initial_research_question"),
        "overall_goal": job_info.get("overall_goal"),
        "decomposed_queries": job_info.get("decomposed_queries", []),
        "subquery_results_stream": job_info.get("subquery_results_stream", []),
        "error": job_info.get("error"),
        "api_error": job_info.get("api_error"),
        "error_details": job_info.get("error_details"),
        "file_path": job_info.get("file_path"),
        "count": job_info.get("count"),
        "message": job_info.get("message") 
    }
    return jsonify(response_data)

@app.route('/result/<job_id>')
def get_result(job_id):
    """Get the result of a completed job"""
    history_item = chat_history.get(job_id)
    if not history_item:
        return jsonify({"status": "not_found", "message": "Result not found or job not completed."}), 404

    subqueries_list = history_item.get("subqueries", [])
    if not subqueries_list: 
        subqueries_file_path = os.path.join(OUTPUT_DIR, "query_specific", job_id, "subqueries.json")
        if os.path.exists(subqueries_file_path):
            try:
                with open(subqueries_file_path, 'r', encoding='utf-8') as f_sub:
                    subqueries_data = json.load(f_sub)
                    subqueries_list = subqueries_data.get('subqueries', subqueries_data.get('original_subqueries', []))
                    history_item["subqueries"] = subqueries_list 
            except Exception as e_sub_load:
                print(f"Error reading subqueries.json for job {job_id} during get_result: {e_sub_load}")
        
    return jsonify({
        "status": "success",
        "question": history_item["question"],
        "answer": history_item["answer"],
        "timestamp": history_item["timestamp"],
        "subqueries": subqueries_list
    })

@app.route('/output/<filename>')
def output_file(filename):
    """Serve output files"""
    return send_from_directory(OUTPUT_DIR, filename)

@app.route('/abstracts')
def abstract_search_page():
    """Abstract collection page"""
    return render_template('abstracts.html')

@app.route('/abstracts/search', methods=['POST'])
def start_abstract_search():
    """Handle new abstract collection requests"""
    query = request.form.get('query', '').strip()
    if not query:
        return jsonify({"status": "error", "message": "Search query cannot be empty"}), 400

    scopus_search_scope = request.form.get('scopus_search_scope', config.SCOPUS_SEARCH_SCOPE)
    
    try:
        year_from_str = request.form.get('year_from', '').strip()
        year_from = int(year_from_str) if year_from_str else None
    except ValueError:
        return jsonify({"status": "error", "message": "Invalid 'Year From'"}), 400
        
    try:
        year_to_str = request.form.get('year_to', '').strip()
        year_to = int(year_to_str) if year_to_str else None
    except ValueError:
        return jsonify({"status": "error", "message": "Invalid 'Year To'"}), 400

    try:
        min_citations_str = request.form.get('min_citations', '').strip()
        min_citations = int(min_citations_str) if min_citations_str else None
    except ValueError:
        return jsonify({"status": "error", "message": "Invalid 'Min Citations'"}), 400

    job_id = get_timestamp()
    
    processing_jobs[job_id] = {
        "status": "Starting", 
        "progress": "Initializing abstract collection...",
        "original_params": { 
            "query": query,
            "scopus_search_scope": scopus_search_scope,
            "year_from": year_from,
            "year_to": year_to,
            "min_citations": min_citations
        }
    }
    
    threading.Thread(target=process_abstract_search, 
                     args=(job_id, query, scopus_search_scope, year_from, year_to, min_citations)).start() 
    
    return jsonify({
        "status": "success",
        "job_id": job_id,
        "message": "Your abstract collection is being processed."
    })

@app.route('/abstracts/status/<job_id>')
def abstract_job_status(job_id):
    """Check the status of an abstract collection job"""
    return job_status(job_id)

@app.route('/abstracts/continue_collection/<job_id>', methods=['POST'])
def continue_abstract_collection(job_id):
    """Continue abstract collection after user confirms large result set"""
    if job_id not in processing_jobs:
        return jsonify({"status": "error", "message": "Job not found"})
    
    if processing_jobs[job_id].get("status") != "AwaitingConfirmation":
        return jsonify({"status": "error", "message": "Job is not awaiting confirmation"})
    
    # Get the original parameters
    original_params = processing_jobs[job_id].get("original_params", {})
    
    if not original_params:
        return jsonify({"status": "error", "message": "Original parameters not found"})
    
    # Update job status
    processing_jobs[job_id]["status"] = "Processing"
    processing_jobs[job_id]["progress"] = "Continuing with search after confirmation..."
    
    # Start a new thread to continue processing with force_continue_large_search=True
    threading.Thread(
        target=process_abstract_search_with_force,
        args=(
            job_id,
            original_params.get("query"),
            original_params.get("scopus_search_scope"),
            original_params.get("year_from"),
            original_params.get("year_to"),
            original_params.get("min_citations")
        )
    ).start()
    
    return jsonify({
        "status": "success",
        "message": "Continuing with collection",
        "job_id": job_id
    })
    
def process_abstract_search_with_force(job_id, query, scopus_search_scope, year_from=None, year_to=None, min_citations=None):
    """Process an abstract collection using obtain_store_abstracts with force_continue_large_search=True"""
    try:
        current_job_data = processing_jobs[job_id]
        
        def update_web_progress(message):
            if job_id in processing_jobs: 
                processing_jobs[job_id]["progress"] = message
                
        update_web_progress("Continuing with large result set collection...")
        
        result = obtain_store_abstracts(
            search_query=query,
            scopus_search_scope=scopus_search_scope,
            year_from=year_from,
            year_to=year_to,
            min_citations_param=min_citations,
            progress_callback=update_web_progress,
            force_continue_large_search=True 
        )
        
        if "status" in result:
            if result["status"] == "SUCCESS":
                current_job_data.update({
                    "status": "Completed",
                    "progress": f"Abstract collection completed! Found {result.get('count', 'N/A')} abstracts.",
                    "file_path": result.get("file_path"),
                    "count": result.get("count")
                })
            else: 
                current_job_data.update({
                    "status": "Error",
                    "error": result.get("message", "Unknown error during forced collection."),
                    "progress": result.get("message", "Error in abstract collection")
                })
        else: 
            current_job_data.update({
                "status": "Error",
                "error": "Unexpected result from abstract collection process.",
                "progress": "Error: Unexpected result."
            })
                
    except Exception as e:
        if job_id in processing_jobs:
            processing_jobs[job_id].update({
                "status": "Error",
                "error": str(e),
                "progress": f"Error: {str(e)}"
            })
        print(f"Error in forced abstract collection job {job_id}: {e}")
        import traceback
        traceback.print_exc()


def process_single_paper_download(job_id: str, doi: str):
    """Downloads a single paper using its DOI."""
    try:
        processing_jobs[job_id] = {
            "status": "Processing",
            "progress": f"Starting download for DOI: {doi}",
            "doi": doi
        }
        print(f"Background job {job_id}: Starting download for DOI {doi} to {PAPER_DOWNLOAD_DIR}")

        # Ensure the output directory for download_dois exists
        os.makedirs(PAPER_DOWNLOAD_DIR, exist_ok=True)

        # download_dois expects a list of DOIs and the output directory
        download_dois([doi], PAPER_DOWNLOAD_DIR) # download_dois handles its own threading for multiple DOIs, for one it's direct.

        # Verify download by checking file existence
        sanitized_doi_filename = sanitize_doi_for_filename(doi) + ".txt"
        file_path = os.path.join(PAPER_DOWNLOAD_DIR, sanitized_doi_filename)

        if os.path.exists(file_path):
            processing_jobs[job_id]["status"] = "Completed"
            processing_jobs[job_id]["progress"] = f"Successfully downloaded DOI: {doi}"
            print(f"Background job {job_id}: Successfully downloaded {file_path}")
        else:
            processing_jobs[job_id]["status"] = "Error"
            processing_jobs[job_id]["error"] = f"File not found after download attempt for DOI: {doi}"
            print(f"Background job {job_id}: Error - File not found for {doi} at {file_path}")

    except Exception as e:
        processing_jobs[job_id]["status"] = "Error"
        processing_jobs[job_id]["error"] = f"Failed to download DOI {doi}: {str(e)}"
        print(f"Background job {job_id}: Exception during download for DOI {doi} - {str(e)}")
        import traceback
        traceback.print_exc()


@app.route('/abstracts/download_paper', methods=['POST'])
def download_paper_route():
    """Initiates the download of a single paper by DOI."""
    doi = request.form.get('doi', '').strip()
    if not doi:
        return jsonify({"status": "error", "message": "DOI cannot be empty"}), 400

    safe_doi_part = sanitize_doi_for_filename(doi)[:30] 
    job_id = f"{get_timestamp()}_dl_{safe_doi_part}"
    
    processing_jobs[job_id] = {
        "status": "Starting",
        "progress": f"Initializing download for DOI: {doi}",
        "doi": doi,
        "type": "single_download" 
    }
    
    thread = threading.Thread(target=process_single_paper_download, args=(job_id, doi))
    thread.start()
    
    return jsonify({
        "status": "success",
        "job_id": job_id,
        "message": f"Download initiated for DOI: {doi}. Check status for progress."
    })

def process_multiple_papers_download(job_id: str, dois: list):
    """Downloads multiple papers using their DOIs."""
    total_dois = len(dois)
    processing_jobs[job_id] = {
        "status": "Processing",
        "progress": f"Downloading {total_dois} paper(s)...",
        "total_dois": total_dois,
        "downloaded_count": 0
    }
    print(f"Background job {job_id}: Starting download for {total_dois} DOIs to {PAPER_DOWNLOAD_DIR}")

    try:
        os.makedirs(PAPER_DOWNLOAD_DIR, exist_ok=True)
        
        # download_dois will handle the list.
        # If download_dois had a progress callback, we could use it here.
        # For now, we assume it processes all or fails.
        download_dois(dois, PAPER_DOWNLOAD_DIR)

        # Verification can be tricky for batch. We'll assume success if no exceptions.
        # A more robust check would iterate through DOIs and check individual files.
        # For simplicity, we'll mark as completed. The UI will refresh and show individual statuses.
        processing_jobs[job_id]["status"] = "Completed"
        processing_jobs[job_id]["progress"] = f"Finished download process for {total_dois} paper(s)."
        processing_jobs[job_id]["downloaded_count"] = total_dois # Assume all attempted
        print(f"Background job {job_id}: Finished download process for {total_dois} DOIs.")

    except Exception as e:
        processing_jobs[job_id]["status"] = "Error"
        processing_jobs[job_id]["error"] = f"Failed during batch download: {str(e)}"
        print(f"Background job {job_id}: Exception during batch download - {str(e)}")
        import traceback
        traceback.print_exc()

@app.route('/abstracts/download_multiple_papers', methods=['POST'])
def download_multiple_papers_route():
    """Initiates the download of multiple papers by a list of DOIs."""
    data = request.get_json()
    dois = data.get('dois', [])

    if not dois or not isinstance(dois, list) or not all(isinstance(d, str) for d in dois):
        return jsonify({"status": "error", "message": "A non-empty list of DOI strings is required"}), 400
    
    job_id = f"{get_timestamp()}_dl_batch_{len(dois)}"
    
    processing_jobs[job_id] = {
        "status": "Starting",
        "progress": f"Initializing batch download for {len(dois)} paper(s)...",
        "total_dois": len(dois),
        "downloaded_count": 0,
        "type": "batch_download" 
    }
    
    thread = threading.Thread(target=process_multiple_papers_download, args=(job_id, dois))
    thread.start()
    
    return jsonify({
        "status": "success",
        "job_id": job_id,
        "message": f"Batch download initiated for {len(dois)} paper(s). Check status for progress."
    })

def process_abstract_search(job_id, query, scopus_search_scope, year_from=None, year_to=None, min_citations=None):
    """Process an abstract collection using the obtain_store_abstracts function"""
    print(f"DEBUG: Starting process_abstract_search for job {job_id}, query: '{query}', scope: {scopus_search_scope}, years: {year_from}-{year_to}, min_citations: {min_citations}")
    
    current_job_data = processing_jobs.get(job_id)
    if not current_job_data:
        print(f"Error: Job {job_id} not found in processing_jobs at start of process_abstract_search.")
        return

    try:
        current_job_data["status"] = "Processing"
        
        def update_web_progress(message):
            if job_id in processing_jobs: 
                processing_jobs[job_id]["progress"] = message
        
        update_web_progress("Initializing abstract collection...")
        
        result = obtain_store_abstracts(
            search_query=query,
            scopus_search_scope=scopus_search_scope,
            year_from=year_from,
            year_to=year_to,
            min_citations_param=min_citations,
            progress_callback=update_web_progress,
            force_continue_large_search=False 
        )

        print(f"DEBUG: Result from obtain_store_abstracts for job {job_id}: {result}")

        if "status" in result:
            result_status = result["status"]
            result_message = result.get("message", "No message provided.")
            result_count = result.get("count")
            
            current_job_data["progress"] = result_message 

            if result_status == "AWAITING_USER_CONFIRMATION_LARGE_RESULTS":
                current_job_data.update({
                    "status": "AwaitingConfirmation",
                    "message": result_message, 
                    "count": result_count,
                })
                print(f"DEBUG: Job {job_id} set to AwaitingConfirmation. Count: {result_count}")
            elif result_status == "SUCCESS":
                current_job_data.update({
                    "status": "Completed",
                    "file_path": result.get("file_path"),
                    "count": result_count,
                    "progress": f"Abstract collection completed! Found {result_count or 'N/A'} abstracts."
                })
            elif result_status.startswith("ERROR_"): 
                current_job_data.update({
                    "status": "Error",
                    "error": result_message,
                    "count": result_count 
                })
            else: 
                current_job_data.update({
                    "status": "Error",
                    "error": f"Unknown status from abstract collection: {result_status}",
                    "progress": f"Unknown status: {result_status}. Message: {result_message}"
                })
        else: 
            current_job_data.update({
                "status": "Error",
                "error": "Invalid response structure from abstract collection process.",
                "progress": "Error: Invalid response from collection process."
            })
            
    except Exception as e:
        print(f"CRITICAL ERROR in process_abstract_search for job {job_id}: {e}")
        import traceback
        traceback.print_exc()
        if job_id in processing_jobs: 
            processing_jobs[job_id].update({
                "status": "Error",
                "error": str(e),
                "progress": f"Critical error: {str(e)}"
            })

@app.route('/static/<path:path>')
def serve_static(path):
    """Serve static files"""
    return send_from_directory(app.static_folder, path)

@app.route('/get_prompt_chunk')
def get_prompt_chunk():
    job_id = request.args.get('job_id')
    citation_key = request.args.get('citation_key')

    if not job_id or not citation_key:
        return jsonify({"status": "error", "message": "Missing job_id or citation_key"}), 400

    prompt_dir = os.path.join(OUTPUT_DIR, "query_specific", job_id)
    if not os.path.isdir(prompt_dir):
        return jsonify({"status": "error", "message": "Invalid job_id or prompt directory not found"}), 400

    # Search each final_prompt file for the citation key
    for file_path in glob.glob(os.path.join(prompt_dir, "final_prompt_*.txt")):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception as e:
            print(f"Error reading prompt file {file_path}: {e}")
            continue # Try next file

        for idx, line in enumerate(lines):
            if line.strip() == citation_key:
                # Found the citation key, now find "Content:"
                i = idx + 1
                while i < len(lines) and not lines[i].strip().startswith("Content:"):
                    i += 1
                
                if i < len(lines) and lines[i].strip().startswith("Content:"):
                    i += 1 # Move past "Content:" line to start of actual content
                    content_lines = []
                    # Collect content until the next '---' separator
                    while i < len(lines) and not lines[i].strip().startswith("---"):
                        content_lines.append(lines[i])
                        i += 1
                    chunk_text = "".join(content_lines).strip()
                    if chunk_text: # Ensure chunk is not empty
                        return jsonify({"status": "success", "chunk": chunk_text})
                    # If chunk_text is empty, it means Content: was followed immediately by --- or EOF.
                    # Continue searching in case of multiple matches or other files.
                # If "Content:" not found or content is empty, this match is invalid.
                # Continue searching in the rest of the file or other files.
    
    return jsonify({"status": "error", "message": "Chunk not found for the given citation key"}), 404

# Helper function to create a sort key (primary: author, secondary: year desc, tertiary: title asc)
def get_abstract_sort_key(abstract_metadata_dict):
    authors_string = abstract_metadata_dict.get('authors', '')
    year_val = abstract_metadata_dict.get('year', '')
    title = abstract_metadata_dict.get('title', '').lower()

    first_author_lastname = 'zzzz'  # Default for no authors or errors, sorts them last
    if authors_string and isinstance(authors_string, str):
        # Handle case where authors_string already contains "et al."
        if ' et al.' in authors_string:
            first_author_lastname = authors_string.split(' et al.')[0].strip().lower()
        else:
            # Split authors by comma if multiple authors exist
            authors_list = authors_string.split(', ')
            if authors_list and authors_list[0].strip(): # Ensure first author is not empty
                first_author = authors_list[0].strip()
                # Extract last name from the first author (everything before the first space)
                name_parts = first_author.split(' ')
                if name_parts and name_parts[0].strip():
                    first_author_lastname = name_parts[0].lower()
    
    try:
        # Ensure year_val is treated as a string before int conversion if it might be numeric type already
        year_int = int(str(year_val))
    except (ValueError, TypeError):
        year_int = 0 # Default for non-integer or missing years

    return (first_author_lastname, -year_int, title)

@app.route('/abstracts/list', methods=['GET'])
def list_abstracts():
    """API endpoint to list abstracts in the database with server-side sorting and filtering."""
    try:
        search_term = request.args.get('search', '').strip().lower()
        search_fields_str = request.args.get('search_fields', 'title,authors')
        search_fields = [field.strip() for field in search_fields_str.split(',') if field.strip()]
        
        try:
            page = int(request.args.get('page', 1))
            per_page = int(request.args.get('per_page', 20))
            if page < 1: page = 1
            if per_page < 1: per_page = 1
            if per_page > 100: per_page = 100 
        except ValueError:
            page, per_page = 1, 20

        collection = get_chroma_collection(
            db_path=os.path.join(_PROJECT_ROOT, "data", "databases", "abstract_chroma_db"),
            collection_name="abstracts",
            execution_mode="query" 
        )

        all_db_items = collection.get(include=["metadatas"])
        
        filtered_items_metadata = []
        if all_db_items and all_db_items.get('ids'):
            for i, doc_id in enumerate(all_db_items['ids']):
                metadata = all_db_items['metadatas'][i]
                item_data_for_filter = {
                    'id': doc_id, 
                    'authors': metadata.get('authors', ''),
                    'year': metadata.get('year', ''),
                    'title': metadata.get('title', '')
                }

                if search_term:
                    matches_search = False
                    if 'title' in search_fields and item_data_for_filter.get('title', '').lower().find(search_term) != -1:
                        matches_search = True
                    if not matches_search and 'authors' in search_fields and item_data_for_filter.get('authors', '').lower().find(search_term) != -1:
                        matches_search = True
                    
                    if matches_search:
                        filtered_items_metadata.append(item_data_for_filter)
                else:
                    filtered_items_metadata.append(item_data_for_filter)
        
        filtered_items_metadata.sort(key=get_abstract_sort_key)
        
        total_filtered_items = len(filtered_items_metadata)

        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        
        ids_for_current_page = [item['id'] for item in filtered_items_metadata[start_idx:end_idx]]
        
        paginated_abstracts_full_data = []
        if ids_for_current_page:
            page_full_data_results = collection.get(ids=ids_for_current_page, include=["metadatas", "documents"])
            
            id_to_full_data_map = {}
            if page_full_data_results.get('ids'):
                for i, doc_id in enumerate(page_full_data_results['ids']):
                    id_to_full_data_map[doc_id] = {
                        'metadata': page_full_data_results['metadatas'][i],
                        'document': page_full_data_results['documents'][i]
                    }

            for item_id in ids_for_current_page: 
                if item_id in id_to_full_data_map:
                    full_data = id_to_full_data_map[item_id]
                    metadata = full_data['metadata']
                    
                    is_downloaded = False
                    doi = metadata.get('doi', '')
                    if doi:
                        sanitized_doi_filename = sanitize_doi_for_filename(doi) + ".txt"
                        txt_path = os.path.join(PAPER_DOWNLOAD_DIR, sanitized_doi_filename)
                        is_downloaded = os.path.exists(txt_path)
                            
                    paginated_abstracts_full_data.append({
                        'id': item_id,
                        'title': metadata.get('title', ''),
                        'authors': metadata.get('authors', ''),
                        'year': metadata.get('year', ''),
                        'source_title': metadata.get('source_title', ''),
                        'cited_by': metadata.get('cited_by', ''),
                        'doi': doi,
                        'document': full_data['document'], 
                        'is_downloaded': is_downloaded
                    })
        
        has_more_pages = end_idx < total_filtered_items
        
        return jsonify({
            'status': 'success',
            'abstracts': paginated_abstracts_full_data,
            'page': page,
            'per_page': per_page,
            'total': total_filtered_items, 
            'has_more': has_more_pages
        })
        
    except Exception as e:
        app.logger.error(f"Error listing abstracts: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': f"Failed to retrieve abstracts: {str(e)}"
        }), 500

def sanitize_doi_for_filename(doi_str):
    """Sanitizes a DOI string to be filesystem-friendly for filenames."""
    if not doi_str:
        return ""
    # Replace characters not suitable for filenames (non-alphanumeric, '.', '-') with underscores.
    # This matches the sanitization used in download_papers.py.
    filename = re.sub(r'[^\w\d.-]', '_', doi_str)
    return filename

if __name__ == '__main__':

    # change run path to current directory
    os.chdir(_PROJECT_ROOT)
    print("Starting CiteSeekerAI Web Interface...")
    print(f"Python path: {sys.path}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Project root: {_PROJECT_ROOT}")
    
    # Try to import one of our project modules to test the Python path
    try:
        import config
        print(f"Successfully imported config module")
    except ImportError as e:
        print(f"Error importing config module: {e}")
        print("Please ensure the project root is in your Python path.")
        print("You can set it manually with: set PYTHONPATH=%CD%")
        sys.exit(1)
    
    # Load existing chat history

    print("Loading chat history...")
    load_chat_history()

    # Perform API key check at startup
    check_api_key_on_startup()

    # Start Flask server
    print("Starting Flask server...")
    print("Access the web interface at http://127.0.0.1:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
