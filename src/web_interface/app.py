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

# --- Ensure project root is in sys.path ---
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, os.pardir, os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
    

# --- Add project root to sys.path ---
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, os.pardir, os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Import project modules
from src.workflows.DeepResearch_squential import run_deep_research
from src.workflows.obtain_store_abstracts import obtain_store_abstracts
from src.rag.chroma_manager import get_chroma_collection

import config

print("Loading packages and modules...")


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


def get_timestamp():
    """Generate a timestamp for job ID"""
    return datetime.now().strftime('%Y%m%d_%H%M%S')

def process_research_question(job_id, question, subquestions_count=3):
    """Process a research question using the CiteSeekerAI pipeline"""
    try:
        # Initialize job structure for structured progress
        processing_jobs[job_id].update({
            "status": "Processing",
            "progress": "Initializing deep research...",
            "progress_message": "Initializing deep research...",
            "initial_research_question": question,
            "overall_goal": None,
            "decomposed_queries": [],
            "subquery_results_stream": [] # To store individual subquery results as they come
        })
        
        # Callback to send structured progress updates to web UI
        def update_web_progress(payload):
            if job_id in processing_jobs:
                if isinstance(payload, dict):
                    payload_type = payload.get("type")
                    
                    if payload_type == "initial_info":
                        data = payload.get("data", {})
                        processing_jobs[job_id]["overall_goal"] = data.get("overall_goal")
                        processing_jobs[job_id]["decomposed_queries"] = data.get("decomposed_queries", [])
                        processing_jobs[job_id]["initial_research_question"] = data.get("initial_research_question", question)
                        processing_jobs[job_id]["progress"] = "Query decomposed. Starting subquery processing."
                        processing_jobs[job_id]["progress_message"] = "Query decomposed. Starting subquery processing."
                    
                    elif payload_type == "subquery_result":
                        data = payload.get("data", {})
                        processing_jobs[job_id]["subquery_results_stream"].append(data)
                        processing_jobs[job_id]["progress"] = f"Subquery {data.get('index', -1) + 1} completed."
                        processing_jobs[job_id]["progress_message"] = f"Subquery {data.get('index', -1) + 1} completed."
                    
                    elif payload_type == "status_update":
                        message = payload.get("message", "Processing...")
                        processing_jobs[job_id]["progress"] = message
                        processing_jobs[job_id]["progress_message"] = message
                
                elif isinstance(payload, str):
                    # Backward compatibility for simple string messages
                    processing_jobs[job_id]["progress"] = payload
                    processing_jobs[job_id]["progress_message"] = payload
        
        # Run deep research with callback and pass the job_id as run_id
        run_deep_research(question, subquestions_count, progress_callback=update_web_progress, run_id=job_id)

        # Find the expected output file directly using job_id instead of finding latest
        output_file = os.path.join(OUTPUT_DIR, f"combined_answers_{job_id}.txt")
        subqueries_file = os.path.join(OUTPUT_DIR, "query_specific", job_id, "subqueries.json")
        
        subqueries_list = []
        if os.path.exists(subqueries_file):
            try:
                with open(subqueries_file, 'r', encoding='utf-8') as f_sub:
                    subqueries_data = json.load(f_sub)
                    # Prefer 'subqueries' key, fallback to 'original_subqueries'
                    if 'subqueries' in subqueries_data and isinstance(subqueries_data['subqueries'], list):
                        subqueries_list = subqueries_data['subqueries']
                    elif 'original_subqueries' in subqueries_data and isinstance(subqueries_data['original_subqueries'], list):
                        subqueries_list = subqueries_data['original_subqueries']
            except Exception as e_sub:
                print(f"Error reading or parsing subqueries.json for job {job_id}: {e_sub}")

        if os.path.exists(output_file):
            with open(output_file, 'r', encoding='utf-8') as f:
                answer = f.read()
            
            # Store in chat history
            chat_history[job_id] = {
                "question": question,
                "answer": answer,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "file_path": output_file,
                "subqueries": subqueries_list
            }
            
            # Update job status
            processing_jobs[job_id]["status"] = "Completed"
            processing_jobs[job_id]["output_file"] = output_file
        else:
            # Fall back to finding the latest file if the expected file doesn't exist
            # This fallback might not have associated subqueries easily, handle gracefully
            print(f"Warning: Expected output file combined_answers_{job_id}.txt not found. Attempting fallback.")
            output_file = find_latest_output_file() # This is a generic fallback
            if output_file:
                with open(output_file, 'r', encoding='utf-8') as f:
                    answer = f.read()
                
                # For fallback, subqueries might be harder to associate if job_id doesn't match file's implicit ID
                # For simplicity, we might not have subqueries here or try to infer if possible
                # For now, it will default to an empty list if subqueries_file for original job_id wasn't found
                chat_history[job_id] = {
                    "question": question,
                    "answer": answer,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "file_path": output_file,
                    "subqueries": subqueries_list # This would be from the original job_id context
                }
                
                processing_jobs[job_id]["status"] = "Completed"
                processing_jobs[job_id]["output_file"] = output_file
                print(f"Warning: Used fallback output file {os.path.basename(output_file)} for job {job_id}")
            else:
                processing_jobs[job_id]["status"] = "Error"
                processing_jobs[job_id]["error"] = "Output file not found"
    
    except Exception as e:
        processing_jobs[job_id]["status"] = "Error"
        processing_jobs[job_id]["error"] = str(e)
        print(f"Error processing question: {str(e)}")
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
    pattern = os.path.join(OUTPUT_DIR, "combined_answers_*.txt")
    files = glob.glob(pattern)
    
    for file_path in sorted(files, key=os.path.getctime):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Extract original question
            question = "Unknown Question"  # Default value
            # Try new format: "## Original Research Question\n{question}\n..."
            marker_new = "## Original Research Question"
            idx_marker_new = content.find(marker_new)

            if idx_marker_new != -1:
                # Calculate start of the content after the marker text itself
                start_of_question_part = idx_marker_new + len(marker_new)
                # Get the rest of the content from this point
                question_section = content[start_of_question_part:]
                
                # The question is expected on the next line.
                # Strip leading whitespace (like the \n immediately after the marker)
                question_section_stripped = question_section.lstrip()
                
                # Take the first line from this stripped section as the question
                question_lines = question_section_stripped.split('\n', 1)
                if question_lines and question_lines[0].strip(): # Check if the line is not empty
                    question = question_lines[0].strip()
            else:
                # Try old format: "Original Research Question: {question}\n"
                marker_old = "Original Research Question:"
                # Split content by the old marker
                question_match_parts = content.split(marker_old, 1)
                if len(question_match_parts) > 1:
                    # The question is in the part after the marker
                    # Strip leading spaces from this part, then take the first line
                    first_line_after_marker = question_match_parts[1].lstrip().split("\n", 1)[0]
                    question = first_line_after_marker.strip()
                
            job_id = os.path.basename(file_path).replace("combined_answers_", "").replace(".txt", "")
            
            chat_history[job_id] = {
                "question": question,
                "answer": content,
                "timestamp": datetime.fromtimestamp(os.path.getctime(file_path)).strftime("%Y-%m-%d %H:%M:%S"),
                "file_path": file_path
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
    return render_template('index.html', chat_history=sorted_history)

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
    subquestions_count = request.form.get('subquestions_count', '3')
    
    try:
        subquestions_count = int(subquestions_count)
        if subquestions_count < 0 or subquestions_count > 10:
            subquestions_count = 3  # Default to 3 if out of range
    except ValueError:
        subquestions_count = 3  # Default to 3 if not a valid integer
    
    if not question:
        return jsonify({"status": "error", "message": "Question cannot be empty"})
    
    # Create job ID
    job_id = get_timestamp()
    
    # Start processing in background
    processing_jobs[job_id] = {
        "status": "Starting", 
        "progress": "Initializing...",
        "progress_message": "Initializing...",
        "initial_research_question": question,
        "overall_goal": None,
        "decomposed_queries": [],
        "subquery_results_stream": []
    }
    
    threading.Thread(target=process_research_question, args=(job_id, question, subquestions_count)).start()
    
    return jsonify({
        "status": "success",
        "job_id": job_id,
        "message": "Your question is being processed. This may take several minutes."
    })

@app.route('/status/<job_id>')
def job_status(job_id):
    """Check the status of a processing job"""
    if job_id not in processing_jobs:
        return jsonify({"status": "not_found", "progress_message": "Job not found."})
    
    # Ensure all relevant keys are present in the response
    job_data = processing_jobs[job_id]
    response_data = {
        "status": job_data.get("status", "Unknown"),
        "progress": job_data.get("progress", ""),  # Keep for backward compatibility
        "progress_message": job_data.get("progress_message", ""),
        "initial_research_question": job_data.get("initial_research_question"),
        "overall_goal": job_data.get("overall_goal"),
        "decomposed_queries": job_data.get("decomposed_queries", []),
        "subquery_results_stream": job_data.get("subquery_results_stream", []),
        "error": job_data.get("error")  # Include error if present
    }
    
    return jsonify(response_data)

@app.route('/result/<job_id>')
def get_result(job_id):
    """Get the result of a completed job"""
    if job_id in chat_history:
        history_item = chat_history[job_id]
        subqueries_list = history_item.get("subqueries", [])

        # If subqueries are not in chat_history (e.g. older history items before this change),
        # try to load them from the subqueries.json file directly.
        if not subqueries_list:
            subqueries_file = os.path.join(OUTPUT_DIR, "query_specific", job_id, "subqueries.json")
            if os.path.exists(subqueries_file):
                try:
                    with open(subqueries_file, 'r', encoding='utf-8') as f_sub:
                        subqueries_data = json.load(f_sub)
                        if 'subqueries' in subqueries_data and isinstance(subqueries_data['subqueries'], list):
                            subqueries_list = subqueries_data['subqueries']
                        elif 'original_subqueries' in subqueries_data and isinstance(subqueries_data['original_subqueries'], list):
                             subqueries_list = subqueries_data['original_subqueries']
                        # Update chat_history entry if we just loaded them
                        chat_history[job_id]["subqueries"] = subqueries_list
                except Exception as e_sub:
                    print(f"Error reading or parsing subqueries.json for job {job_id} during get_result: {e_sub}")
        
        return jsonify({
            "status": "success",
            "question": history_item["question"],
            "answer": history_item["answer"],
            "timestamp": history_item["timestamp"],
            "subqueries": subqueries_list
        })
    
    return jsonify({"status": "not_found"})

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
    scopus_search_scope = request.form.get('scopus_search_scope', config.SCOPUS_SEARCH_SCOPE) # Added, with fallback to config
    
    if not query:
        return jsonify({"status": "error", "message": "Search query cannot be empty"})
    
    # Create job ID
    job_id = get_timestamp()
    
    # Start processing in background
    processing_jobs[job_id] = {"status": "Starting", "progress": "Initializing..."}
    threading.Thread(target=process_abstract_search, args=(job_id, query, scopus_search_scope)).start() # Pass scope
    
    return jsonify({
        "status": "success",
        "job_id": job_id,
        "message": "Your abstract collection is being processed."
    })

@app.route('/abstracts/status/<job_id>')
def abstract_job_status(job_id):
    """Check the status of an abstract collection job"""
    if job_id not in processing_jobs:
        return jsonify({"status": "not_found"})
    
    return jsonify(processing_jobs[job_id])

def process_abstract_search(job_id, query, scopus_search_scope): # Added scopus_search_scope parameter
    """Process an abstract collection using the obtain_store_abstracts function"""
    try:
        # Update job status
        processing_jobs[job_id]["status"] = "Processing"
        # Callback for UI progress updates
        def update_web_progress(message):
            if job_id in processing_jobs:
                processing_jobs[job_id]["progress"] = message
        update_web_progress("Initializing abstract collection...")
        # Temporarily override the SCOPUS_SEARCH_STRING in config
        original_scopus_search_string = config.SCOPUS_SEARCH_STRING
        # The query from the form is now the primary search string for obtain_store_abstracts
        # config.SCOPUS_SEARCH_STRING = query # This line might be redundant if query is passed directly

        # Run with callback to update progress
        obtain_store_abstracts(search_query=query, # Pass query from form
                               scopus_search_scope=scopus_search_scope, # Pass selected scope
                               progress_callback=update_web_progress)

        # Restore original config if it was changed (though direct passing is preferred)
        # config.SCOPUS_SEARCH_STRING = original_scopus_search_string

        # Finalize job status
        processing_jobs[job_id]["status"] = "Completed"
        update_web_progress("Abstract collection completed!")
        # Add more details about the results
        csv_dir = os.path.join(_PROJECT_ROOT, 'data', 'downloads', 'csv')
        latest_csv = max(glob.glob(os.path.join(csv_dir, "*.csv")), key=os.path.getctime, default=None)
        if latest_csv:
            processing_jobs[job_id]["file_path"] = latest_csv
            # Try to count rows in the CSV
            try:
                with open(latest_csv, 'r', encoding='utf-8') as f:
                    count = sum(1 for _ in f) - 1 # -1 for header
                processing_jobs[job_id]["count"] = count
                update_web_progress(f"Abstract collection completed! Found {count} abstracts. File: {os.path.basename(latest_csv)}")
            except Exception as e_count:
                print(f"Could not count rows in {latest_csv}: {e_count}")
                update_web_progress(f"Abstract collection completed! File: {os.path.basename(latest_csv)}")
                pass
        else:
            update_web_progress("Abstract collection completed! No CSV file found to report details.")
            
    except Exception as e:
        processing_jobs[job_id]["status"] = "Error"
        processing_jobs[job_id]["error"] = str(e)
        print(f"Error in abstract collection job {job_id}: {e}")
        import traceback; traceback.print_exc()

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
    year_val = abstract_metadata_dict.get('year', '0') # Default to '0' string
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
    """API endpoint to list abstracts in the database"""
    try:
        search_term = request.args.get('search', '')
        search_fields = request.args.get('search_fields', 'title,authors').split(',')
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 20))
        
        collection = get_chroma_collection(
            db_path=os.path.join(_PROJECT_ROOT, "data", "databases", "abstract_chroma_db"),
            collection_name="abstracts",
            execution_mode="query" 
        )

        total_db_count = collection.count()
        sortable_items = [] # This list will hold items to be sorted.

        # Fetch all IDs and metadatas needed for sorting and filtering.
        # This ensures we operate on the entire dataset for these operations.
        # We only fetch 'metadatas' here to keep it lighter; 'documents' are fetched for the page later.
        all_metadata_results = collection.get(
            limit=total_db_count, # Fetch all
            include=["metadatas"] 
        )

        if all_metadata_results.get('ids'):
            for i, doc_id in enumerate(all_metadata_results['ids']):
                metadata = all_metadata_results['metadatas'][i]
                
                item_data_for_sort_and_filter = {
                    'id': doc_id,
                    'authors': metadata.get('authors', ''),
                    'year': metadata.get('year', ''),
                    'title': metadata.get('title', '')
                }

                if search_term:
                    # Apply search filter if a search term is provided
                    search_term_lower = search_term.lower()
                    matches = False
                    if 'title' in search_fields and item_data_for_sort_and_filter.get('title'):
                        if search_term_lower in item_data_for_sort_and_filter['title'].lower():
                            matches = True
                    
                    if not matches and 'authors' in search_fields and item_data_for_sort_and_filter.get('authors'):
                        if search_term_lower in item_data_for_sort_and_filter['authors'].lower():
                            matches = True
                    
                    if matches:
                        sortable_items.append(item_data_for_sort_and_filter)
                else:
                    # No search term, so all items are candidates for sorting and pagination
                    sortable_items.append(item_data_for_sort_and_filter)
        
        # Sort the (potentially filtered) list of items
        sortable_items.sort(key=get_abstract_sort_key)
        
        # Determine the total count for pagination display
        # If searching, it's the number of matched items. Otherwise, it's the total in DB.
        total_items_for_pagination = len(sortable_items) if search_term else total_db_count

        # Paginate the sorted list of items
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        
        ids_for_page = [item['id'] for item in sortable_items[start_idx:end_idx]]
        
        paginated_abstracts_data = []
        if ids_for_page:
            # Fetch full data (including documents) only for the IDs on the current page
            page_full_data_results = collection.get(ids=ids_for_page, include=["metadatas", "documents"])
            
            id_to_full_data_map = {}
            if page_full_data_results.get('ids'):
                for i, doc_id in enumerate(page_full_data_results['ids']):
                    id_to_full_data_map[doc_id] = {
                        'metadata': page_full_data_results['metadatas'][i],
                        'document': page_full_data_results['documents'][i]
                    }

            for item_id in ids_for_page: 
                if item_id in id_to_full_data_map:
                    full_data = id_to_full_data_map[item_id]
                    metadata = full_data['metadata']
                    paginated_abstracts_data.append({
                        'id': item_id,
                        'title': metadata.get('title', ''),
                        'authors': metadata.get('authors', ''),
                        'year': metadata.get('year', ''),
                        'source_title': metadata.get('source_title', ''),
                        'cited_by': metadata.get('cited_by', ''),
                        'doi': metadata.get('doi', ''),
                        'document': full_data['document']
                    })
        
        has_more = end_idx < len(sortable_items)
        
        return jsonify({
            'status': 'success',
            'abstracts': paginated_abstracts_data,
            'page': page,
            'per_page': per_page,
            'total': total_items_for_pagination, 
            'has_more': has_more
        })
        
    except Exception as e:
        app.logger.error(f"Error listing abstracts: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': f"Failed to retrieve abstracts: {str(e)}"
        }), 500

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

    # Start Flask server
    print("Starting Flask server...")
    print("Access the web interface at http://127.0.0.1:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
