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
    print(f"Added {_PROJECT_ROOT} to sys.path")

# --- Add project root to sys.path ---
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, os.pardir, os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Import project modules
from src.workflows.DeepResearch_squential import run_deep_research
from src.workflows.obtain_store_abstracts import obtain_store_abstracts
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

def get_timestamp():
    """Generate a timestamp for job ID"""
    return datetime.now().strftime('%Y%m%d_%H%M%S')

def process_research_question(job_id, question, subquestions_count=3):
    """Process a research question using the CiteSeekerAI pipeline"""
    try:
        # Update job status
        processing_jobs[job_id]["status"] = "Processing"
        # Callback to send fine-grained progress updates to web UI
        def update_web_progress(message):
            if job_id in processing_jobs:
                processing_jobs[job_id]["progress"] = message
        update_web_progress("Initializing deep research...")
        # Run deep research with callback
        run_deep_research(question, subquestions_count, progress_callback=update_web_progress)

        # Find the output file
        output_file = find_latest_output_file()
        if output_file:
            with open(output_file, 'r', encoding='utf-8') as f:
                answer = f.read()
            
            # Store in chat history
            chat_history[job_id] = {
                "question": question,
                "answer": answer,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "file_path": output_file
            }
            
            # Update job status
            processing_jobs[job_id]["status"] = "Completed"
            processing_jobs[job_id]["output_file"] = output_file
        else:
            processing_jobs[job_id]["status"] = "Error"
            processing_jobs[job_id]["error"] = "Output file not found"
    
    except Exception as e:
        processing_jobs[job_id]["status"] = "Error"
        processing_jobs[job_id]["error"] = str(e)

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
            question_match = content.split("Original Research Question:", 1)
            if len(question_match) > 1:
                question = question_match[1].strip().split("\n", 1)[0].strip()
            else:
                question = "Unknown Question"
                
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
    processing_jobs[job_id] = {"status": "Starting", "progress": "Initializing..."}
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
        return jsonify({"status": "not_found"})
    
    return jsonify(processing_jobs[job_id])

@app.route('/result/<job_id>')
def get_result(job_id):
    """Get the result of a completed job"""
    if job_id in chat_history:
        return jsonify({
            "status": "success",
            "question": chat_history[job_id]["question"],
            "answer": chat_history[job_id]["answer"],
            "timestamp": chat_history[job_id]["timestamp"]
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
    
    if not query:
        return jsonify({"status": "error", "message": "Search query cannot be empty"})
    
    # Create job ID
    job_id = get_timestamp()
    
    # Start processing in background
    processing_jobs[job_id] = {"status": "Starting", "progress": "Initializing..."}
    threading.Thread(target=process_abstract_search, args=(job_id, query)).start()
    
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

def process_abstract_search(job_id, query):
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
        config.SCOPUS_SEARCH_STRING = query

        # Run with callback to update progress
        obtain_store_abstracts(query, progress_callback=update_web_progress)

        # Restore original config
        config.SCOPUS_SEARCH_STRING = original_scopus_search_string

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
