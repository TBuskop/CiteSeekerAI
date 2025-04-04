# Script to generate a search string using Gemini API and then search Scopus with it
import subprocess
import sys
import os
import argparse
from pathlib import Path

# load environment variables from .env file
from dotenv import load_dotenv
load_dotenv(override=True)  # Override existing environment variable

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run search string generation and Scopus search")
    parser.add_argument(
        "--query", 
        type=str, 
        default="what are plausibilistic climate storylines and how are they different from other climate storylines?",
        help="Research question for generating the search string"
    )
    parser.add_argument(
        "--skip-generation", 
        action="store_true",
        help="Skip search string generation and use existing generated_search_string.txt"
    )
    parser.add_argument(
        "--headless", 
        action="store_true", 
        help="Run browser in headless mode for Scopus search"
    )
    parser.add_argument(
        "--year-from",
        type=int,
        help="Start year for filtering Scopus results"
    )
    parser.add_argument(
        "--year-to",
        type=int,
        help="End year for filtering Scopus results"
    )
    return parser.parse_args()

def generate_search_string(query):
    """Run get_search_string.py to generate a search string from the research question."""
    # Ensure the script runs with the correct Python interpreter
    python_executable = sys.executable
    
    # Get the directory of the current script (test.py)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct the path to get_search_string.py
    script_path = os.path.join(current_dir, "get_search_string.py")
    
    # Modify get_search_string.py on the fly to use the provided query
    with open(script_path, 'r') as file:
        script_content = file.read()
    
    # Replace the initial_query with the new one
    script_content = script_content.replace(
        'initial_query = "what are plausibilistic climate storylines and how are they different from other climate storylines? Write me a short literature review on this topic."',
        f'initial_query = "{query} Write me a short literature review on this topic."'
    )
    
    # Create a temporary modified script
    temp_script_path = os.path.join(current_dir, "_temp_get_search_string.py")
    with open(temp_script_path, 'w') as file:
        file.write(script_content)
    
    try:
        print(f"Generating search string from query: '{query}'")
        result = subprocess.run(
            [python_executable, temp_script_path],
            check=True,
            capture_output=True,
            text=True
        )
        print("Search string generation output:")
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
        
        # Check if the generated_search_string.txt file exists
        search_string_file = Path(current_dir) / "generated_search_string.txt"
        if search_string_file.exists():
            with open(search_string_file, 'r') as file:
                search_string = file.read().strip()
            print(f"Generated search string: {search_string}")
            return True
        else:
            print("Error: generated_search_string.txt was not created.")
            return False
    except subprocess.CalledProcessError as e:
        print(f"Search string generation failed with error code {e.returncode}")
        print(f"Output: {e.stdout}")
        print(f"Error: {e.stderr}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred during search string generation: {e}")
        return False
    finally:
        # Remove the temporary script
        if os.path.exists(temp_script_path):
            os.remove(temp_script_path)

def run_scopus_search(headless=False, year_from=None, year_to=None):
    """Run search_scopus.py with the generated search string."""
    # Ensure the script runs with the correct Python interpreter
    python_executable = sys.executable
    
    # Get the directory of the current script (test.py)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct the path to search_scopus.py
    script_path = os.path.join(current_dir, "search_scopus.py")
    
    # Define the command arguments
    command = [
        python_executable,
        script_path
    ]
    
    # Add optional arguments
    if headless:
        command.append("--headless")
    
    if year_from is not None:
        command.extend(["--year-from", str(year_from)])
    
    if year_to is not None:
        command.extend(["--year-to", str(year_to)])
    
    print(f"Running Scopus search: {' '.join(command)}")
    
    # Run the script
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print("Scopus search executed successfully.")
        print("Output:\n", result.stdout)
        if result.stderr:
            print("Errors:\n", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Scopus search failed with error code {e.returncode}")
        print("Output:\n", e.stdout)
        print("Errors:\n", e.stderr)
        return False
    except FileNotFoundError:
        print(f"Error: Could not find the script '{script_path}' or the python executable '{python_executable}'.")
        return False
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False

def main():
    """Main function to run both scripts in sequence."""
    args = parse_arguments()
    
    # First, generate the search string if not skipped
    if not args.skip_generation:
        success = generate_search_string(args.query)
        if not success:
            print("Failed to generate search string. Exiting.")
            return False
    else:
        print("Skipping search string generation. Using existing generated_search_string.txt")
        
    # Then run the Scopus search
    return run_scopus_search(headless=args.headless, year_from=args.year_from, year_to=args.year_to)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)