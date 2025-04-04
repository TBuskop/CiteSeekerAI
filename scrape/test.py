# Script to generate a search string using Gemini API and then search Scopus with it
import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any

# load environment variables from .env file
from dotenv import load_dotenv
load_dotenv(override=True)  # Override existing environment variable

# Import functions from other modules
from get_search_string import generate_search_string
from search_scopus import run_scopus_search

# Configuration variables - modify these directly instead of using CLI arguments
DEFAULT_QUERY = "what are plausibilistic climate storylines and how are they different from other climate storylines?"
SKIP_GENERATION = False  # Set to True to skip search string generation
HEADLESS_BROWSER = False  # Set to True to run browser in headless mode
YEAR_FROM = None  # Start year for filtering (e.g., 2015)
YEAR_TO = None  # End year for filtering (e.g., 2023)
DOWNLOAD_DIR = "data/downloads/csv"  # Directory to save downloaded files

def run_search_process(query: str = None, skip_generation: bool = False, headless: bool = False,
                      year_from: Optional[int] = None, year_to: Optional[int] = None,
                      download_dir: str = "data/downloads/csv") -> Dict[str, Any]:
    """
    Run the complete search process: generate search string and search Scopus.
    
    Args:
        query: Research question for generating the search string
        skip_generation: Skip search string generation and use existing generated_search_string.txt
        headless: Run browser in headless mode for Scopus search
        year_from: Start year for filtering Scopus results
        year_to: End year for filtering Scopus results
        download_dir: Directory to save downloaded files
    
    Returns:
        Dictionary with results:
        {
            'success': bool,  # Overall success
            'search_string': Optional[str],  # Generated search string if available
            'csv_path': Optional[Path],  # Path to the downloaded CSV if available
            'error': Optional[str]  # Error message if any step failed
        }
    """
    result = {
        'success': False,
        'search_string': None,
        'csv_path': None,
        'error': None
    }
    
    # Use default query if none provided
    if query is None:
        query = "what are plausibilistic climate storylines and how are they different from other climate storylines?"
    
    # First, generate the search string if not skipped
    if not skip_generation:
        print(f"Generating search string from query: '{query}'")
        success, search_string = generate_search_string(query)
        
        if not success:
            result['error'] = "Failed to generate search string"
            return result
        
        result['search_string'] = search_string
    else:
        print("Skipping search string generation. Using existing generated_search_string.txt")
        # Try to read the existing file
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            search_string_file = Path(current_dir) / "generated_search_string.txt"
            if search_string_file.exists():
                with open(search_string_file, 'r') as file:
                    result['search_string'] = file.read().strip()
            else:
                result['error'] = "No existing generated_search_string.txt found"
                return result
        except Exception as e:
            result['error'] = f"Error reading existing search string file: {str(e)}"
            return result
    
    # Then run the Scopus search
    print(f"Running Scopus search with the {'generated' if not skip_generation else 'existing'} search string")
    success, csv_path = run_scopus_search(
        query=result['search_string'],  # Use the search string we obtained
        headless=headless,
        year_from=year_from,
        year_to=year_to,
        download_dir=download_dir
    )
    
    if not success:
        result['error'] = "Failed to execute Scopus search"
        return result
    
    result['csv_path'] = csv_path
    result['success'] = True
    return result

def main():
    """Main function to run both scripts in sequence."""
    # Use the globally defined configuration variables
    result = run_search_process(
        query=DEFAULT_QUERY,
        skip_generation=SKIP_GENERATION,
        headless=HEADLESS_BROWSER,
        year_from=YEAR_FROM,
        year_to=YEAR_TO,
        download_dir=DOWNLOAD_DIR
    )
    
    # Return success or failure based on result
    if result['success']:
        print(f"Search process completed successfully.")
        if result['csv_path']:
            print(f"Results saved to: {result['csv_path']}")
        return True
    else:
        print(f"Search process failed: {result['error']}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)