#!/usr/bin/env python
"""
Script to run a search on Scopus and download results as CSV.

This script will:
1. Load configuration and credentials.
2. Initialize the Scopus scraper.
3. Log in to Scopus.
4. Perform a search query.
5. Apply date filters if specified.
6. Export results to CSV.
"""
import sys
import os
import logging
import time
from pathlib import Path
import argparse
from typing import Optional, Tuple, Union

# load environment variables from .env file
from dotenv import load_dotenv

# Import required modules

# --- Add project root to sys.path ---
# This allows absolute imports from 'src' assuming the script is in 'workflows'
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, os.pardir, os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.scrape.scopus_scraper import ScopusScraper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("scopus_search")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run a search on Scopus and download results")
    # read generated_search_string.txt and use it as the default query
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    save_path = os.path.join(PROJECT_ROOT, 'data', 'output', "generated_scopus_search_string.txt")
    default_query_file = Path(save_path)

    print(f"Default query file path: {default_query_file}")
    if default_query_file.exists():
        try:
            default_query = default_query_file.read_text().strip()
            logger.info(f"Using default query from {default_query_file}")
        except Exception as e:
            logger.warning(f"Could not read {default_query_file}, using fallback default query. Error: {e}")
    else:
        logger.info(f"{default_query_file} not found, using fallback default query.")

    parser.add_argument(
        "--query", 
        type=str, 
        help="Search query to use (default: content of generated_search_string.txt or fallback)"
    )
    parser.add_argument(
        "--headless", 
        action="store_true", 
        help="Run browser in headless mode"
    )
    parser.add_argument(
        "--download-dir", 
        type=str, 
        default="data/downloads/csv", # Set default directly
        help="Directory to save downloaded files"
    )
    parser.add_argument(
        "--year-from",
        type=int,
        help="Start year for filtering results"
    )
    parser.add_argument(
        "--year-to",
        type=int,
        help="End year for filtering results"
    )
    parser.add_argument(
        "--scopus-search-scope",
        type=str,
        default=None, # Default to None, will use config if not provided
        help="Scopus search scope (e.g., ALL, TITLE_ABS_KEY)"
    )

    return parser.parse_args()

def run_scopus_search(query: str = None, headless: bool = False, 
                     year_from: Optional[int] = None, year_to: Optional[int] = None, 
                     download_dir: str = "data/downloads/csv",
                     output_csv_path: Optional[Union[str, Path]] = None,
                     scopus_search_scope: Optional[str] = None,
                     force_continue_large_search: bool = False  # Ensure this is part of the definition
                     ) -> Tuple[str, Optional[Path], Optional[int]]:
    """
    Run a search on Scopus and download results as CSV.
    
    Args:
        query: Search query string
        headless: Whether to run browser in headless mode
        year_from: Start year for filtering results
        year_to: End year for filtering results
        download_dir: Directory to save downloaded files (used if output_csv_path is None)
        output_csv_path: Specific path (including filename) to save the CSV. Overrides download_dir and generated filename.
        scopus_search_scope: The scope for the Scopus search (e.g., "ALL", "TITLE_ABS_KEY"). Defaults to config if None.
        force_continue_large_search: If True and search results are many (warning), proceed with export anyway.
        
    Returns:
        Tuple containing:
        - status_code: String indicating search/export success or error
          - "EXPORT_SUCCESS": Export was successful
          - "SEARCH_WARNING_TOO_MANY_RESULTS": Results > 1000 but export succeeded
          - "SEARCH_ERROR_LIMIT_EXCEEDED": Results > 20,000 (export not attempted)
          - Other failure codes from search or export
        - csv_path: Optional[Path] - Path to the exported CSV if successful
        - count: Optional[int] - The number of results (if available)
    """
    try:
        # Change to the directory of this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_dir)
        
        # Load environment variables
        load_dotenv(override=True)
        
        # Create screenshots directory
        screenshots_dir = Path(script_dir) / "screenshots"
        screenshots_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Screenshots will be saved to: {screenshots_dir}")
        
        # Determine download directory and filename
        if output_csv_path:
            output_csv_path = Path(output_csv_path)
            final_download_dir = output_csv_path.parent
            final_filename_base = output_csv_path.stem
            logger.info(f"Using specified output path: {output_csv_path}")
        else:
            final_download_dir = Path(download_dir)
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            safe_query_part = "".join(c if c.isalnum() else '_' for c in query)[:50]
            final_filename_base = f"scopus_{safe_query_part}_{timestamp}"
            output_csv_path = final_download_dir / f"{final_filename_base}.csv"
            logger.info(f"Downloads will be saved to: {final_download_dir} with generated filename.")

        # Set up download directory
        final_download_dir.mkdir(exist_ok=True, parents=True)
        
        # Get default query if none provided
        if query is None:
            # Try to read query from generated_search_string.txt
            default_query_file = Path(script_dir) / "generated_search_string.txt"
            if default_query_file.exists():
                try:
                    query = default_query_file.read_text().strip()
                    logger.info(f"Using query from {default_query_file}")
                except Exception as e:
                    logger.warning(f"Could not read {default_query_file}. Error: {e}")
                    query = "\"climate change\" AND \"coastal erosion\" AND adaptation AND \"Europe\"" # Fallback
            else:
                logger.info(f"{default_query_file} not found, using fallback default query.")
                query = "\"climate change\" AND \"coastal erosion\" AND adaptation AND \"Europe\"" # Fallback
        
        # Initialize the scraper and perform search
        actual_csv_path = None
        with ScopusScraper(
            headless=headless,
            download_dir=final_download_dir,
            screenshots_dir=screenshots_dir
        ) as scraper:              
            logger.info("Scopus scraper initialized successfully.")
            
            # Perform search with year parameters
            search_status, results_count = scraper.search(
                query, 
                search_scope_override=scopus_search_scope,
                year_from=year_from,
                year_to=year_to
            )
            
            # Check for errors or warnings
            if search_status == "SEARCH_ERROR_LIMIT_EXCEEDED":
                logger.error(f"Search returned too many results: {results_count} (limit: 20,000)")
                return search_status, None, results_count
                
            if search_status == "SEARCH_WARNING_TOO_MANY_RESULTS":                
                if not force_continue_large_search:
                    logger.warning(f"Search returned many results: {results_count} (recommended max: 1,000). Pausing for user confirmation.")
                    # Return the warning status so the caller can decide whether to continue
                    return search_status, None, results_count
                else:
                    logger.info(f"Search returned many results: {results_count}. Proceeding with export as force_continue_large_search is True.")
                    # Add more detailed logging for UI progress updates
                    logger.info("=== FORCE CONTINUE WITH LARGE RESULT SET ===")
                    logger.info(f"Processing {results_count} documents for export - this may take longer than usual")
                    logger.info("Please be patient while we prepare the export...")
                    # Do not return here; continue to filtering and export.
                    # The status will be EXPORT_SUCCESS if export is successful.
                
            if search_status.startswith("SEARCH_FAILURE"):
                logger.error(f"Search failed with status: {search_status}")
                return search_status, None, results_count
                  
            logger.info(f"Search completed with status: {search_status}")
              # Date filters are now applied directly in the search method
            # Export results to CSV
            logger.info("===== Starting Scopus Export Process =====")
            logger.info(f"Exporting {results_count if results_count else 'unknown number of'} results to CSV...")
            
            # If we have large results, provide more detailed progress information
            if search_status == "SEARCH_WARNING_TOO_MANY_RESULTS" and force_continue_large_search:
                logger.info("Large result set export in progress - this may take a few minutes")
                logger.info("The browser will select all results and prepare them for download")
            
            actual_csv_path = scraper.export_to_csv(final_filename_base)
            if not actual_csv_path:                
                logger.error("Failed to export results to CSV.")
                return "EXPORT_FAILURE", None, results_count            
            if output_csv_path and actual_csv_path != output_csv_path:
                try:
                    logger.info("Preparing to save CSV to final location...")
                    if actual_csv_path.exists():
                        if output_csv_path.exists():
                            logger.warning(f"Target CSV path {output_csv_path} already exists. Overwriting.")
                            output_csv_path.unlink()
                        logger.info(f"Moving CSV from temporary location to final destination...")
                        actual_csv_path.rename(output_csv_path)
                        logger.info(f"Renamed downloaded file to specified path: {output_csv_path}")
                        actual_csv_path = output_csv_path
                    else:                        
                        logger.error(f"Scraper reported saving to {actual_csv_path}, but file not found.")
                        return "EXPORT_FAILURE", None, results_count
                except OSError as e:
                    logger.error(f"Could not rename {actual_csv_path} to {output_csv_path}: {e}")
                    logger.warning(f"Results saved to CSV at the generated path: {actual_csv_path}")                    # Still success, just at a different path than requested
                    return "EXPORT_SUCCESS", actual_csv_path, results_count
            
            logger.info(f"Results exported to CSV at: {actual_csv_path}")
            return "EXPORT_SUCCESS", actual_csv_path, results_count
            
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")
        return "UNEXPECTED_ERROR", None, None

# Keep the original script functionality when run directly
if __name__ == "__main__":
    args = parse_arguments()
    # Consider adding a CLI arg for force_continue_large_search if needed for direct script runs
    status, csv_file, results_count = run_scopus_search(
        query=args.query,
        headless=args.headless,
        year_from=args.year_from,
        year_to=args.year_to,
        download_dir=args.download_dir,
        scopus_search_scope=args.scopus_search_scope,
        force_continue_large_search=False # Default for CLI run - corrected position
    )
    
    if status == "EXPORT_SUCCESS" and csv_file:
        print(f"\nSearch results saved to: {csv_file}")
        sys.exit(0)  # Success
    else:
        print(f"\nSearch process failed or did not complete successfully. Status: {status}. See log for details.")
        sys.exit(1)  # Error
