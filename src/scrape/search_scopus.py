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
                     scopus_search_scope: Optional[str] = None  # Added parameter
                     ) -> Tuple[bool, Optional[Path]]:
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
        
    Returns:
        Tuple containing (success: bool, csv_path: Optional[Path])
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
            
            # Perform search
            search_success = scraper.search(query, search_scope_override=scopus_search_scope) # Pass scope
            if not search_success:
                logger.error("Search failed. Check your query and try again.")
                return False, None
            logger.info("Search completed successfully.")
            
            # Apply date filters if specified
            if year_from or year_to:
                logger.info(f"Applying date filters: {year_from} to {year_to}")
                filter_success = scraper.apply_date_filter(year_from, year_to)
                if not filter_success:
                    logger.warning("Failed to apply date filters, continuing with unfiltered results")
                else:
                    logger.info("Date filters applied successfully")
            
            # Export results to CSV
            logger.info("Exporting results to CSV...")
            actual_csv_path = scraper.export_to_csv(final_filename_base)
            if not actual_csv_path:
                logger.error("Failed to export results to CSV.")
                return False, None

            if output_csv_path and actual_csv_path != output_csv_path:
                try:
                    if actual_csv_path.exists():
                        if output_csv_path.exists():
                            logger.warning(f"Target CSV path {output_csv_path} already exists. Overwriting.")
                            output_csv_path.unlink()
                        actual_csv_path.rename(output_csv_path)
                        logger.info(f"Renamed downloaded file to specified path: {output_csv_path}")
                        actual_csv_path = output_csv_path
                    else:
                        logger.error(f"Scraper reported saving to {actual_csv_path}, but file not found.")
                        return False, None
                except OSError as e:
                    logger.error(f"Could not rename {actual_csv_path} to {output_csv_path}: {e}")
                    logger.warning(f"Results saved to CSV at the generated path: {actual_csv_path}")
                    return True, actual_csv_path

            logger.info(f"Results exported to CSV at: {actual_csv_path}")
            return True, actual_csv_path
            
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")
        return False, None

# Keep the original script functionality when run directly
if __name__ == "__main__":
    args = parse_arguments()
    success, csv_file = run_scopus_search(
        query=args.query,
        headless=args.headless,
        year_from=args.year_from,
        year_to=args.year_to,
        download_dir=args.download_dir,
        scopus_search_scope=args.scopus_search_scope # Pass from CLI args
    )
    
    if success and csv_file:
        print(f"\nSearch results saved to: {csv_file}")
        sys.exit(0)  # Success
    else:
        print("\nSearch process failed or did not complete successfully. See log for details.")
        sys.exit(1)  # Error
