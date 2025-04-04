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
from typing import Optional

# change run path to current directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# load environment variables from .env file
from dotenv import load_dotenv
load_dotenv(override=True)  # Override existing environment variables

# Load credentials and configuration from environment variables 
# using the new variable names from the .env file
USERNAME = os.getenv("SCOPUS_USERNAME")
PASSWORD = os.getenv("SCOPUS_PASSWORD")
INSTITUTION = os.getenv("SCOPUS_INSTITUTION")

# Import required modules
from scopus_scraper import ScopusScraper

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
    default_query_file = Path("generated_search_string.txt")
    default_query = "\"climate change\" AND \"coastal erosion\" AND adaptation AND \"Europe\"" # Fallback default
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
        default=default_query, 
        help="Search query to use (default: content of generated_search_string.txt or fallback)"
    )
    parser.add_argument(
        "--headless", 
        action="store_true", 
        help="Run browser in headless mode"
    )
    # Institution is now primarily loaded from .env
    parser.add_argument(
        "--institution", 
        type=str,
        default=INSTITUTION,
        help="Institution name (optional, overrides .env)"
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
    # Add arguments for credentials as optional overrides
    parser.add_argument("--username", type=str, default=USERNAME, help="Scopus username (overrides .env)")
    parser.add_argument("--password", type=str, default=PASSWORD, help="Scopus password (overrides .env)")

    return parser.parse_args()

def main():
    """Run Scopus search and download results."""
    args = parse_arguments()
    
    # Use credentials from args (which default to .env values if available)
    username = args.username
    password = args.password
    institution = args.institution
    
    # Add fallback check for old environment variable names
    if not username:
        username = os.getenv("USER_NAME")  # Try old variable name
        logger.warning("SCOPUS_USERNAME not found, falling back to USER_NAME")
    
    if not password:
        password = os.getenv("PASSWORD")  # Try old variable name
        logger.warning("SCOPUS_PASSWORD not found, falling back to PASSWORD")
        
    if not institution:
        institution = os.getenv("UNI_NAME")  # Try old variable name
        logger.warning("SCOPUS_INSTITUTION not found, falling back to UNI_NAME")

    if not username or not password:
        logger.error("Scopus username and password are required. Set SCOPUS_USERNAME and SCOPUS_PASSWORD in .env file.")
        sys.exit(1)

    logger.info(f"Using institution: {institution}")
    logger.info(f"Using username: {username}")
    logger.info("Password: [masked for security]")
    
    # Create screenshots directory
    screenshots_dir = Path(__file__).parent / "screenshots"
    screenshots_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Screenshots will be saved to: {screenshots_dir}")
    
    # Set up download directory from args
    download_dir = Path(args.download_dir)
    download_dir.mkdir(exist_ok=True, parents=True)
    logger.info(f"Downloads will be saved to: {download_dir}")
    
    
    # Step 1: Initialize the Scopus scraper
    logger.info("Step 1: Initializing Scopus scraper...")
    
    csv_path: Optional[Path] = None # Define csv_path before try block
    scraper = None # Define scraper for potential use in except block
    try:
        # Initialize the scraper
        with ScopusScraper(
            headless=args.headless,
            download_dir=download_dir,
            institution=institution, # Pass institution
            screenshots_dir=screenshots_dir # Pass screenshots directory
        ) as scraper:
            logger.info("Scopus scraper initialized successfully.")
            
            # Step 2: Log in to Scopus
            logger.info("Step 2: Logging in to Scopus...")
            # Check return value from login method
            login_success = scraper.login(username=username, password=password)
            if not login_success:
                logger.error("Login failed. Check credentials and network connection.")
                return None
            logger.info("Logged in successfully.")
            
            # Step 3: Perform search query
            logger.info(f"Step 3: Performing search query: {args.query}")
            search_success = scraper.search(args.query)
            if not search_success:
                logger.error("Search failed. Check your query and try again.")
                return None
            logger.info("Search completed successfully.")
            
            # Step 4: Apply date filters if specified
            # if args.year_from or args.year_to:
            #     logger.info(f"Step 4: Applying date filters: {args.year_from} to {args.year_to}")
            #     # Fix method name: apply_date_filters -> apply_date_filter
            #     filter_success = scraper.apply_date_filter(args.year_from, args.year_to)
            #     if not filter_success:
            #         logger.warning("Failed to apply date filters, continuing with unfiltered results")
            #     else:
            #         logger.info("Date filters applied successfully")
            # else:
            #     logger.info("Step 4: No date filters specified, skipping.")
            
            # Step 5: Export results to CSV
            logger.info("Step 5: Exporting results to CSV...")
            # Create a meaningful filename with timestamp
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            safe_query_part = "".join(c if c.isalnum() else '_' for c in args.query)[:50]
            filename = f"scopus_{safe_query_part}_{timestamp}"
            
            # Fix method name: export_results_to_csv -> export_to_csv
            csv_path = scraper.export_to_csv(filename)
            if not csv_path:
                logger.error("Failed to export results to CSV.")
                return None
            
            logger.info(f"Results exported to CSV at: {csv_path}")
            return csv_path
            
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")
        # Try to save screenshot if scraper exists and has a page
        if scraper and hasattr(scraper, 'page') and scraper.page:
            try:
                scraper._save_screenshot("unexpected_error")
            except Exception as screenshot_error:
                logger.error(f"Failed to save error screenshot: {screenshot_error}")
        return None

if __name__ == "__main__":
    csv_file = main()
    if csv_file:
        print(f"\nSearch results saved to: {csv_file}")
        sys.exit(0)  # Success
    else:
        print("\nSearch process failed or did not complete successfully. See log for details.")
        sys.exit(1)  # Error
