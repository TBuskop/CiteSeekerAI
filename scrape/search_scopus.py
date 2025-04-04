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
from typing import Optional, Tuple

# load environment variables from .env file
from dotenv import load_dotenv

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

def run_scopus_search(query: str = None, headless: bool = False, 
                     year_from: Optional[int] = None, year_to: Optional[int] = None, 
                     download_dir: str = "data/downloads/csv",
                     username: Optional[str] = None, 
                     password: Optional[str] = None,
                     institution: Optional[str] = None) -> Tuple[bool, Optional[Path]]:
    """
    Run a search on Scopus and download results as CSV.
    
    Args:
        query: Search query string
        headless: Whether to run browser in headless mode
        year_from: Start year for filtering results
        year_to: End year for filtering results
        download_dir: Directory to save downloaded files
        username: Scopus username (overrides env variables)
        password: Scopus password (overrides env variables)
        institution: Institution name (overrides env variables)
        
    Returns:
        Tuple containing (success: bool, csv_path: Optional[Path])
    """
    try:
        # Change to the directory of this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_dir)
        
        # Load environment variables
        load_dotenv(override=True)
        
        # Get credentials from args or environment variables
        username = username or os.getenv("SCOPUS_USERNAME")
        password = password or os.getenv("SCOPUS_PASSWORD")
        institution = institution or os.getenv("SCOPUS_INSTITUTION")
        
        # Add fallback check for old environment variable names
        if not username:
            username = os.getenv("USER_NAME")
            logger.warning("SCOPUS_USERNAME not found, falling back to USER_NAME")
        
        if not password:
            password = os.getenv("PASSWORD")
            logger.warning("SCOPUS_PASSWORD not found, falling back to PASSWORD")
            
        if not institution:
            institution = os.getenv("UNI_NAME")
            logger.warning("SCOPUS_INSTITUTION not found, falling back to UNI_NAME")

        if not username or not password:
            logger.error("Scopus username and password are required.")
            return False, None

        logger.info(f"Using institution: {institution}")
        logger.info(f"Using username: {username}")
        logger.info("Password: [masked for security]")
        
        # Create screenshots directory
        screenshots_dir = Path(script_dir) / "screenshots"
        screenshots_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Screenshots will be saved to: {screenshots_dir}")
        
        # Set up download directory
        download_dir_path = Path(download_dir)
        download_dir_path.mkdir(exist_ok=True, parents=True)
        logger.info(f"Downloads will be saved to: {download_dir_path}")
        
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
        csv_path = None
        with ScopusScraper(
            headless=headless,
            download_dir=download_dir_path,
            institution=institution,
            screenshots_dir=screenshots_dir
        ) as scraper:
            logger.info("Scopus scraper initialized successfully.")
            
            # Login to Scopus
            login_success = scraper.login(username=username, password=password)
            if not login_success:
                logger.error("Login failed. Check credentials and network connection.")
                return False, None
            logger.info("Logged in successfully.")
            
            # Perform search
            search_success = scraper.search(query)
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
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            safe_query_part = "".join(c if c.isalnum() else '_' for c in query)[:50]
            filename = f"scopus_{safe_query_part}_{timestamp}"
            
            csv_path = scraper.export_to_csv(filename)
            if not csv_path:
                logger.error("Failed to export results to CSV.")
                return False, None
            
            logger.info(f"Results exported to CSV at: {csv_path}")
            return True, csv_path
            
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
        username=args.username,
        password=args.password,
        institution=args.institution
    )
    
    if success and csv_file:
        print(f"\nSearch results saved to: {csv_file}")
        sys.exit(0)  # Success
    else:
        print("\nSearch process failed or did not complete successfully. See log for details.")
        sys.exit(1)  # Error
