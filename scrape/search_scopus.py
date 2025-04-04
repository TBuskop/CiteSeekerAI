#!/usr/bin/env python
"""
Script to run a test search on Scopus and download results as CSV.

This script will:
1. Initialize the Scopus scraper
2. Perform a test search query
3. Export results to CSV
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
load_dotenv()
USERNAME = os.getenv("USERNAME")
PASSWORD = os.getenv("PASSWORD")
UNI_NAME = os.getenv("UNI_NAME")

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
    parser = argparse.ArgumentParser(description="Run a test search on Scopus and download results")
    # read generated_search_string.txt and use it as the default query
    try:
        with open("generated_search_string.txt", "r") as f:
            default_query = f.read().strip()
    except:
        default_query = "\"climate change\" AND \"coastal erosion\" AND adaptation AND \"Europe\""
    
    parser.add_argument(
        "--query", 
        type=str, 
        default=default_query, 
        help="Search query to use"
    )
    parser.add_argument(
        "--headless", 
        action="store_true", 
        help="Run browser in headless mode"
    )
    parser.add_argument(
        "--institution", 
        type=str, 
        help="Institution name"
    )
    parser.add_argument(
        "--download-dir", 
        type=str, 
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
    return parser.parse_args()

def ask_user_confirmation(question: str, default_yes: bool = True) -> bool:
    """
    Ask the user for confirmation with a yes/no question.
    
    Args:
        question: Question to ask the user
        default_yes: Whether the default answer is yes
        
    Returns:
        True if user confirmed, False if not
    """
    prompt = f"{question} [{'Y/n' if default_yes else 'y/N'}]: "
    
    while True:
        user_input = input(prompt).strip().lower()
        
        if not user_input:  # User pressed Enter, use default
            return default_yes
        
        if user_input in ['y', 'yes']:
            return True
        elif user_input in ['n', 'no']:
            return False
        else:
            print("Please respond with 'yes' or 'no' (or 'y' or 'n').")

def main():
    """Run Scopus search and download results."""
    args = parse_arguments()
    
    logger.info("Starting Scopus search process...")
    
    # Create screenshots directory
    screenshots_dir = Path(__file__).parent / "screenshots"
    screenshots_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Screenshots will be saved to: {screenshots_dir}")
    
    # Set up download directory
    # Change download directory to data/downloads/csv
    download_dir = Path("data/downloads/csv")
    download_dir.mkdir(exist_ok=True, parents=True)
    logger.info(f"Downloads will be saved to: {download_dir}")
    
    
    # Step 2: Initialize the Scopus scraper
    logger.info("Step 2: Initializing Scopus scraper...")
    
    try:
        # Initialize the scraper
        with ScopusScraper(
            headless=args.headless,
            download_dir=download_dir,
            institution=args.institution,
            screenshots_dir=screenshots_dir
        ) as scraper:
            
            # Step 3: Verify access to Scopus by logging in first
            logger.info("Step 3: Verifying access to Scopus...")
            login_success = scraper.login()
            
            if not login_success:
                logger.warning("Login may not have completed successfully.")
                print("\nLogin was not successful. The script will try to continue, but")
                print("you may need to complete the login manually in the browser window.")
                print("Please log in to Scopus if a login form appears, then the script will continue.")
                print("Waiting 15 seconds for potential manual login...")
                
                time.sleep(3)
            
            # Step 4: Run the search query
            logger.info(f"Step 4: Running search query: '{args.query}'")
            
            # TESTING CONFIG: Uncomment and modify these lines to use hardcoded date filters
            # Set USE_HARDCODED_YEARS to True to use hardcoded values instead of command line arguments
            USE_HARDCODED_YEARS = False  # Toggle this for testing
            HARDCODED_YEAR_FROM = 2015   # Adjust as needed for testing
            HARDCODED_YEAR_TO = 2023     # Adjust as needed for testing
            
            # Apply year filter parameters - either from command line or hardcoded values
            if USE_HARDCODED_YEARS:
                year_from = HARDCODED_YEAR_FROM
                year_to = HARDCODED_YEAR_TO
                logger.info(f"Using hardcoded year range: {year_from}-{year_to}")
            else:
                year_from = args.year_from
                year_to = args.year_to
                logger.info(f"Using command line year range: {year_from}-{year_to}")
            
            # First perform the basic search
            search_success = scraper.search(args.query)
            
            if not search_success:
                logger.error("Search failed. Please check your query and access permissions.")
                return None
            
            # Then apply date filters if specified
            if year_from or year_to:
                logger.info(f"Applying date range filter: {year_from}-{year_to}")
                filter_success = scraper.apply_date_filter(year_from, year_to)
                if not filter_success:
                    logger.warning("Failed to apply date filter, continuing with unfiltered results")
            
            # Step 5: Export results to CSV
            logger.info("Step 5: Exporting search results to CSV...")
            
            # Create filename from query - make sure to clean it
            query_filename = args.query.replace(" ", "_")[:50]
            
            # Add year range to filename if specified
            if year_from or year_to:
                year_suffix = f"_{year_from if year_from else 'start'}-{year_to if year_to else 'end'}"
                query_filename += year_suffix
                
            csv_path = scraper.export_to_csv(f"scopus_search_{query_filename}")
            
            if csv_path:
                logger.info(f"SUCCESS! Results exported to: {csv_path}")
                logger.info(f"Total file size: {csv_path.stat().st_size / 1024:.1f} KB")
                return csv_path
            else:
                logger.error("Failed to export results to CSV.")
                return None
    
    except Exception as e:
        logger.error(f"Error during Scopus search: {str(e)}")
        return None

if __name__ == "__main__":
    csv_file = main()
    if csv_file:
        print(f"\nSearch results saved to: {csv_file}")
    else:
        print("\nSearch process failed. See log for details.")
        sys.exit(1)
