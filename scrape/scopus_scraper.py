import os
import time
import logging
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
from typing import Dict, Optional, List, Union, Tuple
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ScopusScraper:
    """Class for scraping Scopus search results and exporting them to CSV."""
    
    def __init__(self, 
                 headless: bool = False, 
                 download_dir: Optional[Path] = None,
                 institution: Optional[str] = None,
                 screenshots_dir: Optional[Path] = None):
        """
        Initialize the Scopus scraper.
        
        Args:
            headless: Whether to run the browser in headless mode
            download_dir: Directory where downloads should be saved
            institution: Name of your institution
            screenshots_dir: Directory to save screenshots (defaults to tests/screenshots)
        """
        self.headless = headless
        self.download_dir = download_dir or Path.home() / "Downloads"
        self.institution = institution
        self.browser = None
        self.page = None
        self.playwright = None
        
        # Set screenshots directory
        self.screenshots_dir = screenshots_dir or self._get_default_screenshots_dir()
        # Ensure the screenshots directory exists
        self.screenshots_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_default_screenshots_dir(self) -> Path:
        """
        Get the default directory for saving screenshots.
        
        Returns:
            Path: Path to the screenshots directory
        """
        # Get project root directory (assuming this file is in /scrapers/)
        project_root = Path(__file__).parent.parent
        # Create path to tests/screenshots
        screenshots_dir = project_root / "tests" / "screenshots"
        return screenshots_dir
    
    def _save_screenshot(self, name: str) -> Path:
        """
        Save a screenshot with a given name to the screenshots directory.
        
        Args:
            name: Name of the screenshot file (without extension)
            
        Returns:
            Path: Path to the saved screenshot
        """
        # Add timestamp to prevent overwriting
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"{name}_{timestamp}.png"
        filepath = self.screenshots_dir / filename
        
        try:
            self.page.screenshot(path=str(filepath))
            logger.info(f"Saved screenshot: {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Failed to save screenshot '{name}': {str(e)}")
            return None
            
    def __enter__(self):
        """Context manager entry method."""
        self._start_browser()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit method."""
        self._close_browser()
        
    def _start_browser(self):
        """Start the browser using Playwright."""
        self.playwright = sync_playwright().start()
        
        # Launch the browser
        self.browser = self.playwright.chromium.launch(
            headless=self.headless
        )
        
        # Create a browser context with downloads enabled
        context = self.browser.new_context(
            accept_downloads=True
        )
        
        # Create a new page
        self.page = context.new_page()
        
        # Set timeout
        self.page.context.set_default_timeout(60000)  # 60 seconds timeout
        
        # Ensure download directory exists
        self.download_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure download behavior
        self.page.context.route("**/*.csv", lambda route: route.continue_(
            headers={"Accept": "text/csv", "Content-Type": "text/csv"}
        ))
        
        logger.info("Browser started successfully")
    
    def _close_browser(self):
        """Close the browser."""
        if self.browser:
            self.browser.close()
            self.browser = None
            self.page = None
            
        if self.playwright:
            self.playwright.stop()
            self.playwright = None
            
        logger.info("Browser closed")
    
    def _read_credentials(self) -> Tuple[Optional[str], Optional[str]]:
        """
        Read credentials from credentials.txt
        
        Returns:
            Tuple containing username and password
        """
        from dotenv import load_dotenv
        load_dotenv()
        USER_NAME = os.getenv("USER_NAME")
        PASSWORD = os.getenv("PASSWORD")
        UNI_NAME = os.getenv("UNI_NAME")

        try:   
            username = USER_NAME
            password = PASSWORD
            institution = UNI_NAME
            
            return username, password
        except Exception as e:
            logger.error(f"Error reading credentials: {str(e)}")
            return None, None
    
    def login(self, username: str = None, password: str = None) -> bool:
        """
        Login to Scopus using provided credentials or institutional access.
        
        Args:
            username: Scopus username (optional if using credentials.txt)
            password: Scopus password (optional if using credentials.txt)
            
        Returns:
            bool: True if login successful, False otherwise
        """
        try:
            logger.info("Attempting to access Scopus...")
            
            # If credentials are not provided, try to read from credentials.txt
            if not username or not password:
                username, password = self._read_credentials()
                if username and password:
                    logger.info("Using credentials from credentials.txt")
            
            # Navigate to Scopus
            self.page.goto("https://www-scopus-com.vu-nl.idm.oclc.org/search/form.uri?display=basic")
            
            # Wait for page to load completely
            self.page.wait_for_load_state('networkidle')
            
            # Check if already logged in
            if "search/form.uri" in self.page.url and not ("signin" in self.page.url or "login" in self.page.url):
                logger.info("Already logged in to Scopus")
                return True
            
            # Check if we need to proceed to login
            proceed_button = self.page.query_selector('input[value="PROCEED TO LOGIN"]')
            if proceed_button:
                logger.info("Clicking 'PROCEED TO LOGIN' button")
                proceed_button.click()
                self.page.wait_for_load_state('networkidle')
            
            # Check for institution selection screen
            vu_selector = 'div[data-entityid="http://stsfed.login.vu.nl/adfs/services/trust"]'
            if self.page.query_selector(vu_selector):
                logger.info("Found institution selection screen, selecting VU Amsterdam")
                self.page.click(vu_selector)
                self.page.wait_for_load_state('networkidle')
            # Check if we're on the VU login screen
            if "id.vu.nl" in self.page.url or self.page.query_selector('#userNameInput'):
                logger.info("Found VU login form, entering credentials")
                
                if not username or not password:
                    logger.error("Username and password required for VU login but not provided")
                    return False
                
                # Fill in VUnetID
                self.page.fill('#userNameInput', username)
                
                # Fill in password
                self.page.fill('#passwordInput', password)
                
                # Click sign in
                self.page.click('#submitButton')
                
                # Wait for redirect and authentication to complete
                self.page.wait_for_load_state('networkidle', timeout=30000)
                
                # Give additional time for redirects and processing
                time.sleep(5)
                
                # Check if we reached Scopus successfully
                if "scopus" in self.page.url and "search/form" in self.page.url:
                    logger.info("Login successful, reached Scopus search page")
                    return True
                else:
                    # Check for error messages
                    error_selector = '.error, .errorText, #errorText'
                    if self.page.query_selector(error_selector):
                        error_msg = self.page.text_content(error_selector)
                        logger.error(f"Login error: {error_msg}")
                    else:
                        logger.error(f"Login failed, current URL: {self.page.url}")
                    return False
            # Direct login if needed
            elif username and password:
                return self._login_with_direct_credentials(username, password)
            
            # Unknown login state
            else:
                logger.error(f"Unsupported login page encountered: {self.page.url}")
                return False
                
        except Exception as e:
            logger.error(f"Login failed with error: {str(e)}")
            return False
    
    def _login_with_direct_credentials(self, username: str, password: str) -> bool:
        """
        Login to Scopus using direct username/password.
        
        Args:
            username: Scopus username
            password: Scopus password
            
        Returns:
            bool: True if login successful, False otherwise
        """
        try:
            # Check if credentials were provided
            if not username or not password:
                logger.error("Username and password are required for direct login")
                return False
            logger.info("Attempting direct login to Scopus...")
            self.page.goto("https://id.elsevier.com/as/authorization.oauth2")
            
            # Fill username
            self.page.fill('input[name="username"]', username)
            self.page.click('button[type="submit"]')
            
            # Fill password
            self.page.fill('input[name="password"]', password)
            self.page.click('button[type="submit"]')
            
            # Check if login was successful by looking for a profile element
            try:
                self.page.wait_for_selector('a[title="View your account details"]', timeout=10000)
                logger.info("Login successful")
                return True
            except PlaywrightTimeoutError:
                logger.error("Login failed - could not find profile element")
                return False
                
        except Exception as e:
            logger.error(f"Direct login failed with error: {str(e)}")
            return False
    
    def search(self, query: str, filters: Optional[Dict] = None, 
               year_from: Optional[int] = None, year_to: Optional[int] = None) -> bool:
        """
        Perform a search on Scopus.
        
        Args:
            query: Search query string
            filters: Dictionary of filters to apply
            year_from: Start year for filtering results (will be applied after initial search)
            year_to: End year for filtering results (will be applied after initial search)
            
        Returns:
            bool: True if search successful, False otherwise
        """
        try:
            logger.info(f"Performing search with query: {query}")
            
            # Navigate to Scopus search page
            self.page.goto("https://www-scopus-com.vu-nl.idm.oclc.org/search/form.uri?display=basic")
            
            # Check if we got redirected to a login page due to lack of access
            if "signin" in self.page.url or "login" in self.page.url:
                logger.warning("Redirected to login page. Attempting to authenticate...")
                if not self.login():
                    logger.error("Authentication failed. Cannot access Scopus search.")
                    return False
                
                # After login, navigate back to search page
                self.page.goto("https://www-scopus-com.vu-nl.idm.oclc.org/search/form.uri?display=basic")
            
            # Wait for the page to load completely
            self.page.wait_for_load_state('networkidle')
            
            # Try to find the search input field - could be one of several possible selectors
            # depending on which Scopus interface is shown
            search_selectors = [
                'input.searchterm-input', 
                'input[id^="autosuggest-"][id$="-input"]',
                'input[placeholder=" "][required]'
            ]
            
            found_selector = None
            for selector in search_selectors:
                if self.page.is_visible(selector):
                    found_selector = selector
                    break
                    
            if not found_selector:
                logger.error("Could not find search input field. Page may not have loaded properly.")
                logger.info(f"Current URL: {self.page.url}")
                # Take a screenshot for debugging
                self._save_screenshot("search_field_error")
                return False
                
            logger.info(f"Found search form using selector: {found_selector}")
            
            # Fill in the search query
            self.page.fill(found_selector, query)
            
            # Try different search button selectors
            search_button_selectors = [
                'button.search-submit',
                'button[type="submit"]',
                'button:has-text("Search")'
            ]
            
            # Click the first visible search button
            clicked = False
            for selector in search_button_selectors:
                if self.page.is_visible(selector):
                    logger.info(f"Clicking search button with selector: {selector}")
                    self.page.click(selector)
                    clicked = True
                    break
                    
            if not clicked:
                logger.error("Could not find search button. Search form may have changed.")
                # Take a screenshot for debugging
                self._save_screenshot("search_button_error")
                return False
            
            # Wait for results to load - try different selectors for results
            result_selectors = [
                '.searchResults',
                '.SearchResultsPage',
                'section[data-testid="search-results-section"]',
                'div[data-testid="search-results"]'
            ]
            
            # Wait for any of the result selectors to appear
            try:
                timeout = 2000  # 5 seconds timeout for each selector
                for i, selector in enumerate(result_selectors):
                    try:
                        if i == 0:
                            # First selector, use the full timeout
                            self.page.wait_for_selector(selector, timeout=timeout)
                            logger.info(f"Found search results with selector: {selector}")
                            break
                        else:
                            # For subsequent selectors, use a shorter timeout
                            self.page.wait_for_selector(selector, timeout=2000)
                            logger.info(f"Found search results with selector: {selector}")
                            break
                    except:
                        continue
            except:
                logger.warning("Could not detect standard search results element.")
                # Take a screenshot to help with debugging
                self._save_screenshot("search_results_error")
                
                # Check if we're still on a search-related page
                if "search" in self.page.url:
                    logger.info("We appear to be on a search page, continuing...")
                else:
                    logger.error(f"Unexpected URL after search: {self.page.url}")
                    return False
            
            # Check initial search results count - enhance with logging
            logger.info("Getting initial search results count")
            results_count = self._get_results_count()
            if results_count:
                logger.info(f"Initial search found: {results_count}")
            else:
                # If we couldn't find a count but we're on a search page,
                # it might be in an unexpected format - take a screenshot to help debug
                if "search" in self.page.url:
                    logger.warning("On search page but couldn't find document count")
                    self._save_screenshot("search_results_no_count")
            
            
            # IMPORTANT: Now that we have search results, apply date range filters if provided
            if year_from or year_to:
                logger.info("Initial search completed. Now applying date range filter...")
                date_filter_applied = self._apply_date_range_filter(year_from, year_to)
                if not date_filter_applied:
                    logger.warning("Failed to apply date range filter")
                else:
                    logger.info("Date range filter successfully applied")
                    # Take a screenshot after date filtering
                    self._save_screenshot("after_date_filtering")
                    # Get updated results count after filtering
                    self._get_results_count()
            
            # Apply other filters if provided
            if filters:
                logger.info("Applying additional filters...")
                self._apply_filters(filters)
                
            # Final check if we have results
            return "search" in self.page.url
                
        except Exception as e:
            logger.error(f"Search failed with error: {str(e)}")
            return False
    
    def apply_date_filter(self, year_from: Optional[int] = None, year_to: Optional[int] = None) -> bool:
        """
        Apply date range filter to current search results.
        This method is separate from search() to allow filtering after initial search.
        
        Args:
            year_from: Start year (optional)
            year_to: End year (optional)
            
        Returns:
            bool: True if filter was applied successfully, False otherwise
        """
        if not year_from and not year_to:
            logger.info("No date range specified, skipping date filter")
            return True
            
        logger.info(f"Applying date filter: {year_from if year_from else 'earliest'} to {year_to if year_to else 'latest'}")
        return self._apply_date_range_filter(year_from, year_to)

    def _get_results_count(self) -> Optional[str]:
        """
        Get the number of results from the search results page.
        
        Returns:
            The results count as a string, or None if not found
        """
        try:
            # Try to get the count from H2 element that contains "documents found" text
            h2_selectors = [
                'h2:has-text("documents found")',
                'h2.Typography-module__lVnit',
                'h2[class*="Typography"]'
            ]
            
            # Try H2 selectors first (most accurate)
            for selector in h2_selectors:
                if self.page.is_visible(selector):
                    text = self.page.text_content(selector)
                    logger.info(f"Found results count in H2: '{text}'")
                    return text
            
            # Fallback to other count selectors
            count_selectors = [
                '.resultsCount',
                '[data-testid="results-count"]',
                'span:has-text("results")',
                'div.results-count',
                'span:has-text("documents found")',
                'div:has-text("documents found")'
            ]
            
            for selector in count_selectors:
                if self.page.is_visible(selector):
                    results_count = self.page.text_content(selector)
                    logger.info(f"Search returned {results_count}")
                    return results_count
            
            # Look for any element containing "documents found" text as a last resort
            docs_found_text = self.page.evaluate('''
                () => {
                    const elements = document.querySelectorAll('*');
                    for (const el of elements) {
                        if (el.textContent && el.textContent.includes('documents found')) {
                            return el.textContent.trim();
                        }
                    }
                    return null;
                }
            ''')
            
            if docs_found_text:
                logger.info(f"Found results via JavaScript: {docs_found_text}")
                return docs_found_text
            
            logger.warning("Could not find results count")
            return None
        except Exception as e:
            logger.error(f"Error getting results count: {str(e)}")
            return None

    def _apply_date_range_filter(self, year_from: Optional[int] = None, year_to: Optional[int] = None) -> bool:
        """
        Apply date range filter to search results.
        
        Args:
            year_from: Start year (optional)
            year_to: End year (optional)
            
        Returns:
            bool: True if filter was applied successfully, False otherwise
        """
        try:
            logger.info(f"Applying date range filter: from {year_from if year_from else 'earliest'} "
                        f"to {year_to if year_to else 'latest'}")
            
                       
            # Wait a moment for all UI elements to load
            time.sleep(2)
            
            # Direct selectors for the date range inputs based on the provided HTML
            from_input = 'input[data-testid="input-range-from"]'
            to_input = 'input[data-testid="input-range-to"]'
            
            # Check if the inputs are visible directly without looking for a filter button first
            from_visible = self.page.is_visible(from_input)
            to_visible = self.page.is_visible(to_input)
            
            if from_visible and to_visible:
                logger.info("Found date range inputs directly")
            else:
                # If inputs aren't directly visible, try to find and click a filter button
                logger.info("Date range inputs not immediately visible, looking for filter button...")
                
                # Try various selectors for the date filter button
                date_filter_selectors = [
                    'button[data-testid="date-range-dropdown-button"]',
                    'button:has-text("Date Range")',
                    'button:has-text("Year")',
                    'button:has-text("Publication Date")',
                    '[aria-label="Refine by publication date"]',
                    'button.filter-button-date',
                    'button:has-text("Publication year")',
                    'div[aria-label="Publication year"]',
                    'div:has-text("Publication year")'
                ]
                
                filter_button_clicked = False
                for selector in date_filter_selectors:
                    if self.page.is_visible(selector):
                        logger.info(f"Found date filter button: {selector}")
                        self.page.click(selector)
                        filter_button_clicked = True
                        # Wait for filter dropdown to appear
                        time.sleep(1)
                        break
                
                if not filter_button_clicked:
                    logger.warning("Could not find date range filter button")
                    self._save_screenshot("date_filter_button_not_found")
                    
                    # Check again for inputs after screenshot
                    from_visible = self.page.is_visible(from_input)
                    to_visible = self.page.is_visible(to_input)
                    
                    if not (from_visible and to_visible):
                        logger.error("Could not find date range inputs")
                        return False
            
            # Fill in the date fields if they're provided
            if year_from:
                logger.info(f"Setting start year to {year_from}")
                # Clear the field first and then fill
                self.page.fill(from_input, "")
                self.page.fill(from_input, str(year_from))
            
            if year_to:
                logger.info(f"Setting end year to {year_to}")
                # Clear the field first and then fill
                self.page.fill(to_input, "")
                self.page.fill(to_input, str(year_to))
            
            # Look for the apply button with the correct data-testid
            apply_button = 'button[data-testid="apply-facet-range"]'
            
            if not self.page.is_visible(apply_button):
                # Try alternative apply button selectors
                logger.info("Primary apply button not found, trying alternatives")
                apply_button_selectors = [
                    'button[data-testid="apply-facet-range"]',
                    'button[data-testid="date-range-apply-button"]',
                    'button:has-text("Apply")',
                    'button:has-text("Apply selected year range")',
                    'button:has-text("Limit to")',
                    'button:has-text("Filter")',
                    'button.apply-button',
                    'button.limit-to-button'
                ]
                
                for selector in apply_button_selectors:
                    if self.page.is_visible(selector):
                        apply_button = selector
                        logger.info(f"Found apply button: {selector}")
                        break
            
            if not self.page.is_visible(apply_button):
                logger.error("Could not find apply button for date filter")
                self._save_screenshot("apply_button_not_found")
                return False
            
            # Click the apply button
            logger.info(f"Clicking apply button: {apply_button}")
            self.page.click(apply_button)
            
            # Wait for the filter to be applied and results updated
            time.sleep(3)
            self.page.wait_for_load_state('networkidle', timeout=10000)
                        
            logger.info("Date range filter applied successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error applying date range filter: {str(e)}")
            self._save_screenshot("date_filter_error")
            return False
    
    def _apply_filters(self, filters: Dict):
        """
        Apply filters to the search results.
        
        Args:
            filters: Dictionary of filters to apply
        """
        logger.info("Applying filters to search results")
        
        # Example implementation for date range filter
        if 'year_from' in filters and 'year_to' in filters:
            # Click on date range filter
            self.page.click('button[data-testid="date-range-dropdown-button"]')
            
            # Fill date range
            self.page.fill('input[data-testid="date-range-from-input"]', str(filters['year_from']))
            self.page.fill('input[data-testid="date-range-to-input"]', str(filters['year_to']))
            
            # Apply filter
            self.page.click('button[data-testid="date-range-apply-button"]')
            
            # Wait for results to update
            time.sleep(2)
        
        # Add more filter implementations as needed
    
    def export_to_csv(self, filename: Optional[str] = None) -> Optional[Path]:
        """
        Export search results to CSV.
        
        Args:
            filename: Optional filename for the downloaded CSV
            
        Returns:
            Path to the downloaded file or None if download failed
        """
        try:
            logger.info("Exporting search results to CSV")
            
            
            # First try the new UI export flow
            try:
                # Check if the "Select all" checkbox exists
                select_all_checkbox = self.page.query_selector('#bulkSelectDocument-primary-document-search-results-toolbar')
                
                if select_all_checkbox:
                    logger.info("Found new UI with Select All checkbox - using new export flow")
                    
                    # Click the Select All checkbox
                    self.page.click('#bulkSelectDocument-primary-document-search-results-toolbar')
                    logger.info("Clicked Select All checkbox")
                    
                    # Wait for the selection to apply
                    time.sleep(1)
                    
                    # First, look for the Export button specifically within the primary toolbar
                    export_dropdown_selectors = [
                        'div[data-testid="primary-toolbar"] button:has-text("Export")',
                        'div[data-testid="primary-toolbar"] [aria-label="Export"]',
                        'div[data-testid="primary-toolbar"] button[aria-expanded][aria-haspopup="menu"]',
                        'div[data-testid="primary-toolbar"] button.export-menu',
                        # Fallbacks if the primary-toolbar selector doesn't work
                        'button:has-text("Export")',
                        '[aria-label="Export"]',
                        'button[aria-expanded][aria-haspopup="menu"]',
                    ]
                    
                    dropdown_clicked = False
                    for selector in export_dropdown_selectors:
                        if self.page.is_visible(selector):
                            logger.info(f"Clicking export dropdown menu: {selector}")
                            self.page.click(selector)
                            dropdown_clicked = True
                            # Wait for dropdown to appear
                            time.sleep(1)
                            break
                    
                    if not dropdown_clicked:
                        logger.error("Could not find export dropdown button")
                        self._save_screenshot("export_dropdown_not_found")
                        raise Exception("Could not find export dropdown button")
                    
                    # Now look for the CSV option in the dropdown menu
                    csv_options = [
                        'button[data-testid="export-to-csv"]',
                        'a[data-testid="export-to-csv"]', 
                        'button:has-text("CSV")',
                        'a:has-text("CSV")',
                        'li:has-text("CSV")',
                        '[role="menuitem"]:has-text("CSV")'
                    ]
                    
                    csv_clicked = False
                    for selector in csv_options:
                        if self.page.is_visible(selector):
                            logger.info(f"Clicking CSV option: {selector}")
                            self.page.click(selector)
                            csv_clicked = True
                            break
                    
                    if not csv_clicked:
                        # Take a screenshot showing the open dropdown menu
                        self._save_screenshot("csv_option_not_found")
                        logger.error("Could not find CSV option in dropdown menu")
                        raise Exception("Could not find CSV export option in dropdown menu")
                    
                    # Wait for the export options dialog to appear
                    time.sleep(2)

                    
                    # Check the additional options as requested
                    # Check "Affiliations" checkbox
                    affiliations_checkbox = '#field_group_affiliations'
                    try:
                        logger.info("Checking 'Affiliations' option")
                        # Only check if not already checked
                        if not self.page.is_checked(affiliations_checkbox):
                            self.page.check(affiliations_checkbox)
                    except:
                        logger.warning("Could not check Affiliations checkbox")
                        
                    # Check "Publisher" checkbox
                    publisher_checkbox = '#field_group_publisher'
                    try:
                        logger.info("Checking 'Publisher' option")
                        # Only check if not already checked
                        if not self.page.is_checked(publisher_checkbox):
                            self.page.check(publisher_checkbox)
                    except:
                        logger.warning("Could not check Publisher checkbox")
                        
                    # Check "Abstract & keywords" checkbox
                    abstract_keywords_selectors = [
                        'label:has-text("Abstract & keywords") input',
                        'input[aria-controls="field_group_abstact field_group_authorKeywords field_group_indexedKeywords"]'
                    ]
                    
                    abstract_checked = False
                    for selector in abstract_keywords_selectors:
                        try:
                            if self.page.is_visible(selector):
                                logger.info("Checking 'Abstract & keywords' option")
                                # Only check if not already checked
                                if not self.page.is_checked(selector):
                                    self.page.check(selector)
                                abstract_checked = True
                                break
                        except:
                            continue
                    
                    if not abstract_checked:
                        logger.warning("Could not check Abstract & keywords checkbox")
                        
        
                    
                    # Finally click the Export/Submit button
                    submit_selectors = [
                        'button[data-testid="submit-export-button"]',
                        'button:has-text("Export")',
                        'div:has-text("Export") > button[type="button"]'
                    ]
                    
                    submit_clicked = False
                    with self.page.expect_download() as download_info:
                        for selector in submit_selectors:
                            if self.page.is_visible(selector):
                                logger.info(f"Clicking submit export button: {selector}")
                                self.page.click(selector)
                                submit_clicked = True
                                break
                        
                        if not submit_clicked:
                            logger.error("Could not find submit export button")
                            self._save_screenshot("submit_button_not_found")
                            raise Exception("Could not find submit export button")
                    
                    # Get the download info
                    download = download_info.value
                    
                else:
                    # Fallback to old UI export flow
                    raise Exception("Select All checkbox not found - trying old UI flow")
            
            except Exception as e:
                logger.info(f"New UI export failed: {str(e)}. Trying classic UI export flow...")
                
                # Classic export flow
                try:
                    # Different possible export buttons
                    export_buttons = [
                        'button[title="Export"]',
                        'a[title="Export"]',
                        'button:has-text("Export")',
                        '[aria-label="Export"]'
                    ]
                    
                    # Click the first visible export button
                    export_clicked = False
                    for selector in export_buttons:
                        if self.page.is_visible(selector):
                            logger.info(f"Clicking export button: {selector}")
                            self.page.click(selector)
                            export_clicked = True
                            break
                    
                    if not export_clicked:
                        logger.error("Could not find export button")
                        self._save_screenshot("export_button_not_found")
                        return None
                    
                    # Wait for export options to appear
                    time.sleep(1)
                    
                    # Select CSV option - try different selectors
                    csv_options = [
                        'input[value="CSV"]',
                        'button:has-text("CSV")',
                        'a:has-text("CSV")'
                    ]
                    
                    csv_clicked = False
                    for selector in csv_options:
                        if self.page.is_visible(selector):
                            logger.info(f"Selecting CSV option: {selector}")
                            self.page.click(selector)
                            csv_clicked = True
                            break
                    
                    if not csv_clicked:
                        logger.error("Could not find CSV option")
                        self._save_screenshot("csv_option_not_found")
                        return None
                    
                    # Attempt to select "All information" if it exists
                    try:
                        if self.page.is_visible('input[value="ALL"]'):
                            logger.info("Selecting ALL information option")
                            self.page.click('input[value="ALL"]')
                    except:
                        logger.info("ALL information option not found, continuing...")
                    
                    # Click export/download/submit button - try different selectors
                    submit_buttons = [
                        'button.btnSubmit',
                        'button:has-text("Export")',
                        'button:has-text("Download")',
                        'button[type="submit"]'
                    ]
                    
                    with self.page.expect_download() as download_info:
                        submit_clicked = False
                        for selector in submit_buttons:
                            if self.page.is_visible(selector):
                                logger.info(f"Clicking submit button: {selector}")
                                self.page.click(selector)
                                submit_clicked = True
                                break
                        
                        if not submit_clicked:
                            logger.error("Could not find submit button")
                            self._save_screenshot("submit_button_not_found")
                            return None
                    
                    # Get download info
                    download = download_info.value
                    
                except Exception as inner_e:
                    logger.error(f"Classic UI export failed: {str(inner_e)}")
                    self._save_screenshot("export_failure")
                    return None
            
            # Save the file with the specified name or use default
            if filename:
                # Clean filename to remove invalid characters
                safe_filename = self._sanitize_filename(filename)
                target_path = self.download_dir / f"{safe_filename}.csv"
            else:
                # Use timestamp if no filename is provided
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                target_path = self.download_dir / f"scopus_export_{timestamp}.csv"
            
            download.save_as(target_path)
            logger.info(f"CSV exported successfully to {target_path}")
            
            return target_path
            
        except Exception as e:
            logger.error(f"CSV export failed with error: {str(e)}")
            self._save_screenshot("export_exception")
            return None

    def _sanitize_filename(self, filename: str) -> str:
        """
        Sanitize the filename by removing/replacing characters that are invalid in Windows filenames.
        
        Args:
            filename: The original filename to sanitize
            
        Returns:
            A sanitized filename
        """
        # Define characters that are not allowed in Windows filenames
        invalid_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']
        
        # Replace invalid characters with underscores
        safe_filename = filename
        for char in invalid_chars:
            safe_filename = safe_filename.replace(char, '_')
        
        # Also replace parentheses for safety
        safe_filename = safe_filename.replace('(', '_').replace(')', '_')
        
        # Collapse multiple underscores into one
        while '__' in safe_filename:
            safe_filename = safe_filename.replace('__', '_')
        
        # Trim underscores from start and end
        safe_filename = safe_filename.strip('_')
        
        # Ensure filename isn't too long - Windows limit is 260 total path length
        # Keep it reasonable at 100 chars max
        if len(safe_filename) > 100:
            safe_filename = safe_filename[:97] + '...'
        
        logger.info(f"Sanitized filename: '{filename}' -> '{safe_filename}'")
        return safe_filename

    def perform_search_and_export(self, query: str, filters: Optional[Dict] = None, 
                                  filename: Optional[str] = None) -> Optional[Path]:
        """
        Convenience method to perform search and export in one step.
        
        Args:
            query: Search query string
            filters: Dictionary of filters to apply
            filename: Optional filename for the downloaded CSV
            
        Returns:
            Path to the downloaded file or None if any step failed
        """
        if self.search(query, filters):
            return self.export_to_csv(filename)
        return None
