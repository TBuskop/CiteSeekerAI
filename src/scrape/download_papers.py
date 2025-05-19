#!/usr/bin/env python
import os
import re
import json
import time
import concurrent.futures
import xml.etree.ElementTree as ET # <<< NEW: Import ElementTree
from pathlib import Path # <<< NEW: Import Path
from typing import List, Dict, Optional, Tuple # <<< NEW: Import typing hints
import random # <<< NEW: Import random for header rotation
import sys
import httpx
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright, Cookie, TimeoutError as PlaywrightTimeoutError, Page  # Added Playwright TimeoutError import, Page, Download
from playwright_stealth import stealth_sync
import trafilatura


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, os.pardir, os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import config

# <<< NEW: Import from extract_text >>>
from src.rag.extract_text import extract_text_from_pdf as extract_pdf_text_from_rag

# --- Playwright PDF Download Function (for cookie-preserving PDF downloads) ---
from typing import Optional  # ensure Optional is available at top



# change run path to current file
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

DYNAMIC_URL_PATTERNS: List[str] = [
    # Elsevier ScienceDirect
    r"https://(.*\.)?sciencedirect\.com/.*",
    r"https://(.*\.)?elsevier\.com/.*",
    r"https://validate\.perfdrive\.com/.*", # Radware Bot Manager validation page
    r"https://(.*\.)?onlinelibrary\.wiley\.com/.*",    #onlinelibrary.wiley
    r"https://(.*\.)?ascelibrary\.org/.*", # ascelibrary.org
    r"https://(.*\.)?mdpi\.com/.*", # MDPI
    r"https://(.*\.)?iwaponline\.com/.*", # iwaponline.com
    r"https://(.*\.)?iopscience\.iop\.org/.*", # iopscience
    r"perfdrive", # iopscience.com validation page,
    r"annualreviews\.org/.*", # annualreviews.org,
    r"ametsoc\.org/.*", # ametsoc.org,
    r"https://(.*\.)?pubs\.acs\.org/.*", # pubs.acs
]

# --- Helper function to decide on Playwright usage ---
def _url_needs_playwright(url: str) -> bool:
    if not url:
        return False
    for pattern in DYNAMIC_URL_PATTERNS:
        try:
            if re.search(pattern, url, re.IGNORECASE):
                return True
        except re.error:
            continue
    return False

ROTATING_HEADERS_LIST: List[Dict[str, str]] = [
    {
        # Chrome on Windows
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36',
        'Sec-Ch-Ua': '"Google Chrome";v="125", "Chromium";v="125", "Not.A/Brand";v="24"',
        'Sec-Ch-Ua-Mobile': '?0',
        'Sec-Ch-Ua-Platform': '"Windows"',
        'Upgrade-Insecure-Requests': '1',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
        'Sec-Fetch-Site': 'cross-site', # Assuming navigation from another site (e.g., Google)
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-User': '?1',
        'Sec-Fetch-Dest': 'document',
        'Accept-Encoding': 'gzip, deflate, br, zstd',
        'Accept-Language': 'en-US,en;q=0.9,nl;q=0.8',
        'Referer': 'https://www.google.com/',
        'DNT': '1', # Do Not Track
        'Connection': 'keep-alive',
        'Cache-Control': 'max-age=0', # Simulates a refresh or fresh navigation
    },
    {
        # Safari on macOS
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4.1 Safari/605.1.15',
        # Safari doesn't send Sec-CH-UA headers in the same way as Chromium
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Upgrade-Insecure-Requests': '1', # Still common though Safari is good with HTTPS
        'Sec-Fetch-Site': 'cross-site',
        'Sec-Fetch-Mode': 'navigate',
        # 'Sec-Fetch-User': '?1', # Safari might not always send this for simple navigations
        'Sec-Fetch-Dest': 'document',
        'Accept-Encoding': 'gzip, deflate, br', # Safari added br support
        'Accept-Language': 'en-GB,en;q=0.9',
        'Referer': 'https://duckduckgo.com/',
        'DNT': '1',
        'Connection': 'keep-alive',
        # 'Cache-Control': 'max-age=0', # Less consistently sent by Safari on initial nav compared to Chrome
    },
    {
        # Firefox on Linux
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:126.0) Gecko/20100101 Firefox/126.0',
        # Firefox doesn't send Sec-CH-UA headers
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
        'Upgrade-Insecure-Requests': '1',
        'Sec-Fetch-Site': 'cross-site',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-User': '?1',
        'Sec-Fetch-Dest': 'document',
        'Accept-Encoding': 'gzip, deflate, br, zstd',
        'Accept-Language': 'de-DE,de;q=0.9,en-US;q=0.8,en;q=0.7',
        'Referer': 'https://www.bing.com/',
        'DNT': '1',
        'Connection': 'keep-alive',
        'TE': 'trailers', # Often sent by Firefox
        'Cache-Control': 'max-age=0',
    },
    {
        # Edge on Windows
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36 Edg/125.0.0.0',
        'Sec-Ch-Ua': '"Microsoft Edge";v="125", "Chromium";v="125", "Not.A/Brand";v="24"',
        'Sec-Ch-Ua-Mobile': '?0',
        'Sec-Ch-Ua-Platform': '"Windows"',
        'Sec-Ch-Ua-Platform-Version': '"15.0.0"', # Example, often reflects Windows major build
        'Upgrade-Insecure-Requests': '1',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
        'Sec-Fetch-Site': 'cross-site',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-User': '?1',
        'Sec-Fetch-Dest': 'document',
        'Accept-Encoding': 'gzip, deflate, br, zstd',
        'Accept-Language': 'es-ES,es;q=0.9,en;q=0.8',
        'Referer': 'https://search.yahoo.com/',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Cache-Control': 'max-age=0',
    },
    {
        # Chrome on Android (Example of a mobile User-Agent)
        'User-Agent': 'Mozilla/5.0 (Linux; Android 13; Pixel 7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Mobile Safari/537.36',
        'Sec-Ch-Ua': '"Google Chrome";v="125", "Chromium";v="125", "Not.A/Brand";v="24"',
        'Sec-Ch-Ua-Mobile': '?1', # Mobile
        'Sec-Ch-Ua-Platform': '"Android"',
        'Sec-Ch-Ua-Platform-Version': '"13.0.0"',
        'Sec-Ch-Ua-Model': '"Pixel 7"',
        'Upgrade-Insecure-Requests': '1',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
        'Sec-Fetch-Site': 'cross-site',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-User': '?1',
        'Sec-Fetch-Dest': 'document',
        'Accept-Encoding': 'gzip, deflate, br, zstd',
        'Accept-Language': 'en-US,en;q=0.9',
        'Referer': 'https://www.google.com/',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Cache-Control': 'max-age=0',
    }
]

# <<< NEW: Helper function to get random headers >>>
def get_random_headers() -> Dict[str, str]:
    """Returns a randomly selected header dictionary."""
    return random.choice(ROTATING_HEADERS_LIST)

# --- Helper function to fetch URL metadata via HEAD request ---
def _get_url_metadata(url: str, headers: Dict[str, str]) -> Optional[Dict[str, str]]:
    try:
        head_resp = httpx.head(url, follow_redirects=True, timeout=20, headers=headers)
        final_url = str(head_resp.url)
        # preserve perfdrive fragment
        if "perfdrive" in final_url:
            final_url = url + "#perfdrive"
        content_type = head_resp.headers.get("content-type", "").lower()
        return {"final_url": final_url, "content_type": content_type}
    except Exception as e:
        print(f"Error fetching URL metadata for {url}: {e}")
        return None

###############################
# Headless Browser with Stealth
###############################
####################################
# <<< NEW: Copernicus URL Handling >>>
####################################

def construct_copernicus_url(html_url: str, file_extension: str = ".xml") -> Optional[str]:
    """
    Attempts to construct the Copernicus download URL (XML or PDF) from the article HTML URL.
    Example HTML URL: https://hess.copernicus.org/articles/19/1521/2015/
    Example XML URL:  https://hess.copernicus.org/articles/19/1521/2015/hess-19-1521-2015.xml
    Example PDF URL:  https://hess.copernicus.org/articles/19/1521/2015/hess-19-1521-2015.pdf

    Args:
        html_url: The URL of the HTML article page.
        file_extension: The desired file extension (e.g., ".xml" or ".pdf").

    Returns:
        The constructed download URL or None if the pattern doesn't match.
    """
    # Regex to capture the necessary parts: subdomain (journal), volume, page, year
    match = re.match(r"https://(?P<journal>[^.]+)\.copernicus\.org/articles/(?P<vol>\d+)/(?P<page>\d+)/(?P<year>\d+)/?", html_url)
    if match:
        parts = match.groupdict()
        # Construct the base path
        base_path = f"https://{parts['journal']}.copernicus.org/articles/{parts['vol']}/{parts['page']}/{parts['year']}/"
        # Construct the filename part using the provided extension
        filename = f"{parts['journal']}-{parts['vol']}-{parts['page']}-{parts['year']}{file_extension}"
        return base_path + filename
    return None

def extract_text_from_xml(xml_string: str) -> str:
    """
    Parses XML string, finds the <body> element, and extracts all text
    content within it, attempting to preserve paragraph structure by adding
    newlines after block elements like <p> and <sec>.
    """
    if not xml_string:
        return ""
    try:
        # Remove <?xmltex ...?> processing instructions which can confuse ET
        xml_string_cleaned = re.sub(r'<\?xmltex.*?\?>', '', xml_string)
        root = ET.fromstring(xml_string_cleaned)
        body_element = root.find('body') # Find the <body> tag directly under the root

        if body_element is None:
            print("Warning: <body> tag not found in the XML.")
            return ""

        text_parts = []

        # Recursive function to extract text *only within the body element*
        def get_text_recursive(element):
            # Add text before the first child (strip whitespace)
            if element.text:
                cleaned_text = element.text.strip()
                if cleaned_text: # Only add if not just whitespace
                    text_parts.append(cleaned_text)

            # Process children recursively
            for child in element:
                if child.tag not in {'fig', 'table-wrap', 'disp-formula', 'inline-formula'}:
                    get_text_recursive(child)

                # Add text that comes *after* the child tag (tail)
                if child.tail:
                    cleaned_tail = child.tail.strip()
                    if cleaned_tail: # Only add if not just whitespace
                        text_parts.append(cleaned_tail)

            # Add paragraph breaks after certain block elements if text was added
            if element.tag in {'p', 'sec', 'title'} and text_parts:
                if text_parts[-1] != '\n\n':
                     text_parts.append('\n\n')

        # Start recursion from the found body element
        get_text_recursive(body_element)

        # Join parts intelligently:
        full_text = ' '.join(text_parts)
        full_text = re.sub(r'\s*\n\n\s*', '\n\n', full_text).strip() # Consolidate newlines
        full_text = re.sub(r'[ \t]{2,}', ' ', full_text)            # Consolidate spaces/tabs
        full_text = re.sub(r'(\n\n)+', '\n\n', full_text)           # Ensure max 2 newlines

        return full_text

    except ET.ParseError as e:
        print(f"Error parsing XML: {e}")
        return ""
    except Exception as e:
        import traceback
        print(f"Unexpected error during XML body text extraction: {e}\n{traceback.format_exc()}")
        return ""
    
def get_spoof_cookies(domain: str) -> list[Cookie]:
    """
    Returns a list of common spoofed cookies for the specified domain.
    """
    return [
        Cookie(name="sessionid", value="abcdef1234567890", domain=domain, path="/"),
        Cookie(name="csrftoken", value="XYZ987ABC654", domain=domain, path="/"),
        Cookie(name="CookieConsent", value="yes", domain=domain, path="/"),
        Cookie(name="_ga", value="GA1.2.1234567890.1616161616", domain=domain, path="/"),
        Cookie(name="_gid", value="GA1.2.0987654321.1717171717", domain=domain, path="/"),
    ]

def fetch_with_playwright(url: str) -> str:
    """
    Loads *url* in a stealth‑configured Playwright browser,
    persisting IOP Science / perfdrive cookies in ``iop.json``.
    Determines headless mode based on IOP cookie file status.
    Attempts PDF download via Playwright if HTML extraction fails for specific sites (e.g., Wiley).
    Returns the rendered HTML or extracted PDF text.
    """
    html_str = ""
    pdf_text = "" # <<< NEW: Variable to store potential PDF text
    source = "html_playwright" # <<< NEW: Default source
    browser = None
    context = None
    page: Optional[Page] = None # <<< NEW: Type hint for page

    # <<< NEW: Get random headers for this session >>>
    random_headers = get_random_headers()
    user_agent_to_use = random_headers.get('User-Agent', # Default if not found
                                         "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                         "AppleWebKit/537.36 (KHTML, like Gecko) "
                                         "Chrome/124.0.0.0 Safari/537.36")


    try:
        with sync_playwright() as p:
            # ───────────────────────────────────────────────────────────────
            # 1. Determine IOP state needs and headless mode
            # ───────────────────────────────────────────────────────────────
            needs_iop_state = any(x in url for x in ("iopscience", "perfdrive"))
            storage_kwarg = {}
            headless = True # Default to headless

            if needs_iop_state:
                if os.path.exists(COOKIE_FILE):
                    try:
                        # Check if the file is valid JSON
                        with open(COOKIE_FILE, 'r') as f:
                            json.load(f)
                        storage_kwarg["storage_state"] = str(COOKIE_FILE)
                        print(f"IOP state file found and seems valid: {COOKIE_FILE}. Will load state and run headless.")
                        headless = True # Run headless if state file is valid
                    except (json.JSONDecodeError, OSError) as e:
                        print(f"IOP state file {COOKIE_FILE} exists but is invalid ({e}). Proceeding without state, running non-headless.")
                        headless = False # Run non-headless if state file is invalid
                else:
                    print(f"IOP state file not found: {COOKIE_FILE}. Proceeding without state, running non-headless.")
                    headless = False # Run non-headless if state file is missing

            # ───────────────────────────────────────────────────────────────
            # 2. Launch Browser
            # ───────────────────────────────────────────────────────────────
            print(f"Launching Chromium (headless={headless})...")
            browser = p.chromium.launch(
                headless=headless, # <<< CHANGED: Use determined headless value
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--disable-infobars",
                ],
            )

            # ───────────────────────────────────────────────────────────────
            # 3. Create Context (with or without storage_state)
            # ───────────────────────────────────────────────────────────────
            print("Creating browser context...")
            context = browser.new_context(
                user_agent=user_agent_to_use, # <<< CHANGED: Use selected User-Agent
                locale="nl-NL",
                timezone_id="Europe/Amsterdam",
                viewport={"width": 1280, "height": 800},
                geolocation={"latitude": 52.1326, "longitude": 5.2913},
                permissions=["geolocation"],
                **storage_kwarg, # Pass storage_state if set,
                accept_downloads=True
            )
            if storage_kwarg:
                 print(f"Loaded stored IOP state from {COOKIE_FILE}")
            print("Context created.")


            # ───────────────────────────────────────────────────────────────
            # 4. Add Spoof Cookies if State Wasn't Loaded
            # ───────────────────────────────────────────────────────────────
            if not storage_kwarg:
                # Use the original URL for domain extraction before potential modification
                domain_url = url.replace("#perfdrive", "") # Ensure we get the base domain
                domain = domain_url.split("/")[2]
                print(f"Adding spoof cookies for domain: {domain}")
                context.add_cookies(get_spoof_cookies(domain))

            # ───────────────────────────────────────────────────────────────
            # 5. Create Page, Apply Stealth, Navigate, Extract
            # ───────────────────────────────────────────────────────────────
            page = context.new_page()
            print("Applying stealth...")

            # <<< NEW: Set extra HTTP headers for the page >>>
            # Remove User-Agent from random_headers if it's there, as it's set at context level
            # and Playwright might warn or error if set in both places.
            # However, Playwright's set_extra_http_headers typically overrides or adds,
            # and User-Agent is special. It's safer to let context handle UA.
            # For other headers, this is the place.
            headers_for_page = {k: v for k, v in random_headers.items() if k.lower() != 'user-agent'}
            if headers_for_page:
                print(f"Setting extra HTTP headers for the page: {list(headers_for_page.keys())}")
                page.set_extra_http_headers(headers_for_page)

            stealth_sync(page)



            # Pick the right body selector and adjust URL if needed
            effective_url = url # Start with the original URL
            pdf_download_url: Optional[str] = None # <<< NEW: Initialize pdf_download_url

            if "iopscience" in url or "perfdrive" in url:
                text_selector = 'div[itemprop="articleBody"]'
                # Modify the URL *after* selector logic if needed
                url = url.replace("#perfdrive", "")
            elif "elsevier" in url or "sciencedirect" in url:
                text_selector = "div.Body"
            elif "wiley" in url:
                text_selector = "section.article-section__full"
            elif "ascelibrary" in url:
                text_selector = "section#bodymatter"
            elif "mdpi" in url:
                text_selector = "div.html-body"
            elif "iwaponline" in url:
                text_selector = "div.article-body"
            elif "tandfonline" in url:
                text_selector = "article.article"
            elif "annualreviews" in url:
                text_selector = "div.itemFullTextHtml"
            elif "ametsoc" in url:
                text_selector = "div#articleBody"
            elif "pubs.acs" in url:
                text_selector = "div.article_content"

            print(f"Fetching URL via Playwright (headless={headless}): {effective_url}")
            
            print(f"Navigating to {effective_url}...")
            # Main navigation call. Wait for DOM content to be loaded.
            # Set timeout to 60s.
            page.goto(effective_url, timeout=60000, wait_until="domcontentloaded") 
            print(f"Initial navigation to {effective_url} complete (DOM loaded).")

            # Wait for JavaScript challenges to resolve and network activity to settle.
            # This is crucial for pages that load content dynamically or run anti-bot scripts.
            print("Waiting for network activity to become idle (up to 15s) to allow JS challenges to complete...")
            try:
                page.wait_for_load_state("networkidle", timeout=15000)
                print("Network is now idle.")
            except PlaywrightTimeoutError:
                print("Network did not become idle within 15s. This might happen on pages with continuous background activity or if a challenge is blocking. Proceeding with content extraction.")
            except Exception as e: # Catch other potential errors during wait
                print(f"An error occurred while waiting for network idle: {e}. Proceeding.")

            # reload the page to ensure all content is loaded
            print("Reloading page, to overcome any potential lazy loading issues...")
            page.reload(timeout=60000, wait_until="domcontentloaded")
            # An additional short, fixed wait can sometimes be beneficial,
            # especially if running non-headless to allow for any final rendering or manual checks/CAPTCHAs.
            final_wait_duration = 5 if not headless else 2
            print(f"Performing a final wait of {final_wait_duration}s...")
            time.sleep(final_wait_duration)

            try:
                print(f"Attempting to extract text using selector: '{text_selector}'")
                html_str = (
                    f"<html><body>{page.locator(text_selector).inner_text()}</body></html>"
                )
                print("Successfully extracted text content via selector.")
                source = "html_playwright_selector"
            except Exception as e:
                    print("Selector extraction failed. Attempting to extract full HTML...")
                    print(f"Error: {e}")
                    html_str = page.content()
                    source = "html_playwright_full_content"


            # ───────────────────────────────────────────────────────────────
            # 6. Persist cookies back to disk if this was an IOP visit
            # ───────────────────────────────────────────────────────────────
            if needs_iop_state:
                print(f"Saving updated state to {COOKIE_FILE}...")
                # check if cookies folder exists, if not create it
                cookies_folder = Path(COOKIE_FILE).parent
                if not cookies_folder.exists():
                    cookies_folder.mkdir(parents=True, exist_ok=True)
                    print(f"Created cookies folder: {cookies_folder}")
                context.storage_state(path=str(COOKIE_FILE)) # Use str() for path object
                print(f"Successfully saved state to {COOKIE_FILE}")

            # ───────────────────────────────────────────────────────────────
            # 7. Cleanup (within try block before exception handling)
            # ───────────────────────────────────────────────────────────────
            print("Closing context and browser...")
            context.close()
            context = None # Prevent use in finally if closed successfully
            browser.close()
            browser = None # Prevent use in finally if closed successfully

    except Exception as e:
        print(f"Playwright error for {url}: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging
    finally:
        # Ensure resources are closed even if errors occurred mid-process
        if context:
            print("Closing context due to error or incomplete exit...")
            context.close()
        if browser:
            print("Closing browser due to error or incomplete exit...")
            browser.close()

    # <<< NEW: Return PDF text if extracted, otherwise HTML string >>>
    return html_str


#############################################
# Helper Function: Meta Refresh Extraction
#############################################

def extract_meta_redirect(html: str, base_url: str) -> Optional[str]:
    """
    Parses the HTML for a meta refresh tag and extracts the redirect URL.
    If the URL is relative, it resolves it against the given base URL.
    Returns None if no redirect is found.
    """
    if not html: # Added check for empty HTML
        return None
    try:
        soup = BeautifulSoup(html, "html.parser")
        meta = soup.find("meta", attrs={"http-equiv": re.compile(r"refresh", re.I)})
        if meta:
            content = meta.get("content", "")
            m = re.search(r'url\s*=\s*([^"]+)', content, re.IGNORECASE)
            if m:
                redirect_path = m.group(1).strip().strip("'\"") # Remove quotes
                if not redirect_path or redirect_path.lower() == 'about:blank':
                    return None
                if not redirect_path.lower().startswith(("http://", "https://")):
                    try:
                        base = httpx.URL(base_url)
                        redirect_url = base.join(redirect_path)
                        return str(redirect_url)
                    except Exception as e:
                        print(f"Error resolving relative redirect '{redirect_path}' from base '{base_url}': {e}")
                        return None
                else:
                    return redirect_path
    except Exception as e:
        print(f"Error parsing HTML for meta redirect in URL {base_url}: {e}")
    return None

####################################
# PDF Extraction with PyMuPDF
####################################
def remove_references_section(text):
    """
    Detects references-related section headers and removes text starting from the last one
    ONLY IF it appears after the first 1500 characters.
    """
    pattern = r'^\s*(References|Bibliography|Acknowledgement|Acknowledgements|\*\*References Used:\*\*)\s*$'
    matches = list(re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE))

    if matches:
        print(f"[*] Found {len(matches)} references-related section header(s).")

        for i, match in enumerate(matches):
            print(f"  [{i+1}] Found '{match.group().strip()}' at index {match.start()}")

        last_match = matches[-1]
        last_index = last_match.start()

        if last_index > 1500:
            print(f"[*] Removing text from last header '{last_match.group().strip()}' at index {last_index}.")
            return text[:last_index].strip()
        else:
            print(f"[*] Last header found at index {last_index} is within the first 1500 characters. No text removed.")
            return text
    else:
        print("[*] No references section header found. No text removed.")
        return text

def _handle_pdf_url(url: str, output_directory, headers: Dict[str, str]) -> Dict[str, str]:
    """Downloads PDF from URL and extracts text using httpx (default) or Playwright."""
    pdf_bytes = None
    source_prefix = "pdf_httpx" # Default source

    # --- Existing httpx download logic ---
    print(f"[*] Attempting PDF download via httpx for: {url}")
    source_prefix = "pdf_httpx"
    try:
        with httpx.stream("GET", url, headers=headers, follow_redirects=True, timeout=40.0) as response:
            response.raise_for_status()
            # Check content type again just to be sure
            content_type = response.headers.get("content-type", "").lower()
            if "application/pdf" not in content_type:
                print(f"Warning: Expected PDF but got Content-Type '{content_type}' from {url}. Attempting to process anyway.")

            pdf_bytes_list = []
            for chunk in response.iter_bytes():
                pdf_bytes_list.append(chunk)
            pdf_bytes = b"".join(pdf_bytes_list)
            print(f"PDF downloaded successfully via httpx from: {url} ({len(pdf_bytes)} bytes)")

    except httpx.HTTPStatusError as e:
        print(f"HTTP Status Error {e.response.status_code} downloading PDF from {url}: {e}")
    except httpx.RequestError as e:
        print(f"Request Error downloading PDF from {url}: {e}")
    except Exception as e:
        print(f"General Error downloading PDF via httpx from {url}: {e}")
    # --- End httpx download logic ---

    # Process downloaded bytes (common logic for both methods)
    if pdf_bytes:
        # create a temp folder in the output directory if it doesn't exist
        
        temp_dir = Path(output_directory) / "temp" # <<< NEW: Use output directory for temp files
        temp_dir.mkdir(parents=True, exist_ok=True) # Create temp directory if it doesn't exist
        # Use the temp path if created by Playwright, otherwise create one
        temp_file_path = temp_dir / f"temp_extracted_pdf_{int(time.time())}.pdf"
        try:
            # If using httpx, write the bytes to the temp file
            with open(temp_file_path, "wb") as f:
                f.write(pdf_bytes)

            pdf_text = extract_pdf_text_from_rag(temp_file_path)
            if pdf_text:
                print(f"[*] Successfully extracted text from PDF ({source_prefix}): {url}")
                return {"text": pdf_text, "source": source_prefix}
            else:
                print(f"[!] Failed to extract text from PDF ({source_prefix}): {url}")
                return {"text": "", "source": f"{source_prefix}_extract_failed"}
        except Exception as e:
            print(f"[!] Error processing PDF after download ({source_prefix}): {e}")
            return {"text": "", "source": f"{source_prefix}_processing_error"}
        finally:
            # Clean up the temp file used for extraction
            if temp_file_path.exists():
                 try:
                    temp_file_path.unlink()
                    print(f"[*] Removed temporary PDF: {temp_file_path}")
                 except OSError as e_unlink:
                    print(f"[!] Error removing temporary PDF {temp_file_path}: {e_unlink}")
    else:
        # This case means download failed (either httpx or Playwright)
        print(f"[!] PDF download failed for {url} using")
        return {"text": "", "source": f"{source_prefix}_download_failed"}


def _handle_copernicus_xml(html_url: str, headers: Dict[str, str]) -> Optional[Dict[str, str]]:
    """
    Attempts to fetch and parse Copernicus XML from a corresponding HTML URL.
    Returns extracted text and source if successful, None otherwise (indicating fallback).
    """
    print(f"Detected Copernicus article page: {html_url}. Attempting XML fetch.")
    # <<< CHANGED: Use new function name and specify .xml extension >>>
    xml_url = construct_copernicus_url(html_url, file_extension=".xml")
    if not xml_url:
        print(f"Could not construct Copernicus XML URL from {html_url}.")
        return None # Signal to fallback to HTML

    print(f"Constructed XML URL: {xml_url}")
    try:
        xml_resp = httpx.get(xml_url, timeout=30, follow_redirects=True, headers=headers)
        if xml_resp.status_code == 404:
            print(f"Copernicus XML not found (404) at: {xml_url}. Falling back to HTML.")
            return None # Signal fallback
        xml_resp.raise_for_status() # Raise other errors

        print(f"Successfully downloaded XML from {xml_url}")
        xml_content = xml_resp.content.decode(xml_resp.encoding or 'utf-8', errors='replace')
        xml_text = extract_text_from_xml(xml_content)

        if xml_text and len(xml_text) > 100:
            print("Successfully extracted text from Copernicus XML.")
            return {"text": xml_text, "source": "copernicus_xml"}
        else:
            print("Failed to extract sufficient text from Copernicus XML. Falling back to PDF download.")
            return None # Signal fallback

    except httpx.HTTPStatusError as e:
        print(f"HTTP Status Error {e.response.status_code} fetching Copernicus XML from {xml_url}. Falling back to HTML. Error: {e}")
        return None # Signal fallback
    except httpx.RequestError as e:
        print(f"Request Error fetching Copernicus XML from {xml_url}. Falling back to HTML. Error: {e}")
        return None # Signal fallback
    except Exception as e:
        print(f"General Error during Copernicus XML fetch/parse for {xml_url}. Falling back to HTML. Error: {e}")
        return None # Signal fallback


def _fetch_html_content(url: str, headers: Dict[str, str], is_copernicus_fallback: bool = False) -> Dict[str, str]:
    """
    Fetches HTML content, deciding between httpx/Playwright and handling meta-redirects.
    Returns a dict with 'html_str' and 'source_method'.
    """
    html_str = ""
    source_method = ""
    current_url_to_fetch = url

    use_playwright = _url_needs_playwright(current_url_to_fetch)

    if use_playwright:
        print(f"Fetching HTML (dynamic) via Playwright: {current_url_to_fetch}")
        source_method = "html_playwright"
        if is_copernicus_fallback: source_method = "copernicus_html_playwright"
        html_str = fetch_with_playwright(current_url_to_fetch)
    else:
        print(f"Fetching HTML (static) via httpx: {current_url_to_fetch}")
        source_method = "html_httpx"
        if is_copernicus_fallback: source_method = "copernicus_html_httpx"
        try:
            r = httpx.get(current_url_to_fetch, timeout=20, follow_redirects=True, headers=headers)
            r.raise_for_status()
            current_url_to_fetch = str(r.url) # Update after redirects
            html_str = r.content.decode(r.encoding or 'utf-8', errors='replace')

            # Check for meta refresh AFTER httpx GET
            redirect_url = extract_meta_redirect(html_str, current_url_to_fetch)
            if redirect_url and redirect_url != current_url_to_fetch:
                print(f"Found meta refresh redirect to {redirect_url}. Following...")
                use_playwright_redirect = _url_needs_playwright(redirect_url)
                if use_playwright_redirect:
                    print(f"Fetching redirected URL (dynamic) via Playwright: {redirect_url}")
                    source_method = "html_playwright_redirect"
                    html_str = fetch_with_playwright(redirect_url)
                    current_url_to_fetch = redirect_url # Update URL context
                else:
                    print(f"Fetching redirected URL (static) via httpx: {redirect_url}")
                    source_method = "html_httpx_redirect"
                    r2 = httpx.get(redirect_url, timeout=20, follow_redirects=True, headers=headers)
                    r2.raise_for_status()
                    current_url_to_fetch = str(r2.url) # Update final URL again
                    html_str = r2.content.decode(r2.encoding or 'utf-8', errors='replace')

        except httpx.RequestError as e:
            print(f"httpx GET request error for {current_url_to_fetch}: {e}")
            html_str = ""
            source_method += "_fetch_failed"
        except httpx.HTTPStatusError as e:
            print(f"httpx Status error {e.response.status_code} for {e.request.url}")
            if e.response.status_code == 403:
                print(f"Received 403 from httpx for {current_url_to_fetch}. Retrying with Playwright...")
                source_method = "html_playwright_403_retry"
                if is_copernicus_fallback: source_method = "copernicus_html_playwright_403_retry"
                html_str = fetch_with_playwright(current_url_to_fetch)
                if not html_str:
                    source_method += "_fetch_failed"
            else:
                html_str = ""
                source_method += "_fetch_failed"

    return {"html_str": html_str, "source_method": source_method, "final_url": current_url_to_fetch}


def _extract_text_from_html(html_str: str, url: str, source_method: str) -> Dict[str, str]:
    """Extracts text from HTML using Trafilatura and performs basic validation."""
    if not html_str:
        print(f"No HTML content to extract for {url}. Source method: {source_method}")
        return {"text": "", "source": f"{source_method}_error_empty_html"} # Already failed

    try:
        article_text = trafilatura.extract(
            html_str,
            include_tables=False,
            include_comments=False,
            output_format="txt",
            url=url # Provide context
        )
        if article_text and len(article_text) > 50:
            common_errors = ["encountered an error", "enable javascript", "checking your browser", "please enable cookies", "cloudflare"]
            lower_text = article_text.lower()
            if any(err in lower_text for err in common_errors):
                if "playwright" in source_method:
                    print(f"Warning: Playwright fetch for {url} might have resulted in an error/challenge page text.")
                else:
                    print(f"Warning: httpx fetch for {url} resulted in error/challenge page text. Might need Playwright.")
            return {"text": article_text, "source": source_method}
        else:
            print(f"Trafilatura extraction failed or text too short for {url}. Source method: {source_method}")
            return {"text": "", "source": f"{source_method}_extract_failed"}
    except Exception as e:
        print(f"Error during Trafilatura extraction for {url}: {e}")
        return {"text": "", "source": f"{source_method}_extract_error"}


def fetch_article_from_url(url: str, doi: str, output_directory) -> Dict[str, str]:
    """
    Fetches article text from a URL by orchestrating helper functions.
    Handles PDFs, Copernicus XML, dynamic/static HTML, and meta redirects.
    """
    if not url:
        print("Error: fetch_article_from_url called with empty URL.")
        return {"text": "", "source": "error_empty_url"}

    headers = get_random_headers() # <<< CHANGED: Use random headers

    # 1. Get URL metadata (final URL, content type)
    metadata = _get_url_metadata(url, headers)
    if not metadata:
        # Error occurred during HEAD request, source determined by _get_url_metadata logs
        return {"text": "", "source": "error_head_request"}

    final_url = metadata["final_url"]
    content_type = metadata["content_type"]

    # 2. Handle PDF
    if "application/pdf" in content_type:
        return _handle_pdf_url(final_url, output_directory, headers)

    # 3. Handle Copernicus (Check for XML first)
    is_copernicus_html = False
    if ("text/html" in content_type or "application/xhtml+xml" in content_type) and \
       "copernicus.org/articles/" in final_url:
        copernicus_result = _handle_copernicus_xml(final_url, headers)
        if copernicus_result:
            return copernicus_result # XML success
        else:
            # XML failed or not found, try to download pdf
            copernicus_pdf_url = construct_copernicus_url(final_url, file_extension=".pdf")
            copernicus_result = _handle_pdf_url(copernicus_pdf_url, output_directory,headers)
            if copernicus_result:
                return copernicus_result # XML success
            else:
                is_copernicus_html = True
                print(f"Falling back to HTML processing for Copernicus URL: {final_url}")

    # 4. Handle HTML (or fallback for Copernicus/unknown types)
    if is_copernicus_html or ("text/html" in content_type or "application/xhtml+xml" in content_type or not content_type):
        # Fetch HTML content (handles Playwright/httpx/redirects)
        html_fetch_result = _fetch_html_content(final_url, headers, is_copernicus_fallback=is_copernicus_html)
        html_str = html_fetch_result["html_str"]
        source_method = html_fetch_result["source_method"]
        final_html_url = html_fetch_result["final_url"] # URL after potential meta-redirects

        # Extract text from the fetched HTML
        html_extraction_result = _extract_text_from_html(html_str, final_html_url, source_method)

        # --- NEW: PDF Fallback Logic ---
        min_text_length_threshold = 3000 # Define a threshold for acceptable text length
        html_text = html_extraction_result.get("text", "")
        is_html_sufficient = html_text and len(html_text) >= min_text_length_threshold and "fetch_failed" not in html_extraction_result.get("source", "") and "extract_failed" not in html_extraction_result.get("source", "")

        if is_html_sufficient:
            print(f"[*] HTML extraction successful for {final_html_url}")
            return html_extraction_result # Return successful HTML extraction
        else:
            print(f"[!] HTML extraction failed or insufficient text for {final_html_url}")

####################################
# DOI Resolution and Processing
####################################

def resolve_doi(doi: str) -> Optional[str]:
    """
    Resolves a DOI via https://doi.org and returns the final URL.
    Uses httpx with increased timeout, redirect following, and enhanced headers.
    """
    if not doi: return None
    doi_url = f"https://doi.org/{doi}"

    headers = get_random_headers() # <<< CHANGED: Use random headers

    try:
        print(f"Resolving DOI {doi} via {doi_url}...")
        response = httpx.get(doi_url, follow_redirects=True, timeout=30, headers=headers)
        # response.raise_for_status()

        final_url = str(response.url)
        if "perfdrive" in final_url:
            final_url = doi_url # Reset to original DOI URL if it resolves to a validation page
        print(f"DOI {doi} resolved to intermediate/final URL: {final_url} (Status: {response.status_code})")

    
        return final_url

    except httpx.HTTPStatusError as e:
         failed_url = e.request.url
         print(f"HTTP Status Error {e.response.status_code} while resolving DOI {doi} at URL: {failed_url}")
         return None
    except httpx.RequestError as e:
        failed_url = e.request.url
        print(f"Request Exception while resolving DOI {doi} for URL {failed_url}: {e}")
        return None
    except Exception as e:
         print(f"General Exception while resolving DOI {doi}: {e}")
         import traceback
         traceback.print_exc()
         return None

# <<< NEW HELPER FUNCTION FOR PARALLEL PROCESSING >>>
def _process_single_doi(doi: str, output_directory: str, min_text_length: int = 100) -> Tuple[str, Dict[str, str]]:
    """
    Processes a single DOI: resolves, fetches content, saves if successful.
    Returns a tuple (doi, result_dict).
    """
    print(f"--- Starting processing for DOI: {doi} ---")
    resolved_url = resolve_doi(doi)
    result_dict = {}

    if resolved_url:
        print(f"Resolved DOI {doi} to URL: {resolved_url}")
        article_data = fetch_article_from_url(resolved_url, doi, output_directory)
        # remove everything after references in the article text
        if article_data and isinstance(article_data.get("text"), str):
            # references could be written in different ways, so we use regex to find them. They end with references\n or \nn
            article_data["text"] = remove_references_section(article_data["text"])


        if article_data and isinstance(article_data.get("text"), str) and len(article_data["text"]) >= min_text_length:
            print(f"Successfully extracted text for DOI {doi} (Source: {article_data.get('source', 'N/A')})")
            result_dict = article_data
            # Sanitize DOI for filename and join with output directory
            filename_doi = re.sub(r'[^\w\d.-]', '_', doi) + ".txt" # Add .txt extension
            path_to_save = os.path.join(output_directory, filename_doi) # Use output_directory
            try:
                with open(path_to_save, "w", encoding="utf-8") as f:
                        f.write(article_data["text"])
                print(f"Saved text content for DOI {doi} to: {path_to_save}") # Show full path
            except IOError as e:
                print(f"Error saving file {path_to_save}: {e}")
                # Add failure info to results even if saving failed
                result_dict["save_error"] = str(e)

        elif article_data:
             print(f"Failed to extract sufficient text for DOI {doi}. Source: {article_data.get('source', 'failed_extraction')}, URL: {resolved_url}")
             result_dict = {"text": "", "source": article_data.get('source', 'failed_extraction'), "resolved_url": resolved_url}
        else:
             print(f"Failed to process URL obtained for DOI {doi}. URL: {resolved_url}")
             result_dict = {"text": "", "source": "fetch_failed", "resolved_url": resolved_url}

    else:
        print(f"DOI {doi} could not be resolved.")
        result_dict = {"text": "", "source": "doi_resolve_failed"}

    print(f"--- Finished processing for DOI: {doi} ---")
    return doi, result_dict

def process_dois(dois: List[str], output_directory: str, max_workers: int = 8) -> Dict[str, Dict[str, str]]:
    """
    Processes a list of DOIs in parallel: resolves each DOI, fetches its content dynamically,
    and returns a dictionary mapping each DOI to its extracted text and source.
    Saves successful extractions to the specified directory.
    """
    # save cookie file at output path
    global COOKIE_FILE 
    COOKIE_FILE = os.path.join(output_directory, "cookies", "iop.json")

    results = {}
    min_text_length = 100

    if not dois:
        print("Warning: No DOIs provided for processing.")
        return results

    # Ensure output directory exists
    os.makedirs(output_directory, exist_ok=True)

    print(f"Processing {len(dois)} DOIs using up to {max_workers} workers...")
    processed_count = 0
    total_count = len([doi for doi in dois if doi and isinstance(doi, str)])
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all DOI processing tasks
        future_to_doi = {
            executor.submit(_process_single_doi, doi, output_directory, min_text_length): doi
            for doi in dois if doi and isinstance(doi, str) # Basic validation before submitting
        }

        # Process completed futures
        for future in concurrent.futures.as_completed(future_to_doi):
            original_doi = future_to_doi[future]
            try:
                # Retrieve the result tuple (doi, result_dict)
                processed_doi, result_data = future.result()
                results[processed_doi] = result_data
            except Exception as exc:
                print(f"DOI {original_doi} generated an exception during processing: {exc}")
                results[original_doi] = {"text": "", "source": "processing_exception", "error": str(exc)}
            processed_count += 1
            print(f"Processed {processed_count} / {total_count} DOIs...", flush=True)
            # else: # Optional: Log successful completion from the main loop if needed
            #     print(f"Successfully completed processing for DOI: {original_doi}")

    # remove iop.json if it exists
    if os.path.exists(COOKIE_FILE):
        try:
            os.remove(COOKIE_FILE)
            print(f"Removed temporary cookie file: {COOKIE_FILE}")
        except OSError as e:
            print(f"Error removing temporary cookie file {COOKIE_FILE}: {e}")
    

    print(f"Finished parallel processing for {total_count} DOIs.")
    return results

def download_dois(proposed_dois: List[str], output_directory: str) -> None:
    """
    Downloads articles for a list of DOIs, checking if they are already downloaded.
    If not, it processes them in parallel to extract text and saves the results.
    """
    print(f"Checking for existing files in: {output_directory}")
    # remove dois that are already in the downloads folder
    dois_to_process = []
    for doi in proposed_dois:
        if not doi or not isinstance(doi, str):
            print(f"Skipping invalid proposed DOI: {doi}")
            continue
        filename_doi = re.sub(r'[^\w\d.-]', '_', doi) + ".txt" # Add .txt extension but replacing / with _
        path_to_check = os.path.join(output_directory, filename_doi) # Use output_directory
        if not os.path.exists(path_to_check):
            dois_to_process.append(doi)
        else:
            print(f"Skipping DOI {doi} as file already exists: {path_to_check}")

    if not dois_to_process:
        print("No new DOIs to process.")
        return

    print(f"Starting parallel DOI processing for {len(dois_to_process)} DOIs...") # Updated print
    print("-" * 30)

    # Call the parallelized process_dois function
    # You can adjust max_workers here if needed, e.g., process_dois(dois_to_process, output_directory, max_workers=10)
    extracted_articles = process_dois(dois_to_process, output_directory)

    # output_filename = "articles_extracted.json" # Consider if this summary file is still needed/where to save it
    print("-" * 30)
    try:
        # with open(output_filename, "w", encoding="utf-8") as f:
        #     json.dump(extracted_articles, f, ensure_ascii=False, indent=4)
        print(f"\n--- Processing Complete ---")
        success_count = sum(1 for doi, data in extracted_articles.items() if data.get("text") and len(data["text"]) >= 100 and "save_error" not in data)
        total_processed = len(dois_to_process) # Use the count of DOIs intended for processing
        print(f"Successfully extracted and saved text for {success_count} out of {total_processed} processed DOIs.")
    except TypeError as e:
         print(f"\n--- Processing Complete (with summary error) ---")
         print(f"Error summarizing results: {e}. Check data structure.")
         print("Dumping raw results:")
         print(extracted_articles)
    except Exception as e:
        print(f"\n--- Processing Complete (with summary error) ---")
        print(f"Error summarizing results: {e}")

    if extracted_articles:
        print("\n--- Source Summary ---")
        source_counts = {}
        for doi, data in extracted_articles.items():
            source = data.get("source", "unknown")
            source_counts[source] = source_counts.get(source, 0) + 1
        # Sort sources for consistent output, handling potential None or non-string keys gracefully
        sorted_sources = sorted(source_counts.keys(), key=lambda x: str(x))
        for source in sorted_sources:
            count = source_counts[source]
            print(f"- {source}: {count}")


