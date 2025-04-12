#!/usr/bin/env python
import re
import json
import concurrent.futures
from typing import List, Dict, Optional
import xml.etree.ElementTree as ET # <<< NEW: Import ElementTree


import httpx
import trafilatura
import fitz  # PyMuPDF for PDF extraction
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
from playwright_stealth import stealth_sync
from langchain.text_splitter import RecursiveCharacterTextSplitter

# <<< NEW: Define patterns for URLs that likely require Playwright >>>
# Add regex patterns for domains or URL structures known to need JS rendering.
# Examples:
# - r"https://(.*\.)?twitter\.com/.*"
# - r"https://(.*\.)?facebook\.com/.*"
# - r"https://www\.some-spa-site\.com/app/.*"
# - r"https://(.*\.)?javascript-heavy-domain\.org/.*"
# Assume most don't need it, so keep this list specific.

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
]

###############################
# Headless Browser with Stealth
###############################
####################################
# <<< NEW: Copernicus XML Handling >>>
####################################

def construct_copernicus_xml_url(html_url: str) -> Optional[str]:
    """
    Attempts to construct the Copernicus XML download URL from the article HTML URL.
    Example HTML URL: https://hess.copernicus.org/articles/19/1521/2015/
    Example XML URL:  https://hess.copernicus.org/articles/19/1521/2015/hess-19-1521-2015.xml
    """
    # Regex to capture the necessary parts: subdomain (journal), volume, page, year
    match = re.match(r"https://(?P<journal>[^.]+)\.copernicus\.org/articles/(?P<vol>\d+)/(?P<page>\d+)/(?P<year>\d+)/?", html_url)
    if match:
        parts = match.groupdict()
        # Construct the base path
        base_path = f"https://{parts['journal']}.copernicus.org/articles/{parts['vol']}/{parts['page']}/{parts['year']}/"
        # Construct the filename part
        filename = f"{parts['journal']}-{parts['vol']}-{parts['page']}-{parts['year']}.xml"
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

def fetch_with_playwright(url: str) -> str:
    """
    Uses Playwright in headless mode with stealth modifications to load the URL,
    waits until the network is idle, and returns the fully rendered HTML.
    """
    html_str = "" # Initialize html_str
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                args=["--disable-blink-features=AutomationControlled"]
            )
            context = browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/105.0.5195.102 Safari/537.36"
                )
            )
            page = context.new_page()
            stealth_sync(page)  # Apply stealth modifications
            try:
                # check if url contains elsevier or sciencedirect
                if "elsevier" in url or "sciencedirect" in url:
                    text_selector = 'div.Body'
                elif "wiley" in url:
                    text_selector = 'section.article-section__full'
                elif "ascelibrary" in url:
                    text_selector = 'section#bodymatter'
                elif "mdpi" in url:
                    text_selector = 'div.html-body'
                elif "iwaponline" in url:
                    text_selector = 'div.article-body'


                print(f"Fetching URL via Playwright (headless browser): {url}")
                page.goto(url, timeout=30000, wait_until='networkidle')
                try:
                    print(f"Found content in selector: {text_selector}")
                    html_str = page.locator(text_selector).inner_text()
                    # wrap in html tags to ensure valid HTML
                    html_str = f"<html><body>{html_str}</body></html>"
                    print(f"Fetched content from selector: {text_selector}")
                except:
                    html_str = page.content()
            except Exception as e:
                print(f"Error in Playwright fetch for {url}: {e}")
                try:
                    html_str = page.content()
                except Exception as page_content_error:
                    print(f"Could not get page content after Playwright error for {url}: {page_content_error}")
                    html_str = "" # Ensure it's empty on failure
            finally:
                try:
                    context.close()
                except Exception as ctx_close_err:
                    print(f"Error closing Playwright context for {url}: {ctx_close_err}")
                try:
                    browser.close()
                except Exception as browser_close_err:
                    print(f"Error closing Playwright browser for {url}: {browser_close_err}")
    except Exception as e:
        print(f"General Playwright error for {url}: {e}")
        html_str = "" # Ensure it's empty on failure

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

def extract_pdf_text(pdf_bytes: bytes) -> str:
    """
    Extracts text from PDF bytes using PyMuPDF.
    """
    if not pdf_bytes:
        return ""
    text = ""
    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            for page in doc:
                text += page.get_text()
        return text
    except Exception as e:
        print(f"Error extracting PDF text: {e}")
        return "" # Return empty string on error

# =====================================================
# WebPageHelper Class (Optional - not used by process_dois)
# =====================================================
class WebPageHelper:
    """Helper class to process web pages and split text into chunks."""
    def __init__(
        self,
        min_char_count: int = 150,
        snippet_chunk_size: int = 1000,
        max_thread_num: int = 10,
    ):
        self.httpx_client = httpx.Client(verify=False, follow_redirects=True, timeout=15.0)
        self.min_char_count = min_char_count
        self.max_thread_num = max_thread_num
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=snippet_chunk_size,
            chunk_overlap=0,
            length_function=len,
            is_separator_regex=False,
            separators=[
                "\n\n", "\n", ".", "\uff0e", "\u3002", ",", "\uff0c", "\u3001",
                " ", "\u200B", "",
            ],
        )

    def _needs_playwright(self, url: str) -> bool:
        """Checks if the URL matches any dynamic pattern."""
        for pattern in DYNAMIC_URL_PATTERNS:
            if re.search(pattern, url, re.IGNORECASE):
                return True
        return False

    def download_webpage_content(self, url: str) -> Optional[bytes]:
        """Downloads content, choosing method based on URL patterns."""
        if self._needs_playwright(url):
            print(f"[WebPageHelper] Using Playwright for {url}")
            html_str = fetch_with_playwright(url)
            return html_str.encode('utf-8') if html_str else None
        else:
            print(f"[WebPageHelper] Using httpx for {url}")
            try:
                res = self.httpx_client.get(url, timeout=15.0)
                res.raise_for_status()
                potential_html = res.content.decode(res.encoding or 'utf-8', errors='ignore')
                redirect_url = extract_meta_redirect(potential_html, str(res.url))
                if redirect_url:
                    print(f"[WebPageHelper] Found meta redirect from {url} to {redirect_url}. Following...")
                    return self.download_webpage_content(redirect_url)
                else:
                    return res.content
            except httpx.RequestError as exc:
                print(f"[WebPageHelper] httpx request error for {url}: {exc}")
                return None
            except httpx.HTTPStatusError as exc:
                print(f"[WebPageHelper] httpx status error for {url}: {exc.response.status_code} - {exc}")
                return None
            except Exception as exc:
                print(f"[WebPageHelper] General error downloading {url}: {exc}")
                return None

    def urls_to_articles(self, urls: List[str]) -> Dict[str, Dict]:
        """Processes multiple URLs to extract article text."""
        articles = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_thread_num) as executor:
            future_to_url = {executor.submit(self.download_webpage_content, url): url for url in urls}

            for future in concurrent.futures.as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    content_bytes = future.result()
                    if content_bytes:
                        try:
                            content_str = content_bytes.decode('utf-8', errors='ignore')
                            article_text = trafilatura.extract(
                                content_str,
                                include_tables=False,
                                include_comments=False,
                                output_format="txt",
                                url=url
                            )
                            if article_text and len(article_text) > self.min_char_count:
                                source_type = "html/dynamic" if self._needs_playwright(url) else "html/static"
                                articles[url] = {"text": article_text, "source": source_type}
                            else:
                                print(f"[WebPageHelper] Extracted text too short or None for {url}")
                        except Exception as e:
                            print(f"[WebPageHelper] Error processing content from {url}: {e}")
                    else:
                        print(f"[WebPageHelper] Failed to download content for {url}")
                except Exception as exc:
                    print(f"[WebPageHelper] Error processing URL {url}: {exc}")
        return articles

    def close(self):
        """Closes the httpx client."""
        self.httpx_client.close()
        print("[WebPageHelper] httpx client closed.")

####################################
# Fetch Article from URL Function (Refactored)
####################################

def _url_needs_playwright(url: str) -> bool:
    """Helper function to check if a URL matches dynamic patterns."""
    if not url: return False # Handle None or empty URLs
    for pattern in DYNAMIC_URL_PATTERNS:
        # Add robustness for potential malformed patterns
        try:
            if re.search(pattern, url, re.IGNORECASE):
                return True
        except re.error as e:
            print(f"Warning: Invalid regex pattern '{pattern}': {e}")
    return False

def _get_url_metadata(url: str, headers: Dict[str, str]) -> Optional[Dict[str, str]]:
    """
    Performs a HEAD request to get the final URL and content type after redirects.
    Returns a dictionary with 'final_url' and 'content_type' or None on error.
    """
    try:
        print(f"Checking URL type: {url}")
        head_resp = httpx.head(url, follow_redirects=True, timeout=20, headers=headers)
        # head_resp.raise_for_status()
        final_url = str(head_resp.url)
        content_type = head_resp.headers.get("content-type", "").lower()
        print(f"URL resolved to: {final_url}, Content-Type: {content_type}")
        return {"final_url": final_url, "content_type": content_type}
    except httpx.HTTPStatusError as e:
        print(f"HTTP Status Error during HEAD request for {url}: {e.response.status_code} - {e.request.url}")
        return None # Indicate error with None
    except httpx.RequestError as e:
        print(f"Request Error during HEAD request for {url}: {e}")
        return None # Indicate error with None
    except Exception as e:
        print(f"General Error during HEAD request for {url}: {e}")
        return None # Indicate error with None


def _handle_pdf_url(url: str, headers: Dict[str, str]) -> Dict[str, str]:
    """Downloads PDF from URL and extracts text."""
    print(f"Fetching PDF content from {url} using httpx")
    try:
        r = httpx.get(url, timeout=30, follow_redirects=True, headers=headers)
        r.raise_for_status()
        pdf_text = extract_pdf_text(r.content)
        if pdf_text:
            return {"text": pdf_text, "source": "pdf"}
        else:
            print(f"Failed to extract text from PDF: {url}")
            return {"text": "", "source": "pdf_extract_failed"}
    except httpx.HTTPStatusError as e:
        print(f"HTTP Status Error downloading PDF {url}: {e.response.status_code}")
        return {"text": "", "source": "pdf_download_failed_status"}
    except httpx.RequestError as e:
        print(f"Request Error downloading PDF {url}: {e}")
        return {"text": "", "source": "pdf_download_failed_request"}
    except Exception as e:
        print(f"General Error downloading/processing PDF {url}: {e}")
        return {"text": "", "source": "pdf_download_failed_general"}


def _handle_copernicus_xml(html_url: str, headers: Dict[str, str]) -> Optional[Dict[str, str]]:
    """
    Attempts to fetch and parse Copernicus XML from a corresponding HTML URL.
    Returns extracted text and source if successful, None otherwise (indicating fallback).
    """
    print(f"Detected Copernicus article page: {html_url}. Attempting XML fetch.")
    xml_url = construct_copernicus_xml_url(html_url)
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
            print("Failed to extract sufficient text from Copernicus XML. Falling back to HTML.")
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
            html_str = ""
            source_method += "_fetch_failed"

    return {"html_str": html_str, "source_method": source_method, "final_url": current_url_to_fetch}


def _extract_text_from_html(html_str: str, url: str, source_method: str) -> Dict[str, str]:
    """Extracts text from HTML using Trafilatura and performs basic validation."""
    if not html_str:
        print(f"No HTML content to extract for {url}. Source method: {source_method}")
        return {"text": "", "source": f"{source_method}_fetch_failed"} # Already failed

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


def fetch_article_from_url(url: str) -> Dict[str, str]:
    """
    Fetches article text from a URL by orchestrating helper functions.
    Handles PDFs, Copernicus XML, dynamic/static HTML, and meta redirects.
    """
    if not url:
        print("Error: fetch_article_from_url called with empty URL.")
        return {"text": "", "source": "error_empty_url"}

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'DNT': '1',
        'Upgrade-Insecure-Requests': '1',
        'Referer': 'https://www.google.com/'
    }

    # 1. Get URL metadata (final URL, content type)
    metadata = _get_url_metadata(url, headers)
    if not metadata:
        # Error occurred during HEAD request, source determined by _get_url_metadata logs
        return {"text": "", "source": "error_head_request"}

    final_url = metadata["final_url"]
    content_type = metadata["content_type"]

    # 2. Handle PDF
    if "application/pdf" in content_type:
        return _handle_pdf_url(final_url, headers)

    # 3. Handle Copernicus (Check for XML first)
    is_copernicus_html = False
    if ("text/html" in content_type or "application/xhtml+xml" in content_type) and \
       "copernicus.org/articles/" in final_url:
        copernicus_result = _handle_copernicus_xml(final_url, headers)
        if copernicus_result:
            return copernicus_result # XML success
        else:
            # XML failed or not found, mark for HTML fallback
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
        return _extract_text_from_html(html_str, final_html_url, source_method)

    # 5. Handle other unsupported content types
    else:
        print(f"Skipping unsupported content-type '{content_type}' for URL {final_url}")
        return {"text": "", "source": f"skipped_{content_type.replace('/', '_')}"}

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

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'DNT': '1',
        'Upgrade-Insecure-Requests': '1',
        'Referer': 'https://www.google.com/'
    }

    try:
        print(f"Resolving DOI {doi} via {doi_url}...")
        response = httpx.get(doi_url, follow_redirects=True, timeout=30, headers=headers)
        # response.raise_for_status()

        final_url = str(response.url)
        print(f"DOI {doi} resolved to intermediate/final URL: {final_url} (Status: {response.status_code})")

        if "doi.org/" in final_url and doi in final_url:
            print(f"Warning: DOI {doi} resolution might have stalled at {final_url}. Checking response body...")
            try:
                soup = BeautifulSoup(response.text, 'html.parser')
                link = soup.find('a', href=re.compile(r'https?://(?!doi\.org)'))
                if link and link.get('href'):
                    potential_url = link['href']
                    print(f"Attempting fallback resolution via link found in body: {potential_url}")
                    if potential_url.startswith('http'):
                         return potential_url
                    else:
                         print(f"Fallback link '{potential_url}' is not a valid absolute URL.")
            except Exception as parse_err:
                print(f"Could not parse doi.org page for fallback link: {parse_err}")
            return None
        else:
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


def process_dois(dois: List[str], output_directory: str) -> Dict[str, Dict[str, str]]:
    """
    Processes a list of DOIs: resolves each DOI, fetches its content dynamically,
    and returns a dictionary mapping each DOI to its extracted text and source.
    Saves successful extractions to the specified directory.
    Uses sequential processing for simplicity.
    """
    results = {}
    min_text_length = 100

    if not dois:
        print("Warning: No DOIs provided for processing.")
        return results

    # Ensure output directory exists
    os.makedirs(output_directory, exist_ok=True)

    for doi in dois:
        if not doi or not isinstance(doi, str):
             print(f"Skipping invalid DOI entry: {doi}")
             continue

        print(f"\n--- Processing DOI: {doi} ---")
        resolved_url = resolve_doi(doi)

        if resolved_url:
            print(f"Resolved DOI {doi} to URL: {resolved_url}")
            article_data = fetch_article_from_url(resolved_url)

            if article_data and isinstance(article_data.get("text"), str) and len(article_data["text"]) >= min_text_length:
                print(f"Successfully extracted text for DOI {doi} (Source: {article_data.get('source', 'N/A')})")
                results[doi] = article_data
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
                    results[doi]["save_error"] = str(e)


            elif article_data:
                 print(f"Failed to extract sufficient text for DOI {doi}. Source: {article_data.get('source', 'failed_extraction')}, URL: {resolved_url}")
                 results[doi] = {"text": "", "source": article_data.get('source', 'failed_extraction'), "resolved_url": resolved_url}
            else:
                 print(f"Failed to process URL obtained for DOI {doi}. URL: {resolved_url}")
                 results[doi] = {"text": "", "source": "fetch_failed", "resolved_url": resolved_url}

        else:
            print(f"DOI {doi} could not be resolved.")
            results[doi] = {"text": "", "source": "doi_resolve_failed"}

    return results

def download_dois(proposed_dois: List[str], output_directory: str) -> None:
    """
    Downloads articles for a list of DOIs, checking if they are already downloaded.
    If not, it processes them to extract text and saves the results to the specified directory.
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

    print(f"Starting DOI processing for {len(dois_to_process)} DOIs...")
    print("-" * 30)

    extracted_articles = process_dois(dois_to_process, output_directory) # Pass output_directory

    # output_filename = "articles_extracted.json" # Consider if this summary file is still needed/where to save it
    print("-" * 30)
    try:
        # with open(output_filename, "w", encoding="utf-8") as f:
        #     json.dump(extracted_articles, f, ensure_ascii=False, indent=4)
        print(f"\n--- Processing Complete ---")
        success_count = sum(1 for doi, data in extracted_articles.items() if data.get("text") and len(data["text"]) >= 100 and "save_error" not in data)
        total_processed = len(dois_to_process)
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
        for source in sorted(source_counts.keys()):
            count = source_counts[source]
            print(f"- {source}: {count}")


