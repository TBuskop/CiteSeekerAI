import os
import re
import argparse
import sys
from pathlib import Path
from pdfminer.high_level import extract_text as pdfminer_extract_text
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions

# --- Previous code remains the same (Imports, Gemini Config, Functions) ---

# ------------------------------
# Gemini Configuration
# ------------------------------

# --- Configuration ---
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable or direct assignment in script is required.")
try:
    genai.configure(api_key=GOOGLE_API_KEY)
except Exception as e:
    print(f"Error configuring Gemini: {e}")
    exit(1)

DEFAULT_MODEL_NAME = "gemini-1.5-flash" # Set to the cheapest model as requested

SAFETY_SETTINGS = [

]
GENERATION_CONFIG = {
    "temperature": 0.7, "top_p": 1.0, "top_k": 1,
}

# ------------------------------
# Extraction and Cleaning Functions (Keep as before)
# ------------------------------
def extract_text_from_pdf(pdf_path):
    print(f"[*] Extracting text from: {pdf_path.name}")
    try:
        text = pdfminer_extract_text(pdf_path)
        print("[+] Text extraction successful.")
        return text
    except FileNotFoundError:
        print(f"[!] Error: PDF file not found at {pdf_path}")
        return None
    except Exception as e:
        print(f"[!] Error during PDF text extraction ({pdf_path.name}): {e}")
        return None

import re

def clean_academic_text(text):
    if not text:
        return ""
    lines = text.splitlines()
    cleaned_lines = []
    potential_header_footer = {}
    
    # Identify potential header/footer lines (e.g., page numbers)
    for line in lines:
        stripped_line = line.strip()
        if stripped_line and len(stripped_line) < 50:
            if re.fullmatch(r'-?\s*\d+\s*-?', stripped_line):
                potential_header_footer[stripped_line] = potential_header_footer.get(stripped_line, 0) + 1
                
    frequent_threshold = 3
    lines_to_remove = {line for line, count in potential_header_footer.items() if count >= frequent_threshold}
    
    # Remove header/footer lines
    for line in lines:
        stripped_line = line.strip()
        if stripped_line and stripped_line not in lines_to_remove:
            cleaned_lines.append(stripped_line)
            
    
    # Join the lines into a single text and do additional clean-up.
    text = "\n".join(cleaned_lines)
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n\s*\n\s*\n*', '\n\n', text)
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    
    return text.strip()


def remove_references_section(text):
    """
    Attempts to remove text starting from the line containing the
    'References' or 'Bibliography' section header onwards.
    Uses a case-insensitive, multiline search.
    """
    # Pattern explanation:
    # ^       - Matches the beginning of a line (due to re.MULTILINE)
    # \s*     - Matches zero or more whitespace characters (spaces, tabs)
    # (References|Bibliography) - Matches either "References" or "Bibliography"
    # \s*     - Matches zero or more whitespace characters after the keyword
    # $       - Matches the end of the line (due to re.MULTILINE)
    pattern = r'^\s*(References|Bibliography|Acknowledgement|Acknowledgements)\s*$'

    # Perform a multiline, case-insensitive search
    match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)

    if match:
        # Found the start of the references section header line
        refs_start_index = match.start()
        found_header = match.group().strip() # Get the actual header found (e.g., "References")
        print(f"[*] Found '{found_header}' section header at index {refs_start_index}. Removing text from this line onwards.")

        # Return the text *before* the start of the matched line
        # Slicing up to match.start() ensures the header line itself is removed
        return text[:refs_start_index].strip()
    else:
        # Optional: Add a print statement for debugging if needed
        # print("[*] Did not find a standard 'References' or 'Bibliography' section header line.")
        print("[*] No references section header found. No text removed.")
        return text # Return original text if no pattern matched


# ------------------------------
# Gemini Conversion Function (Keep as before)
# ------------------------------
def convert_to_markdown_gemini(text_content, model_name):
    if not text_content: return None
    print(f"[*] Converting text to Markdown using Gemini model: {model_name}")
    try:
        model = genai.GenerativeModel(model_name=model_name, safety_settings=SAFETY_SETTINGS, generation_config=GENERATION_CONFIG)
        prompt = f"""Please convert the following text, extracted and cleaned from an academic PDF document, into well-structured Markdown format.

Focus on preserving the semantic structure:
- Identify and format title, authors, abstract if present.
- Use appropriate Markdown heading levels (#, ##, ###) for sections and subsections.
- Format bullet points (* or -) and numbered lists (1., 2.).
- Preserve paragraph breaks (use double line breaks in Markdown).
- Format code blocks using triple backticks (```).
- Attempt to format table-like structures using Markdown table syntax (| Header | ...). Acknowledge if the table structure is unclear from the text.
- Retain emphasis like **bold** or *italics* if discernible.
- Ensure mathematical formulas are represented clearly, perhaps using LaTeX syntax within $...$ or $$...$$ delimiters if you identify them, or using unicode characters if appropriate. Indicate if formulas are complex "[Complex Formula Cannot Be Represented]".
- Handle citations appropriately, perhaps keeping them inline like [1] or (Author, Year).
- Remove any remaining extraction artifacts like stray page numbers or headers/footers if you identify them.
- DO NOT add any introductory or concluding remarks. Output only the Markdown content.

Here is the text to convert:
--- START OF TEXT ---
{text_content}
--- END OF TEXT ---
"""
        response = model.generate_content(prompt, request_options={'timeout': 600})

        if response.prompt_feedback and response.prompt_feedback.block_reason:
             print(f"[!] Generation blocked. Reason: {response.prompt_feedback.block_reason}")
             if response.prompt_feedback.safety_ratings:
                  print("Safety Ratings:", [f"{r.category}: {r.probability}" for r in response.prompt_feedback.safety_ratings])
             return None
        if response.parts:
            markdown_content = response.text
            lines = markdown_content.strip().splitlines()
            if lines:
                 if "markdown" in lines[0].lower() or lines[0].strip() in ["```", "markdown"]: lines.pop(0)
            if lines:
                 if lines[-1].strip() == "```": lines.pop()
            markdown_content = "\n".join(lines).strip()
            print("[+] Markdown conversion successful.")
            return markdown_content
        else:
            print("[!] Error: Received an empty or unexpected response from Gemini API.")
            return None
    except google_exceptions.DeadlineExceeded as e: print(f"[!] Gemini API Call Error: Timeout exceeded. ({e})"); return None
    except google_exceptions.ResourceExhausted as e: print(f"[!] Gemini API Call Error: Resource exhausted (likely quota). ({e})"); return None
    except google_exceptions.InvalidArgument as e: print(f"[!] Gemini API Call Error: Invalid argument. ({e}) \n[!] Possible cause: Input text > token limit?"); return None
    except google_exceptions.GoogleAPICallError as e: print(f"[!] Gemini API Call Error: {e}"); return None
    except Exception as e: print(f"[!] An unexpected error during Gemini conversion: {e}"); return None

# ------------------------------
# File Handling Functions (Keep as before)
# ------------------------------
def save_text_file(text, output_path):
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"[+] Successfully saved: {output_path.name}")
    except IOError as e:
        print(f"[!] Error writing file {output_path}: {e}")

# ------------------------------
# Main Execution Logic (Modified argparse and logic)
# ------------------------------

def main():
    # --- Define Default Base Project Name ---
    default_project = "droughts" # Changed from "test" for clarity

    # --- Define Default Paths using the project name ---
    # Use current working directory as base for defaults
    default_pdf_dir = Path(f"./pdfs/{default_project}")
    default_output_dir = Path(f"./markdown_output/{default_project}")
    default_cleaned_dir = Path(f"./cleaned_text/{default_project}") # Default location for cleaned text
    skip_references_default = True
    convert_to_markdown_gemini_default = False

    parser = argparse.ArgumentParser(
        description="Extract text from PDFs, clean it, and convert to Markdown using Gemini.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # --- Arguments with updated defaults ---
    parser.add_argument("--pdf-dir", type=str, default=str(default_pdf_dir),
                        help="Directory containing the input PDF files.")
    parser.add_argument("--output-dir", type=str, default=str(default_output_dir),
                        help="Directory to save the output Markdown (.md) files.")
    # --- MODIFIED: Set default path for cleaned text dir ---
    parser.add_argument("-c", "--cleaned-text-dir", type=str,
                        default=str(default_cleaned_dir), # SAVE BY DEFAULT to default_cleaned_dir
                        help="Directory to save intermediate cleaned text (.txt) files. "
                             "Provide an empty string ('') or specific path like 'None' to disable saving.")
    parser.add_argument("-m", "--model", type=str, default=DEFAULT_MODEL_NAME,
                        help="Gemini model name to use for conversion.")
    parser.add_argument("--skip-references", action='store_true', default=skip_references_default,
                        help="Attempt to remove the References/Bibliography section before conversion.")
    parser.add_argument("--overwrite", action='store_true',
                        help="Overwrite existing output files without asking.")
    # --- Optional: Argument to explicitly set project name ---
    parser.add_argument("--project", type=str, default=None,
                        help="Specify a project name to organize input/output folders "
                             "(e.g., 'paper_xyz'). Overrides default paths if set.")


    args = parser.parse_args()

    # --- Determine effective paths based on --project argument ---
    if args.project:
        print(f"[*] Using project name: {args.project}")
        # Override default paths if project name is given
        pdf_dir = Path(f"./pdfs/{args.project}")
        output_dir_md = Path(f"./markdown_output/{args.project}")
        # Only override cleaned dir default if it wasn't explicitly set differently
        if args.cleaned_text_dir == str(default_cleaned_dir):
             cleaned_dir_base = Path(f"./cleaned_text/{args.project}")
        else:
             cleaned_dir_base = Path(args.cleaned_text_dir) # User explicitly set it
    else:
        # Use paths derived from command-line args or their defaults
        pdf_dir = Path(args.pdf_dir)
        output_dir_md = Path(args.output_dir)
        cleaned_dir_base = Path(args.cleaned_text_dir) # Use the default or user-provided path

    # --- Determine if cleaned text should be saved ---
    # Check if the user explicitly disabled saving
    disable_clean_save = args.cleaned_text_dir in ['', 'None', 'none', 'false', 'False']
    output_dir_cleaned = None
    if not disable_clean_save:
        # Ensure cleaned_dir_base is a valid Path if not disabled
        if isinstance(cleaned_dir_base, Path):
            output_dir_cleaned = cleaned_dir_base
        else: # Should not happen with current defaults, but safety check
             print(f"[!] Warning: Invalid path provided for --cleaned-text-dir ('{args.cleaned_text_dir}'). Disabling cleaned text saving.")


    # --- Print effective paths ---
    print(f"[*] Using PDF input directory: {pdf_dir.resolve()}")
    print(f"[*] Using Markdown output directory: {output_dir_md.resolve()}")
    if output_dir_cleaned:
        print(f"[*] Saving cleaned text to: {output_dir_cleaned.resolve()}")
    else:
        print("[*] Intermediate cleaned text files will NOT be saved.") # Updated log message

    # --- Directory creation and checks (as before) ---
    if not pdf_dir.is_dir():
        print(f"\n[!] Error: Input PDF directory not found: {pdf_dir}")
        print(f"[!] Please create it or specify a valid directory using --pdf-dir (or --project).")
        try:
            pdf_dir.mkdir(parents=True, exist_ok=True)
            print(f"[*] Created input directory: {pdf_dir}. Please place PDFs inside.")
        except OSError as e:
            print(f"[!] Could not create input directory: {e}")
            return
    try:
         output_dir_md.mkdir(parents=True, exist_ok=True)
         if output_dir_cleaned:
             output_dir_cleaned.mkdir(parents=True, exist_ok=True)
    except OSError as e:
         print(f"[!] Error creating output directories: {e}")
         return

    # --- Process PDF files (rest of the loop remains the same) ---
    pdf_files = sorted(list(pdf_dir.glob("*.pdf")))
    if not pdf_files:
        print(f"\n[!] No PDF files found in {pdf_dir}")
        print("[!] Ensure PDF files are present in the input directory.")
        return

    print(f"\n[*] Found {len(pdf_files)} PDF files to process.")

    for pdf_file in pdf_files:
        print(f"\n--- Processing: {pdf_file.name} ---")

        md_output_path = output_dir_md / f"{pdf_file.stem}.md"
        cleaned_output_path = output_dir_cleaned / f"{pdf_file.stem}_cleaned.txt" if output_dir_cleaned else None

        if not args.overwrite and cleaned_output_path.exists():
            print(f"[*] Skipping {pdf_file.name}: Output Markdown file already exists ({md_output_path.name}). Use --overwrite.")
            continue

        raw_text = extract_text_from_pdf(pdf_file)
        if not raw_text:
            print(f"[!] Skipping {pdf_file.name} due to text extraction error.")
            continue

        cleaned_text = clean_academic_text(raw_text)

        if args.skip_references:
            cleaned_text = remove_references_section(cleaned_text)

        if not cleaned_text:
             print(f"[!] Skipping {pdf_file.name}: Text content is empty after cleaning.")
             continue

        # --- Save cleaned text IF output_dir_cleaned is set ---
        if cleaned_output_path:
             print(f"[*] Saving cleaned text for {pdf_file.name}...")
             save_text_file(cleaned_text, cleaned_output_path) # save_text_file handles dir creation

        if convert_to_markdown_gemini_default:
            markdown_content = convert_to_markdown_gemini(cleaned_text, model_name=args.model)

            if markdown_content:
                print(f"[*] Saving Markdown output for {pdf_file.name}...")
                save_text_file(markdown_content, md_output_path) # save_text_file handles dir creation
            else:
                print(f"[!] Failed to generate Markdown for {pdf_file.name}. Skipping save.")

    print("\n--- Processing Complete ---")

if __name__ == "__main__":
    main()