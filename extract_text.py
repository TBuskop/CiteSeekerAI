import os
import re
from pathlib import Path
from PyPDF2 import PdfReader

# ------------------------------
# Extraction and cleaning functions
# ------------------------------

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except Exception as e:
        print(f"Error processing {pdf_path}: {str(e)}")
        return ""

def clean_academic_text(text):
    """Clean extracted academic text."""
    # Remove page numbers on separate lines
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
    
    # Remove headers/footers (common patterns in academic papers)
    text = re.sub(r'\n.+(?:vol\.|volume|pp\.|pages?).*\n', '\n', text, flags=re.IGNORECASE)
    
    # Replace multiple spaces with a single space
    text = re.sub(r' +', ' ', text)
    
    # Replace multiple newlines with two newlines
    text = re.sub(r'\n\s*\n\s*\n*', '\n\n', text)
    
    # Remove hyphenation at end of lines (e.g., "exam-\nple" becomes "example")
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    
    # Strip whitespace from each line and join them back
    lines = [line.strip() for line in text.split('\n')]
    return "\n".join(lines).strip()

def remove_references_section(text):
    """
    Remove all references sections from academic text.
    The function identifies and removes text between various forms of "References" headers
    and the next chapter, section, or end of file.
    """
    # Pattern to match common reference section headers
    references_patterns = [
        r'References\s*\n',
        r'REFERENCES\s*\n',
        ]
    
    # Pattern to match start of a new section (chapter, section heading, etc.)
    section_patterns = [
        r'Chapter\s+\d+\s*\n',
        r'\n\d+\.\s+[A-Z]',  # Section numbering like "1. Introduction"
        r'\n[A-Z][A-Z\s]+\n'  # ALL CAPS section headers
    ]
    
    # Combine section patterns with OR logic
    section_pattern = '|'.join(section_patterns)
    
    # Process text iteratively to find and remove multiple reference sections
    result_text = text
    for ref_pattern in references_patterns:
        # Process all instances of this reference pattern
        last_position = 0
        while True:
            # Find the next references section after last position
            refs_match = re.search(ref_pattern, result_text[last_position:])
            if not refs_match:
                break
                
            # Calculate absolute position in the text
            refs_start = last_position + refs_match.start()
            
            # Find the next section header after this references section
            section_match = re.search(section_pattern, result_text[refs_start:])
            
            if section_match:
                # Found next section - remove text between references and next section
                refs_end = refs_start + section_match.start()
                result_text = result_text[:refs_start] + result_text[refs_end:]
                # We removed some text, so don't advance the position
            else:
                # No next section found - remove everything after references
                result_text = result_text[:refs_start]
                break
                
            # Update last position to continue searching
            last_position = refs_start
    
    return result_text

# ------------------------------
# Settings and Directories
# ------------------------------

# Set the project name and word limit per merged document
project_name = "paper_2_intro"
WORD_LIMIT = 100000

# Define directories for PDFs and merged output
pdf_dir = Path(f"pdfs/{project_name}")
output_dir = Path(f"merged_texts/{project_name}")
output_dir_extracted = Path(f"extracted_texts/{project_name}")

# Create output directory if it doesn't exist
output_dir.mkdir(parents=True, exist_ok=True)
output_dir_extracted.mkdir(parents=True, exist_ok=True)

# ------------------------------
# Merge cleaned texts with word-limit constraint
# ------------------------------

merged_text = ""         # accumulator for the current merged file
merged_word_count = 0    # current word count in accumulator
file_index = 1           # merged file counter

# Function to write current merged_text to disk and reset the accumulator
def write_merged_file(text, index):
    output_file = output_dir / f"{project_name}_merged_{index}.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Saved merged text to: {output_file}")

# Process each PDF file in sorted order (optional: sort by name)
for pdf_file in sorted(pdf_dir.glob("*.pdf")):
    print(f"Processing: {pdf_file.name}")
    
    # Extract and clean the text from the current PDF
    raw_text = extract_text_from_pdf(pdf_file)
    cleaned_text = clean_academic_text(raw_text)
    cleaned_text = remove_references_section(cleaned_text)  # Remove references section

    # save the cleaned text to a file
    output_file = output_dir_extracted / f"{pdf_file.stem}.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(cleaned_text)
    
    # Prepare an entry header so you know which PDF the text came from
    entry_header = f"\n\n--- {pdf_file.name} ---\n\n"
    entry_text = entry_header + cleaned_text
    
    # Count the words in the new entry (using split by whitespace)
    entry_word_count = len(entry_text.split())
    
    # If adding this entry would exceed our word limit and we already have text, write current file
    if merged_word_count + entry_word_count > WORD_LIMIT and merged_word_count > 0:
        write_merged_file(merged_text, file_index)
        file_index += 1
        merged_text = ""
        merged_word_count = 0
    
    # Append the entry text to the merged accumulator
    merged_text += entry_text
    merged_word_count += entry_word_count

# Write any remaining merged text to a final file
if merged_text.strip():
    write_merged_file(merged_text, file_index)

print("Merging complete.")
