from google import genai
import os
import re
from pathlib import Path
from typing import Tuple, Optional

# load environment variables from .env file
from dotenv import load_dotenv

def generate_scopus_search_string(query: str, save_to_file: bool = True) -> Tuple[bool, Optional[str]]:
    """
    Generate a search string from a research question using the Gemini API.
    
    Args:
        query: The research question
        save_to_file: Whether to save the search string to a file
        
    Returns:
        Tuple containing (success: bool, search_string: Optional[str])
    """
    try:
        # Change to the directory of this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_dir)
        
        # Load environment variables
        load_dotenv(override=True)
        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        
        if not GEMINI_API_KEY:
            print("Error: GEMINI_API_KEY not found in environment variables")
            return False, None
        
        temperature = 1
        max_tokens = 6000
        model = "gemini-1.5-flash-8b" #"gemini-2.5-pro-exp-03-25"
        
        # Add "Write me a short literature review on this topic" to the query
        full_query = f"{query} Write me a short literature review on this topic."
        
        # Open prompt file and replace placeholder with the query
        with open(os.path.join(script_dir, "../llm_prompts/search_string_prompt.txt"), "r") as f:
            prompt = f.read()
        
        prompt = prompt.replace("[User Climate Research Question Here]", full_query)
        
        # Initialize Gemini client
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        
        generate_content_config = genai.types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=6000,
            response_mime_type="text/plain",
        )
        
        response = gemini_client.models.generate_content(
            model=model,
            contents=prompt,
            config=generate_content_config
        )
        
        print("Search string generated successfully:")
        
        # Remove all backticks including those that might be in code blocks (```)
        cleaned_response = re.sub(r'`+', '', response.text)
        
        print(cleaned_response)
        
        if save_to_file:
            # Save the cleaned response to a file
            output_file = os.path.join(script_dir, "generated_search_string.txt")
            with open(output_file, "w") as f:
                f.write(cleaned_response)
            
            print(f"Generated search string saved to {output_file}")
        
        return True, cleaned_response
    
    except genai.exceptions.InvalidArgument as e:
        print(f"Invalid argument: {e}")
        return False, None
    except Exception as e:
        print(f"Error generating search string: {e}")
        return False, None

# Keep the original script functionality when run directly
if __name__ == "__main__":
    initial_query = "what are plausibilistic climate storylines and how are they different from other climate storylines?"
    success, search_string = generate_search_string(initial_query)
    
    if success:
        print(f"Generated search string ready for use: {search_string}")
    else:
        print("Failed to generate search string")