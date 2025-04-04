from google import genai
import os
import re

# change run to current directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

temperature = 1
max_tokens = 6000
model = "gemini-2.5-pro-exp-03-25"
initial_query = "what are plausibilistic climate storylines and how are they different from other climate storylines? Write me a short literature review on this topic."
# open prompt file and replace [User Climate Research Question Here] with initial_query
with open("search_string_prompt.txt", "r") as f:
    prompt = f.read()

prompt = prompt.replace("[User Climate Research Question Here]", initial_query)

# load environment variables from .env file
from dotenv import load_dotenv
load_dotenv(override=True)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

generate_content_config = genai.types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=6000,
            response_mime_type="text/plain",
        )

try: # Add a try/except block specifically around the API call for better debugging
    response = gemini_client.models.generate_content(
        model=model,
        contents=prompt,
        config=generate_content_config
    )
    print("Search string generated successfully:")
    
    # Remove all backticks including those that might be in code blocks (```)
    cleaned_response = re.sub(r'`+', '', response.text)
    
    print(cleaned_response)

    # Save the cleaned response to a file
    with open("generated_search_string.txt", "w") as f:
        f.write(cleaned_response)

    # Log the final result after cleaning
    with open("generated_search_string.txt", "r") as f:
        generated_search_string = f.read().strip()
        print(f"Generated search string ready for use: {generated_search_string}")

except genai.exceptions.InvalidArgument as e:
    print(f"Invalid argument: {e}")
except Exception as e:
    print(f"Error generating search string: {e}")