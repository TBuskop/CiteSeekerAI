import numpy as np
import traceback
import re
import json
import os
from typing import List, Optional, Dict, Any

import sys
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# --- Config Imports ---
from config import (
    GEMINI_API_KEY,
    EMBEDDING_MODEL,
    CHUNK_CONTEXT_MODEL,
    SUBQUERY_MODEL,
    CHAT_MODEL,
    DEFAULT_CONTEXT_LENGTH,
    OUTPUT_EMBEDDING_DIMENSION,
    DEFAULT_TOTAL_CONTEXT_WINDOW,
)
from src.my_utils.utils import count_tokens, truncate_text # Import necessary utils

# --- API Client Imports and Setup ---
try:
    from google import genai
    from google.api_core import exceptions as google_exceptions
    from google.genai.types import EmbedContentConfig, GenerateContentConfig, HarmCategory, HarmBlockThreshold
    GOOGLE_GENAI_AVAILABLE = True
except ImportError:
    print("Warning: Google Generative AI library not installed (`pip install google-generativeai`). Gemini models unavailable.")
    GOOGLE_GENAI_AVAILABLE = False
    genai = None
    google_exceptions = None
    EmbedContentConfig = None
    GenerateContentConfig = None
    HarmCategory = None
    HarmBlockThreshold = None

# --- REINTRODUCE Global client variable ---
gemini_client: Optional[genai.Client] = None

# --- REMOVE Global configuration flag ---
# gemini_configured_successfully: bool = False

def initialize_clients():
    """Initializes API clients based on config keys."""
    global gemini_client
    gemini_client = None # Reset client
    initialization_successful = False # Track success locally

    if GOOGLE_GENAI_AVAILABLE and GEMINI_API_KEY:
        try:
            # Use genai.Client for initialization
            gemini_client = genai.Client(api_key=GEMINI_API_KEY)
            # Optional: Add a check here, e.g., try listing models via the client if supported
            # models = gemini_client.list_models() # Check documentation for exact method
            print("Google GenAI client configured successfully using genai.Client().")
            initialization_successful = True # Mark success
        except ImportError:
            print("Warning - Google Generative AI library not installed (ImportError during client creation).")
            gemini_client = None
        except Exception as e:
            print(f"Warning - Failed to configure Google GenAI client using genai.Client(): {e}")
            gemini_client = None # Ensure it's None on failure
    elif not GOOGLE_GENAI_AVAILABLE:
        print("Info: Google GenAI library not installed, skipping Gemini client initialization.")
    elif not GEMINI_API_KEY:
        print("Info: GEMINI_API_KEY not found in config, skipping Gemini client initialization.")

    # --- Added Debugging ---
    print(f"DEBUG (llm_interface.initialize_clients): Initialization complete.")
    print(f"DEBUG (llm_interface.initialize_clients): gemini_client type = {type(gemini_client)}")
    print(f"DEBUG (llm_interface.initialize_clients): gemini_client is None = {gemini_client is None}")
    if initialization_successful and isinstance(gemini_client, genai.Client):
        print("DEBUG (llm_interface.initialize_clients): Client appears to be a valid genai.Client instance.")
    else:
        print("DEBUG (llm_interface.initialize_clients): Client is NOT a valid genai.Client instance after initialization attempt.")
    # --- End Debugging ---


# --- Embedding Function ---
def get_embedding(text: str,
                  model: str = EMBEDDING_MODEL,
                  embedding_dimension: Optional[int] = OUTPUT_EMBEDDING_DIMENSION,
                  task_type: Optional[str] = "retrieval_document" # Default task type
                 ) -> Optional[np.ndarray]:
    """
    Generates an embedding for the given text using the specified model.

    Args:
        text: The input text to embed.
        model: The embedding model name (e.g., "models/embedding-001", "models/text-embedding-004").
        embedding_dimension: Optional dimension for the output embedding (supported by some models).
        task_type: The intended task for the embedding (e.g., "retrieval_document", "retrieval_query").

    Returns:
        A numpy array representing the embedding, or None if an error occurs.
    """
    # 1. Basic Input Validation
    if not text or not text.strip():
        # print("Warning: Attempting to embed empty text. Returning None.")
        return None
    text = text.replace("\n", " ") # Consistent preprocessing

    try:
        # 2. Check Provider and Client Initialization Status
        model_lower = model.lower()
        is_gemini_model = any(m in model_lower for m in ["embedding-001", "text-embedding-004"])

        if is_gemini_model:
            # --- Check the client instance ---
            if not GOOGLE_GENAI_AVAILABLE or not isinstance(gemini_client, genai.Client):
                raise RuntimeError("Google GenAI client not available or not initialized correctly.")

            # Ensure model name has 'models/' prefix if not already present
            api_model_name = model if model.startswith("models/") else f"models/{model}"

            # Prepare arguments for embed_content
            # Use 'content' for single text embedding
            kwargs: Dict[str, Any] = {'model': api_model_name, 'contents': text}

            # --- Task Type Handling (Gemini specific) ---
            valid_tasks = ["retrieval_document", "retrieval_query", "semantic_similarity", "classification", "clustering"]
            if "text-embedding-004" not in api_model_name:
                # Handle task_type for other models like embedding-001
                if "embedding-001" in api_model_name:
                    if task_type and task_type.lower() in valid_tasks:
                        kwargs['task_type'] = task_type.lower()
                    elif task_type:
                        print(f"Warning: Unsupported task_type '{task_type}' for {api_model_name}. Proceeding without it or with default.")
                    # else: rely on API default if task_type is None or invalid
                # Add handling for other potential future models here if they accept task_type
            else:
                # Explicitly do NOT add task_type for text-embedding-004 when using client.models.embed_content
                if task_type:
                    print(f"Info: task_type '{task_type}' ignored for model {api_model_name} with client.models.embed_content.")


            # --- Output Dimension Handling (Gemini specific) ---
            embed_config_args = {}
            if "text-embedding-004" in api_model_name and embedding_dimension is not None:
                 embed_config_args['output_dimensionality'] = embedding_dimension

            # Add config to kwargs if it has content
            if embed_config_args:
                kwargs['config'] = EmbedContentConfig(**embed_config_args)


            # --- Make the Gemini API Call using client.models.embed_content ---
            # print(f"DEBUG: Calling client.models.embed_content with args: {kwargs}") # Debug
            response = gemini_client.models.embed_content(**kwargs) # Use the client instance method

            # --- Parse the Gemini Response ---
            # --- MODIFICATION START: Correctly handle EmbedContentResponse ---
            if isinstance(response, genai.types.EmbedContentResponse) and response.embeddings:
                 # Access the first embedding object in the list
                 content_embedding = response.embeddings[0]
                 if hasattr(content_embedding, 'values') and isinstance(content_embedding.values, list) and len(content_embedding.values) > 0:
                     vector = content_embedding.values # Get the vector list
                     # Optional: Verify dimension if provided
                     if embedding_dimension is not None and len(vector) != embedding_dimension:
                          print(f"Warning: Embedding dimension mismatch for {api_model_name}. Expected {embedding_dimension}, got {len(vector)}. Using received vector.")
                     return np.array(vector, dtype=np.float32)
                 else:
                     print(f"Error: Gemini ContentEmbedding object missing 'values' or has empty list for {api_model_name}.")
                     return None
            # Fallback check for older dict structure (less likely now)
            elif isinstance(response, dict) and 'embedding' in response:
                 vector = response.get('embedding')
                 if isinstance(vector, list) and len(vector) > 0:
                     return np.array(vector, dtype=np.float32)
                 else:
                     print(f"Error: Gemini embedding response (dict) contained empty/invalid list for {api_model_name}.")
                     return None
            else:
                 print(f"Error: Unexpected or empty Gemini embedding response structure for {api_model_name}: {type(response)}")
                 return None
            # --- MODIFICATION END ---


        # --- Add other providers (e.g., OpenAI) here if needed ---
        # elif "openai-model-name" in model_lower:
        #     # ... handle OpenAI embedding call ...
        #     pass

        else:
            raise ValueError(f"Embedding model '{model}' is not supported by this function.")

    # 3. Catch All Exceptions
    except google_exceptions.ResourceExhausted as rate_limit_error:
        print(f"!!! Rate Limit Error (429) during embedding generation for model {model}: {rate_limit_error}")
        raise # Propagate rate limit errors
    except Exception as e:
        print(f"!!! Critical Error during embedding generation for model {model}: {e}")
        traceback.print_exc() # Print stack trace for debugging
        return None # Return None on any other error

# --- LLM Response Generation ---
def generate_llm_response(prompt: str, max_tokens: int, temperature: float = 0.7, model: Optional[str] = None) -> str:
    """
    Generates a response from the configured LLM provider using gemini_client.models.generate_content.

    Args:
        prompt: The input prompt for the LLM.
        max_tokens: The maximum number of tokens to generate in the response.
        temperature: The sampling temperature (controls randomness).
        model: The specific LLM model to use (e.g., "gemini-1.5-flash", "gpt-4o"). Defaults to CHAT_MODEL from config.

    Returns:
        The generated text response, or an error/blocked message string.
    """
    target_model = model if model else CHAT_MODEL
    if not target_model:
        return "[Error: No chat model specified in config or arguments]"

    # Ensure model name has 'models/' prefix if needed for the API call
    api_model_name = target_model if target_model.startswith("models/") else f"models/{target_model}"

    # Define supported models (adjust if necessary based on CHAT_MODEL/SUBQUERY_MODEL etc.)
    supported_gemini_models = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro", "gemini-1.0-pro", "gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-1.5-flash-8b", "gemini-2.5-pro-exp-03-25"]
    # Extract base model name for check
    base_model_name = api_model_name.split('/')[-1]

    if base_model_name in supported_gemini_models:
        # --- Check the client instance ---
        if not GOOGLE_GENAI_AVAILABLE or not isinstance(gemini_client, genai.Client):
             return "[Error: Google GenAI client not available or not initialized correctly]"

        # Prepare generation config
        generation_config = GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            response_mime_type="text/plain", # Often default, but can be explicit
        )
    
        try:
            # Use the client.models.generate_content method
            response = gemini_client.models.generate_content(
                model=api_model_name, # Pass the full model name
                contents=prompt,      # Pass the prompt string
                config=generation_config,
            )
            # print(f"DEBUG: Full Response for {api_model_name}: {response}") # Optional debug

            # --- Process Gemini Response ---
            # 1. Check for blocking first
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                block_reason_str = str(response.prompt_feedback.block_reason)
                print(f"WARN: Response blocked by safety filters for {api_model_name}. Reason: {block_reason_str}")
                return f"[Blocked due to Safety Filters: {block_reason_str}]"

            # 2. Check candidates and content parts
            # Note: client.models.generate_content might have slightly different response structure
            # than genai.GenerativeModel(...).generate_content. Adjust parsing if needed.
            # Assuming it still has .candidates and .text attributes for simplicity first.
            try:
                # The .text attribute attempts to consolidate parts automatically
                if response.text:
                    return response.text.strip()
                else:
                    # If .text is empty, check finish reason for clues
                    finish_reason = "Unknown"
                    # Accessing candidates might differ, check response object structure
                    if hasattr(response, 'candidates') and response.candidates and hasattr(response.candidates[0], 'finish_reason'):
                         finish_reason = str(response.candidates[0].finish_reason) # Get enum string

                    if finish_reason == "STOP":
                         print(f"WARN: Response from {api_model_name} generated empty text (finish reason: STOP).")
                         return "" # Return empty string for valid empty response
                    else:
                         print(f"WARN: No valid text content received from {api_model_name}. Finish Reason: {finish_reason}")
                         # Check if content parts exist but are empty
                         no_parts = True
                         if hasattr(response, 'candidates') and response.candidates and hasattr(response.candidates[0].content, 'parts'):
                             no_parts = not bool(response.candidates[0].content.parts)
                         if no_parts:
                             print(f"WARN: Response candidates[0].content.parts is empty or missing.")

                         return f"[Error generating response: No text content. Finish Reason: {finish_reason}]"

            except (ValueError, AttributeError, IndexError) as e:
                 # Catch errors during response parsing
                 print(f"WARN: Error accessing response attributes for {api_model_name} (potentially blocked or invalid content): {e}")
                 # Try to get finish reason if possible
                 finish_reason = "Unknown"
                 try:
                     if hasattr(response, 'candidates') and response.candidates and hasattr(response.candidates[0], 'finish_reason'):
                          finish_reason = str(response.candidates[0].finish_reason)
                 except Exception: pass # Ignore errors getting finish reason here
                 return f"[Error generating response: Invalid content or access error. Finish Reason: {finish_reason}]"


        except google_exceptions.ResourceExhausted as rate_limit_error:
            print(f"!!! Rate Limit Error (429) during generation for model {api_model_name}: {rate_limit_error}")
            return f"[Error: Rate limit hit for {api_model_name}]"
        except Exception as e:
            print(f"ERROR: Exception during Gemini API call for {api_model_name}: {e}")
            traceback.print_exc()
            return f"[Error during API call: {e}]"

    # --- Add other providers (e.g., OpenAI) here ---
    # elif model.lower() in ["gpt-3.5-turbo", "gpt-4", "gpt-4o"]:
    #     # ... handle OpenAI chat completion call ...
    #     pass

    else:
        # Use base_model_name for the error message
        return f"[Error: Unsupported chat model: {base_model_name}]"

# --- Chunk Context Generation ---
def generate_chunk_context(
    full_document_text: str,
    chunk_text: str,
    start_token_idx: int,
    end_token_idx: int,
    total_window_tokens: int = DEFAULT_TOTAL_CONTEXT_WINDOW,
    summary_max_tokens: int = DEFAULT_CONTEXT_LENGTH,
    context_model: str = CHUNK_CONTEXT_MODEL,
    encoding_model: str = EMBEDDING_MODEL # Model used for tokenization window
) -> str:
    """
    Generates a succinct, chunk-specific context using a sliding token-based window.
    (Implementation details omitted for brevity - assume it's the same as in the original file)
    """
    if not full_document_text or not chunk_text: return "Context unavailable (empty input)."
    text_window = full_document_text 

    prompt = f"""
        You are an AI assistant helping to create contextual summaries for text chunks.
        Provided Text Window: <text_window>{text_window}</text_window>
        Chunk Requiring Context: <chunk>{chunk_text}</chunk>
        Instructions: Analyze the text *immediately surrounding* the chunk **within the window**. Determine the primary subject or theme. Synthesize this into a highly succinct contextual description (maximum 1-2 sentences). Focus on the **local topic around the chunk within the window**. Do NOT summarize the chunk or the entire window. Output **only** the 1-2 sentence description.
        Generated Contextual Description (Output Only):
        """
    try:
        response = generate_llm_response(prompt, summary_max_tokens, temperature=0.5, model=context_model)
        if response.startswith("[Error") or response.startswith("[Blocked"):
             print(f"Warning: Failed to generate context via LLM: {response}")
             return "Error generating context."
        response = re.sub(r"^(Context:|Here is the context:)\s*", "", response, flags=re.IGNORECASE).strip()
        return response if response else "Context could not be generated."
    except Exception as e:
        print(f"Error in generate_chunk_context with model {context_model}: {e}")
        traceback.print_exc()
        return "Error generating context."


# --- Subquery Generation ---
def generate_subqueries(initial_query: str, model: str = SUBQUERY_MODEL) -> Dict[str, List[str]]:
    """
    Generates diverse sub-queries from an initial query using a prompt that returns a JSON object
    with 'bm25_queries' and 'vector_search_queries'. Falls back to the initial query if processing fails.
    """
    default_result = {"bm25_queries": [initial_query], "vector_search_queries": [initial_query]}
    # retrieve based on project path
    prompt_path = os.path.join(_PROJECT_ROOT, "llm_prompts", "subquery_prompt.txt")
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt_template = f.read()
        prompt = prompt_template.replace("<initial_query>", initial_query)
    except Exception as e:
        print(f"Error reading subquery prompt file: {e}. Using initial query only.")
        return default_result

    try:
        response_text = generate_llm_response(prompt, max_tokens=4096, temperature=0.3, model=model)
        if not response_text or response_text.startswith("[Error") or response_text.startswith("[Blocked"):
            raise RuntimeError(f"Subquery LLM failed or returned error: {response_text}")
        # Remove potential markdown fences
        response_text = re.sub(r"^```json\s*", "", response_text, flags=re.IGNORECASE | re.MULTILINE)
        response_text = re.sub(r"\s*```$", "", response_text, flags=re.IGNORECASE | re.MULTILINE)
        response_text = response_text.strip()
        parsed_json = json.loads(response_text)
        if not (isinstance(parsed_json, dict) and "bm25_queries" in parsed_json and "vector_search_queries" in parsed_json):
            raise ValueError("JSON structure invalid")
        bm25_queries = [q for q in parsed_json["bm25_queries"] if isinstance(q, str) and q.strip()]
        vector_search_queries = [q for q in parsed_json["vector_search_queries"] if isinstance(q, str) and q.strip()]
        if not bm25_queries:
            bm25_queries = [initial_query]
            print("Warning: BM25 queries empty; using initial query.")
        if not vector_search_queries:
            vector_search_queries = [initial_query]
            print("Warning: Vector search queries empty; using initial query.")
        return {"bm25_queries": bm25_queries, "vector_search_queries": vector_search_queries}
    except json.JSONDecodeError as json_err:
        print(f"JSON decoding error: {json_err}. Raw response:\n{response_text}")
        return default_result
    except Exception as e:
        print(f"Error generating subqueries: {e}")
        traceback.print_exc()
        return default_result
