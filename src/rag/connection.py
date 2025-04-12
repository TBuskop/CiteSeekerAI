from google import genai
import os

temperature = 0.5
max_tokens = 1000
model = "gemini-2.5-pro-exp-03-25"
n_queries = 5
initial_query = "what are plausibilistic climate storylines and how are they different from other climate storylines? Write me a short literature review on this topic."
prompt = f"""You are an expert research assistant specializing in generating precise search queries for climate science, climate impacts, climate change literature databases.
                Your task is to decompose the user's original query into {n_queries} specific and diverse sub-queries optimized for a hybrid retrieval system combining lexical (like BM25) and semantic (vector-based) search.

                The goal is to retrieve distinct, relevant chunks of information from academic papers on climate science, climate change, and climate impacts, which will then be reranked to best answer the original query.

                **Query Optimization Guidelines:**
                *   **Hybrid System Focus:** Generate queries that work well for *both* keyword matching and semantic understanding. Prioritize precise keywords and concepts.
                *   **Keyword & Operator Driven:** Formulate queries primarily using essential keywords. Strategically use boolean operators (`AND`, `OR`) and exact phrase matching (`"..."`) to increase precision for the lexical search component. For example, `"sea level rise" AND "future projections" AND "Mediterranean"`.
                *   **Facet Coverage & Diversity:** Ensure each sub-query targets a *distinct facet* or angle of the original query. Use the decomposition strategies below to ensure variety. Avoid significant overlap between sub-queries.
                *   **Specificity:** Use terminology common in academic climate science literature. Avoid vague terms.
                *   **Format:** Generate search terms/phrases, not full natural language questions.

                **Decomposition Strategies to Consider:**
                *   Specific climate phenomena (e.g., `"sea level rise"`, `"extreme heat events"`, `"ocean acidification"`)
                *   Geographic regions or ecosystems (e.g., `"Arctic"`, `"Sahel region"`, `"coral reefs"`)
                *   Time scales or periods (e.g., `"paleo-climate"`, `"future projections"`, `"21st century"`)
                *   Methodologies (e.g., `"climate modeling"`, `"observational data analysis"`, `"impact assessment methodology"`)
                *   Specific impacts (e.g., `impacts AND "agriculture"`, `impacts AND "human health"`, `impacts AND "biodiversity"`)
                *   Underlying mechanisms or drivers (e.g., `"greenhouse gas emissions"`, `"aerosol effects"`, `"climate feedbacks"`)
                *   **Policy impacts or mitigation/adaptation strategies** (e.g., `"climate adaptation strategies"`, `"disaster risk management" AND "climate change"`, `"climate finance" AND "developing countries"`, `"mitigation policy"`) # <-- Added this line
                *   Comparisons or Definitions (e.g., `"X definition"`, `"comparison between X and Y"`, `"X versus Y"`)

                Original Query:
                '{initial_query}'

                Generate exactly {n_queries} focused sub-queries based on the Original Query, optimized according to the guidelines above.
                Return ONLY the sub-queries, each on a new line. Do not include numbering, bullet points, explanations, or introductory text. Avoid overly broad reformulations of the original query.
                """

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

generate_content_config = genai.types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            response_mime_type="text/plain",
        )

try: # Add a try/except block specifically around the API call for better debugging
    response = gemini_client.models.generate_content(
        model=model,
        contents=prompt,
        config=generate_content_config
    )
    print("Sub-queries generated successfully:")
    print(response.text)

except genai.exceptions.InvalidArgument as e:
    print(f"Invalid argument: {e}")