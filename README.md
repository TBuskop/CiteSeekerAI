# CiteSeekerAI: A Tool for Cited LLM Responses 

## Overview

Imagine you want to research a topic but do not know where to begin. You could either go through a mountain of academic papers to find the needle in the haystack or you could use this tool. This tool acts like a **research assistant** to help you find answers much faster and more effectively.

**What does it do?**

Think of it working in these steps:

1.  **Finds Relevant Papers:**
    *   You give it your research question (or a general topic).
    *   It intelligently searches a massive online library of academic papers (like Scopus).
    *   It picks out the summaries (abstracts) of papers that seem most relevant to your question.

2.  **Gets the Full Story:**
    *   For the most promising paper summaries it found, the tool tries to download the full academic papers.

3.  **Breaks Down Your Big Question:**
    *   Your main research question might be quite broad or complex.
    *   The tool uses smart AI (similar to what powers ChatGPT) to break your big question down into several smaller, more specific sub-questions. This makes finding precise answers easier.

4.  **Reads and Understands the Papers (The Smart Part!):**
    *   The tool then "reads" through the full papers it downloaded.
    *   It's clever enough to understand the *meaning and context* of the text, not just looking for keywords.
    *   It then creates a special, organized mini-library containing only the most relevant pieces of information from these papers that relate to your overall research question.

5.  **Answers Your Questions, Step-by-Step:**
    *   For each of those smaller sub-questions, the tool dives into its specially created mini-library of relevant information.
    *   It uses AI to find the exact snippets from the papers that answer that sub-question.
    *   It then uses AI to write an answer to that sub-question, making sure the answer is based on the information it found in the papers.

6.  **Puts It All Together for a Final Answer:**
    *   Finally, it takes all the answers to the smaller sub-questions and combines them.
    *   This creates a comprehensive, well-supported answer to your original, big research question, all backed by the academic literature it processed.

**In simple terms, you ask a tough research question, and this tool:**

*   🔎 Searches for relevant academic articles.
*   📄 Downloads the most important ones.
*   🧠 "Reads" and "understands" these articles to find the key information.
*   💡 Uses AI to piece together that information and provide you with a detailed answer.

This saves you many hours of manual searching, sifting through papers, and trying to connect all the dots yourself!
  

## Features
  *   **Automated Scopus Search:** Can generate Scopus search strings from a research question.
*   **Full-Text Download:** Automates downloading papers based on DOIs (requires network access to journals, e.g., via university VPN/eduroam).
*   **Sequential Processing & Refinement:** Handles sub-queries one after another and allowing for  refinement of later sub-queries based on prior results.
*   **Configurable:** Uses a central `config.py` queries, , LLM models and default parameters.
*   **Adjusment of LLM system prompts to your needs:** System prompts are provided and can be edited to suit your needs

## Setup
1.  **Clone the repository:**

 ```bash
 
 git clone https://github.com/TBuskop/CiteSeekerAI.git
 
 cd academic_lit_llm_2
 ```

2.  **Install dependencies:**

```bash

 pip install -r requirements.txt

 ```
 
3.  **Environment Variables:**
    Create a `.env` file in the project root directory and add the api key (you can also rename `.env.example` to .env)

```env

 GEMINI_API_KEY="YOUR_GEMINI_API_KEY_HERE"

 ```
  
## Configuration
Adjust the search query for scopus, your question and how many iterations for the question it should go through. The script will recognise if abstracts need te be collected based on previous queries and will only download papers relevant to the question and that are not in the databse already.    

## Running the Workflow

The main workflow orchestrates the entire process from abstract collection using scopus, downloading relevant papers, query decomposition, al the way to final answer generation. It is executed via:

```bash

python src/main.py

```

## Disclaimer
This project includes features for downloading and processing academic papers. Users are solely responsible for ensuring that their use of these features complies with all applicable laws, including copyright regulations, and the terms of service of any websites or APIs accessed. The authors of this project are not liable for any misuse or legal issues arising from the use of this software. Always respect publisher copyrights and terms of use.

