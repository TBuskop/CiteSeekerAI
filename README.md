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
*   **Web Interface:** Provides a user-friendly web interface for interacting with the research assistant, viewing chat history, and collecting abstracts.

## Setup
1.  **Clone the repository:**

 ```bash
 
 git clone https://github.com/TBuskop/CiteSeekerAI.git
 
 ```

2.  **Install dependencies:**

```bash
cd your_path/CiteSeekerAI
conda env create -f environment.yml
conda activate citeseeker
playwright install

 ```
 
3.  **Environment Variables:**
    Create a `.env` file (or rename `.env.example` to .env) in the project root directory and add the [api key](https://ai.google.dev/gemini-api/docs/api-key). Be sure to setup the billing information.

```env

 GEMINI_API_KEY="YOUR_GEMINI_API_KEY_HERE"

 ```
  
## Configuration
Adjust the search query for scopus, your question and how many iterations for the question it should go through. The script will recognise if abstracts need te be collected based on previous queries and will only download papers relevant to the question and that are not in the databse already.    

## Running the Workflow

There are two main ways to interact with CiteSeekerAI:

### 1. Command-Line Interface (CLI)

The main workflow orchestrates the entire process from abstract collection using Scopus, downloading relevant papers, query decomposition, all the way to final answer generation. It is executed via:

```bash

python src/main.py

```

### 2. Web Interface

CiteSeekerAI also offers a web-based interface for a more interactive experience.

**Features of the Web Interface:**
*   **Chat-based Interaction:** Ask research questions directly in a chat window.
*   **View Previous Questions:** Access and review the history of your past research queries and their answers.
*   **Citation Context:** Hover over citations to see the original text to get a better understanding of the statement. 
*   **Abstract Collection:** Collecting abstracts based on your search terms.

**How to Run the Web Interface:**

Simply execute or double click the `run_web_interface.bat` script located in the root directory of the project:

```bash
run_web_interface.bat
```
This script will:
1. Activate the necessary conda environment.
2. Start the Flask server.
3. Automatically open the web interface in your default browser (usually at `http://127.0.0.1:5000`).

Or run

```bash

python src/web_interface/app.py

```

Once opened collect abstracts first before asking a question.

![image](https://github.com/user-attachments/assets/d917e679-0341-428d-ac37-9f8144c6e438)


## Disclaimer
This project includes features for downloading and processing academic papers. Users are solely responsible for ensuring that their use of these features complies with all applicable laws, including copyright regulations, and the terms of service of any websites or APIs accessed. The authors of this project are not liable for any misuse or legal issues arising from the use of this software. Always respect publisher copyrights and terms of use.

## To Note
- Sometimes the process takes a while. Run the script, get a coffee and come back to your result.
- Each query does cost a little bit of money. A query can be very cheap (less than a cent) or be very complex involving advanced reasoning models and many subqueries and a large amount of chunks being sent to the LLM to generate and answer.
