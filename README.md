# 🚀 CiteSeekerAI: Your AI-powered Research Sidekick

Tired of drowning in a sea of academic papers? Let **CiteSeekerAI** be your AI research buddy, helping you quickly find clear answers backed by real citations!

![image](https://github.com/user-attachments/assets/d917e679-0341-428d-ac37-9f8144c6e438)

## ✨ How It Works

Just ask your question, sit back, and let CiteSeekerAI:

1. **🔎 Find the Good Stuff**

   * Takes your research question or topic.
   * Searches huge academic libraries (like Scopus).
   * Picks the best paper summaries for your needs.

2. **📥 Grab the Papers**

   * Automatically fetches full-text versions of promising papers.

3. **🧩 Break it Down**

   * Splits your big question into smaller, bite-sized queries using smart AI.

4. **📖 Read & Understand (Yep, Smart!)**

   * Actually "reads" and comprehends full papers (not just keyword matching).
   * Organizes a neat mini-library focused on your topic.

5. **🎯 Find Exact Answers**

   * Picks out precise snippets that directly answer each sub-question.
   * Uses AI to clearly and concisely summarize answers.

6. **🌟 Deliver One Awesome Answer**

   * Combines these detailed answers into one comprehensive, fully-cited response.

Forget manual searches and long hours reading—let CiteSeekerAI connect the dots!

## 🔥 Key Features

* **Auto Scopus Search:** Turn questions directly into Scopus queries.
* **Paper Fetching:** Automatically grab full-text papers (needs university access).
* **AI-driven Q\&A:** Decompose complex questions into manageable tasks.
* **Fully Customizable:** Easily adjust prompts, parameters, and searches via `config.py`.
* **Web & CLI Interfaces:** Easy-to-use web app and straightforward command-line tool.

## 🛠️ Quick Setup

Clone, install, configure—done!

```bash
git clone https://github.com/TBuskop/CiteSeekerAI.git
cd CiteSeekerAI
conda env create -f environment.yml
conda activate citeseeker
playwright install
```

### 🔑 Set Your Google Gemini [API Key](https://ai.google.dev/gemini-api/docs/api-key) (be sure to add billing information):

Create a `.env` file with your API key:

```env
GEMINI_API_KEY="YOUR_GEMINI_API_KEY_HERE"
```

## 🚦 Running CiteSeekerAI

### 💻 Command Line (CLI)

Simply run:

```bash
python src/main.py
```

### 🌐 Web Interface (Interactive & Fun!)

Run the easy-start script:

```bash
run_web_interface.bat
```

Or manually:

```bash
python src/web_interface/app.py
```

Then visit [http://127.0.0.1:5000](http://127.0.0.1:5000) and start exploring!

### Web Highlights

* Chat-style interaction
* Citation previews on hover
* Customizable research parameters
* Simple abstract management
* Easy batch paper downloads

## ☕ Coffee Break Alert!

* Some queries take a moment—perfect for a coffee run!
* Remember, advanced queries may incur small costs (usually just a few cents).

🎉 Happy researching!


## ⚠️ Disclaimer

This project includes features for downloading and processing academic papers. Users are solely responsible for ensuring that their use of these features complies with all applicable laws, including copyright regulations, and the terms of service of any websites or APIs accessed. The authors of this project are not liable for any misuse or legal issues arising from the use of this software. Always respect publisher copyrights and terms of use.

---
