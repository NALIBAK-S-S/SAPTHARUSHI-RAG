# RAG for Saptharushi

This app uses Retrieval-Augmented Generation (RAG) with Gemini and LanceDB to answer questions about the `saptharushi.pdf` document. It includes a Streamlit UI for interactive Q&A.

---

## Setup Instructions

### 1. Clone or Download the Project

Place all files (including `saptharushi.pdf`) in a folder, 

### 2. Create and Activate a Virtual Environment

Open a terminal in the project folder and run:

**Windows:**
```sh
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```sh
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Requirements

```sh
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Set Up API Keys

- **Gemini API Key:**  
  Get your Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey).
- **Nomic API Key:**  
  Get your Nomic API key from [Nomic](https://platform.nomic.ai/).

Set them as environment variables:

**Windows (Command Prompt):**
```sh
set GEMINI_API_KEY=your-gemini-key
set NOMIC_API_KEY=your-nomic-key
```

**Linux/Mac (Bash):**
```sh
export GEMINI_API_KEY=your-gemini-key
export NOMIC_API_KEY=your-nomic-key
```

### 5. Prepare the Vector Database

Run the following to process the PDF, create contextualized chunks, and store them in LanceDB:

```sh
python creation.py
```

### 6. Start the Streamlit App

```sh
streamlit run streamlit.py
```

The app will open in your browser. You can ask questions about the document and see the conversation history.

---

## Troubleshooting

- Make sure `saptharushi.pdf` is present in the project folder.
- Ensure your API keys are set before running the scripts.
- If you see missing package errors, run `pip install -r requirements.txt` again.

---

## File Overview

- `requirements.txt` – Python dependencies
- `creation.py` – Processes PDF, creates and embeds contextualized chunks
- `streamlit.py` – Streamlit Q&A interface
- `saptharushi.pdf` – Source document

---
