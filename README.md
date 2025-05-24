# KCC Query Assistant 🚜

## Harvesting Knowledge: Your AI-Powered Agricultural Guide

The KCC Query Assistant is a locally-run AI application designed to empower users with agricultural insights from the Kisan Call Center (KCC) dataset. It offers a seamless way to ask farming-related questions in natural language and receive answers derived from a vast collection of real-world farmer queries and solutions. When specific information isn't found within this local knowledge base, the assistant intelligently switches to a live Internet search to provide the most relevant and up-to-date information possible.

Built with a "local-first" philosophy, this tool prioritizes offline capability and data privacy by utilizing a local Large Language Model (LLM) through Ollama, ensuring that sensitive queries and data interactions remain on your machine.

## Key Features & Advancements

* **Natural Language Queries:** Ask questions just as you would in a normal conversation. [cite: 9]
* **Offline KCC Database Search:** Leverages a preprocessed and indexed version of the KCC dataset for quick, local information retrieval. [cite: 1, 5]
* **Intelligent Web Fallback:**
    * Automatically performs a live web search (via DuckDuckGo) if the KCC dataset doesn't contain a sufficiently relevant answer. [cite: 8]
    * Specific queries recognized as time-sensitive (e.g., "weather," "price") are directly routed to web search.
* **Source Transparency:** The user interface clearly indicates whether an answer is sourced from the KCC Dataset or from a Web Search. [cite: 10]
* **Local & Private LLM:** Powered by Gemma (a state-of-the-art open model) running locally via Ollama, ensuring your queries are processed offline. [cite: 3, 4]
* **CPU Optimized:** Designed and configured to run efficiently on standard CPU hardware. [cite: 4]
* **Enhanced Data Quality:** The KCC dataset has undergone significant cleaning (both advanced external methods and programmatic filtering) to improve the reliability of retrieved information.
* **Multi-Layered Relevance Assessment:** A sophisticated pipeline evaluates KCC context using semantic similarity thresholds, programmatic filters for non-substantive answers, and an LLM-based "Judge" with few-shot prompting to assess the quality and relevance of KCC answers before use.

## System Architecture & Workflow

The KCC Query Assistant employs a Retrieval-Augmented Generation (RAG) strategy:

1.  **Data Ingestion & Preprocessing (`src/preprocess.py`):**
    * The raw KCC dataset (CSV) is loaded. [cite: 11]
    * It undergoes extensive cleaning: HTML removal, text normalization (lowercase, punctuation removal), and advanced filtering to remove empty, non-informative, or very short Q&A pairs. [cite: 2] The cleaned, structured Q&A data is saved.
2.  **Embedding Generation & Indexing (`src/indexer.py`):**
    * The 'answer' portions of the cleaned KCC Q&A pairs are converted into numerical vector embeddings using the `all-mpnet-base-v2` sentence-transformer model. [cite: 5]
    * These embeddings are stored in a local FAISS vector index (`index/kcc_faiss.index`) for efficient semantic search. [cite: 6] A corresponding CSV (`index/indexed_data.csv`) stores the Q&A pairs for easy lookup.
3.  **Query Handling & RAG Pipeline (`src/rag_pipeline.py`):**
    * The user submits a query via the Streamlit UI. [cite: 11]
    * **Keyword Routing:** Time-sensitive queries (containing keywords like "weather," "price," "latest news") are immediately routed to web search.
    * **KCC Semantic Search:** For other queries, a semantic search is performed on the FAISS index to find the top-K most relevant KCC Q&A pairs. [cite: 7, 11]
    * **KCC Context Evaluation (Multi-Layered):**
        * **High-Confidence Path:** If a retrieved KCC entry has an exceptionally strong semantic match (below `HIGH_CONFIDENCE_L2_THRESHOLD`) AND passes programmatic quality filters, its context is used directly.
        * **LLM Judge Path:** If matches are good (below general `L2_DISTANCE_THRESHOLD`) but not high-confidence, and they pass programmatic filters, the context is presented to an LLM "Judge" (Gemma 3, guided by few-shot examples) to assess its sufficiency and directness in answering the user's query.
    * **Web Fallback Decision:** If no KCC context is used (either due to forced web search, failing thresholds, failing programmatic filters, or LLM Judge deeming it insufficient), the system performs a live web search using DuckDuckGo. [cite: 8, 11] The user is implicitly notified by the answer source indication.
    * **Answer Generation:** The selected context (from KCC or web, with the current date added for web context) is combined with the user's query into a prompt for the local Gemma 3 LLM, which generates the final answer.
4.  **User Interface (`app.py`):**
    * A simple and intuitive web interface built with Streamlit allows users to input queries and view answers. [cite: 9]
    * The source of each answer (KCC Dataset or Web Search) is clearly displayed. [cite: 10]

## Project Structure
KCCQueryAssistant/
├── app.py                    # Main Streamlit application UI
├── data/
│   ├── kcc_raw.csv           # User-provided raw KCC dataset
│   └── kcc_cleaned.csv       # Cleaned data (output of preprocess.py, input to indexer.py)
├── index/
│   ├── kcc_faiss.index       # FAISS vector index
│   └── indexed_data.csv      # Q&A data mapped to the index
├── src/
│   ├── preprocess.py         # Cleans and prepares kcc_raw.csv
│   ├── indexer.py            # Generates embeddings and FAISS index
│   └── rag_pipeline.py       # Core RAG logic and LLM interaction
├── README.md                 # This guide
├── requirements.txt          # Python dependencies for the project
└── (demo.mp4)                # User-created demo video

## Getting Started: Setup & Launch

**1. Prerequisites:**
    * Python (version 3.9 recommended).
    * Ollama: Ensure it's installed and running. Download from [ollama.com](https://ollama.com).
    * Git (optional, for version control).

**2. Download the LLM Model:**
    Open your terminal or command prompt and run:
    ```bash
    ollama pull gemma3:1b-it-qat
    ```

**3. Environment & Dependencies:**
    Navigate to the project root directory (e.g., `D:\KCCQueryAssistant`). It's best practice to use a Python virtual environment:
    ```bash
    python -m venv .venv
    # Activate:
    # Windows: .venv\Scripts\activate
    # macOS/Linux: source .venv/bin/activate
    ```
    Install the necessary packages:
    ```bash
    pip install -r requirements.txt
    ```

**4. Prepare the KCC Dataset:**
    * Obtain the raw KCC dataset (as a CSV file).
    * Place it in the `D:\KCCQueryAssistant\data\` folder and name it `kcc_raw.csv`.

**5. Running the KCC Query Assistant (Order is Important):**

    * **Step A: Preprocess Data**
        Run from the project root directory:
        ```bash
        python src/preprocess.py
        ```
        *(Or using your specific interpreter: `"C:/Program Files/Python39/python.exe" src/preprocess.py`)*
        This will generate `data/kcc_cleaned.csv`.

    * **Step B: Create Embeddings & Index**
        This can take a significant amount of time (1-2+ hours). Run from the project root:
        ```bash
        python src/indexer.py
        ```
        *(Or: `"C:/Program Files/Python39/python.exe" src/indexer.py`)*
        This creates files in the `index/` directory.

    * **Step C: Launch the Application**
        Ensure Ollama is running. Then, from the project root:
        ```bash
        streamlit run app.py
        ```
        *(Or: `"C:/Program Files/Python39/python.exe" -m streamlit run app.py`)*
        The application will open in your web browser.

## Key Technologies

* **LLM:** Gemma 3 (`gemma3:1b-it-qat`) via Ollama
* **Embeddings:** `all-mpnet-base-v2` (Sentence Transformers)
* **Vector Store:** FAISS
* **Web Search:** DuckDuckGo (`duckduckgo-search` library)
* **UI:** Streamlit
* **Core Libraries:** Pandas, NumPy, Requests, BeautifulSoup4

## Sample Queries to Demonstrate Capabilities

*(Please curate a list of 10-12 diverse queries that best showcase your system, including successful KCC retrievals with good answers, intelligent web fallbacks for out-of-scope topics, and correct routing for time-sensitive queries like weather/price. Below are examples to adapt.)*

1.  `control measure for aphid infestation in cotton`
2.  `how to get kisan credit card loan`
3.  `fertilizer dose for knol khol 1 bigha and planting distance`
4.  `where to get training on prawn culture`
5.  `weather update for [Your Local Area]`
6.  `current market price of tomatoes in [Your Local Market]`
7.  `latest news on organic farming subsidies`
8.  `who is the current agriculture minister of India`
9.  `Explain the benefits of drip irrigation`
10. `What are common soil deficiencies in [A Specific Region in India]?`
11. `How to manage drought stress in groundnut cultivation?` [cite: 12]
12. `What pest-control methods are recommended for paddy in Tamil Nadu?` [cite: 11]

## System Limitations & Future Directions

* **KCC Data Quality:** The accuracy of KCC-based answers is fundamentally tied to the quality, completeness, and timeliness of the information within the indexed KCC dataset. While significant cleaning has been performed, residual issues or outdated information in the source data may still arise.
* **Keyword Routing Scope:** The current list for forcing web search (weather, price, etc.) is effective for those terms but isn't exhaustive for all possible time-sensitive queries.
* **LLM Evaluation Nuances:** The LLM Judge, while guided by few-shot examples, is still a probabilistic model, and its evaluation of KCC context sufficiency might not be perfect in every edge case.
* **Web Information Veracity:** The web search fallback provides recent information snippets but does not include an advanced, independent fact-verification layer for all web content.
* **Semantic Search Specificity:** While powerful, the semantic search might not always retrieve the single most ideal document if other documents are numerically closer in the embedding space due to query phrasing.

**Potential Future Enhancements:**
* Ongoing curation and updates to the KCC knowledge base.
* Implementation of more dynamic query understanding (e.g., LLM-based intent classification) for smarter routing to KCC, specific APIs (e.g., dedicated weather/market APIs), or general web search.
* Addition of a re-ranking stage for retrieved documents to further improve relevance.
* User feedback mechanisms to refine answers and identify problematic KCC entries.

This KCC Query Assistant represents a robust effort to build an intelligent, local-first agricultural information tool, balancing a rich offline dataset with the ability to tap into real-time web knowledge.#   D i b y e n d u D e b _ K C C Q u e r y A s s i s t a n t  
 