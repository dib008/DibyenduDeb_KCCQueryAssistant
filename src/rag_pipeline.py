# D:\KCCQueryAssistant\src\rag_pipeline.py

import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import json
import requests # For Ollama API
from duckduckgo_search import DDGS # For actual web search fallback
import datetime # To get the current date
import re # For more advanced programmatic filtering

# --- Configuration ---
index_folder = r"D:\KCCQueryAssistant\index"
faiss_index_path = os.path.join(index_folder, "kcc_faiss.index")
indexed_data_path = os.path.join(index_folder, "indexed_data.csv") 

embedding_model_name = 'all-mpnet-base-v2'
ollama_model_name = 'gemma3:1b-it-qat'

TOP_K_RESULTS_KCC = 3
TOP_K_RESULTS_WEB = 3

L2_DISTANCE_THRESHOLD = 0.8 
HIGH_CONFIDENCE_L2_THRESHOLD = 0.4 

FORCE_WEB_SEARCH_KEYWORDS = [
    "weather", "forecast", 
    "price", "market price", "rate", "cost of", "value of",
    "latest news", "current events", "update on", "what's new", "today's news"
] 

# MODIFIED: Refined regex for "asking for/about"
PROGRAMMATIC_NON_ANSWER_PATTERNS = [
    r"^\s*explain(ed)?(\s.*)?$", r"^\s*describe(d)?(\s.*)?$",
    r"^\s*details given.*", r"^\s*information given.*", r"^\s*will be provided.*",
    r"^\s*(please\s+)?contact\s+.*(department|office|kvk| Kendra|university|college|expert|specialist|officer|personnel|help desk|helpline|center|centre).*$",
    r"^\s*(please\s+)?visit\s+.*(department|office|kvk| Kendra|university|college|expert|specialist|officer|personnel|help desk|helpline|center|centre).*$",
    r"^\s*(please\s+)?consult\s+.*(department|office|kvk| Kendra|university|college|expert|specialist|officer|personnel|help desk|helpline|center|centre).*$",
    r"^\s*(please\s+)?approach\s+.*(department|office|kvk| Kendra|university|college|expert|specialist|officer|personnel|help desk|helpline|center|centre).*$",
    r"^\s*(please\s+)?refer\s+to\s+.*$", r"^\s*see\s+(above|below|details|document|website).*$",
    r"^\s*adviced?(\s+to)?\s+(contact|visit|consult|approach).*$", 
    r"^\s*suggested(\s+(him|her|them|to))?\s+(contact|visit|consult|approach).*$",
    r"^\s*(yes|no|ok|c|na|n\/a|nil|none)\s*$", 
    r"^\s*not\s+available\s*$", r"^\s*not\s+applicable\s*$", r"^\s*no\s+information\s*$",
    r"^\s*asking(\s+about|\s+for)?\s+.*", # MODIFIED: Made more general (removed "the\s+")
    r"^\s*query(\s+regarding|\s+about)?\s+.*"
]
MIN_KCC_ANSWER_SUBSTANCE_LENGTH = 25 

# ... (Rest of the script remains IDENTICAL to the version where I introduced HIGH_CONFIDENCE_L2_THRESHOLD) ...
# (Includes load_resources, embed_query, search_faiss, _ollama_llm_call, generate_llm_response,
# is_kcc_context_sufficient, perform_web_search, the detailed answer_question logic,
# and the if __name__ == "__main__" block)
# For brevity, I will not repeat the entire script if only this list of patterns changed.
# Please make sure you make this change in the last complete script I provided.

# --- Global variables ---
sentence_model = None
faiss_index = None
df_indexed_data = None

def load_resources():
    global sentence_model, faiss_index, df_indexed_data
    print("Loading resources for KCC Query Assistant...")
    sentence_model = SentenceTransformer(embedding_model_name, device='cpu')
    if not os.path.exists(faiss_index_path):
        raise FileNotFoundError(f"FAISS index not found at {faiss_index_path}.")
    faiss_index = faiss.read_index(faiss_index_path)
    if not os.path.exists(indexed_data_path):
        raise FileNotFoundError(f"Indexed data CSV ({indexed_data_path}) not found.")
    df_indexed_data = pd.read_csv(indexed_data_path)
    if not {'questions', 'answers'}.issubset(df_indexed_data.columns):
        raise ValueError(f"Indexed data CSV ({indexed_data_path}) must contain 'questions' and 'answers' columns.")
    print(f"Resources loaded: FAISS index with {faiss_index.ntotal} vectors, {len(df_indexed_data)} indexed Q&A pairs.")

def embed_query(query_text):
    if sentence_model is None: load_resources()
    query_embedding = sentence_model.encode([query_text], convert_to_numpy=True)
    return query_embedding.astype(np.float32)

def search_faiss(query_embedding, k=TOP_K_RESULTS_KCC):
    if faiss_index is None: load_resources()
    distances, indices = faiss_index.search(query_embedding, k)
    return distances[0], indices[0]

def _ollama_llm_call(prompt_text, temperature=0.2, model=ollama_model_name):
    payload = {
        "model": model, "prompt": prompt_text, "stream": False,
        "options": {"temperature": temperature}
    }
    try:
        response = requests.post("http://localhost:11434/api/generate", json=payload, timeout=90)
        response.raise_for_status()
        response_data = response.json()
        return response_data.get("response", "Error: No 'response' field in LLM output.")
    except requests.exceptions.Timeout:
        print("Error: Ollama API request timed out.")
        return "LLM_TIMEOUT_ERROR"
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to Ollama API: {e}")
        return "LLM_CONNECTION_ERROR"
    except json.JSONDecodeError:
        print(f"Error decoding JSON from Ollama. Response: {response.text if response else 'No response'}")
        return "LLM_JSON_DECODE_ERROR"

def generate_llm_response(prompt_text):
    return _ollama_llm_call(prompt_text, temperature=0.2)

def is_kcc_context_sufficient(user_query, kcc_context_str): 
    print("DEBUG: Asking LLM (with few-shot examples) to judge KCC context sufficiency...")
    max_context_len_for_judge = 2000 
    if len(kcc_context_str) > max_context_len_for_judge:
        kcc_context_str = kcc_context_str[:max_context_len_for_judge] + "..."
    prompt = f"""You are an AI evaluator. Your task is to determine if the 'Retrieved KCC Context' provides a direct, self-contained, and truly informative answer to the 'User Query'.

Here are some examples of how to evaluate:
Example 1: User Query: "control of gandhi insect" / Retrieved KCC Context: "KCC Answer: spray rogor @2ml/litre" / Evaluation: YES
Example 2: User Query: "kcc loan details" / Retrieved KCC Context: "KCC Answer: explain in detail about the kcc loan." / Evaluation: NO
Example 3: User Query: "paddy variety for my area" / Retrieved KCC Context: "KCC Answer: contact kvk for details" / Evaluation: NO
Example 4: User Query: "information about soil testing" / Retrieved KCC Context: "KCC Answer: soil testing is important for good crop yield. it helps determine nutrient levels." / Evaluation: YES
Example 5: User Query: "fertilizer dose for knol khol 1 bigha and planting distance" / Retrieved KCC Context: "KCC Answer: 23, 41, 13 and 200 kilo grams of urea, ssp, mop and farm yard manure respectively have been suggested for 1 bigha of land for knol khol cultivation. planting distance suggested is 30 centimetre from row to row and 25 centimetre from plant to plant." / Evaluation: YES

Now, evaluate the following:
User Query: "{user_query}"
Retrieved KCC Context:
---
{kcc_context_str}
---
Carefully consider if the 'Retrieved KCC Context' *actually provides the specific information or details* asked for in the 'User Query'.
Is it a direct answer, or is it merely a statement about providing information, a generic referral, a placeholder, too vague, or otherwise does not substantively answer the query?
Respond with only the word "YES" if the context is a direct, substantive, and informative answer. Respond with only the word "NO" otherwise.
"""
    judgement = _ollama_llm_call(prompt, temperature=0.0) 
    cleaned_judgement = judgement.strip().lower().split(' ')[0] if judgement.strip() else "" 
    cleaned_judgement = re.sub(r'[^\w]', '', cleaned_judgement) 
    print(f"DEBUG: LLM Judgement for KCC context (raw: '{judgement.strip()}', processed: '{cleaned_judgement}'): '{cleaned_judgement}'")
    return cleaned_judgement == "yes"

def perform_web_search(query_text, max_results=TOP_K_RESULTS_WEB):
    print(f"Performing live web search for: '{query_text}' using DuckDuckGo (max_results={max_results})...")
    try:
        with DDGS() as ddgs: results = list(ddgs.text(query_text, max_results=max_results))
        if not results: return f"No web results for '{query_text}' via DuckDuckGo."
        snippets = [res.get('body', '') for res in results if res.get('body')]
        if not snippets: return f"No usable snippets for '{query_text}' via DuckDuckGo."
        print(f"DEBUG: Snippets from web search ({len(snippets)} retrieved): {snippets}")
        return "\n---\n".join(snippets)
    except Exception as e:
        print(f"DuckDuckGo Web search error: {e}")
        return f"Failed web search for '{query_text}'. Error: {e}"

def answer_question(user_query):
    print(f"\nUser Query: {user_query}")
    if sentence_model is None: load_resources()

    llm_answer_text = ""
    answer_source = "System (No Answer Found)" 
    kcc_context_was_used = False 

    query_lower = user_query.lower()
    force_web = any(keyword in query_lower for keyword in FORCE_WEB_SEARCH_KEYWORDS)

    if force_web:
        matched_keywords = [k for k in FORCE_WEB_SEARCH_KEYWORDS if k in query_lower] 
        print(f"DEBUG: Query contains keyword(s) ({matched_keywords}). Forcing web search directly.")
    else:
        print("DEBUG: Not a forced web search query. Attempting KCC search & evaluation...")
        query_emb = embed_query(user_query)
        distances, indices = search_faiss(query_emb)
        print(f"Retrieved top raw KCC results (distances, indices): {distances}, {indices}")
        
        retrieved_kcc_context_parts_for_high_conf = []
        retrieved_kcc_context_parts_for_judge = []
        
        best_initial_kcc_distance = float('inf')
        used_high_confidence_kcc = False

        if len(indices) > 0 and indices[0] != -1:
            best_initial_kcc_distance = distances[0]
            
            for i in range(len(indices)): 
                idx = indices[i]
                if not (0 <= idx < len(df_indexed_data)):
                    print(f"Warning: KCC index {idx} out of bounds.")
                    continue

                q = df_indexed_data.iloc[idx].get('questions', 'N/A')
                a_raw = df_indexed_data.iloc[idx].get('answers', 'N/A')
                a_cleaned_for_check = a_raw.strip().lower()
                is_substantive = True

                if a_cleaned_for_check == 'n/a' or len(a_cleaned_for_check) < MIN_KCC_ANSWER_SUBSTANCE_LENGTH:
                    is_substantive = False
                    print(f"DEBUG: KCC Answer snippet (idx {idx}) '{a_cleaned_for_check[:50]}...' FAILED length/NA check.")
                else:
                    for pattern in PROGRAMMATIC_NON_ANSWER_PATTERNS:
                        if re.match(pattern, a_cleaned_for_check):
                            is_substantive = False
                            print(f"DEBUG: KCC Answer snippet (idx {idx}) '{a_cleaned_for_check[:50]}...' matched non-answer pattern: '{pattern}'")
                            break 
                
                if is_substantive:
                    # Check for high confidence first
                    if distances[i] < HIGH_CONFIDENCE_L2_THRESHOLD:
                        print(f"DEBUG: KCC Answer (idx {idx}) MET HIGH_CONFIDENCE_L2_THRESHOLD ({distances[i]:.4f} < {HIGH_CONFIDENCE_L2_THRESHOLD}) AND passed programmatic filter.")
                        retrieved_kcc_context_parts_for_high_conf.append(f"KCC Question: {q}\nKCC Answer: {a_raw}")
                        used_high_confidence_kcc = True # Flag that at least one high-confidence snippet was found and used
                    # Else if it meets general threshold, add for LLM Judge
                    elif distances[i] < L2_DISTANCE_THRESHOLD:
                        retrieved_kcc_context_parts_for_judge.append(f"KCC Question: {q}\nKCC Answer: {a_raw}")
                        print(f"DEBUG: KCC Answer (idx {idx}) passed programmatic filter and general threshold ({distances[i]:.4f}). Adding for LLM Judge consideration.")
                    else: # distance[i] >= L2_DISTANCE_THRESHOLD
                        print(f"DEBUG: KCC Answer (idx {idx}) distance {distances[i]:.4f} exceeds L2_DISTANCE_THRESHOLD {L2_DISTANCE_THRESHOLD}. Stopping KCC context collection.")
                        break # No need to check further KCC results for this path
            
            if used_high_confidence_kcc: # If any snippet met high confidence, use collected high-confidence parts
                kcc_context_for_generation = "\n\n---\n\n".join(retrieved_kcc_context_parts_for_high_conf)
                prompt = f"User's Question: {user_query}\n\nBased *only* on the following highly relevant information from the KCC (Kisan Call Centre) database (High Confidence), provide a direct and concise answer:\n\nKCC Information:\n{kcc_context_for_generation}\n\nAnswer:"
                print(f"\nDEBUG: USING HIGHLY CONFIDENT KCC CONTEXT (Best Initial Dist: {best_initial_kcc_distance:.4f}):\n---\n{kcc_context_for_generation}\n---\n")
                llm_answer_text = generate_llm_response(prompt)
                answer_source = "KCC Dataset (High Confidence)"
                kcc_context_was_used = True
            elif retrieved_kcc_context_parts_for_judge: # No high-confidence, but have candidates for judge
                kcc_context_for_judging = "\n\n---\n\n".join(retrieved_kcc_context_parts_for_judge)
                print(f"\nDEBUG: KCC CONTEXT CANDIDATE (passed programmatic filter, best initial distance: {best_initial_kcc_distance:.4f}). SENDING TO LLM JUDGE:\n---\n{kcc_context_for_judging}\n---\n")
                if is_kcc_context_sufficient(user_query, kcc_context_for_judging):
                    print("DEBUG: LLM Judge deemed KCC context SUFFICIENT.")
                    prompt = f"User's Question: {user_query}\n\nBased *only* on the following relevant information from the KCC (Kisan Call Centre) database (LLM Judged), provide a direct and concise answer:\n\nKCC Information:\n{kcc_context_for_judging}\n\nAnswer:"
                    llm_answer_text = generate_llm_response(prompt)
                    answer_source = "KCC Dataset (LLM Judged)"
                    kcc_context_was_used = True
                else: # LLM Judge deemed KCC insufficient
                    print("DEBUG: LLM Judge deemed KCC context INSUFFICIENT. Will proceed to web search.")
                    kcc_context_was_used = False 
            else: # No snippets passed programmatic filters or L2 threshold
                 print(f"DEBUG: No KCC context passed programmatic filters or met general threshold ({L2_DISTANCE_THRESHOLD}) (best KCC distance: {best_initial_kcc_distance:.4f if best_initial_kcc_distance != float('inf') else 'N/A'}). Will proceed to web search.")
                 kcc_context_was_used = False

        # If KCC was used, return the answer now
        if kcc_context_was_used:
            return llm_answer_text, answer_source

    # Fallback to web search
    # This block is reached if force_web OR if kcc_context_was_used is False
    if force_web or not kcc_context_was_used:
        if not force_web:
             print("INFO: KCC context not used or deemed insufficient. Proceeding to web search.")
        # else: # Message for forced web already printed in the `if force_web:` block.
        #      print("INFO: Forced web query. Proceeding to web search.")

        web_snippets_str = perform_web_search(user_query)
        current_date_str = datetime.datetime.now().strftime("%Y-%m-%d (%A)")
        no_web_results_indicators = ["Failed to perform web search", "No web results for", "No usable snippets for"]
        is_error_or_no_results = any(indicator in web_snippets_str for indicator in no_web_results_indicators)

        if is_error_or_no_results or not web_snippets_str.strip():
            llm_answer_text = "I could not find a specific answer in the KCC database, and my attempt to search the web did not yield useful results or encountered an error. Please try rephrasing your question."
            answer_source = "System (No Answer)"
        else:
            print(f"\nDEBUG: WEB CONTEXT TO LLM:\n---\n{web_snippets_str}\n---\n")
            prompt = f"Today's date is {current_date_str}. Please answer the user's question: '{user_query}'. Use the following web search snippets as your primary source of information. Synthesize a concise answer. If the snippets provide conflicting information or are insufficient for a clear answer, please state that. Do not invent information not present in the snippets.\n\nWeb Search Snippets:\n{web_snippets_str}\n\nAnswer:"
            print("Prompting LLM with web search data...")
            llm_answer_text = generate_llm_response(prompt)
            answer_source = "Web Search"
            
    return llm_answer_text, answer_source

if __name__ == "__main__":
    load_resources()
    test_queries = [
        "fertilizer dose of knol khol to be cultivated in a land of 1 bigha along with the planting distance.",
        "asking for the fertilizers doses to be applied in the 6th year stage of coconut trees along with the procedure of application.", 
        "asked about kcc loan", 
        "control measure for aphid infestation in cotton", 
        "how to get kisan credit card loan",
        "where to get training on prawn culture.",
        "weather update Punjab", 
        "what is the market price of onions in Nashik today",
        "current gold rate in India",
        "who is the current president of Brazil", 
        "latest government schemes for organic farming 2024",
        "Explain the process of photosynthesis",
        "What is the capital of Japan?"
    ]
    for query in test_queries:
        final_answer_text, final_answer_source = answer_question(query)
        print(f"\nFinal Answer for '{query}' (Source: {final_answer_source}):\n{final_answer_text}\n{'='*50}\n")