# D:\KCCQueryAssistant\src\indexer.py

import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import time
import os

# --- Configuration ---
# MODIFIED: Path to point to your new ChatGPT-cleaned data file
cleaned_data_path = r"D:\KCCQueryAssistant\data\kcc_cleaned.csv" 

# Define model name - 'all-mpnet-base-v2' is a good starting point
embedding_model_name = 'all-mpnet-base-v2'

# Path to save the FAISS index and mapping
output_folder = r"D:\KCCQueryAssistant\index" 
faiss_index_path = os.path.join(output_folder, "kcc_faiss.index")
# This CSV will store the questions and answers that correspond to the indexed embeddings.
# It will be overwritten with data from the new cleaned file.
indexed_data_correspondence_path = os.path.join(output_folder, "indexed_data.csv") 

# IMPORTANT: Define your question and answer column names as they appear in your new kcc_cleaned.csv
# Assuming they are still 'questions' and 'answers'. If not, change these:
question_col = 'questions' 
answer_col = 'answers'     

# Ensure output directory exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"Created output directory: {output_folder}")

# --- 1. Load Cleaned Data ---
print(f"Loading cleaned data from {cleaned_data_path}...")
try:
    df = pd.read_csv(cleaned_data_path)
    print(f"Successfully loaded cleaned data. Shape: {df.shape}")
except FileNotFoundError:
    print(f"ERROR: Cleaned data file not found at {cleaned_data_path}. Please ensure the file exists at this path.")
    exit()
except Exception as e:
    print(f"ERROR: Could not load cleaned data. {e}")
    exit()

if df.empty:
    print(f"ERROR: The cleaned data file at {cleaned_data_path} is empty. Cannot proceed with indexing.")
    exit()

# Validate columns
if not {question_col, answer_col}.issubset(df.columns):
    print(f"ERROR: Expected columns '{question_col}' and/or '{answer_col}' not found in {cleaned_data_path}.")
    print(f"Available columns are: {df.columns.tolist()}")
    print(f"Please update the 'question_col' and 'answer_col' variables in this script if needed.")
    exit()
    
# We will index the 'answers' column.
texts_to_index = df[answer_col].astype(str).tolist() # Convert to list of strings
print(f"Loaded {len(texts_to_index)} texts from column '{answer_col}' to index.")

if not texts_to_index:
    print(f"ERROR: No texts found in column '{answer_col}' to index. Please check the CSV file and column name.")
    exit()

# --- 2. Load Embedding Model ---
print(f"Loading sentence transformer model: {embedding_model_name}...")
start_time = time.time()
model = SentenceTransformer(embedding_model_name, device='cpu') 
end_time = time.time()
print(f"Model loaded in {end_time - start_time:.2f} seconds.")

# --- 3. Generate Embeddings ---
print(f"Generating embeddings for {len(texts_to_index)} texts... This might take a while.")
start_time = time.time()
embeddings = model.encode(texts_to_index, show_progress_bar=True, convert_to_numpy=True)
end_time = time.time()
print(f"Embeddings generated in {end_time - start_time:.2f} seconds.")

if embeddings.size == 0:
    print("ERROR: Embedding generation resulted in an empty array. Cannot proceed.")
    exit()
print(f"Shape of embeddings: {embeddings.shape}")

# --- 4. Set up & Build FAISS Index ---
embedding_dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(embedding_dimension) 
print(f"FAISS index created with dimension {embedding_dimension} using IndexFlatL2.")

print("Adding embeddings to FAISS index...")
start_time = time.time()
index.add(embeddings.astype(np.float32)) 
end_time = time.time()
print(f"Embeddings added to index in {end_time - start_time:.2f} seconds.")
print(f"Total vectors in index: {index.ntotal}")

# --- 5. Save the Index and Data Correspondence File ---
print(f"Saving FAISS index to {faiss_index_path}...")
faiss.write_index(index, faiss_index_path)

print(f"Saving the corresponding Q&A data (used for indexing) to {indexed_data_correspondence_path}...")
df_to_save_correspondence = df[[question_col, answer_col]]
df_to_save_correspondence.to_csv(indexed_data_correspondence_path, index=False)

print("\nEmbedding generation and indexing complete!")
print(f"FAISS index saved at: {faiss_index_path}")
print(f"Indexed Q&A data correspondence file saved at: {indexed_data_correspondence_path}")