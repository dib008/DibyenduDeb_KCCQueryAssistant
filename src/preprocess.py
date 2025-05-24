# D:\KCCQueryAssistant\src\preprocess.py

import pandas as pd
from bs4 import BeautifulSoup # To remove HTML
import re # For regular expressions (stripping punctuation, etc.)
import os

# --- Configuration ---
# Path to your raw KCC data file
raw_data_path = r"D:\KCCQueryAssistant\data\kcc_raw.csv"
# Path to save the cleaned data
cleaned_data_output_path = r"D:\KCCQueryAssistant\data\kcc_cleaned_for_indexing.csv" # New name to avoid confusion

# Define your question and answer column names based on your CSV
# From your previous output, these were 'questions' and 'answers'
question_col = 'questions'
answer_col = 'answers'

# List of known non-informative answers (all lowercase)
# Add to this list if you find more such terms in your data
NON_INFORMATIVE_ANSWERS = [
    "explain", "explained", "describe him", "described", "given", "c", 
    "ok", "yes", "no", "na", "n/a", "", "none", "nil", "as above",
    "see above", "not available", "not applicable", "no information",
    "advised", "suggested" # These might be too broad, review their impact
]

MIN_ANSWER_LENGTH = 15  # Minimum number of characters for an answer to be considered informative (after cleaning)

# --- Helper Functions for Cleaning ---
def remove_html_tags(text):
    if isinstance(text, str):
        return BeautifulSoup(text, "html.parser").get_text()
    return text

def normalize_text(text):
    if isinstance(text, str):
        text = text.lower()  # Convert to lowercase
        text = re.sub(r'[^\w\s]', '', text)  # Strip punctuation (keeps letters, numbers, underscore, whitespace)
        text = " ".join(text.split())  # Remove extra whitespace
        return text
    return "" # Return empty string if not a string (e.g. if NaN made it through)

def is_informative(answer_text):
    """Checks if an answer is informative."""
    if not isinstance(answer_text, str) or not answer_text.strip():
        return False # Empty or whitespace only
    
    # Check against the list of known non-informative standalone answers
    if answer_text in NON_INFORMATIVE_ANSWERS:
        return False
    
    # Check for minimum length
    if len(answer_text) < MIN_ANSWER_LENGTH:
        return False
        
    return True

# --- Main Preprocessing Logic ---
print(f"Starting preprocessing of KCC data from: {raw_data_path}")

# 1. Load Data
try:
    df = pd.read_csv(raw_data_path, encoding='utf-8')
    print(f"Successfully loaded dataset. Original shape: {df.shape}")
except FileNotFoundError:
    print(f"ERROR: Raw data file not found at {raw_data_path}. Exiting.")
    exit()
except Exception as e:
    print(f"ERROR: Could not load raw data. {e}. Exiting.")
    exit()

# 2. Validate Columns
if question_col not in df.columns or answer_col not in df.columns:
    print(f"ERROR: Specified question ('{question_col}') or answer ('{answer_col}') columns not found in the CSV.")
    print(f"Available columns are: {df.columns.tolist()}")
    exit()

# 3. Initial Cleaning & Normalization
print("Removing HTML and normalizing text (lowercase, strip punctuation, extra whitespace)...")
df[question_col] = df[question_col].apply(remove_html_tags).apply(normalize_text)
df[answer_col] = df[answer_col].apply(remove_html_tags).apply(normalize_text)

# 4. Drop rows with empty questions or answers *after* initial normalization
print("Dropping rows with empty questions or answers after initial normalization...")
# An empty string was returned by normalize_text if input was not a string or became empty
df.dropna(subset=[question_col, answer_col], inplace=True) # Handles actual NaN values if any
df = df[df[question_col].str.strip() != '']
df = df[df[answer_col].str.strip() != '']
print(f"Shape after dropping empty Q/A: {df.shape}")

# 5. Filter out non-informative answers
print(f"Filtering out non-informative answers (e.g., based on keywords like {NON_INFORMATIVE_ANSWERS} or length < {MIN_ANSWER_LENGTH} chars)...")
# The is_informative function expects the already normalized answer_text
df['is_answer_informative'] = df[answer_col].apply(is_informative)
df_cleaned = df[df['is_answer_informative']].copy() # Use .copy() to avoid SettingWithCopyWarning
df_cleaned.drop(columns=['is_answer_informative'], inplace=True)
print(f"Shape after filtering non-informative answers: {df_cleaned.shape}")

# 6. Remove Duplicates
# Based on both question and the (now cleaner) answer
print("Removing duplicate question-answer pairs...")
df_cleaned.drop_duplicates(subset=[question_col, answer_col], inplace=True, keep='first')
print(f"Shape after removing duplicates: {df_cleaned.shape}")

# 7. Final Check for Empty Strings (just in case)
df_cleaned = df_cleaned[df_cleaned[question_col].str.strip().str.len() > 0]
df_cleaned = df_cleaned[df_cleaned[answer_col].str.strip().str.len() > 0]
print(f"Final shape after all cleaning: {df_cleaned.shape}")

# 8. Save Cleaned Data
if df_cleaned.empty:
    print("ERROR: No data remaining after cleaning. Output file will not be saved. Please check your data and cleaning parameters.")
else:
    try:
        # Ensure output directory exists (if data directory is specified in the path)
        output_dir = os.path.dirname(cleaned_data_output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        df_cleaned[[question_col, answer_col]].to_csv(cleaned_data_output_path, index=False, encoding='utf-8')
        print(f"\nCleaned and filtered data saved successfully to: {cleaned_data_output_path}")
        print(f"The cleaned dataset contains {len(df_cleaned)} rows.")
        print("\nSample of the final cleaned data:")
        print(df_cleaned[[question_col, answer_col]].head())
    except Exception as e:
        print(f"ERROR: Could not save cleaned data. {e}")

print("\nPreprocessing script finished.")