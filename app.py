# D:\KCCQueryAssistant_V2\app.py

import streamlit as st
import sys
import os

# Add the 'src' directory to Python's path so it can find rag_pipeline
# This assumes app.py is in D:\KCCQueryAssistant_V2\ 
# and rag_pipeline.py is in D:\KCCQueryAssistant_V2\src\
current_dir = os.path.dirname(__file__)
src_path = os.path.join(current_dir, 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

# Try to import with error handling
try:
    from rag_pipeline import answer_question, load_resources
except ImportError as e:
    st.error(f"Failed to import RAG pipeline from 'src' directory: {e}")
    st.error(f"Ensure rag_pipeline.py is in '{src_path}' and all its dependencies are installed in your Python environment.")
    st.error(f"Current sys.path: {sys.path}")
    st.stop()
except Exception as e:
    st.error(f"An unexpected error occurred during import: {e}")
    st.stop()


# --- Page Configuration ---
st.set_page_config(page_title="KCC Query Assistant", layout="wide")

# --- Load Resources ---
@st.cache_resource # Cache the loaded resources for efficiency
def load_all_resources_cached():
    try:
        load_resources() # This is the function from your rag_pipeline.py
        st.success("AI Resources loaded successfully!")
        return True 
    except FileNotFoundError as e:
        st.error(f"ERROR LOADING RESOURCES (FileNotFound): {e}. Please ensure index files are present in '{os.path.join(current_dir, 'index')}'. Have you run indexer.py for V2?")
        return False
    except ValueError as e:
        st.error(f"ERROR LOADING RESOURCES (ValueError): {e}. Check columns in 'indexed_data.csv'.")
        return False
    except Exception as e:
        st.error(f"An unexpected error occurred while loading AI resources: {e}")
        return False

resources_loaded = load_all_resources_cached()

# --- App UI ---
st.title("🚜 KCC Query Assistant")
st.markdown("Ask a question related to farming. The assistant will try to answer based on the KCC dataset or a web search fallback.")

if not resources_loaded:
    st.error("Critical error: AI Resources could not be loaded. The application cannot function. Please check the console from where you ran Streamlit for detailed error messages.")
else:
    # --- User Input ---
    user_query = st.text_input("Enter your question here:", placeholder="e.g., What are the control measures for aphids in cotton?", key="query_input")

    if st.button("Get Answer", key="get_answer_button"):
        if user_query:
            with st.spinner("Analysing your query and finding the best answer... Please wait."):
                try:
                    # Call your RAG pipeline's main function
                    # It now returns a tuple: (answer_text, source_text)
                    answer_text, source_text = answer_question(user_query)
                    
                    st.subheader("Answer:")
                    st.markdown(answer_text) # Use markdown for better formatting potential
                    
                    # MODIFIED: Display the source clearly
                    if source_text == "KCC Dataset":
                        st.info("ℹ️ This answer is based on information from the KCC Dataset.")
                    elif source_text == "Web Search":
                        st.warning("ℹ️ This answer is based on information retrieved from a web search (DuckDuckGo).")
                    elif source_text == "System (No Answer)":
                        st.error("ℹ️ No specific answer could be formulated based on available KCC data or web search.")
                    else: # Fallback for any other source string
                        st.caption(f"Source: {source_text}")

                except Exception as e:
                    st.error(f"An error occurred while generating the answer: {e}")
                    st.exception(e) # Show full traceback in the app for debugging
        else:
            st.warning("Please enter a question.")

# --- Footer (Optional) ---
st.markdown("---")
st.caption("KCC Query Assistant  - Powered by Ollama, FAISS, Sentence Transformers, and DuckDuckGo.")