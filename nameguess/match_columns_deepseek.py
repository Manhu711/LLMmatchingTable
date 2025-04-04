import os
import argparse
import pandas as pd
import logging
import requests
import time
from sentence_transformers import SentenceTransformer, util
import numpy as np
from column_memory import ColumnMatchMemory
import streamlit as st
from simple_medical_rag import SimpleMedicalRAG

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Predefined dictionary of known expansions for validation
KNOWN_EXPANSIONS = {
    "financial": {
        "net_ret": "Net Return",
        "net_rtn": "Net Return",
        "cct": "Credit Card Type",
        "card_type": "Card Type"
    },
    "general": {
        "net_ret": "Net Return",
        "net_rtn": "Net Return",
        "cct": "Customer Contact",
        "card_type": "Card Type"
    }
}

# Add a global variable for the RAG system
medical_rag = None

def init_rag():
    """Initialize the simplified medical RAG system."""
    global medical_rag
    try:
        medical_rag = SimpleMedicalRAG()
        logger.info("Initialized Medical Abbreviation RAG System")
        return medical_rag.is_available
    except Exception as e:
        logger.error(f"Error initializing Medical Abbreviation RAG System: {str(e)}")
        return False

# DeepSeek integration using direct API calls
class DeepSeekLLM:
    def __init__(self, api_key=None):
        """Initialize the DeepSeek LLM client."""
        # Try to get API key from different sources
        try:
            secrets_key = st.secrets.get("DEEPSEEK_API_KEY", None)
        except Exception:
            secrets_key = None
        
        self.api_key = (
            api_key or 
            os.environ.get("DEEPSEEK_API_KEY") or 
            secrets_key
        )
        
        if not self.api_key:
            raise ValueError(
                "DeepSeek API key not found. Please set the DEEPSEEK_API_KEY "
                "environment variable or add it to .streamlit/secrets.toml"
            )
        self.api_base = "https://api.deepseek.com/v1"
        self.model_name = "deepseek-coder"
        logger.info(f"Initialized DeepSeek LLM with model: {self.model_name}")
        
    def __call__(self, prompt, temperature=0.0, max_tokens=1024):
        """Call the model with the given prompt using direct API requests."""
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            data = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            # Make the API request
            response = requests.post(
                f"{self.api_base}/chat/completions", 
                headers=headers,
                json=data
            )
            
            # Check if the request was successful
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                error_msg = f"API request failed with status code {response.status_code}: {response.text}"
                logger.error(error_msg)
                return f"Error: {error_msg}"
                
        except Exception as e:
            logger.error(f"Error calling DeepSeek API: {str(e)}")
            return f"Error: {str(e)}"

def extract_answer(raw_answer_str: str, sep_token: str, abbreviations: list):
    """Process the raw model output into a list of expanded column names."""
    try:
        # Log the raw answer for debugging
        logger.info(f"Raw answer from LLM: {raw_answer_str}")
        
        # Clean the answer string
        raw_answer_str = raw_answer_str.strip()
        
        # First attempt: Split on sep_token (expected format)
        # Try to find the actual list of expansions - most direct format
        answer_list = [_ans.strip() for _ans in raw_answer_str.split(sep_token)]
        if len(answer_list) == len(abbreviations):
            logger.info(f"Successfully parsed with separator token: {answer_list}")
            return answer_list
        
        # Second attempt: Look for a clear list-like format with the same number of items
        import re
        list_pattern = r"(?:^|\n)(?:\d+[\.\)]\s*)?(.+?)(?:$|\n)"
        matches = re.findall(list_pattern, raw_answer_str)
        if len(matches) == len(abbreviations):
            logger.info(f"Successfully parsed with list pattern: {matches}")
            return [match.strip() for match in matches]
        
        # Third attempt: Look for direct mappings in the text
        predictions = []
        for abbr in abbreviations:
            # Try multiple patterns to capture different response formats
            patterns = [
                rf"`{abbr}`[^-]*stands for \"([^\"]+)\"",  # `BP` stands for "Blood Pressure"
                rf"{abbr}[^-]*means \"([^\"]+)\"",         # BP means "Blood Pressure"
                rf"{abbr}[^-]*:[ \t]*([^\n\.,]+)",         # BP: Blood Pressure
                rf"{abbr}[ \t]+->[ \t]+([^\n\.,]+)"        # BP -> Blood Pressure
            ]
            
            for pattern in patterns:
                match = re.search(pattern, raw_answer_str, re.IGNORECASE)
                if match:
                    expanded_name = match.group(1).strip()
                    predictions.append(expanded_name)
                    logger.info(f"Matched {abbr} to {expanded_name} using pattern {pattern}")
                    break
            else:
                # If no match found, try to find any mention of this abbreviation followed by text
                general_pattern = rf"(?:^|\n|\s){re.escape(abbr)}(?:\s|:)+(.*?)(?:\n|$|\.|,)"
                match = re.search(general_pattern, raw_answer_str, re.IGNORECASE)
                if match:
                    expanded_name = match.group(1).strip()
                    predictions.append(expanded_name)
                    logger.info(f"Matched {abbr} to {expanded_name} using general pattern")
                else:
                    predictions.append(abbr)  # Fall back to the abbreviation itself
                    logger.warning(f"Could not find expansion for {abbr}")
        
        if len(predictions) == len(abbreviations):
            logger.info(f"Successfully parsed with direct mapping patterns: {predictions}")
            return predictions
            
        # If we still don't have enough predictions, fill with empty strings
        if len(predictions) < len(abbreviations):
            logger.warning(f"Not enough predictions ({len(predictions)}) for abbreviations ({len(abbreviations)})")
            predictions.extend([" "] * (len(abbreviations) - len(predictions)))
        
        return predictions[:len(abbreviations)]  # Ensure correct length
        
    except Exception as e:
        logger.error(f"Error extracting answer: {str(e)}")
        return [" "] * len(abbreviations)

class PromptTemplate:
    @property
    def demos(self):
        _demo = (
            "As abbreviations of column names from a table, "
            "c_name | pCd | dt stand for Customer Name | Product Code | Date. "
        )
        return _demo

    @property
    def sep_token(self):
        _sep_token = " | "
        return _sep_token

def expand_abbreviations(abbreviations: list, context: str, model: DeepSeekLLM, 
                        prompt_template: PromptTemplate, verbose: bool = False):
    """Expand abbreviations using the LLM and simplified medical RAG."""
    global medical_rag
    
    # Initialize RAG if not already done
    if medical_rag is None:
        init_rag()
    
    # Construct prompt
    query = f"Expand: {' | '.join(abbreviations)} "
    context_part = f"Context: {context}. " if context else ""
    
    # Add RAG context if available
    rag_context = ""
    if medical_rag is not None and medical_rag.is_available:
        for abbr in abbreviations:
            context = medical_rag.get_context_for_llm(abbr)
            if context and "No abbreviation information found" not in context:
                rag_context += context
    
    if rag_context:
        rag_context = f"Medical abbreviation reference:\n{rag_context}\n"
        if verbose:
            print("Using medical RAG context:", rag_context)
    
    prompt = (
        f"{context_part}{rag_context}{prompt_template.demos}{query}"
        "If available, provide the first three most likely expanded names as a list separated by ' | '. "
        "Do not include additional explanations."
    )

    if verbose:
        print("\nPrompt:", prompt)

    raw_answer = model(prompt)
    time.sleep(1)  # Rate limiting
    
    # Extract the top three expanded names
    predictions = extract_answer(raw_answer, prompt_template.sep_token, abbreviations)
    
    # Ensure only the top three predictions are returned
    predictions = [pred.split(prompt_template.sep_token)[:3] for pred in predictions]

    # Join the top three predictions back into a single string for each abbreviation
    predictions = [' | '.join(pred) for pred in predictions]

    if len(predictions) != len(abbreviations):
        logger.warning(f"Mismatch in predictions length. Expected {len(abbreviations)}, got {len(predictions)}")
        predictions = [" "] * len(abbreviations)

    return predictions

@st.cache_resource
def load_models():
    """Cached model loading"""
    deepseek_model = DeepSeekLLM()
    prompt_template = PromptTemplate()
    semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
    return deepseek_model, prompt_template, semantic_model

def match_columns(source_cols, dest_cols, source_expanded, dest_expanded, 
                 semantic_model, verbose=False, memory_matches=None, similarity_threshold=0.5):
    """Match columns between source and destination tables using semantic similarity."""
    if memory_matches is None:
        memory_matches = {}
        
    # Ensure all expanded names are strings
    source_expanded = [str(exp) if exp and exp.strip() else f"Column {src}" 
                      for src, exp in zip(source_cols, source_expanded)]
    dest_expanded = [str(exp) if exp and exp.strip() else f"Column {dst}" 
                    for dst, exp in zip(dest_cols, dest_expanded)]
    
    # Compute embeddings and similarities
    source_embeddings = semantic_model.encode(source_expanded, convert_to_tensor=True)
    dest_embeddings = semantic_model.encode(dest_expanded, convert_to_tensor=True)
    similarity_matrix = util.cos_sim(source_embeddings, dest_embeddings)
    
    # Generate all comparisons
    matches = []
    all_comparisons = []
    highest_similarity = 0
    highest_similarity_pair = None
    
    for i, (src_col, src_exp) in enumerate(zip(source_cols, source_expanded)):
        # Check if we have a memory match first
        if src_col in memory_matches:
            dest_col = memory_matches[src_col]
            dest_idx = dest_cols.index(dest_col)
            match = {
                "source_column": src_col,
                "source_expanded": src_exp,
                "dest_column": dest_col,
                "dest_expanded": dest_expanded[dest_idx],
                "similarity_score": 1.0,  # Perfect score for memory matches
                "from_memory": True
            }
            matches.append(match)
            continue
            
        # If no memory match, use semantic matching
        similarities = similarity_matrix[i]
        best_match_idx = similarities.argmax().item()
        best_score = similarities[best_match_idx].item()
        
        # Track highest similarity pair
        if best_score > highest_similarity:
            highest_similarity = best_score
            highest_similarity_pair = {
                "source_column": src_col,
                "source_expanded": src_exp,
                "dest_column": dest_cols[best_match_idx],
                "dest_expanded": dest_expanded[best_match_idx],
                "similarity_score": best_score
            }
        
        # Store all comparisons
        for j, (dst_col, dst_exp) in enumerate(zip(dest_cols, dest_expanded)):
            score = similarities[j].item()
            all_comparisons.append({
                "source_column": src_col,
                "source_expanded": src_exp,
                "dest_column": dst_col,
                "dest_expanded": dst_exp,
                "similarity_score": score
            })
        
        # Create match entry
        if best_score < similarity_threshold:
            match = {
                "source_column": src_col,
                "source_expanded": src_exp,
                "dest_column": "NO_MATCH",
                "dest_expanded": "NO_MATCH",
                "similarity_score": best_score
            }
        else:
            match = {
                "source_column": src_col,
                "source_expanded": src_exp,
                "dest_column": dest_cols[best_match_idx],
                "dest_expanded": dest_expanded[best_match_idx],
                "similarity_score": best_score
            }
        matches.append(match)
    
    return matches, all_comparisons, highest_similarity_pair

def process_files(source_file, dest_file, context=None, use_memory=True, memory_file="data/column_matches.json", save_new_matches=True):
    """Process source and destination files to match columns."""
    # Initialize our column memory
    if use_memory:
        os.makedirs(os.path.dirname(os.path.abspath(memory_file)), exist_ok=True)
        column_memory = ColumnMatchMemory(memory_file)
    else:
        column_memory = None
    
    # Load files and get columns
    source_df = pd.read_csv(source_file)
    dest_df = pd.read_csv(dest_file)
    source_cols = list(source_df.columns)
    dest_cols = list(dest_df.columns)
    
    # Check memory for existing matches
    memory_matches = {}
    if use_memory and column_memory:
        memory_matches = column_memory.find_matches(source_cols, dest_cols)
        
    # Determine which columns need expansion through the LLM
    columns_to_expand = [col for col in source_cols if col not in memory_matches]
    
    # Only process columns that need expansion
    if columns_to_expand:
        source_expanded = expand_abbreviations(columns_to_expand, context, deepseek_model, prompt_template, verbose)
        all_expanded = []
        expansion_index = 0
        for col in source_cols:
            if col in memory_matches:
                all_expanded.append(f"MEMORY_MATCH:{col}")
            else:
                all_expanded.append(source_expanded[expansion_index])
                expansion_index += 1
    else:
        all_expanded = [f"MEMORY_MATCH:{col}" for col in source_cols]
    
    # Expand destination columns
    dest_expanded = expand_abbreviations(dest_cols, context, deepseek_model, prompt_template, verbose)
    
    # Ensure only the top three predictions are used for destination columns
    dest_expanded = [' | '.join(exp.split(prompt_template.sep_token)[:3]) for exp in dest_expanded]
    
    # Match columns
    matches, all_comparisons, highest_similarity_pair = match_columns(
        source_cols, dest_cols, all_expanded, dest_expanded, 
        semantic_model, memory_matches=memory_matches
    )
    
    # Save new matches if requested
    if use_memory and save_new_matches and column_memory:
        for match in matches:
            source_col = match["source_column"]
            dest_col = match["dest_column"]
            if match["similarity_score"] > 0.85:
                column_memory.add_match(source_col, dest_col)
    
    return matches, all_comparisons, highest_similarity_pair

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-cols", type=str, required=True,
        help="Pipe-separated list of source table column names (e.g., 'net_ret | cct')"
    )
    parser.add_argument(
        "--dest-cols", type=str, required=True,
        help="Pipe-separated list of destination table column names (e.g., 'net_rtn | card_type')"
    )
    parser.add_argument(
        "--context", type=str, default="",
        help="Optional context for the LLM prompt (e.g., 'financial data')"
    )
    parser.add_argument(
        "--model_name", type=str, default="deepseek-coder",
        help="DeepSeek model to use (default: deepseek-coder)"
    )
    parser.add_argument(
        "--api_key", type=str, default=None,
        help="DeepSeek API key (if not provided, will use environment variable or default)"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print detailed debug information"
    )
    args = parser.parse_args()

    # Process input column names
    source_cols = [col.strip() for col in args.source_cols.split("|")]
    dest_cols = [col.strip() for col in args.dest_cols.split("|")]
    if not source_cols or not dest_cols:
        print("No columns provided for source or destination table. Exiting.")
        exit(1)

    # Debug: Print the columns
    if args.verbose:
        print("\nDebug: Source columns:")
        print(source_cols)
        print("Debug: Destination columns:")
        print(dest_cols)

    # Initialize models
    deepseek_model, prompt_template, semantic_model = load_models()

    print(f"Using model: {args.model_name}")
    print("Expanding source table columns...")
    matches, all_comparisons, highest_similarity_pair = process_files(source_cols, dest_cols, args.context, args.verbose)

    # Print all comparisons
    print("\nAll Comparisons:")
    for match in matches:
        print(f"{match['source_column']} ({match['source_expanded']}) vs. {match['dest_column']} ({match['dest_expanded']}), Similarity = {match['similarity_score']:.3f}")

    # Print best matches for each source column
    print("\nBest Matches for Each Source Column:")
    for match in matches:
        print(f"{match['source_column']} ({match['source_expanded']}) --> {match['dest_column']} ({match['dest_expanded']}), Similarity = {match['similarity_score']:.3f}")

    # Save best matches to CSV
    results_df = pd.DataFrame(matches)
    output_dir = "outputs"
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, "matched_columns.csv")
    if os.path.exists(output_file):
        os.remove(output_file)
    results_df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"\nBest matches saved to {output_file}")

    print("Done!") 