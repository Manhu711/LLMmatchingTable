import streamlit as st
import pandas as pd
import os
from pathlib import Path

# Define the memory file path as a constant to ensure consistency
MEMORY_FILE_PATH = os.path.join(os.path.expanduser("~"), ".column_matcher", "memory.json")

# Set API key for development
if not os.environ.get("DEEPSEEK_API_KEY"):
    # Try to get from Streamlit secrets
    try:
        os.environ["DEEPSEEK_API_KEY"] = st.secrets["DEEPSEEK_API_KEY"]
    except:
        st.error("DeepSeek API key not found. Please configure it in your environment or Streamlit secrets.")

from match_columns_deepseek import DeepSeekLLM, PromptTemplate, expand_abbreviations, match_columns, init_rag
from sentence_transformers import SentenceTransformer
from simple_medical_rag import SimpleMedicalRAG
import re

st.set_page_config(page_title="Column Matcher", layout="wide")

def create_aligned_tables(source_cols, source_expanded, dest_cols, dest_expanded):
    """Create two aligned tables showing original and expanded column names"""
    # Create DataFrames for both tables with unique indices
    source_df = pd.DataFrame({
        'Original Column': source_cols,
        'Expanded Name': source_expanded
    }).drop_duplicates()
    
    dest_df = pd.DataFrame({
        'Original Column': dest_cols,
        'Expanded Name': dest_expanded
    }).drop_duplicates()
    
    # Apply styling
    def style_df(df):
        return df.style.set_properties(**{
            'background-color': '#f0f2f6',
            'color': '#1e1e1e',
            'border': '1px solid #e6e9ef',
            'text-align': 'left',
            'padding': '0.5rem',
            'font-size': '0.9rem'
        }).set_properties(
            subset=['Original Column'],
            **{
                'font-weight': 'bold',
                'background-color': '#e6e9ef'
            }
        ).set_table_styles([
            {'selector': 'thead th', 'props': [
                ('background-color', '#4e7496'), 
                ('color', 'white'),
                ('font-weight', 'bold'),
                ('text-align', 'center'),
                ('padding', '0.5rem')
            ]},
            {'selector': 'tbody tr:nth-of-type(odd)', 'props': [
                ('background-color', '#f8f9fa')
            ]}
        ])
    
    return style_df(source_df), style_df(dest_df)

def get_dest_options(source_col, all_comparisons):
    """Get all destination options for a source column sorted by similarity"""
    dest_options = []
    for comp in all_comparisons:
        if comp['source_column'] == source_col:
            dest_options.append({
                'column': comp['dest_column'],
                'expanded': comp['dest_expanded'],
                'score': comp['similarity_score']
            })
    # Sort options by similarity score
    dest_options.sort(key=lambda x: x['score'], reverse=True)
    return dest_options

def display_matches(matches, all_comparisons):
    """Display matches with editing capability"""
    st.subheader("Column Matching Results")
    
    # Ensure we have edited_matches in session state
    if 'edited_matches' not in st.session_state:
        st.session_state.edited_matches = matches.copy()
    
    # Ensure we have editing_state in session state
    if 'editing_state' not in st.session_state:
        st.session_state.editing_state = {i: False for i in range(len(matches))}
    
    # Get destination columns from session state
    if 'dest_df' in st.session_state:
        dest_cols = st.session_state.dest_df.columns.tolist()
    else:
        # Fallback options
        dest_cols = list(set([comp["dest_column"] for comp in all_comparisons if comp["dest_column"] != "NO_MATCH"]))
        if not dest_cols:
            dest_cols = [match["dest_column"] for match in matches if match["dest_column"] != "NO_MATCH"]

    # Add memory matches to the top of dropdown if available
    memory_options = []
    if 'editing_source_col' in st.session_state:
        try:
            from column_memory import ColumnMatchMemory
            memory = ColumnMatchMemory(MEMORY_FILE_PATH)
            src_col = st.session_state.editing_source_col
            memory_matches = memory.get_match(src_col)
            
            if memory_matches is not None:  # Check explicitly for None
                for mem_match in memory_matches:
                    if mem_match in dest_cols:
                        memory_options.append(f"🔄 {mem_match} (from memory)")
        except Exception as e:
            pass

    # Create the dropdown options
    dropdown_options = memory_options + dest_cols + ["-- No match --"]
    
    # Create a set of already matched destination columns
    matched_dest_cols = {
        match['dest_column'] for match in st.session_state.edited_matches 
        if match['dest_column'] != "NO_MATCH"
    }
    
    # Now display the matches
    for i, match in enumerate(st.session_state.edited_matches):
        col1, col2, col3, col4, col5, col6 = st.columns([3, 3, 2, 1, 1, 1])  # Added an extra column for No Match button
        
        with col1:
            st.text(f"{match['source_column']} ({match['source_expanded']})")
        
        with col2:
            if st.session_state.editing_state.get(i, False):
                # Set the correct default selection
                if match['dest_column'] == "NO_MATCH":
                    selected_index = len(dropdown_options) - 1  # Index of "-- No match --"
                else:
                    try:
                        selected_index = dropdown_options.index(match['dest_column'])
                    except ValueError:
                        selected_index = 0  # Default to first option if not found
                
                # Create the dropdown
                selected_dest = st.selectbox(
                    "Select destination column",
                    options=dropdown_options,
                    index=selected_index,
                    key=f"edit_dropdown_{i}"
                )
                
                # Update the match immediately when the dropdown value changes
                if selected_dest == "-- No match --":
                    st.session_state.edited_matches[i]['dest_column'] = "NO_MATCH"
                    st.session_state.edited_matches[i]['dest_expanded'] = "NO_MATCH"
                    st.session_state.edited_matches[i]['similarity_score'] = 0.0
                else:
                    st.session_state.edited_matches[i]['dest_column'] = selected_dest
                    
                    # Try to get the expanded name
                    try:
                        if 'dest_expanded' in st.session_state and 'dest_df' in st.session_state:
                            dest_idx = st.session_state.dest_df.columns.get_loc(selected_dest)
                            if dest_idx < len(st.session_state.dest_expanded):
                                st.session_state.edited_matches[i]['dest_expanded'] = st.session_state.dest_expanded[dest_idx]
                            else:
                                st.session_state.edited_matches[i]['dest_expanded'] = selected_dest
                        else:
                            st.session_state.edited_matches[i]['dest_expanded'] = selected_dest
                    except Exception:
                        st.session_state.edited_matches[i]['dest_expanded'] = selected_dest
                    
                    # Set high similarity for manually selected
                    st.session_state.edited_matches[i]['similarity_score'] = 1.0
            else:
                # Display the current match
                if match['dest_column'] == "NO_MATCH":
                    st.text("-- No match --")
                else:
                    st.text(f"{match['dest_column']} ({match['dest_expanded']})")
        
        with col3:
            st.text(f"Similarity: {match['similarity_score']:.3f}")
        
        with col4:
            # Instead of trying to access the button's state, use a callback
            if st.button("Edit", key=f"edit_button_{i}"):
                st.session_state.editing_state[i] = True
                # Store the source column being edited for memory lookup
                st.session_state.editing_source_col = match['source_column']
                st.rerun()
        
        with col5:
            if st.session_state.editing_state.get(i, False):
                if st.button("Save", key=f"save_button_{i}"):
                    st.session_state.editing_state[i] = False
                    st.rerun()
        
        with col6:
            # Add a "No Match" button that directly sets this column to have no match
            # Only show if not already set to NO_MATCH and not in editing mode
            if not st.session_state.editing_state.get(i, False) and match['dest_column'] != "NO_MATCH":
                if st.button("No Match", key=f"no_match_button_{i}", 
                            help="Mark this column as having no match in the destination table"):
                    # Set to NO_MATCH
                    st.session_state.edited_matches[i]['dest_column'] = "NO_MATCH"
                    st.session_state.edited_matches[i]['dest_expanded'] = "NO_MATCH"
                    st.session_state.edited_matches[i]['similarity_score'] = 0.0
                    st.rerun()
            # If already NO_MATCH, show a disabled button or indicator
            elif not st.session_state.editing_state.get(i, False) and match['dest_column'] == "NO_MATCH":
                st.markdown("✓ No Match")

        # In the column display loop, after selected_dest is chosen:
        if selected_dest != "-- No match --":
            # Remove "(from memory)" suffix if present
            clean_dest = selected_dest.replace(" (from memory)", "").replace("🔄 ", "")
            
            # Check if this destination column is already matched
            if clean_dest in matched_dest_cols:
                current_match = next(
                    match for match in st.session_state.edited_matches 
                    if match['dest_column'] == clean_dest
                )
                st.warning(f"""
                    ⚠️ Warning: Column `{clean_dest}` is already matched to source column `{current_match['source_column']}`.
                    Matching multiple source columns to the same destination column is not allowed.
                    Please choose a different destination column or use "No match" if there's no appropriate match.
                    """)
                # Don't update the match
                continue

@st.cache_resource
def load_semantic_model():
    """Load and cache the semantic model"""
    return SentenceTransformer('all-MiniLM-L6-v2')

def load_models():
    """Initialize the required models"""
    # Set API key directly from environment variable
    deepseek_model = DeepSeekLLM()  # The model will get the API key from environment
    prompt_template = PromptTemplate()
    semantic_model = load_semantic_model()
    return deepseek_model, prompt_template, semantic_model

@st.cache_data
def cached_expansion(abbreviations: list, context: str, verbose: bool = False):
    """Cached wrapper for abbreviation expansion"""
    if 'models' not in st.session_state:
        st.session_state.models = load_models()
    
    deepseek_model, prompt_template, _ = st.session_state.models
    return expand_abbreviations(
        abbreviations, context, deepseek_model, prompt_template, verbose
    )

def process_files(source_df, dest_df, context, models, use_memory=True, save_new_matches=True):
    """Process the uploaded files and match columns"""
    deepseek_model, prompt_template, semantic_model = models
    verbose = False  # Hard-code to False instead of using a checkbox
    
    # Get column names
    source_cols = source_df.columns.tolist()
    dest_cols = dest_df.columns.tolist()
    
    # Initialize column memory if enabled
    memory_matches = {}  # Default to empty dict
    column_memory = None
    if use_memory:
        try:
            from column_memory import ColumnMatchMemory
            memory_file = MEMORY_FILE_PATH
            os.makedirs(os.path.dirname(os.path.abspath(memory_file)), exist_ok=True)
            column_memory = ColumnMatchMemory(memory_file)
            
            # Get matches from memory but ensure it's not None
            found_matches = column_memory.find_matches(source_cols, dest_cols)
            if found_matches is not None:
                memory_matches = found_matches
            
            if verbose and memory_matches:
                st.write("Using matches from memory:")
                for src, dst in memory_matches.items():
                    st.write(f"- '{src}' → '{dst}'")
        except Exception as e:
            if verbose:
                st.warning(f"Error accessing column memory: {str(e)}")
            memory_matches = {}  # Ensure it's an empty dict if there was an error
    
    # Determine which columns need expansion through the LLM
    columns_to_expand_source = [col for col in source_cols if col not in memory_matches]
    
    # Expand column names using cached function
    with st.spinner('Expanding source table columns...'):
        if columns_to_expand_source:
            source_expanded_new = cached_expansion(
                columns_to_expand_source, context, verbose
            )
            
            # Create full source_expanded list by combining memory and new expansions
            source_expanded = []
            expansion_index = 0
            for col in source_cols:
                if col in memory_matches:
                    # Use a placeholder - we'll use memory for matching directly
                    source_expanded.append(f"MEMORY_MATCH:{col}")
                else:
                    source_expanded.append(source_expanded_new[expansion_index])
                    expansion_index += 1
        else:
            # If all columns are in memory, we still need a placeholder list
            source_expanded = [f"MEMORY_MATCH:{col}" for col in source_cols]
    
    with st.spinner('Expanding destination table columns...'):
        dest_expanded = cached_expansion(
            dest_cols, context, verbose
        )
        
        # Debug output if needed
        if verbose:
            st.write("Destination Columns Expansion:")
            for orig, exp in zip(dest_cols, dest_expanded):
                st.write(f"{orig} -> {exp}")
    
    # Match columns
    with st.spinner('Matching columns...'):
        matches, all_comparisons, highest_similarity_pair = match_columns(
            source_cols, dest_cols, source_expanded, dest_expanded, 
            semantic_model, verbose, memory_matches, similarity_threshold=0.3
        )
    
    # Save new matches if requested - only save valid matches, not NO_MATCH
    if use_memory and save_new_matches and column_memory:
        saved_count = 0
        for match in matches:
            source_col = match["source_column"]
            dest_col = match["dest_column"]
            # Only save matches with high confidence and not NO_MATCH
            if dest_col != "NO_MATCH" and match["similarity_score"] > 0.85 and not match.get("from_memory", False):
                column_memory.add_match(source_col, dest_col)
                saved_count += 1
        
        if verbose and saved_count > 0:
            st.info(f"Saved {saved_count} new high-confidence matches to memory.")
    
    return source_expanded, dest_expanded, matches, all_comparisons, highest_similarity_pair

def create_matched_source_file(source_df, matches):
    """Create a new DataFrame with matched column names and handle missing/extra columns"""
    matched_df = source_df.copy()
    
    # Create a mapping from source column to destination column
    # Only include columns that have a valid match (not NO_MATCH)
    column_mapping = {
        match['source_column']: match['dest_column'] 
        for match in matches 
        if match['dest_column'] != "NO_MATCH"
    }
    
    # Step 1: Rename the matched columns
    matched_df = matched_df.rename(columns=column_mapping)
    
    # Step 2: Get list of columns we're keeping as-is (marked as NO_MATCH)
    unmatched_source_columns = [
        match['source_column'] 
        for match in matches 
        if match['dest_column'] == "NO_MATCH"
    ]
    
    # Step 3: Find ALL destination columns not present in the renamed dataframe
    # Get the complete list of destination columns from the original destination DataFrame
    if 'dest_df' in st.session_state:
        all_dest_columns = st.session_state.dest_df.columns.tolist()
        # Find which destination columns are missing in our matched dataframe
        missing_dest_columns = [col for col in all_dest_columns if col not in matched_df.columns]
    else:
        # Fallback to the old approach if dest_df is not available
        dest_columns = [match['dest_column'] for match in matches if match['dest_column'] != "NO_MATCH"]
        missing_dest_columns = [col for col in dest_columns if col not in matched_df.columns]
    
    # Step 4: Add missing destination columns with "MISSING" values (uppercase as requested)
    for col in missing_dest_columns:
        matched_df[col] = "MISSING"  # Changed from "Missing" to "MISSING" as requested
    
    # Log the column operations for debugging and user information
    col_operations = {
        "matched_columns": column_mapping,
        "kept_source_columns": unmatched_source_columns,
        "added_missing_columns": missing_dest_columns
    }
    
    # For verbose output, add to session state
    st.session_state.column_operations = col_operations
    
    return matched_df

def safe_edit_match(i, selected_dest):
    """Safely edit a match with comprehensive error handling"""
    try:
        if selected_dest == "-- No match --":
            st.session_state.edited_matches[i]['dest_column'] = "NO_MATCH"
            st.session_state.edited_matches[i]['dest_expanded'] = "NO_MATCH"
            st.session_state.edited_matches[i]['similarity_score'] = 0.0
        else:
            st.session_state.edited_matches[i]['dest_column'] = selected_dest
            
            # Try multiple approaches to get the expanded name
            try:
                if 'dest_expanded' in st.session_state and 'dest_df' in st.session_state:
                    dest_idx = st.session_state.dest_df.columns.get_loc(selected_dest)
                    if dest_idx < len(st.session_state.dest_expanded):
                        st.session_state.edited_matches[i]['dest_expanded'] = st.session_state.dest_expanded[dest_idx]
                    else:
                        st.session_state.edited_matches[i]['dest_expanded'] = selected_dest
                else:
                    st.session_state.edited_matches[i]['dest_expanded'] = selected_dest
            except Exception as e:
                # If any error occurs during expanded name lookup, just use the column name
                st.session_state.edited_matches[i]['dest_expanded'] = selected_dest
                
            # Set a high similarity score for manually selected matches
            st.session_state.edited_matches[i]['similarity_score'] = 1.0
    except Exception as e:
        st.error(f"Error editing match: {str(e)}")

def clear_column_memory():
    """Function to completely clear the column matching memory"""
    memory_file = MEMORY_FILE_PATH
    
    try:
        # Write an empty dictionary to the file
        import json
        with open(memory_file, 'w') as f:
            json.dump({}, f)
        return True, "Memory cleared successfully"
    except Exception as e:
        return False, f"Error clearing memory: {str(e)}"

def validate_matches(matches):
    """Validate matches to detect duplicate destination columns"""
    dest_columns = {}  # Dictionary to track destination columns and their sources
    duplicates = {}    # Dictionary to track duplicate matches
    
    for match in matches:
        dest_col = match['dest_column']
        source_col = match['source_column']
        
        if dest_col != "NO_MATCH":
            if dest_col in dest_columns:
                # Found a duplicate destination column
                if dest_col not in duplicates:
                    duplicates[dest_col] = [dest_columns[dest_col]]
                duplicates[dest_col].append(source_col)
            else:
                dest_columns[dest_col] = source_col
    
    return duplicates

def main():
    st.title("LLM Based CSV Table Matcher")
    
    # Add RAG status indicator in sidebar
    with st.sidebar:
        st.header("System Status")
        rag_available = init_rag()
        if rag_available:
            st.success("✅ Medical Abbreviation System (Simplified): Active")
        else:
            st.warning("⚠️ Medical Abbreviation System: Not Available")
            st.info("To enable medical abbreviation support, ensure the simplified medical abbreviations dictionary is present.")
    
    # Add deployment info
    is_cloud = os.environ.get('STREAMLIT_DEPLOYMENT', '') == 'cloud'
    if is_cloud:
        st.sidebar.success("🌐 Running on Streamlit Cloud")
    else:
        st.sidebar.info("💻 Running Locally")
    
    # Add brief introduction with better styling
    st.markdown("""
    <div style="background-color: #f0f7fb; padding: 1.5rem; border-radius: 0.5rem; border-left: 5px solid #4e7496; margin-bottom: 1.5rem;">
        <p style="font-size: 1.1rem; line-height: 1.6; color: #1e3a5f; margin: 0;">
            This app matches your source data table to a destination table format by using Large Language Models (LLMs) 
            to interpret column names and find semantic matches between tables.
        </p>
        <p style="font-size: 1.1rem; line-height: 1.6; color: #1e3a5f; margin-top: 1rem;">
            The app includes a Medical Abbreviation RAG System that enhances column name interpretation based on a medical abbreviation database 
            (Grossman Liu, L., et al. A deep database of medical abbreviations and acronyms for natural language processing. 
            Sci Data 8, 149 (2021). https://doi.org/10.1038/s41597-021-00929-4).
        </p>
        <p style="font-size: 1.1rem; line-height: 1.6; color: #1e3a5f; margin-top: 1rem;">
            The matching memory feature learns from your confirmed column matches, saving them for future use to improve 
            accuracy and efficiency across similar datasets over time.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize models and RAG
    if 'models' not in st.session_state:
        with st.spinner('Loading models...'):
            st.session_state.models = load_models()
    
    # Initialize session state for matched file
    if 'matched_df' not in st.session_state:
        st.session_state.matched_df = None
    if 'matching_confirmed' not in st.session_state:
        st.session_state.matching_confirmed = False
    if 'source_filename' not in st.session_state:
        st.session_state.source_filename = ""
    
    # File uploaders section with guidance
    st.markdown("### Step 1: Upload Your Data")
    st.markdown("""
    Upload the source CSV file you want to transform and the destination CSV file 
    with the target column structure.
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        source_file = st.file_uploader("Upload Source CSV", type=['csv'])
        if source_file is not None and st.session_state.source_filename != source_file.name:
            st.session_state.source_filename = source_file.name
            st.session_state.matching_confirmed = False
    with col2:
        dest_file = st.file_uploader("Upload Destination CSV", type=['csv'])
    
    # Context input with explanation
    st.markdown("### Step 2: Provide Context (Optional)")
    st.markdown("""
    Providing context about your data domain helps the AI better interpret column names. 
    For example, "financial data", "clinical study", "customer records", etc.
    """)
    
    context = st.text_input(
        "Context", 
        placeholder="e.g., medical data, clinical study, etc."
    )
    
    # In the sidebar
    with st.sidebar:
        st.header("App Settings")
        use_memory = st.checkbox("Use column matching memory", value=True)
        save_matches = st.checkbox("Save new matches to memory", value=True)
        
        if use_memory:
            st.info("Column matching memory is enabled.")
            
            # Memory management section
            if st.button("View Memory Contents"):
                try:
                    from column_memory import ColumnMatchMemory
                    memory = ColumnMatchMemory(MEMORY_FILE_PATH)
                    matches = memory.get_all_matches()
                    
                    if matches:
                        st.write("### Stored Matches")
                        for src, dest in matches.items():
                            st.write(f"- {src} → {dest}")
                    else:
                        st.info("No matches stored in memory yet.")
                except Exception as e:
                    st.error(f"Error accessing memory: {str(e)}")
    
    if source_file and dest_file:
        try:
            # Load DataFrames
            source_df = pd.read_csv(source_file)
            dest_df = pd.read_csv(dest_file)
            
            # Store in session state
            st.session_state.source_df = source_df
            st.session_state.dest_df = dest_df
            
            # Process button
            if st.button("Match Columns"):
                st.session_state.matching_confirmed = False
                
                # Process files
                results = process_files(
                    source_df, dest_df, context, 
                    st.session_state.models,
                    use_memory=use_memory,
                    save_new_matches=save_matches
                )
                
                source_expanded, dest_expanded, matches, all_comparisons, highest_similarity_pair = results
                
                # Store results in session state
                st.session_state.matches = matches
                st.session_state.all_comparisons = all_comparisons
                st.session_state.source_expanded = source_expanded
                st.session_state.dest_expanded = dest_expanded
                
                # Reset edited matches
                st.session_state.edited_matches = matches.copy()
                st.session_state.editing_state = {i: False for i in range(len(matches))}
            
            # Display results if available
            if 'matches' in st.session_state:
                # Display column interpretations
                st.markdown("### Step 3: Review Column Interpretations")
                source_table, dest_table = create_aligned_tables(
                    source_df.columns.tolist(),
                    st.session_state.source_expanded,
                    dest_df.columns.tolist(),
                    st.session_state.dest_expanded
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Source Table Columns")
                    st.dataframe(source_table, hide_index=True)
                with col2:
                    st.subheader("Destination Table Columns")
                    st.dataframe(dest_table, hide_index=True)
                
                # Display matches
                st.markdown("""
                ### Step 4: Review and Edit Matches

                **Important Notes:**
                - Each destination column can only be matched to one source column
                - If multiple source columns seem to match the same destination column:
                  - Choose the most appropriate match
                  - Mark others as "No match" or find alternative matches
                - Use the "Edit" button to change matches
                - Use the "No Match" button to mark columns that don't have a corresponding match
                """)
                display_matches(st.session_state.matches, st.session_state.all_comparisons)
                
                # Confirm matching button
                if st.button("Confirm Matching"):
                    # Check for duplicate matches
                    duplicates = validate_matches(st.session_state.edited_matches)
                    
                    if duplicates:
                        # Show warning about duplicate matches
                        st.error("⚠️ Invalid Matching Detected!")
                        st.markdown("#### Duplicate Destination Columns Found:")
                        for dest_col, source_cols in duplicates.items():
                            st.markdown(f"""
                            Multiple source columns matched to `{dest_col}`:
                            {', '.join(f'`{src}`' for src in source_cols)}
                            """)
                        st.info("Please fix the matches before confirming. Each destination column can only be matched to one source column.")
                        
                        # Highlight the problematic matches in the display
                        for i, match in enumerate(st.session_state.edited_matches):
                            if match['dest_column'] in duplicates:
                                st.session_state.editing_state[i] = True
                    else:
                        # Proceed with normal confirmation
                        st.session_state.matching_confirmed = True
                        st.session_state.matched_df = create_matched_source_file(
                            st.session_state.source_df,
                            st.session_state.edited_matches
                        )
                        
                        # Save matches to memory immediately after confirmation
                        if use_memory and save_matches:
                            try:
                                from column_memory import ColumnMatchMemory
                                memory = ColumnMatchMemory(MEMORY_FILE_PATH)
                                saved_count = 0
                                
                                for match in st.session_state.edited_matches:
                                    source_col = match["source_column"]
                                    dest_col = match["dest_column"]
                                    # Save all confirmed matches except NO_MATCH
                                    if dest_col != "NO_MATCH":
                                        memory.add_match(source_col, dest_col)
                                        saved_count += 1
                                
                                if saved_count > 0:
                                    st.success(f"Matching confirmed! {saved_count} matches saved to memory.")
                                else:
                                    st.success("Matching confirmed! No new matches to save.")
                            except Exception as e:
                                st.warning(f"Could not save matches to memory: {str(e)}")
                                st.success("Matching confirmed! Your source data has been transformed.")
                        else:
                            st.success("Matching confirmed! Your source data has been transformed.")
                    
                    # Display matched file if confirmed
                    if st.session_state.matching_confirmed:
                        if st.session_state.matched_df is not None:  # Add this check
                            # First, display the column operations summary
                            st.markdown("### Column Operations Summary")
                            with st.expander("View Details", expanded=True):
                                if hasattr(st.session_state, 'column_operations'):
                                    # Show matched columns
                                    if st.session_state.column_operations["matched_columns"]:
                                        st.markdown("#### ✅ Matched and Renamed Columns")
                                        for src, dest in st.session_state.column_operations["matched_columns"].items():
                                            st.markdown(f"- `{src}` → `{dest}`")
                                    else:
                                        st.info("No columns were matched/renamed.")

                                    # Show kept source columns
                                    if st.session_state.column_operations["kept_source_columns"]:
                                        st.markdown("#### 🔄 Kept Original Columns (No Match)")
                                        for col in st.session_state.column_operations["kept_source_columns"]:
                                            st.markdown(f"- `{col}`")
                                    else:
                                        st.info("No original columns were kept unchanged.")

                                    # Show added missing columns
                                    if st.session_state.column_operations["added_missing_columns"]:
                                        st.markdown("#### ➕ Added Missing Columns (Filled with 'MISSING')")
                                        for col in st.session_state.column_operations["added_missing_columns"]:
                                            st.markdown(f"- `{col}`")
                                    else:
                                        st.info("No missing columns were added.")

                                    # Add a summary of the changes
                                    st.markdown("---")
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Matched Columns", 
                                                len(st.session_state.column_operations["matched_columns"]))
                                    with col2:
                                        st.metric("Kept Original", 
                                                len(st.session_state.column_operations["kept_source_columns"]))
                                    with col3:
                                        st.metric("Added Missing", 
                                                len(st.session_state.column_operations["added_missing_columns"]))

                            # Then show the preview with highlighted columns
                            st.markdown("### Preview of Matched File")
                            if hasattr(st.session_state, 'column_operations'):
                                try:
                                    # Get lists of columns for highlighting
                                    kept_columns = st.session_state.column_operations["kept_source_columns"]
                                    missing_columns = st.session_state.column_operations["added_missing_columns"]
                                    matched_columns = list(st.session_state.column_operations["matched_columns"].values())
                                    
                                    # Check for duplicate columns
                                    if st.session_state.matched_df.columns.duplicated().any():
                                        st.warning("Duplicate column names detected. Displaying without highlighting.")
                                        st.dataframe(st.session_state.matched_df.head(10), use_container_width=True)
                                    else:
                                        # Display the styled dataframe
                                        def highlight_columns(df):
                                            if df.index.duplicated().any() or df.columns.duplicated().any():
                                                # If there are duplicate indices or columns, return unstyled
                                                return pd.DataFrame('', index=df.index, columns=df.columns)
                                            
                                            styles = pd.DataFrame('', index=df.index, columns=df.columns)
                                            
                                            try:
                                                # Highlight matched columns (light green)
                                                for col in matched_columns:
                                                    if col in df.columns:
                                                        styles[col] = 'background-color: #e6ffe6'
                                                
                                                # Highlight kept original columns (light yellow)
                                                for col in kept_columns:
                                                    if col in df.columns:
                                                        styles[col] = 'background-color: #ffffcc'
                                                
                                                # Highlight added missing columns (light red)
                                                for col in missing_columns:
                                                    if col in df.columns:
                                                        styles[col] = 'background-color: #ffcccc'
                                            except Exception as e:
                                                st.warning(f"Could not apply highlighting: {str(e)}")
                                            
                                            return styles

                                        # Add a legend for the colors
                                        st.markdown("""
                                        **Color Legend:**
                                        - <span style='background-color: #e6ffe6; padding: 2px 6px;'>Matched Columns</span>
                                        - <span style='background-color: #ffffcc; padding: 2px 6px;'>Kept Original Columns</span>
                                        - <span style='background-color: #ffcccc; padding: 2px 6px;'>Added Missing Columns</span>
                                        """, unsafe_allow_html=True)

                                        # Display the styled dataframe
                                        styled_df = st.session_state.matched_df.head(10).style.apply(highlight_columns, axis=None)
                                        st.dataframe(styled_df, use_container_width=True)
                                except Exception as e:
                                    st.error(f"Error displaying preview: {str(e)}")
                                    # Fallback to unstyled display
                                    st.dataframe(st.session_state.matched_df.head(10), use_container_width=True)
                            else:
                                st.error("No matched data available. Please confirm the matching first.")

                            # Download section - Add error handling
                            try:
                                st.markdown("### Download Transformed Data")
                                default_filename = f"{os.path.splitext(st.session_state.source_filename)[0]}_matched.csv"
                                
                                # Verify data is available before creating download button
                                if not st.session_state.matched_df.empty:
                                    st.download_button(
                                        label="Download Matched File",
                                        data=st.session_state.matched_df.to_csv(index=False).encode('utf-8'),
                                        file_name=default_filename,
                                        mime="text/csv",
                                        key='download-matched-file'
                                    )
                                else:
                                    st.warning("No data available for download.")
                            except Exception as e:
                                st.error(f"Error preparing download: {str(e)}")
                        else:
                            st.error("No matched data available. Please confirm the matching first.")
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.exception(e)
    
    else:
        st.info("Please upload both source and destination CSV files to begin.")

if __name__ == "__main__":
    main()