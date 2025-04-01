import json
import os
from typing import Dict, List, Optional, Tuple
import streamlit as st

class ColumnMatchMemory:
    """Stores and retrieves previously successful column matches using local file storage.
    Supports storing multiple destination mappings for the same source column."""
    
    def __init__(self, memory_file: str = "data/column_matches.json", user_id: str = None):
        """Initialize the column matching memory.
        
        Args:
            memory_file: Path to the JSON file storing the matches
            user_id: Optional user identifier for user-specific memory
        """
        if user_id:
            self.memory_file = f"data/user_{user_id}_matches.json"
        else:
            self.memory_file = memory_file
        self.matches = self._load_matches()
    
    def _load_matches(self) -> Dict[str, List[str]]:
        """Load previously saved matches from disk.
        
        Returns:
            Dictionary mapping source columns to lists of destination columns
        """
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r') as f:
                    data = json.load(f)
                    
                    # Convert old format (single mapping) to new format (list of mappings)
                    # This ensures backward compatibility
                    result = {}
                    for src, dst in data.items():
                        if isinstance(dst, list):
                            result[src] = dst
                        else:
                            result[src] = [dst]
                    return result
            except Exception as e:
                print(f"Error loading column matches: {str(e)}")
                return {}
        return {}
    
    def _save_matches(self):
        """Save current matches to disk."""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(self.memory_file)), exist_ok=True)
            with open(self.memory_file, 'w') as f:
                json.dump(self.matches, f, indent=2)
        except Exception as e:
            print(f"Error saving column matches: {str(e)}")
    
    def get_match(self, source_col: str) -> Optional[List[str]]:
        """Get all matched destination columns for a source column.
        
        Args:
            source_col: The source column name
            
        Returns:
            List of matched destination column names, or None if no match exists
        """
        result = self.matches.get(source_col)
        # Ensure we always return a list if there's a match
        if result is not None and not isinstance(result, list):
            return [result]
        return result
    
    def add_match(self, source_col: str, dest_col: str):
        """Add a new match to the memory. If the source column already exists,
        add the destination column to its list of mappings if not already present.
        
        Args:
            source_col: The source column name
            dest_col: The destination column name
        """
        if source_col not in self.matches:
            self.matches[source_col] = [dest_col]
        else:
            # Only add if not already in the list
            if dest_col not in self.matches[source_col]:
                self.matches[source_col].append(dest_col)
        self._save_matches()
    
    def get_all_matches(self) -> Dict[str, List[str]]:
        """Get all stored matches.
        
        Returns:
            Dictionary mapping source columns to lists of destination columns
        """
        return self.matches.copy()
    
    def find_matches(self, source_cols: list, dest_cols: list) -> Dict[str, str]:
        """Find the best matches from memory for the given source and destination columns.
        For each source column, return the most appropriate destination column if available.
        
        Args:
            source_cols: List of source column names
            dest_cols: List of destination column names
            
        Returns:
            Dictionary mapping source columns to destination columns for found matches
        """
        found_matches = {}  # Always start with an empty dictionary
        
        try:
            # For each source column, find the best match among available destination columns
            for src_col in source_cols:
                if src_col in self.matches:
                    # Get all potential matches for this source column
                    potential_matches = self.matches[src_col]
                    
                    # Find the first match that exists in destination columns
                    for dst_col in potential_matches:
                        if dst_col in dest_cols:
                            found_matches[src_col] = dst_col
                            break
        except Exception as e:
            print(f"Error finding matches: {str(e)}")
            # Return empty dict in case of any error
        
        return found_matches  # Always return the dictionary, even if empty 