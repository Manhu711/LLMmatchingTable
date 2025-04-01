import json
import os
import re

class SimpleMedicalRAG:
    def __init__(self, data_path=None):
        """Initialize the simple medical RAG system with a dictionary lookup approach."""
        if data_path is None:
            # Try multiple possible locations
            possible_paths = [
                "data/medical_abbreviations_dict.json",
                "../data/medical_abbreviations_dict.json",
                os.path.join(os.path.dirname(__file__), "data/medical_abbreviations_dict.json")
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    data_path = path
                    break
        
        self.abbr_dict = {}
        
        try:
            with open(data_path, 'r') as f:
                self.abbr_dict = json.load(f)
            self.is_available = True
            print(f"Loaded {len(self.abbr_dict)} medical abbreviations")
        except FileNotFoundError:
            self.is_available = False
            print(f"Medical abbreviation dictionary not found at {data_path}")
    
    def query(self, query_text, top_k=5):
        """Query the abbreviation dictionary for expansions."""
        if not self.is_available:
            return []
        
        # Clean query - extract just the abbreviation
        query = query_text.strip().upper()
        # If query contains non-letter chars, try to extract just the abbreviation
        match = re.search(r'\b([A-Z0-9]{2,})\b', query)
        if match:
            query = match.group(1)
        
        # Direct dictionary lookup (no embedding needed)
        if query in self.abbr_dict:
            # Sort by confidence if available
            results = sorted(self.abbr_dict[query], 
                           key=lambda x: x.get('confidence', 0), 
                           reverse=True)
            return results[:top_k]
        
        # No exact match found
        return []
    
    def get_context_for_llm(self, query_text, top_k=5):
        """Format abbreviation expansions for the LLM prompt."""
        results = self.query(query_text, top_k)
        
        if not results:
            return ""
        
        context = f"Potential meanings of '{query_text}' from medical abbreviation database:\n"
        
        # Create a set to track unique long forms we've added
        added_expansions = set()
        
        for i, result in enumerate(results, 1):
            long_form = result['long_form']
            domain = result.get('domain', 'Medical')
            confidence = result.get('confidence', 0.9)
            
            # Skip if we've already added this expansion
            if long_form in added_expansions:
                continue
                
            added_expansions.add(long_form)
            context += f"{i}. {long_form} ({domain}, confidence: {confidence:.2f})\n"
            
            # Break if we've added enough unique expansions
            if len(added_expansions) >= top_k:
                break
                
        return context