# rag_components/query_expander.py
import os
from typing import List, Optional
from dotenv import load_dotenv
from groq import Groq

class QueryExpander:
    """Query expansion for automotive AI domain"""
    
    def __init__(self, api_key: Optional[str] = None):
        load_dotenv()
        self.api_key = api_key or os.getenv('GROQ_API_KEY')
        if self.api_key:
            self.client = Groq(api_key=self.api_key)
        else:
            self.client = None
        
        # Domain-specific keyword mappings for automotive AI
        self.domain_synonyms = {
            "autonomous": ["self-driving", "driverless", "automated", "robotic"],
            "vehicle": ["car", "automobile", "truck", "transport", "EV", "electric vehicle"],
            "AI": ["artificial intelligence", "machine learning", "ML", "deep learning", "neural network"],
            "sensor": ["lidar", "radar", "camera", "ultrasonic", "perception"],
            "startup": ["company", "venture", "business", "firm", "enterprise", "scale-up"],
            "patent": ["intellectual property", "IP", "invention", "technology protection"],
            "trend": ["development", "advancement", "progress", "innovation", "emerging"]
        }
    
    def expand_with_llm(self, query: str, num_variations: int = 3) -> List[str]:
        """Generate query variations using LLM"""
        if not self.client:
            return [query]  # Fallback to original query
        
        prompt = f"""Generate {num_variations} different ways to search for information about this query in an automotive technology database.
        Focus on synonyms, technical terms, alternative phrasing, and related concepts in automotive AI.
        
        Original query: "{query}"
        
        Generate {num_variations} search variations (one per line):"""
        
        try:
            response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.7
            )
            
            variations = response.choices[0].message.content.strip().split('\n')
            # Clean up variations
            clean_variations = [v.strip().strip('"').strip("'") for v in variations if v.strip()]
            # Add original query and ensure uniqueness
            all_queries = [query] + clean_variations
            return list(dict.fromkeys(all_queries))[:num_variations + 1]
            
        except Exception as e:
            print(f"LLM query expansion failed: {e}")
            return [query]
    
    def expand_with_keywords(self, query: str) -> List[str]:
        """Expand query using domain-specific keyword replacements"""
        query_lower = query.lower()
        variations = [query]
        
        # Generate variations by replacing domain keywords
        for term, synonyms in self.domain_synonyms.items():
            if term in query_lower:
                for synonym in synonyms:
                    # Simple replacement (case-insensitive)
                    variation = self._replace_word(query, term, synonym)
                    if variation and variation != query:
                        variations.append(variation)
        
        # Add variations with additional related terms
        related_terms = []
        if "automotive" in query_lower or "car" in query_lower:
            related_terms.extend(["technology", "innovation", "system"])
        if "ai" in query_lower or "artificial intelligence" in query_lower:
            related_terms.extend(["algorithm", "model", "system"])
        
        for term in related_terms:
            if term not in query_lower:
                variations.append(f"{query} {term}")
        
        return list(dict.fromkeys(variations))  # Remove duplicates
    
    def _replace_word(self, text: str, old: str, new: str) -> str:
        """Case-insensitive word replacement"""
        import re
        return re.sub(rf'\b{re.escape(old)}\b', new, text, flags=re.IGNORECASE)
    
    def expand_query(self, query: str, use_llm: bool = True) -> List[str]:
        """Main expansion method - combines both approaches"""
        # Always start with keyword-based expansion
        keyword_variations = self.expand_with_keywords(query)
        
        if use_llm and self.client:
            try:
                llm_variations = self.expand_with_llm(query, num_variations=2)
                # Combine and deduplicate
                all_variations = keyword_variations + llm_variations
                unique_variations = []
                seen = set()
                for v in all_variations:
                    if v.lower() not in seen:
                        seen.add(v.lower())
                        unique_variations.append(v)
                return unique_variations[:5]  # Limit to 5 total variations
            except:
                return keyword_variations[:3]  # Fallback to keyword variations
        else:
            return keyword_variations[:3]