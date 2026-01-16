# src/hybrid_matcher.py
from semantic_matcher import SemanticMatcher
import sqlite3
import re

class HybridMatcher:
    def __init__(self):
        self.semantic_matcher = SemanticMatcher()
    
    def extract_keywords(self, text):
        """Extract important keywords from text."""
        # Simple keyword extraction 
        text_lower = text.lower()
        keywords = set(re.findall(r'\b[a-z]{3,}\b', text_lower))
        return keywords
    
    def keyword_matching_score(self, cv_text, job_description):
        """
        Calculate keyword overlap score.
        
        Returns:
            Score between 0 and 1
        """
        cv_keywords = self.extract_keywords(cv_text)
        job_keywords = self.extract_keywords(job_description)
        
        if not job_keywords:
            return 0.0
        
        # Jaccard similarity
        intersection = len(cv_keywords & job_keywords)
        union = len(cv_keywords | job_keywords)
        
        return intersection / union if union > 0 else 0.0
    
    def hybrid_rank(self, job_description, semantic_weight=0.7, 
                    keyword_weight=0.3, db_path="database/cv_database.db"):
        """
        Rank candidates using both semantic and keyword matching.
        
        Args:
            job_description: Job description text
            semantic_weight: Weight for semantic score (0-1)
            keyword_weight: Weight for keyword score (0-1)
            
        Returns:
            List of (candidate_id, name, semantic_score, keyword_score, combined_score)
        """
        # Validate weights
        total_weight = semantic_weight + keyword_weight
        if abs(total_weight - 1.0) > 0.01:
            raise ValueError("Weights must sum to 1.0")
        
        # Get semantic rankings
        semantic_results = self.semantic_matcher.rank_candidates(job_description, db_path)
        
        # Connect to database for full text
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        hybrid_results = []
        
        for candidate_id, name, semantic_score in semantic_results:
            # Get full CV text
            cursor.execute("SELECT full_text FROM candidates WHERE id = ?", (candidate_id,))
            result = cursor.fetchone()
            
            if not result:
                continue
                
            cv_text = result[0]
            
            # Calculate keyword score
            keyword_score = self.keyword_matching_score(cv_text, job_description)
            
            # Combined score
            combined_score = (semantic_weight * semantic_score + 
                            keyword_weight * keyword_score)
            
            hybrid_results.append((
                candidate_id, name, semantic_score, 
                keyword_score, combined_score
            ))
        
        conn.close()
        
        # Sort by combined score
        hybrid_results.sort(key=lambda x: x[4], reverse=True)
        return hybrid_results

