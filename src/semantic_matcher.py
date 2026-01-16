# src/semantic_matcher.py
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
import sqlite3
import os

class SemanticMatcher:
    def __init__(self, model_name='all-MiniLM-L6-v2', embeddings_dir='database'):
        """
        Initialize semantic matcher with pre-trained model.
        
        Model info:
        - all-MiniLM-L6-v2: Fast, 384 dimensions, good for similarity
        - Confirmed from sentence-transformers documentation
        """
        print(f"Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.embeddings_dir = embeddings_dir
        self.embeddings_file = os.path.join(embeddings_dir, 'cv_embeddings.npz')
        self.metadata_file = os.path.join(embeddings_dir, 'cv_metadata.json')
        
        # Create embeddings directory if it doesn't exist
        os.makedirs(embeddings_dir, exist_ok=True)
    
    def generate_embedding(self, text):
        """
        Generate embedding vector for text.
        
        Returns:
            numpy array of embeddings
        """
        embedding = self.model.encode(text, convert_to_tensor=False)
        return embedding
    
    def compute_similarity(self, embedding1, embedding2):
        """
        Compute cosine similarity between two embeddings.
        
        Returns:
            Similarity score between 0 and 1
        """
        # Reshape for sklearn
        emb1 = embedding1.reshape(1, -1)
        emb2 = embedding2.reshape(1, -1)
        
        similarity = cosine_similarity(emb1, emb2)[0][0]
        return float(similarity)
    
    def load_embeddings(self):
        """
        Load embeddings from NumPy file.
        
        Returns:
            Dictionary mapping candidate_id to embedding vector
        """
        if not os.path.exists(self.embeddings_file):
            return {}
        
        data = np.load(self.embeddings_file)
        return {int(k): data[k] for k in data.files}
    
    def precompute_all_embeddings(self, db_path="database/cv_database.db"):
        """
        Pre-compute embeddings and store in NumPy compressed file.
        This is MUCH more efficient than storing in SQLite.
        """
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT id, full_text FROM candidates")
        candidates = cursor.fetchall()
        conn.close()
        
        if not candidates:
            print("No candidates found in database!")
            return
        
        embeddings_dict = {}
        metadata = {}
        
        print(f"Processing {len(candidates)} candidates...")
        
        for i, (candidate_id, full_text) in enumerate(candidates, 1):
            if i % 100 == 0:
                print(f"Processing candidate {i}/{len(candidates)}...")
            
            # Generate embedding
            embedding = self.generate_embedding(full_text)
            embeddings_dict[str(candidate_id)] = embedding
            metadata[str(candidate_id)] = {'model': self.model_name}
        
        # Save to NumPy compressed file
        print("Saving embeddings to file...")
        np.savez_compressed(
            self.embeddings_file,
            **embeddings_dict
        )
        
        # Save metadata
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f)
        
        file_size = os.path.getsize(self.embeddings_file) / (1024 * 1024)
        print(f"✓ Embeddings saved to {self.embeddings_file} ({file_size:.2f} MB)")
        print(f"✓ Total candidates processed: {len(candidates)}")
    
    def rank_candidates(self, job_description, db_path="database/cv_database.db"):
        """
        Rank all candidates against a job description using pre-computed embeddings.
        
        Args:
            job_description: Job description text
            db_path: Path to SQLite database
            
        Returns:
            List of tuples (candidate_id, name, score) sorted by score
        """
        # Generate job description embedding
        job_embedding = self.generate_embedding(job_description)
        
        # Load pre-computed embeddings from file
        embeddings = self.load_embeddings()
        
        if not embeddings:
            raise ValueError(
                "No embeddings found! Run precompute_all_embeddings() first.\n"
                "Usage: matcher.precompute_all_embeddings()"
            )
        
        # Get candidate names from database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id, name FROM candidates")
        candidates = {cid: name for cid, name in cursor.fetchall()}
        conn.close()
        
        # Compute similarities
        results = []
        for candidate_id, cv_embedding in embeddings.items():
            if candidate_id in candidates:
                score = self.compute_similarity(job_embedding, cv_embedding)
                results.append((candidate_id, candidates[candidate_id], score))
        
        # Sort by score (descending)
        results.sort(key=lambda x: x[2], reverse=True)
        return results
    
    def get_embedding_stats(self):
        """
        Get statistics about stored embeddings.
        """
        if not os.path.exists(self.embeddings_file):
            return {
                'exists': False,
                'count': 0,
                'file_size_mb': 0
            }
        
        embeddings = self.load_embeddings()
        file_size = os.path.getsize(self.embeddings_file) / (1024 * 1024)
        
        return {
            'exists': True,
            'count': len(embeddings),
            'file_size_mb': round(file_size, 2),
            'model': self.model_name
        }

