# src/db_manager.py
import sqlite3
from datetime import datetime
import json

class CVDatabase:
    def __init__(self, db_path="database/cv_database.db"):
        self.db_path = db_path
        self.create_tables()
    
    def get_connection(self):
        """Create database connection."""
        return sqlite3.connect(self.db_path)
    
    def create_tables(self):
        """
        Create normalized database schema.
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Main CV table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS candidates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT UNIQUE,
                phone TEXT,
                education TEXT,
                experience_years INTEGER DEFAULT 0,
                category TEXT,
                full_text TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Skills table (many-to-many relationship)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS skills (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                skill_name TEXT UNIQUE NOT NULL
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS candidate_skills (
                candidate_id INTEGER,
                skill_id INTEGER,
                FOREIGN KEY (candidate_id) REFERENCES candidates(id),
                FOREIGN KEY (skill_id) REFERENCES skills(id),
                PRIMARY KEY (candidate_id, skill_id)
            )
        """)
        
        # Embeddings table (for semantic search)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cv_embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                candidate_id INTEGER UNIQUE,
                embedding_vector TEXT,  -- Stored as JSON array
                model_name TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (candidate_id) REFERENCES candidates(id)
            )
        """)
        
        # Job descriptions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS job_descriptions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                description TEXT NOT NULL,
                required_skills TEXT,
                embedding_vector TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Matching results table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS matching_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                candidate_id INTEGER,
                job_id INTEGER,
                semantic_score REAL,
                keyword_score REAL,
                combined_score REAL,
                matched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (candidate_id) REFERENCES candidates(id),
                FOREIGN KEY (job_id) REFERENCES job_descriptions(id)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def insert_candidate(self, parsed_cv):
        """
        Insert parsed CV data into database.
        
        Args:
            parsed_cv: Dictionary from CVParser.parse_cv()
            
        Returns:
            Candidate ID
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            # INSERT WITH CATEGORY 
            cursor.execute("""
                INSERT INTO candidates (name, email, phone, education, experience_years, category, full_text)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                parsed_cv['name'],
                parsed_cv['email'],
                parsed_cv['phone'],
                parsed_cv['education'],
                parsed_cv['experience_years'],
                parsed_cv.get('category'),  
                parsed_cv['full_text']
            ))
            
            candidate_id = cursor.lastrowid
            
            # Insert skills
            if parsed_cv['skills']:
                skills_list = [s.strip() for s in parsed_cv['skills'].split(',')]
                for skill in skills_list:
                    # Insert skill if not exists
                    cursor.execute("""
                        INSERT OR IGNORE INTO skills (skill_name) VALUES (?)
                    """, (skill,))
                    
                    # Get skill ID
                    cursor.execute("SELECT id FROM skills WHERE skill_name = ?", (skill,))
                    skill_id = cursor.fetchone()[0]
                    
                    # Link candidate to skill
                    cursor.execute("""
                        INSERT OR IGNORE INTO candidate_skills (candidate_id, skill_id)
                        VALUES (?, ?)
                    """, (candidate_id, skill_id))
            
            conn.commit()
            return candidate_id
            
        except sqlite3.IntegrityError as e:
            print(f"Database error: {e}")
            return None
        finally:
            conn.close()
    
    def store_embedding(self, candidate_id, embedding_vector, model_name):
        """
        Store embedding vector for a candidate.
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Convert numpy array to JSON string
        embedding_json = json.dumps(embedding_vector.tolist())
        
        cursor.execute("""
            INSERT OR REPLACE INTO cv_embeddings (candidate_id, embedding_vector, model_name)
            VALUES (?, ?, ?)
        """, (candidate_id, embedding_json, model_name))
        
        conn.commit()
        conn.close()
    
    def get_all_candidates(self):
        """Retrieve all candidates with their skills."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT c.id, c.name, c.email, c.experience_years, c.category,
                   GROUP_CONCAT(s.skill_name) as skills
            FROM candidates c
            LEFT JOIN candidate_skills cs ON c.id = cs.candidate_id
            LEFT JOIN skills s ON cs.skill_id = s.id
            GROUP BY c.id
        """)
        
        results = cursor.fetchall()
        conn.close()
        return results

# Usage
if __name__ == "__main__":
    db = CVDatabase()
    print("Database initialized successfully!")