"""
Gradio Demo for CV Parsing and Semantic Matching System
Run with: python gradio_demo.py
"""
import gradio as gr
import pandas as pd
import sqlite3
from pathlib import Path
import json
from parser import CVParser
from db_manager import CVDatabase
from semantic_matcher import SemanticMatcher
from hybrid_matcher import HybridMatcher


# Initialize components
parser = CVParser()
db = CVDatabase()
semantic_matcher = SemanticMatcher()
hybrid_matcher = HybridMatcher()

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_database_stats():
    """Get statistics from the database."""
    conn = sqlite3.connect("database/cv_database.db")
    cursor = conn.cursor()
    
    # Total candidates
    cursor.execute("SELECT COUNT(*) FROM candidates")
    total_candidates = cursor.fetchone()[0]
    
    # Total skills
    cursor.execute("SELECT COUNT(*) FROM skills")
    total_skills = cursor.fetchone()[0]
    
    # Average experience
    cursor.execute("SELECT AVG(experience_years) FROM candidates")
    avg_experience = cursor.fetchone()[0] or 0
    
    conn.close()
    
    return {
        "total_candidates": total_candidates,
        "total_skills": total_skills,
        "avg_experience": round(avg_experience, 1)
    }

def format_match_results(results, top_n=10):
    data = []

    for rank, (cid, name, sem, kw, total) in enumerate(results[:top_n], 1):
        row = {
            "Rank": rank,
            "Candidate Name": name,
            "ID": cid,
            "Semantic Score": f"{sem:.4f}",
            "Final Score": f"{total:.4f}"
        }
        if kw is not None:
            row["Keyword Score"] = f"{kw:.4f}"

        data.append(row)

    return pd.DataFrame(data)


def get_candidate_details(candidate_id):
    """Get detailed information about a candidate."""
    conn = sqlite3.connect("database/cv_database.db")
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT c.id, c.name, c.email, c.phone, c.education, 
               c.experience_years, c.category,
               GROUP_CONCAT(DISTINCT s.skill_name) as skills
        FROM candidates c
        LEFT JOIN candidate_skills cs ON c.id = cs.candidate_id
        LEFT JOIN skills s ON cs.skill_id = s.id
        WHERE c.id = ?
        GROUP BY c.id
    """, (candidate_id,))
    
    result = cursor.fetchone()
    conn.close()
    
    if result:
        return {
            "ID": result[0],
            "Name": result[1],
            "Email": result[2] or "N/A",
            "Phone": result[3] or "N/A",
            "Education": result[4] or "N/A",
            "Experience": f"{result[5]} years",
            "Category": result[6] or "N/A",
            "Skills": result[7] or "N/A"
        }
    return None

# =============================================================================
# TAB 1: JOB MATCHING
# =============================================================================

def match_job_description(job_desc, matching_type, top_n, semantic_weight):
    if not job_desc.strip():
        return None, "Please enter a job description."

    try:
        if matching_type == "Semantic Only":
            semantic_results = semantic_matcher.rank_candidates(job_desc)
            results = [
                (cid, name, score, None, score)
                for cid, name, score in semantic_results
            ]
            method_text = "Semantic Similarity Only"
        else:
            keyword_weight = 1 - semantic_weight
            results = hybrid_matcher.hybrid_rank(
                job_desc,
                semantic_weight=semantic_weight,
                keyword_weight=keyword_weight
            )
            method_text = "Hybrid (Semantic + Keyword)"

        df = format_match_results(results, top_n)

        summary = f"""
### Matching Complete ✓

**Method:** {method_text}  
**Candidates Found:** {len(results)}  
**Showing Top:** {min(top_n, len(results))}

**Top Match:** {results[0][1]}  
**Score:** {results[0][4]:.4f}
        """

        return df, summary

    except Exception as e:
        return None, f"Error: {str(e)}"


# =============================================================================
# TAB 2: CANDIDATE SEARCH
# =============================================================================

def search_candidates(search_query, category_filter):
    conn = sqlite3.connect("database/cv_database.db")
    cursor = conn.cursor()

    query = """
        SELECT c.id, c.name, c.email, c.experience_years, c.category,
               GROUP_CONCAT(DISTINCT s.skill_name) AS skills
        FROM candidates c
        LEFT JOIN candidate_skills cs ON c.id = cs.candidate_id
        LEFT JOIN skills s ON cs.skill_id = s.id
        WHERE 1=1
    """
    params = []

    if search_query:
        query += """
        AND (
            c.name LIKE ?
            OR EXISTS (
                SELECT 1
                FROM candidate_skills cs2
                JOIN skills s2 ON cs2.skill_id = s2.id
                WHERE cs2.candidate_id = c.id
                AND s2.skill_name LIKE ?
            )
        )
        """
        term = f"%{search_query}%"
        params.extend([term, term])

    if category_filter != "All":
        query += " AND LOWER(c.category) = LOWER(?)"
        params.append(category_filter)

    query += " GROUP BY c.id LIMIT 50"

    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return pd.DataFrame(), "No candidates found."

    df = pd.DataFrame(rows, columns=[
        "ID", "Name", "Email", "Experience (years)", "Category", "Skills"
    ])

    return df, f"Found {len(rows)} candidates"


    df = pd.DataFrame(rows, columns=[
        "ID", "Name", "Email", "Experience (years)", "Category", "Skills"
    ])

    return df, f"Found {len(rows)} candidates"


def show_candidate_profile(candidate_id):
    """Display detailed candidate profile."""
    if not candidate_id:
        return "Please enter a candidate ID."
    
    try:
        candidate_id = int(candidate_id)
        details = get_candidate_details(candidate_id)
        
        if not details:
            return "Candidate not found."
        
        profile = f"""
# 👤 Candidate Profile

---

**ID:** {details['ID']}  
**Name:** {details['Name']}  
**Email:** {details['Email']}  
**Phone:** {details['Phone']}  

---

### 💼 Professional Information

**Category:** {details['Category']}  
**Experience:** {details['Experience']}  

---

### 🎓 Education

{details['Education']}

---

### 🛠️ Skills

{details['Skills']}
        """
        
        return profile
        
    except ValueError:
        return "Invalid candidate ID. Please enter a number."
    except Exception as e:
        return f"Error: {str(e)}"

# =============================================================================
# TAB 3: UPLOAD & PROCESS NEW CVS
# =============================================================================

def process_single_cv(cv_text):
    """Process a single CV from text input."""
    if not cv_text.strip():
        return "Please enter CV text.", None
    
    try:
        # Parse CV
        parsed_data = parser.parse_cv(cv_text)
        
        # Insert into database
        candidate_id = db.insert_candidate(parsed_data)
        
        if candidate_id:
            # Generate embedding
            embedding = semantic_matcher.generate_embedding(cv_text)
            db.store_embedding(candidate_id, embedding, semantic_matcher.model_name)
            
            # Format output
            result = f"""
### ✓ CV Processed Successfully!

**Candidate ID:** {candidate_id}  
**Name:** {parsed_data['name']}  
**Email:** {parsed_data['email']}  
**Experience:** {parsed_data['experience_years']} years  
**Skills:** {parsed_data['skills']}  

The candidate has been added to the database and is ready for matching.
            """
            
            # Also return as structured data
            df = pd.DataFrame([{
                "Field": k.replace('_', ' ').title(),
                "Value": v if v else "N/A"
            } for k, v in parsed_data.items() if k != 'full_text'])
            
            return result, df
        else:
            return "Error: Could not insert candidate into database.", None
            
    except Exception as e:
        return f"Error processing CV: {str(e)}", None

def batch_process_csv(csv_file):
    """Process multiple CVs from uploaded CSV file."""
    if csv_file is None:
        return "Please upload a CSV file.", None
    
    try:
        df = pd.read_csv(csv_file.name)
        
        # Check for required column
        if 'Resume_str' not in df.columns:
            return "Error: CSV must have 'Resume_str' column.", None
        
        processed = 0
        errors = 0
        
        for idx, row in df.iterrows():
            try:
                cv_text = row['Resume_str']
                parsed_data = parser.parse_cv(cv_text)
                
                # Add category if available
                if 'Category' in row and pd.notna(row['Category']):
                    parsed_data['category'] = str(row['Category']).strip()
                
                candidate_id = db.insert_candidate(parsed_data)
                
                if candidate_id:
                    embedding = semantic_matcher.generate_embedding(cv_text)
                    db.store_embedding(candidate_id, embedding, semantic_matcher.model_name)
                    processed += 1
            except Exception as e:
                errors += 1
                continue
        
        result = f"""
### Batch Processing Complete!

**Total CVs in file:** {len(df)}  
**Successfully processed:** {processed}  
**Errors:** {errors}  

All candidates have been added to the database.
        """
        
        # Get updated stats
        stats = get_database_stats()
        stats_df = pd.DataFrame([stats])
        
        return result, stats_df
        
    except Exception as e:
        return f"Error processing CSV: {str(e)}", None

# =============================================================================
# TAB 4: DATABASE STATISTICS
# =============================================================================

def show_statistics():
    """Display database statistics and visualizations."""
    conn = sqlite3.connect("database/cv_database.db")
    
    # Overall stats
    stats = get_database_stats()
    
    # Category distribution
    query = """
        SELECT category, COUNT(*) as count
        FROM candidates
        WHERE category IS NOT NULL
        GROUP BY category
        ORDER BY count DESC
    """
    category_df = pd.read_sql_query(query, conn)
    
    # Top skills
    query = """
        SELECT s.skill_name, COUNT(*) as count
        FROM skills s
        JOIN candidate_skills cs ON s.id = cs.skill_id
        GROUP BY s.skill_name
        ORDER BY count DESC
        LIMIT 20
    """
    skills_df = pd.read_sql_query(query, conn)
    
    # Experience distribution
    query = """
        SELECT experience_years, COUNT(*) as count
        FROM candidates
        GROUP BY experience_years
        ORDER BY experience_years
    """
    exp_df = pd.read_sql_query(query, conn)
    
    conn.close()
    
    summary = f"""
# 📊 Database Statistics

---

### Overview

- **Total Candidates:** {stats['total_candidates']}
- **Total Unique Skills:** {stats['total_skills']}
- **Average Experience:** {stats['avg_experience']} years

---
    """
    
    return summary, category_df, skills_df, exp_df

# =============================================================================
# GRADIO INTERFACE
# =============================================================================

# Custom CSS for better styling
custom_css = """
.gradio-container {
    font-family: 'Arial', sans-serif;
}
.gr-button-primary {
    background-color: #2563eb !important;
}
"""

# Create Gradio interface
with gr.Blocks(theme=gr.themes.Soft(), css=custom_css, title="CV Matching System") as demo:
    
    gr.Markdown("""
    # 🎯 CV Parsing & Semantic Matching System
    ### Database-Oriented Resume Analysis with AI-Powered Matching
    """)
    
    with gr.Tabs():
        
        # =====================================================================
        # TAB 1: JOB MATCHING
        # =====================================================================
        with gr.Tab("🔍 Job Matching"):
            gr.Markdown("### Match candidates to your job description using semantic AI")
            
            with gr.Row():
                with gr.Column(scale=2):
                    job_input = gr.Textbox(
                        label="Job Description",
                        placeholder="Enter the job description here...\n\nExample:\nWe are looking for a Senior Python Developer with 5+ years experience in machine learning, data analysis, and cloud platforms (AWS/Azure). Must have strong SQL skills and experience with Docker/Kubernetes.",
                        lines=10
                    )
                    
                    with gr.Row():
                        matching_type = gr.Radio(
                            choices=["Hybrid (Recommended)", "Semantic Only"],
                            value="Hybrid (Recommended)",
                            label="Matching Method"
                        )
                        top_n = gr.Slider(
                            minimum=5,
                            maximum=50,
                            value=10,
                            step=5,
                            label="Number of Results"
                        )
                    
                    semantic_weight = gr.Slider(
                        minimum=0.3,
                        maximum=1.0,
                        value=0.7,
                        step=0.1,
                        label="Semantic Weight (vs Keyword)",
                        info="Higher = more focus on meaning, Lower = more focus on exact keywords"
                    )
                    
                    match_btn = gr.Button("🚀 Find Matching Candidates", variant="primary", size="lg")
                
                with gr.Column(scale=1):
                    match_summary = gr.Markdown("### Results will appear here")
            
            match_results = gr.Dataframe(
                label="Top Candidates",
                wrap=True,
                interactive=False
            )
            
            match_btn.click(
                fn=match_job_description,
                inputs=[job_input, matching_type, top_n, semantic_weight],
                outputs=[match_results, match_summary]
            )
        
        # =====================================================================
        # TAB 2: CANDIDATE SEARCH
        # =====================================================================
        with gr.Tab("👥 Search Candidates"):
            gr.Markdown("### Search and view candidate profiles")
            
            with gr.Row():
                search_input = gr.Textbox(
                    label="Search by Name or Skills",
                    placeholder="e.g., 'Python' or 'John Smith'"
                )
                category_filter = gr.Dropdown(
                    choices=["All", "Information-Technology", "Engineering", "Healthcare", 
                            "Finance", "Sales", "Business-Development", "HR", "Designer",
                            "Teacher", "Advocate", "Fitness", "Agriculture", "BPO",
                            "Consultant", "Digital-Media", "Automobile", "Chef", "Apparel",
                            "Accountant", "Construction", "Public-Relations", "Banking",
                            "Arts", "Aviation"],
                    value="All",
                    label="Filter by Category"
                )
                search_btn = gr.Button("🔎 Search", variant="primary")
            
            search_status = gr.Textbox(label="Status", interactive=False)
            search_results = gr.Dataframe(label="Search Results", wrap=True)
            
            gr.Markdown("---")
            gr.Markdown("### View Candidate Profile")
            
            with gr.Row():
                candidate_id_input = gr.Textbox(
                    label="Enter Candidate ID",
                    placeholder="e.g., 42"
                )
                profile_btn = gr.Button("📋 View Profile", variant="secondary")
            
            candidate_profile = gr.Markdown()
            
            search_btn.click(
                fn=search_candidates,
                inputs=[search_input, category_filter],
                outputs=[search_results, search_status]
            )
            
            profile_btn.click(
                fn=show_candidate_profile,
                inputs=[candidate_id_input],
                outputs=[candidate_profile]
            )
        
        # =====================================================================
        # TAB 3: UPLOAD & PROCESS
        # =====================================================================
        with gr.Tab("📤 Add New CVs"):
            gr.Markdown("### Process new candidate CVs")
            
            with gr.Tab("Single CV"):
                cv_text_input = gr.Textbox(
                    label="Paste CV Text",
                    placeholder="Paste the complete CV text here...",
                    lines=15
                )
                process_single_btn = gr.Button("✅ Process CV", variant="primary")
                
                single_result = gr.Markdown()
                single_data = gr.Dataframe(label="Extracted Information")
                
                process_single_btn.click(
                    fn=process_single_cv,
                    inputs=[cv_text_input],
                    outputs=[single_result, single_data]
                )
            
            with gr.Tab("Batch Upload (CSV)"):
                gr.Markdown("""
                Upload a CSV file with the following columns:
                - **Resume_str** (required): Full CV text
                - **Category** (optional): Job category
                """)
                
                csv_upload = gr.File(
                    label="Upload CSV File",
                    file_types=[".csv"]
                )
                process_batch_btn = gr.Button("⚡ Process Batch", variant="primary")
                
                batch_result = gr.Markdown()
                batch_stats = gr.Dataframe(label="Updated Statistics")
                
                process_batch_btn.click(
                    fn=batch_process_csv,
                    inputs=[csv_upload],
                    outputs=[batch_result, batch_stats]
                )
        
        # =====================================================================
        # TAB 4: STATISTICS
        # =====================================================================
        with gr.Tab("📊 Database Statistics"):
            gr.Markdown("### View database analytics and insights")
            
            stats_btn = gr.Button("🔄 Refresh Statistics", variant="primary")
            
            stats_summary = gr.Markdown()
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### Category Distribution")
                    category_stats = gr.Dataframe(label="Candidates by Category")
                
                with gr.Column():
                    gr.Markdown("#### Top 20 Skills")
                    skills_stats = gr.Dataframe(label="Most Common Skills")
            
            gr.Markdown("#### Experience Distribution")
            exp_stats = gr.Dataframe(label="Years of Experience")
            
            stats_btn.click(
                fn=show_statistics,
                outputs=[stats_summary, category_stats, skills_stats, exp_stats]
            )
            
            # Auto-load on tab open
            demo.load(
                fn=show_statistics,
                outputs=[stats_summary, category_stats, skills_stats, exp_stats]
            )
    
    gr.Markdown("""
    ---
    ### 💡 Tips
    - **Hybrid matching** combines semantic understanding with keyword matching for best results
    - **Semantic weight**: 0.7 works well for most cases (70% meaning, 30% keywords)
    - Use **category filters** to narrow down search results
    - **Batch processing** is recommended for large datasets
    
    ### 🔧 Technical Details
    - **Semantic Model:** sentence-transformers (all-MiniLM-L6-v2)
    - **Database:** SQLite with normalized schema
    - **Similarity Metric:** Cosine similarity on embedding vectors
    """)

# Launch the demo
if __name__ == "__main__":
    # Check if database exists, if not show warning
    db_path = Path("database/cv_database.db")
    if not db_path.exists():
        print("⚠️  Warning: Database not found. Please run the main pipeline first:")
        print("   python src/main.py")
        print()
    
    demo.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,
        share=True,  # Set to True to create public link
        show_error=True
    )