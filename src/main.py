# src/main.py
from parser import CVParser
from db_manager import CVDatabase
from semantic_matcher import SemanticMatcher
from hybrid_matcher import HybridMatcher


def pipeline_process_cvs(csv_path="/kaggle/input/resume-dataset/Resume/Resume.csv"):
    print("[1/4] Parsing CVs from CSV...")
    parser = CVParser()
    parsed_cvs = parser.parse_from_csv(csv_path)
    
    print("[2/4] Storing in database...")
    db = CVDatabase()
    for cv in parsed_cvs:
        candidate_id = db.insert_candidate(cv)
        if candidate_id:
            print(f"✓ Stored: {cv['name']} (ID: {candidate_id})")
    
    print("[3/4] Generating embeddings...")
    matcher = SemanticMatcher()
    matcher.precompute_all_embeddings()
    
    print("[4/4] Complete!")

def match_job(job_description, use_hybrid=True):
    """
    Match candidates to a job description.
    """
    print("\n" + "=" * 60)
    print("JOB MATCHING")
    print("=" * 60)
    print(f"\nJob Description:\n{job_description}\n")
    
    if use_hybrid:
        matcher = HybridMatcher()
        results = matcher.hybrid_rank(job_description)
        
        print("\nTop 5 Matches (Hybrid Scoring):")
        print(f"{'Rank':<6}{'Name':<20}{'Semantic':<12}{'Keyword':<12}{'Combined':<12}")
        print("-" * 62)
        
        for rank, (cid, name, sem, kw, combined) in enumerate(results[:5], 1):
            print(f"{rank:<6}{name:<20}{sem:<12.4f}{kw:<12.4f}{combined:<12.4f}")
    else:
        matcher = SemanticMatcher()
        results = matcher.rank_candidates(job_description)
        
        print("\nTop 5 Matches (Semantic Only):")
        for rank, (cid, name, score) in enumerate(results[:5], 1):
            print(f"{rank}. {name} - Score: {score:.4f}")
    
    print("=" * 60)

# Example usage
if __name__ == "__main__":
    # Run the complete pipeline
    pipeline_process_cvs()
    
    # Example job matching
    example_job = """
    We are looking for an experienced HR Administrator to manage employee relations, training, and benefits. The ideal candidate
    has a strong background in the hospitality industry (Hilton, IHG) with 5+ years experience and is proficient with systems like HRIS, Micros, and Opera PMS. 
    Skills in conflict resolution,team management, and customer service are essential. You will be responsible for onboarding,
    policy development, and loss prevention strategies.
    """
    
    match_job(example_job, use_hybrid=True)

    