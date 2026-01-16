# src/parser.py
import re
import spacy
import pandas as pd

class CVParser:
    def __init__(self):
        # Load spaCy model for Named Entity Recognition
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            print("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
    
    def extract_name(self, text):
        """
        Extract name using spaCy NER.
        """
        if not self.nlp:
            return "Unknown"
        
        doc = self.nlp(text[:500])  # Check first 500 chars
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                return ent.text
        return "Unknown"
    
    def extract_email(self, text):
        """
        Extract email using regex.
        """
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        return emails[0] if emails else None
    
    def extract_phone(self, text):
        # Match +country code followed by digits, spaces, dashes, or dots
        phone_pattern = r'(\+\d{1,3}[ \-]?\d{1,4}(?:[ \-]?\d{2,4}){1,3}|\(\d{3}\)[ \-]?\d{3}[ \-]?\d{4}|\d{3}[ \-]?\d{3}[ \-]?\d{4})'
        matches = re.findall(phone_pattern, text)
        if matches:
            # Return the first full match as-is
            return matches[0].strip()
        return None

    
    def extract_education(self, text):
        """
        Extract education section using keywords.
        """
        education_keywords = [
            'Bachelor', 'Master', 'PhD', 'B.Sc', 'M.Sc', 'MBA',
            'University', 'College', 'Institute', 'Degree'
        ]
        
        lines = text.split('\n')
        education_lines = []
        
        for i, line in enumerate(lines):
            if any(keyword.lower() in line.lower() for keyword in education_keywords):
                # Include context (2 lines before and after)
                start = max(0, i-1)
                end = min(len(lines), i+3)
                education_lines.extend(lines[start:end])
        
        return ' '.join(education_lines) if education_lines else "Not found"
    
    def extract_skills(self, text):
        """
        Extract skills using keyword matching.
        """
        # Common technical skills (extend this list)
        skill_keywords = [
            'python', 'java', 'javascript', 'sql', 'machine learning',
            'data analysis', 'docker', 'kubernetes', 'aws', 'azure',
            'react', 'angular', 'node.js', 'tensorflow', 'pytorch',
            'git', 'agile', 'scrum', 'project management', 'c++', 'c#',
            'ruby', 'php', 'swift', 'kotlin', 'go', 'rust', 'scala',
            'tableau', 'power bi', 'excel', 'powerpoint', 'word',
            'jira', 'confluence', 'slack', 'teams', 'salesforce',
            'oracle', 'mongodb', 'postgresql', 'mysql', 'redis',
            'spark', 'hadoop', 'kafka', 'airflow', 'jenkins',
            'ci/cd', 'devops', 'linux', 'windows', 'macos',
            'rest api', 'graphql', 'microservices', 'html', 'css'
        ]
        
        text_lower = text.lower()
        found_skills = [skill for skill in skill_keywords if skill in text_lower]
        
        return ', '.join(found_skills) if found_skills else "Not specified"
    
    def extract_experience_years(self, text):
        """
        Estimate years of experience from text.
        """
        # Look for patterns like "5 years experience", "3+ years"
        year_patterns = [
            r'(\d+)\+?\s*years?\s*(?:of\s*)?experience',
            r'experience\s*:?\s*(\d+)\+?\s*years?'
        ]
        
        for pattern in year_patterns:
            match = re.search(pattern, text.lower())
            if match:
                return int(match.group(1))
        
        return 0  # Default if not found
    
    def parse_cv(self, text, category=None):
        """
        Parse all information from CV text.
        
        Args:
            text: CV text to parse
            category: Optional job category
        
        Returns:
            Dictionary with extracted fields
        """
        parsed_data = {
            'name': self.extract_name(text),
            'email': self.extract_email(text),
            'phone': self.extract_phone(text),
            'education': self.extract_education(text),
            'skills': self.extract_skills(text),
            'experience_years': self.extract_experience_years(text),
            'full_text': text
        }
        
        # Add category if provided
        if category:
            parsed_data['category'] = category
        
        return parsed_data
    
    def parse_from_csv(self, csv_path, resume_column='Resume_str', 
                   category_column='Category', id_column='ID'):
        print(f"Reading CSV from: {csv_path}")
        df = pd.read_csv(csv_path)
    
        print(f"Found {len(df)} resumes in CSV")
    
        parsed_cvs = []
    
        for idx, row in df.iterrows():
            try:
                cv_text = str(row[resume_column])
                category = None
                if category_column in df.columns:
                    category = row[category_column]
                parsed_data = self.parse_cv(cv_text, category=category)
                # FIX: Use ID as name if name is "Unknown"
                if id_column in df.columns and parsed_data['name'] == "Unknown":
                    resume_id = row[id_column]
                    parsed_data['name'] = f"Candidate_{resume_id}"
                parsed_cvs.append(parsed_data)
            
                if (idx + 1) % 100 == 0:
                    print(f"Processed {idx + 1}/{len(df)} CVs...")
            except Exception as e:
                print(f"Error parsing CV at row {idx}: {str(e)}")
                continue
        print(f"Successfully parsed {len(parsed_cvs)} CVs")
        return parsed_cvs
        
