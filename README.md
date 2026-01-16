# CV Parsing and Semantic Matching System

**A database-oriented resume analysis system with AI-powered semantic matching for job-candidate alignment.**

---

## **📌 Overview**
This project is a **CV Parsing and Semantic Matching System** designed to:
- Parse resumes (CVs) from text or CSV files.
- Store structured data in a **SQLite database**.
- Generate **semantic embeddings** for CVs and job descriptions.
- Match candidates to job descriptions using **hybrid (semantic + keyword) scoring**.
- Provide a **Gradio-based web interface** for interactive use.

<img width="1353" height="637" alt="CVMATCH1" src="https://github.com/user-attachments/assets/3de6be5f-7eab-487b-886c-d5156cf24272" />
<img width="1339" height="633" alt="CVMATCH3" src="https://github.com/user-attachments/assets/771ee821-dbfd-4352-9b5f-797d8755bc9f" />
<img width="1349" height="633" alt="CVMATCH4" src="https://github.com/user-attachments/assets/a625c94e-7b32-4cb6-b74b-d54640dc8d80" />
<img width="1344" height="634" alt="CVMATCH5" src="https://github.com/user-attachments/assets/b4b89d8c-215a-47dc-b71a-67fba515e971" />





---

## **🔧 Features**
| Feature | Description |
|---------|-------------|
| **CV Parsing** | Extracts names, emails, skills, education, and experience from raw text. |
| **Database Storage** | Uses SQLite for structured storage of candidates, skills, and embeddings. |
| **Semantic Matching** | Uses `sentence-transformers` to generate embeddings and compute cosine similarity. |
| **Hybrid Matching** | Combines semantic and keyword-based matching for improved accuracy. |
| **Gradio UI** | Interactive web interface for job matching, candidate search, and database stats. |

---

## **Installation**

### **Prerequisites**
- Python 3.8+
- SQLite3
- Required libraries (see `requirements.txt`)

### **Setup**
1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-repo/cv-matching-system.git
   cd cv-matching-system
   ```
2.**Install dependencies:**
```bash
pip install -r requirements.txt
```
3.**Initialize the pipeline:**
```bash
python src/main.py
python src/app-demp.py
```
 ## **Usage**
### 1. Process CVs
**Single CV:**
Paste CV text in the Gradio UI under "Add New CVs".
**Batch Processing (CSV):**
Upload a CSV file with a Resume_str column (and optional Category column).

### **2. Match Candidates to Jobs**

Enter a job description in the "Job Matching" tab.
Select matching method (Hybrid or Semantic Only).
Adjust semantic weight (default: 0.7).
Click "Find Matching Candidates" to get ranked results.

### **3. Search Candidates**
Use the "Search Candidates" tab to filter by name, skills, or category. and use Id candidate to see all its information profile .

### **4. View Database Stats**

Check category distribution, top skills, and experience levels in the "Database Statistics" tab.

## **Technical Details**

### **Database Schema**

Candidates: Stores parsed CV data (name, email, skills, etc.).
Skills: Normalized table for skills (many-to-many with candidates).
Embeddings: Stores semantic vectors for CVs.
Job Descriptions: Stores job postings for matching.
Matching Results: Stores scores for candidate-job pairs.

### **Semantic Matching**

Uses all-MiniLM-L6-v2 (Sentence Transformers) for embeddings.
Computes cosine similarity between job and CV embeddings.

### **Hybrid Matching**

Combines semantic similarity (70%) and keyword overlap (30%) for robust ranking.

## **Contributing**

Contributions are welcome! Open an issue or submit a PR.

 ## **License**
MIT License. See LICENSE for details.





















