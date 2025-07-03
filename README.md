
# MatchPulse

**MatchPulse** is a hybrid resume–job description matching system that combines transformers, weak supervision, and multi-dimensional similarity scoring to solve real-world hiring challenges.

Developed as part of my Master's dissertation at Alexandru Ioan Cuza University, Faculty of Computer Science.

---

## 🌟 Key Features

- 🔍 **Resume & Job Description Parsing** using DistilBERT, RoBERTa, and ELECTRA
- 🏷️ **Entity Extraction** for Skills, Degree, Designation, and Years of Experience
- 🤖 **Hybrid Annotation Pipeline** using Snorkel weak supervision + GPT-based labeling
- 🔗 **Similarity Scoring Pipeline** combining semantic (BERT), lexical (TF-IDF), and entity-based scoring
- 🛡️ **Explainability**: JSON outputs explaining match scores and contributing factors
- 🌐 **REST API** for programmatic interaction
- 💻 **Demo UI** (minimal frontend)

---

## 🔧 Project Structure

```
src/
├── api/                # REST API (Flask/FastAPI)
├── data_analysis/      # Resume & annotation EDA
├── data_generation/    # Synthetic data generation (GPT)
├── data_preprocess/    # Text cleaning, weak supervision (Snorkel)
├── ner/                # NER training & evaluation
├── similarity/         # Resume–JD similarity pipeline
└── utils/              # Helper functions

ui/                     # Demo frontend (HTML/JS)

notebooks/              # Example Jupyter notebooks
data/                   # Example input files (optional, add your own)
```

---

## 🚀 Quickstart

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/matchpulse.git
cd matchpulse
```

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the API (Example)
```bash
cd src/api
python get_similarity.py
```

The API will be available at:  
`http://localhost:5000/similarity`

Send a POST request with your resume and job description to get a match score.

### 4. Explore the Demo UI
Open `ui/index.html` in your browser. *(Requires the API running locally)*

---

## 🔍 Example API Usage

```bash
curl -X POST http://localhost:5000/similarity   -H "Content-Type: application/json"   -d '{"resume": "Experienced Java Developer...", "job_description": "Looking for a backend engineer..."}'
```

Example Response:
```json
{
  "match": true,
  "score": 0.89,
  "explanation": {
    "Skills": ["Java", "Spring Boot"],
    "Degree": ["Bachelor's in Computer Science"],
    "Designation": ["Backend Engineer"]
  }
}
```

---

## 📊 Demo Video

🎥 [Watch the MatchPulse Demo](https://your-video-link.com)

---

## 🧠 Models & Techniques Used
- **Transformers:** RoBERTa, DistilBERT, ELECTRA
- **Weak Supervision:** Snorkel for labeling functions
- **NER Training:** Focal Loss to mitigate class imbalance
- **Similarity:** BERT embeddings, TF-IDF, entity matching
- **Explainability:** JSON output for transparency

---

## 📂 Datasets
See `src/data_generation/` for generating your own datasets using GPT.  
Example datasets used:
- Kaggle Resume Dataset
- Custom GPT-generated similarity pairs

---

## 🛡 License

MIT License.

---

## 🤝 Contributing

Contributions are welcome! Feel free to open Issues and Pull Requests.

---

## 📚 Academic Reference

This project is based on my dissertation:  
**MatchPulse: A Hybrid Approach Using Transformers and Weak Supervision for Resume–Job Matching**  
Supervised by Conf. Dr. Diana-Maria Trandabăț  
Alexandru Ioan Cuza University, Iași — 2025.

---

## 🔗 Related Keywords

`#NLP #ResumeParsing #JobMatching #RoBERTa #Snorkel #Transformers #OpenSource #ExplainableAI`
