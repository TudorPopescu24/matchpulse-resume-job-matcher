import json
from typing import Dict, List
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz

bert_model = SentenceTransformer("all-MiniLM-L6-v2")

DEGREES = {
    "b.sc", "bachelor", "bachelors", "btech", "b.e", "m.sc", "mtech", "m.e",
    "mba", "phd", "ph.d", "master", "masters", "bsdc", "msdc",
    "bachelor's degree", "bachelor's", "master's degree", "m.sc.", "b.sc."
}
degree_canonical_map = {
    "b.sc": "Bachelor", "b.sc.": "Bachelor", "bachelor": "Bachelor", "bachelors": "Bachelor",
    "bachelor's": "Bachelor", "bachelor's degree": "Bachelor", "btech": "Bachelor", "b.e": "Bachelor",
    "m.sc": "Master", "m.sc.": "Master", "mtech": "Master", "m.e": "Master", "master": "Master",
    "masters": "Master", "master's degree": "Master", "phd": "PhD", "ph.d": "PhD", "mba": "Master",
    "bsdc": "Bachelor", "msdc": "Master"
}

def normalize_degree_text(text: str) -> str:
    text_clean = text.lower()
    for deg in sorted(DEGREES, key=len, reverse=True):
        if deg in text_clean:
            canonical = degree_canonical_map.get(deg, deg.title())
            remaining = text_clean.replace(deg, "").strip(" ,:-")
            return f"{canonical} in {remaining.title()}" if remaining else canonical
    return text.strip().title()

def semantic_fuzzy_match(jd_skill, resume_skills, fuzzy_weight=0.3, min_combined_threshold=0.5):
    jd_text = jd_skill.get("text", "").strip().lower()
    jd_years = jd_skill.get("years_of_experience")
    jd_vec = bert_model.encode([jd_text])[0]

    best_score, best_match = 0.0, None

    for r in resume_skills:
        r_text = r.get("text", "").strip().lower()
        r_years = r.get("years_of_experience")
        r_vec = bert_model.encode([r_text])[0]

        bert_score = cosine_similarity([jd_vec], [r_vec])[0][0]
        fuzzy_score = fuzz.token_sort_ratio(jd_text, r_text) / 100.0
        combined = fuzzy_weight * fuzzy_score + (1 - fuzzy_weight) * bert_score

        if jd_years and r_years:
            combined *= 0.5 + 0.5 * min(1.0, r_years / jd_years)
        elif jd_years or r_years:
            combined *= 0.9

        if combined > best_score and combined > min_combined_threshold:
            best_score, best_match = combined, r_text

    return {
        "jd_skill": jd_text,
        "matched_resume_skill": best_match,
        "similarity_score": round(best_score, 2) if best_match else 0.0
    }

def flatten_for_cosine(entities: Dict[str, List[Dict]], label: str, normalize_fn=None) -> str:
    items = entities.get(label, [])
    texts = []
    for e in items:
        txt = e["text"].strip()
        if normalize_fn:
            txt = normalize_fn(txt)
        if txt:
            section = e.get("section", "general")
            weight = 1.4 if section == "requirements" else 1.2 if section == "preferred" else 1.0
            texts.extend([txt.lower()] * round(weight))
    return " ".join(texts)

def compute_cosine(text1: str, text2: str) -> float:
    if not text1.strip() or not text2.strip():
        return 0.0
    tfidf = TfidfVectorizer().fit_transform([text1, text2])
    return cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]

def compute_designation_similarity(resume_designations: List[Dict], jd_designations: List[Dict]) -> float:
    if not resume_designations or not jd_designations:
        return 0.0
    resume_text = " ".join(d["text"].strip().lower() for d in resume_designations if d["text"].strip())
    jd_text = " ".join(d["text"].strip().lower() for d in jd_designations if d["text"].strip())
    return compute_cosine(resume_text, jd_text)

def weighted_skill_similarity(jd_skills, resume_skills, resume_entities) -> (float, List[Dict]):
    if not jd_skills:
        return 1.0, []
    matches = [semantic_fuzzy_match(j, resume_skills) for j in jd_skills]
    avg = round(sum(m["similarity_score"] for m in matches) / len(jd_skills), 2)
    return avg, matches

def get_all_text(entities: Dict[str, List[Dict]]) -> str:
    return " ".join(e["text"].strip() for spans in entities.values() for e in spans if e["text"].strip())

def compute_bert_similarity(resume: Dict, jd: Dict) -> float:
    r_text, j_text = get_all_text(resume), get_all_text(jd)
    emb = bert_model.encode([r_text, j_text])
    return cosine_similarity([emb[0]], [emb[1]])[0][0]

def compute_tf_idf_similarity(resume: Dict, jd: Dict) -> float:
    r_text = get_all_text(resume)
    j_text = get_all_text(jd)
    return compute_cosine(r_text, j_text)

def match_resume_to_jd(resume_path: str, jd_path: str):
    with open(resume_path, encoding="utf-8") as f:
        resume = json.load(f)
    with open(jd_path, encoding="utf-8") as f:
        jd = json.load(f)

    jd_skills = jd.get("Skills", [])
    resume_skills = resume.get("Skills", [])
    skill_score, matches = weighted_skill_similarity(jd_skills, resume_skills)

    degree_sim = compute_cosine(
        flatten_for_cosine(resume, "Degree", normalize_degree_text),
        flatten_for_cosine(jd, "Degree", normalize_degree_text)
    )
    designation_sim = compute_designation_similarity(
        resume.get("Designation", []),
        jd.get("Designation", [])
    )
    tfidf_sim = compute_tf_idf_similarity(resume, jd)
    bert_sim = compute_bert_similarity(resume, jd)

    print("=== Cosine Similarity ===")
    print(f"Degree similarity:         {degree_sim:.2f}")
    print(f"Designation similarity:    {designation_sim:.2f}")
    print(f"Weighted skill similarity: {skill_score:.2f}")

    print("\n=== Matched Skills ===")
    for m in matches:
        jd_s, resume_s, score = m["jd_skill"], m["matched_resume_skill"], m["similarity_score"]
        if resume_s:
            print(f"JD: '{jd_s}' ↔ Resume: '{resume_s}' (Score: {score})")
        else:
            print(f"JD: '{jd_s}' ↔ No match found (Best Score: {score})")

    print("\n=== TF-IDF Similarity ===")
    print(f"TF-IDF Full Text similarity:  {tfidf_sim:.2f}")

    print("\n=== BERT-Based Semantic Similarity ===")
    print(f"Overall Resume–JD similarity: {bert_sim:.2f}")

if __name__ == "__main__":
    match_resume_to_jd("../ner/eval/examples/resume_entities.json", "../ner/eval/examples/jd_entities.json")
