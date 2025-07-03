from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
from src.ner.eval.eval_ner_jd import extract_entities_from_jd
from src.ner.eval.eval_ner_resumes import extract_entities_from_resume
from src.similarity.resumes_jd_similarity import (
    compute_bert_similarity,
    compute_cosine,
    weighted_skill_similarity,
    compute_designation_similarity,
    normalize_degree_text,
    flatten_for_cosine,
)
from src.data_preprocess.text_preprocessor import TextPreprocessor
from fastapi.middleware.cors import CORSMiddleware
from rapidfuzz import fuzz

import nltk
nltk.download('punkt')

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalyzeRequest(BaseModel):
    resume: str
    job_description: str

import numpy as np

def convert_numpy_types(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    elif isinstance(obj, np.float32) or isinstance(obj, np.float64):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    return obj


def fuzzy_match(a, b, threshold=80):
    norm_a = normalize_degree_text(a)
    norm_b = normalize_degree_text(b)
    return fuzz.token_sort_ratio(norm_a, norm_b) >= threshold

def find_entity_matches(jd_entities, resume_entities, key, match_fn):
    matched = []
    unmatched_jd = []
    used_resume_indices = set()

    for i, jd_ent in enumerate(jd_entities.get(key, [])):
        found = False
        for j, resume_ent in enumerate(resume_entities.get(key, [])):
            if j in used_resume_indices:
                continue
            if match_fn(jd_ent, resume_ent):
                matched.append({"jd": jd_ent, "resume": resume_ent})
                used_resume_indices.add(j)
                found = True
                break
        if not found:
            unmatched_jd.append(jd_ent)

    return {"matched": matched, "unmatched": unmatched_jd}


@app.post("/api/analyze")
def analyze(request: AnalyzeRequest) -> Dict[str, Any]:
    try:
        preprocessor = TextPreprocessor()
        resume_text = preprocessor.preprocess(request.resume)
        jd_text = preprocessor.preprocess(request.job_description)

        threshold = 0.5

        jd_entities = extract_entities_from_jd(jd_text)
        resume_entities = extract_entities_from_resume(resume_text)

        components = []
        weights = []
        result = {}

        # BERT similarity
        bert_sim = compute_bert_similarity(resume_entities, jd_entities)
        components.append(bert_sim)
        weights.append(0.3174)
        result["bert_similarity"] = round(bert_sim, 2)

        # Skill similarity
        jd_skills = jd_entities.get("Skills", [])
        if jd_skills:
            skill_score, _ = weighted_skill_similarity(
                jd_skills, resume_entities.get("Skills", []), resume_entities
            )
            components.append(skill_score)
            weights.append(0.2678)
            result["skill_similarity"] = round(skill_score, 2)

        # TF-IDF lexical similarity
        tfidf_sim = compute_cosine(resume_text, jd_text)
        components.append(tfidf_sim)
        weights.append(0.1126)
        result["tfidf_similarity"] = round(tfidf_sim, 2)

        # Designation similarity
        jd_designations = jd_entities.get("Designation", [])
        if jd_designations:
            desig_sim = compute_designation_similarity(
                resume_entities.get("Designation", []), jd_designations
            )
            components.append(desig_sim)
            weights.append(0.1817)
            result["designation_similarity"] = round(desig_sim, 2)

        # Degree similarity
        jd_degrees = jd_entities.get("Degree", [])
        if jd_degrees:
            degree_sim = compute_cosine(
                flatten_for_cosine(resume_entities, "Degree", normalize_degree_text),
                flatten_for_cosine(jd_entities, "Degree", normalize_degree_text),
            )
            components.append(degree_sim)
            weights.append(0.1204)
            result["degree_similarity"] = round(degree_sim, 2)

        combined_score = round(sum(c * w for c, w in zip(components, weights)) / sum(weights), 2)
        predicted_label = "matched" if combined_score >= threshold else "mismatched"
        result["combined_score"] = combined_score
        result["predicted_label"] = predicted_label

        result["resume_entities"] = resume_entities
        result["jd_entities"] = jd_entities

        result["preprocessed_resume_text"] = resume_text
        result["preprocessed_jd_text"] = jd_text

        def normalize(text):
            return text.lower().strip()

        result["matches"] = {
            "Skills": find_entity_matches(jd_entities, resume_entities, "Skills", lambda a, b: normalize(a["text"]) == normalize(b["text"])),
            "Designation": find_entity_matches(jd_entities, resume_entities, "Designation", lambda a, b: normalize(a["text"]) == normalize(b["text"])),
            "Degree": find_entity_matches(jd_entities, resume_entities, "Degree", lambda a, b: normalize_degree_text(a["text"]) == normalize_degree_text(b["text"])),
            "Years of experience": find_entity_matches(
                jd_entities, resume_entities, "Years of experience",
                lambda a, b: "value" in a and "value" in b and isinstance(a["value"], (int, float)) and isinstance(
                    b["value"], (int, float)) and b["value"] >= a["value"]
            )
        }

        return convert_numpy_types(result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
