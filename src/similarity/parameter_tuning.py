import csv
import json
import random
import numpy as np
from typing import List, Dict, Tuple
from sklearn.metrics import classification_report, f1_score
from tqdm import tqdm

from src.ner.eval.eval_ner_jd_only_transformers import extract_entities_from_jd
from src.ner.eval.eval_ner_resumes_only_transformers import extract_entities_from_resume
from src.similarity.resumes_jd_similarity import (
    compute_bert_similarity,
    compute_cosine,
    weighted_skill_similarity,
    compute_designation_similarity,
    normalize_degree_text,
    flatten_for_cosine,
)


def compute_score(jd_text, resume_text, jd_entities, resume_entities, weights: Dict[str, float], threshold: float) -> Tuple[str, str, float]:
    components = []

    bert_sim = compute_bert_similarity(resume_entities, jd_entities)
    components.append(bert_sim * weights["bert"])

    jd_skills = jd_entities.get("Skills", [])
    skill_score = 0.0
    if jd_skills:
        skill_score, _ = weighted_skill_similarity(jd_skills, resume_entities.get("Skills", []), resume_entities)
    components.append(skill_score * weights["skill"])

    tfidf_sim = compute_cosine(resume_text, jd_text)
    components.append(tfidf_sim * weights["tfidf"])

    jd_designations = jd_entities.get("Designation", [])
    desig_sim = 0.0
    if jd_designations:
        desig_sim = compute_designation_similarity(resume_entities.get("Designation", []), jd_designations)
    components.append(desig_sim * weights["designation"])

    jd_degrees = jd_entities.get("Degree", [])
    degree_sim = 0.0
    if jd_degrees:
        degree_sim = compute_cosine(
            flatten_for_cosine(resume_entities, "Degree", normalize_degree_text),
            flatten_for_cosine(jd_entities, "Degree", normalize_degree_text)
        )
    components.append(degree_sim * weights["degree"])

    combined_score = sum(components) / sum(weights.values()) * 10
    predicted_label = "matched" if combined_score >= threshold else "mismatched"
    return predicted_label, combined_score, skill_score


def sample_and_evaluate(csv_path: str, weights: Dict[str, float], sample_size: int = 100, threshold: float = 5.0) -> Tuple[float, List[str], List[str]]:
    y_true: List[str] = []
    y_pred: List[str] = []

    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    sampled = random.sample(rows, sample_size)

    for i, row in enumerate(tqdm(sampled, desc="Evaluating sampled rows")):
        try:
            jd_text = row["job_description"]
            resume_text = row["resume"]
            true_label = "matched" if row["label"].strip() == "1" else "mismatched"

            jd_entities = extract_entities_from_jd(jd_text)
            resume_entities = extract_entities_from_resume(resume_text)

            pred_label, _, _ = compute_score(jd_text, resume_text, jd_entities, resume_entities, weights, threshold)

            y_true.append(true_label)
            y_pred.append(pred_label)
        except Exception as e:
            print(f"Row {i} skipped: {e}")
            continue

    macro_f1 = f1_score(y_true, y_pred, average="macro")
    return macro_f1, y_true, y_pred


def tune_weights(csv_path: str, iterations: int = 30, sample_size: int = 100, threshold: float = 5.0):
    best_f1 = 0.0
    best_weights = {}

    for _ in range(iterations):
        raw_weights = {
            "bert": random.uniform(0.1, 0.4),
            "skill": random.uniform(0.1, 0.4),
            "tfidf": random.uniform(0.05, 0.3),
            "designation": random.uniform(0.05, 0.3),
            "degree": random.uniform(0.05, 0.3),
        }

        total = sum(raw_weights.values())
        weights = {k: round(v / total, 4) for k, v in raw_weights.items()}

        f1, y_true, y_pred = sample_and_evaluate(csv_path, weights, sample_size, threshold)

        print(f"F1={f1:.4f} | Weights={weights}")
        if f1 > best_f1:
            best_f1 = f1
            best_weights = weights

    print("\nBest configuration:")
    print(json.dumps(best_weights, indent=2))
    print(f"Macro F1: {best_f1:.4f}")



if __name__ == "__main__":
    tune_weights("../../data/eval/generated/jd_resume_dataset_v2.csv")
