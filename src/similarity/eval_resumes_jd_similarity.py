import csv
import json
from typing import List
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report

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

def evaluate_csv(csv_path: str, threshold: float = 5.0):
    y_true: List[str] = []
    y_pred: List[str] = []
    results = []

    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    for i, row in enumerate(tqdm(rows, desc="Evaluating CSV rows")):
        try:
            jd_text = row["job_description"]
            resume_text = row["resume"]
            true_label = "matched" if row["label"].strip() == "1" else "mismatched"

            jd_entities = extract_entities_from_jd(jd_text)
            resume_entities = extract_entities_from_resume(resume_text)

            components = []
            weights = []
            result_json = {
                "row": i,
                "true_label": true_label
            }

            bert_sim = compute_bert_similarity(resume_entities, jd_entities)
            components.append(bert_sim)
            weights.append(0.35)
            result_json["bert"] = float(round(bert_sim, 2))

            jd_skills = jd_entities.get("Skills", [])
            if jd_skills:
                skill_score, _ = weighted_skill_similarity(jd_skills, resume_entities.get("Skills", []), resume_entities)
                components.append(skill_score)
                weights.append(0.25)
                result_json["skill"] = float(round(skill_score, 2))

            tfidf_sim = compute_cosine(resume_text, jd_text)
            components.append(tfidf_sim)
            weights.append(0.15)
            result_json["tfidf"] = float(round(tfidf_sim, 2))

            jd_designations = jd_entities.get("Designation", [])
            if jd_designations:
                desig_sim = compute_designation_similarity(resume_entities.get("Designation", []), jd_designations)
                components.append(desig_sim)
                weights.append(0.15)
                result_json["designation"] = float(round(desig_sim, 2))

            jd_degrees = jd_entities.get("Degree", [])
            if jd_degrees:
                degree_sim = compute_cosine(
                    flatten_for_cosine(resume_entities, "Degree", normalize_degree_text),
                    flatten_for_cosine(jd_entities, "Degree", normalize_degree_text)
                )
                components.append(degree_sim)
                weights.append(0.10)
                result_json["degree"] = float(round(degree_sim, 2))

            combined_score = float(round(sum(c * w for c, w in zip(components, weights)) / sum(weights) * 10, 2))
            predicted_label = "matched" if combined_score >= threshold else "mismatched"

            result_json["predicted_label"] = predicted_label
            result_json["combined_score"] = combined_score

            y_true.append(true_label)
            y_pred.append(predicted_label)
            results.append(result_json)

        except Exception as e:
            print(f"[Row {i}] Skipped due to error: {e}")
            continue

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred, labels=["matched", "mismatched"]))

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, digits=2))

    with open("matching_results_eval_from_csv_v2.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    evaluate_csv("../../data/eval/generated/jd_resume_dataset_v2.csv", threshold=5.0)
