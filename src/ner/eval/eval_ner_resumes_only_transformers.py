import re
import json
from typing import List, Dict
from collections import defaultdict
from nltk import sent_tokenize
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

from src.data_preprocess.text_preprocessor import TextPreprocessor

MODEL_PATH = "../../ner/train/ner_roberta_focal_final"

def extract_year_value(text: str) -> int | None:
    match = re.search(r"\b(\d{1,2})\s*(\+|plus)?\s*(years?|yrs?)", text)
    return int(match.group(1)) if match else None

def link_skills_to_experience(entities: List[Dict], max_distance: int = 120) -> List[Dict]:
    years = [e for e in entities if e['label'] == 'Years of experience' and e.get('value') is not None]
    skills = [e for e in entities if e['label'] == 'Skills']
    links = []
    for s in skills:
        best_year = None
        min_dist = float('inf')
        for y in years:
            dist = min(abs(s['start'] - y['end']), abs(y['start'] - s['end']))
            if dist <= max_distance and dist < min_dist:
                min_dist = dist
                best_year = y
        result = {
            "start": s['start'],
            "end": s['end'],
            "text": s['text'],
            "years_of_experience": best_year["value"] if best_year else None
        }
        links.append(result)
    return links

def extract_entities_from_resume(text: str) -> Dict:
    preprocessor = TextPreprocessor()
    preprocessed = preprocessor.preprocess(text)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
    ner = pipeline("token-classification", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

    entities = []
    offset = 0
    for sentence in sent_tokenize(preprocessed):
        ents = ner(sentence)
        ents = [ent for ent in ents if ent["score"] >= 0.5]
        for ent in ents:
            ent["start"] += offset
            ent["end"] += offset
            entities.append({
                "label": ent["entity_group"],
                "text": ent["word"].strip(),
                "score": ent["score"],
                "start": ent["start"],
                "end": ent["end"],
                "source": "bert"
            })
        offset += len(sentence) + 1

    simplified = defaultdict(list)
    for e in entities:
        base = {
            "start": e["start"],
            "end": e["end"],
            "text": e["text"]
        }
        if e["label"] == "Years of experience":
            base["value"] = extract_year_value(e["text"])
        simplified[e["label"]].append(base)

    if "Skills" in simplified:
        enriched_skills = link_skills_to_experience(entities)
        simplified["Skills"] = enriched_skills

    return simplified

if __name__ == "__main__":
    with open("./examples/resume_sample.txt", "r", encoding="utf-8") as f:
        sample_text = f.read()

    result = extract_entities_from_resume(sample_text)

    with open("./examples/resume_entities.json", "w", encoding="utf-8") as out_file:
        json.dump(result, out_file, indent=2, ensure_ascii=False)

    print("Entities saved.")
