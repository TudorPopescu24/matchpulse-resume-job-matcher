import re
import json
import pandas as pd
from typing import List, Dict
from collections import defaultdict
from nltk import sent_tokenize

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from src.data_preprocess.text_preprocessor import TextPreprocessor

MODEL_PATH = "../../ner/train/ner_roberta_focal_final"

REQUIREMENTS_HEADERS = ["requirements", "must have", "you must"]
PREFERRED_HEADERS = ["preferred qualifications", "nice to have", "bonus points", "would be a plus"]

def split_requirements_preferred(text: str) -> Dict[str, str]:
    lines = text.splitlines()
    sections = defaultdict(str)
    current = None
    for line in lines:
        line_clean = line.strip().lower()
        if any(h in line_clean for h in REQUIREMENTS_HEADERS):
            current = "requirements"
        elif any(h in line_clean for h in PREFERRED_HEADERS):
            current = "preferred"
        elif re.match(r"^[A-Z][a-z]+:", line):
            current = None
        if current:
            sections[current] += line + "\n"
    return sections

def extract_year_value(text: str) -> int | None:
    match = re.search(r"\b(\d{1,2})\s*(\+|plus)?\s*(years?|yrs?)", text)
    return int(match.group(1)) if match else None

def extract_entities_from_jd(text: str) -> Dict[str, List[Dict]]:
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
                "start": ent["start"],
                "end": ent["end"],
                "score": ent["score"],
                "source": "bert"
            })
        offset += len(sentence) + 1

    section_spans = split_requirements_preferred(text)
    requirement_spans = section_spans.get("requirements", "")
    preferred_spans = section_spans.get("preferred", "")

    structured = defaultdict(list)
    for e in entities:
        span_text = preprocessed[e["start"]:e["end"]].lower()
        if span_text in preferred_spans.lower():
            section = "preferred"
        elif span_text in requirement_spans.lower():
            section = "requirements"
        else:
            section = "general"

        item = {
            "text": e["text"],
            "start": e["start"],
            "end": e["end"],
            "section": section
        }

        if e["label"] == "Years of experience":
            item["value"] = extract_year_value(e["text"])

        structured[e["label"]].append(item)

    return dict(structured)

if __name__ == "__main__":
    with open("./examples/jd_sample.txt", "r", encoding="utf-8") as f:
        jd_text = f.read()

    result = extract_entities_from_jd(jd_text)

    with open("./examples/jd_entities.json", "w", encoding="utf-8") as out_file:
        json.dump(result, out_file, indent=2, ensure_ascii=False)

    print("JD entities with section classification saved.")
