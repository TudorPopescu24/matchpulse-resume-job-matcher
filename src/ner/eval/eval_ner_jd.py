import re
import json
import pandas as pd
from typing import List, Dict
from collections import defaultdict
from nltk import sent_tokenize
import os

from snorkel.labeling import labeling_function, PandasLFApplier, LabelingFunction
from snorkel.labeling.model import LabelModel
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

from src.data_preprocess.text_preprocessor import TextPreprocessor

MODEL_PATH = "./src/ner/train/ner_roberta_focal_final"

REQUIREMENTS_HEADERS = ["requirements", "must have", "you must"]
PREFERRED_HEADERS = ["preferred qualifications", "nice to have", "bonus points", "would be a plus"]

LABELS = {
    "Skills": 0,
    "Designation": 1,
    "Years of experience": 2,
    "Degree": 3
}
ABSTAIN = -1

DEGREES = {
    "b.sc", "bachelor", "bachelors", "btech", "b.e", "m.sc", "mtech", "m.e",
    "mba", "phd", "ph.d", "master", "masters", "bsdc", "msdc",
    "bachelor's degree", "bachelor's", "master's degree", "phd", "m.sc.", "b.sc."
}

@labeling_function()
def lf_skills(x):
    text = x.text.lower()
    return LABELS["Skills"] if any(k in text for k in SKILLS) and len(text.split()) < 10 else ABSTAIN

@labeling_function()
def lf_designation(x):
    text = x.text.lower()
    return LABELS["Designation"] if any(k in text for k in JOB_TITLES) and len(text.split()) < 10 else ABSTAIN

@labeling_function()
def lf_years(x):
    text = x.text.lower()
    patterns = [
        r"\b\d{1,2}\+?\s*(years?|yrs?)\b",
        r"^\s*\d{1,2}\s*(years?|yrs?)\s*$",
        r"\b\d{1,2}\s*(\+|plus)?\s*(years?|yrs?)\s+of\s+experience",
        r"experience\s+(of\s+)?\d{1,2}\s+(years?|yrs?)"
    ]
    return LABELS["Years of experience"] if any(re.search(p, text) for p in patterns) else ABSTAIN

@labeling_function()
def lf_degree(x):
    text = x.text.lower()
    return LABELS["Degree"] if any(k in text for k in DEGREES) else ABSTAIN

def safe_read_csv(path, expected_columns):
    encodings_to_try = ['utf-8', 'cp1252', 'ISO-8859-1']
    for enc in encodings_to_try:
        try:
            df = pd.read_csv(path, encoding=enc, encoding_errors='ignore', on_bad_lines='skip')
            df = df[[col for col in expected_columns if col in df.columns]]
            return df.dropna()
        except Exception:
            continue
    return pd.DataFrame(columns=expected_columns)

lfs = [lf_skills, lf_designation, lf_years, lf_degree]

skills_df = safe_read_csv("./data/annotation/skills_dataset.csv", ["name:String"])
SKILLS = set(skills_df["name:String"].str.lower().unique())

jobtitles_df = safe_read_csv("./data/annotation/it_jobtitles_dataset.csv", ["Job Title"])
JOB_TITLES = set(jobtitles_df["Job Title"].str.lower().unique())

def extract_degree_sentences(text: str) -> List[str]:
    blocks = re.split(r"\n{2,}|EDUCATION|EDUCATIE|UCATION|ED", text, flags=re.IGNORECASE)
    candidate_blocks = []
    for block in blocks:
        lines = block.strip().splitlines()
        chunk = " ".join(line.strip() for line in lines if line.strip())
        if any(degree in chunk.lower() for degree in DEGREES):
            candidate_blocks.append(chunk)
    return candidate_blocks

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

def merge_adjacent_labels(entities: List[Dict], label: str, max_gap: int = 2) -> List[Dict]:
    merged = []
    temp = None
    for e in sorted([e for e in entities if e['label'] == label], key=lambda x: x['start']):
        if temp and e['start'] - temp['end'] <= max_gap:
            temp['text'] += ' ' + e['text'].strip()
            temp['end'] = e['end']
        else:
            if temp:
                merged.append(temp)
            temp = e
    if temp:
        merged.append(temp)
    return merged

def rule_based_extraction(text: str) -> List[Dict]:
    matches = []
    for skill in SKILLS:
        for m in re.finditer(r'\b' + re.escape(skill) + r'\b', text, flags=re.IGNORECASE):
            matches.append({"label": "Skills", "text": m.group(), "start": m.start(), "end": m.end(), "score": 1.0, "source": "rule"})
    for title in JOB_TITLES:
        for m in re.finditer(r'\b' + re.escape(title) + r'\b', text, flags=re.IGNORECASE):
            matches.append({"label": "Designation", "text": m.group(), "start": m.start(), "end": m.end(), "score": 1.0, "source": "rule"})

    degree_pattern = r"\b(bachelor(?:'s)?|master(?:'s)?|ph\.?d|phd|m\.?sc|b\.?sc|m\.?tech|b\.?tech)[^\n]{0,120}?(in|of)?\s+(?P<field>[A-Z][a-zA-Z &]{2,}(?:\s+[A-Z][a-zA-Z &]{2,})*)"
    for m in re.finditer(degree_pattern, text, flags=re.IGNORECASE):
        matches.append({"label": "Degree", "text": m.group().strip(), "start": m.start(), "end": m.end(), "score": 1.0, "source": "rule"})

    year_patterns = [
        r"\b\d{1,2}\+?\s*(years?|yrs?)\b",
        r"\b\d{1,2}\s*(\+|plus)?\s*(years?|yrs?)\s+of\s+experience",
        r"experience\s+(of\s+)?\d{1,2}\s+(years?|yrs?)"
    ]
    for pattern in year_patterns:
        for m in re.finditer(pattern, text, flags=re.IGNORECASE):
            matches.append({
                "label": "Years of experience",
                "text": m.group(),
                "start": m.start(),
                "end": m.end(),
                "score": 1.0,
                "source": "rule"
            })

    return matches

def clean_skill_text(text: str) -> str:
    return re.sub(r"^[*]+|[*]+$", "", text.strip(" *"))

def resolve_conflicts(predictions: List[Dict]) -> List[Dict]:
    grouped = defaultdict(list)
    for r in predictions:
        key = (r["label"], r["start"], r["end"], r["text"].lower())
        grouped[key].append(r)
    resolved = []
    for group in grouped.values():
        label_count = defaultdict(int)
        for r in group:
            label_count[r["label"]] += 1
        majority_label = max(label_count.items(), key=lambda x: x[1])[0]
        best = max((r for r in group if r["label"] == majority_label), key=lambda x: x["score"])
        resolved.append(best)
    return resolved

def resolve_and_group(predictions: List[Dict]) -> List[Dict]:
    seen = set()
    predictions = [r for r in predictions if r["source"] != "bert" or r["score"] >= 0.7]
    predictions = sorted(predictions, key=lambda x: (x["start"], -(x["end"] - x["start"])))
    merged = []
    for current in predictions:
        key = (current["label"], current["start"], current["end"])
        if key in seen:
            continue
        seen.add(key)
        overlap = False
        for accepted in merged:
            if not (current["end"] <= accepted["start"] or current["start"] >= accepted["end"]):
                if current["label"] == accepted["label"] and current["label"] not in {"Years of experience", "Degree"}:
                    overlap = True
                    if (current["end"] - current["start"] > accepted["end"] - accepted["start"]) or \
                       (current["score"] > accepted["score"]):
                        merged.remove(accepted)
                        merged.append(current)
                    break
        if not overlap:
            merged.append(current)
    return merged

def extract_year_value(text: str) -> int | None:
    match = re.search(r"\b(\d{1,2})\s*(\+|plus)?\s*(years?|yrs?)", text)
    return int(match.group(1)) if match else None

def link_skills_to_experience(entities: List[Dict], max_distance: int = 120) -> List[Dict]:
    years = [e for e in entities if e['label'] == 'Years of experience']
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
            "text": s['text']
        }
        if best_year:
            result["years_of_experience"] = extract_year_value(best_year["text"])
        links.append(result)
    return links


def split_long_degrees(text_block: str, base_start: int) -> List[Dict]:
    patterns = [
        r"(Bachelor(?:'s)? (?:Degree)?(?: in [A-Z][a-z]+(?: [A-Z][a-z]+)*)?)",
        r"(Master(?:'s)? (?:Degree)?(?: in [A-Z][a-z]+(?: [A-Z][a-z]+)*)?)",
        r"(Ph\\.?D(?: in [A-Z][a-z]+(?: [A-Z][a-z]+)*)?)",
        r"(MSc(?: in [A-Z][a-z]+(?: [A-Z][a-z]+)*)?)",
        r"(BSc(?: in [A-Z][a-z]+(?: [A-Z][a-z]+)*)?)"
    ]
    found = []
    for pattern in patterns:
        for m in re.finditer(pattern, text_block):
            found.append({
                "label": "Degree",
                "text": m.group(),
                "start": base_start + m.start(),
                "end": base_start + m.end(),
                "score": 1.0,
                "source": "split"
            })
    return found

def extract_entities_from_jd(text: str) -> Dict[str, List[Dict]]:
    preprocessor = TextPreprocessor()
    preprocessed = preprocessor.preprocess(text)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
    ner = pipeline("token-classification", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

    bert_results = []
    offset = 0
    for sentence in sent_tokenize(preprocessed):
        ents = ner(sentence)
        ents = [ent for ent in ents if ent["score"] >= 0.5]
        for ent in ents:
            ent["start"] += offset
            ent["end"] += offset
            ent["text"] = ent["word"].strip()
            bert_results.append({
                "label": ent["entity_group"],
                "text": ent["text"],
                "score": ent["score"],
                "start": ent["start"],
                "end": ent["end"],
                "source": "bert"
            })
        offset += len(sentence) + 1

    spans = [s for s in sent_tokenize(preprocessed) if 5 < len(s.split()) < 20]
    spans += extract_degree_sentences(preprocessed)
    df = pd.DataFrame({"text": spans})

    L = PandasLFApplier(lfs).apply(df)
    label_model = LabelModel(cardinality=4, verbose=False)
    label_model.fit(L_train=L, n_epochs=300)
    df["weak_label"] = label_model.predict(L)

    weak_results = []
    for _, row in df.iterrows():
        if row.weak_label != ABSTAIN:
            label = list(LABELS.keys())[row.weak_label]
            match = re.search(re.escape(row.text), preprocessed)
            if match:
                weak_results.append({
                    "label": label,
                    "text": row.text,
                    "score": 1.0,
                    "start": match.start(),
                    "end": match.end(),
                    "source": "weak"
                })

    rule_results = rule_based_extraction(preprocessed)
    all_preds = bert_results + weak_results + rule_results
    entities = resolve_and_group(resolve_conflicts(all_preds))

    designations = merge_adjacent_labels(entities, "Designation")
    entities = [e for e in entities if e["label"] != "Designation"] + designations

    split_entities = []
    for e in entities:
        if e['label'] == 'Degree' and (e['end'] - e['start'] > 100):
            split_entities.extend(split_long_degrees(e['text'], e['start']))
        else:
            split_entities.append(e)
    entities = split_entities

    section_spans = split_requirements_preferred(text)
    requirement_spans = section_spans.get("requirements", "")
    preferred_spans = section_spans.get("preferred", "")

    for e in entities:
        span_text = preprocessed[e["start"]:e["end"]].lower()
        if span_text in preferred_spans.lower():
            e["section"] = "preferred"
        elif span_text in requirement_spans.lower():
            e["section"] = "requirements"
        else:
            e["section"] = "general"

    structured = defaultdict(list)
    for e in entities:
        structured[e["label"]].append({
            "text": e["text"],
            "start": e["start"],
            "end": e["end"],
            "section": e["section"]
        })

    if "Skills" in structured:
        linked_skills = link_skills_to_experience(entities)
        for skill in structured["Skills"]:
            for linked in linked_skills:
                if skill["start"] == linked["start"] and skill["end"] == linked["end"]:
                    skill["years_of_experience"] = linked.get("years_of_experience")

        seen = {}
        filtered_skills = []
        for skill in structured["Skills"]:
            key = skill["text"].lower()
            if key not in seen or (not seen[key].get("years_of_experience") and skill.get("years_of_experience")):
                seen[key] = skill
        structured["Skills"] = [
            {
                **s,
                "text": clean_skill_text(s["text"])
            } for s in seen.values()
        ]

        STOPWORDS = {"in", "on", "at", "of"}
        structured["Skills"] = [
            s for s in structured["Skills"]
            if s["text"].strip().lower() not in STOPWORDS and len(s["text"]) > 2
        ]

    if "Designation" in structured:
        structured["Designation"] = [d for d in structured["Designation"] if d["text"].strip().lower() not in {"engineer", "engineers", "developer", "developers"}]

    if "Years of experience" in structured:
        for e in structured["Years of experience"]:
            e["value"] = extract_year_value(e["text"])

    return dict(structured)

if __name__ == "__main__":
    with open("./examples/jd_sample.txt", "r", encoding="utf-8") as f:
        jd_text = f.read()

    result = extract_entities_from_jd(jd_text)

    with open("./examples/jd_entities.json", "w", encoding="utf-8") as out_file:
        json.dump(result, out_file, indent=2, ensure_ascii=False)

    print("JD entities with section classification saved.")
