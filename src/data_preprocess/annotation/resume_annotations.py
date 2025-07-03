import re
import json
import pandas as pd
import numpy as np
from typing import List, Dict
from snorkel.labeling import PandasLFApplier
from snorkel.labeling.model import LabelModel
from pydantic import BaseModel
import openai

from weak_supervision_lfs import (
    lf_email, lf_graduation_year,
    lf_skills, lf_job_titles, lf_years_of_experience, lf_designation,
    lf_location, lf_name_like, LABELS, ABSTAIN, SKILLS
)

class EntityAnnotation(BaseModel):
    entity: str
    text: str
    start: int
    end: int

def extract_skill_spans(text: str, skills: List[str]) -> List[Dict]:
    found = []
    seen_spans = set()
    lowered_text = text.lower()
    for skill in sorted(skills, key=len, reverse=True):
        pattern = re.escape(skill.lower())
        for match in re.finditer(rf'\b{pattern}\b', lowered_text):
            start = match.start()
            end = match.end()
            if (start, end) not in seen_spans:
                seen_spans.add((start, end))
                found.append({"text": text[start:end], "start": start, "end": end})
    return found

class WeakSupervisionEngine:
    def __init__(self, entity_types: List[str]):
        self.entity_types = entity_types
        self.lfs = self._create_labeling_functions()

    def _create_labeling_functions(self):
        return [
            lf_email, lf_graduation_year, lf_company_names, lf_university_names,
            lf_skills, lf_job_titles, lf_years_of_experience, lf_designation,
            lf_location, lf_name_like
        ]

    def generate_labels(self, data: pd.DataFrame) -> np.ndarray:
        applier = PandasLFApplier(lfs=self.lfs)
        L_train = applier.apply(df=data)
        label_model = LabelModel(cardinality=len(self.entity_types), verbose=True)
        label_model.fit(L_train=L_train, n_epochs=500, seed=42)
        return label_model.predict_proba(L=L_train)

    def extract_weak_annotations(self, data: pd.DataFrame) -> List[List[EntityAnnotation]]:
        annotations = []
        for text in data['text']:
            local_anns = []
            for match in re.finditer(r"(\d+\s+years?|Less than 1 year)", text, re.IGNORECASE):
                local_anns.append(EntityAnnotation(entity="Years of experience", text=match.group(), start=match.start(), end=match.end()))
            for skill in extract_skill_spans(text, list(SKILLS)):
                local_anns.append(EntityAnnotation(entity="Skills", text=skill['text'], start=skill['start'], end=skill['end']))
            annotations.append(local_anns)
        return annotations

class LLMAugmentor:
    def __init__(self, api_key: str, entity_types: List[str]):
        self.client = openai.OpenAI(api_key=api_key)
        self.entity_types = entity_types

    def _create_prompt(self, text: str) -> str:
        return f"""
        Extract entities from this resume text. Entity types: {', '.join(self.entity_types)}.
        Return JSON array with entities in format: {{"entity": "type", "text": "string", "start": int, "end": int}}.

        Resume:
        {text}
        """

    def get_annotations(self, text: str, max_retries: int = 3) -> List[EntityAnnotation]:
        for attempt in range(max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo-0125",
                    messages=[{"role": "user", "content": self._create_prompt(text)}]
                )
                content = response.choices[0].message.content.strip()
                parsed = try_repair_json_array(content)
                return [EntityAnnotation(**e) for e in parsed]
            except json.JSONDecodeError as jde:
                print(f"JSON decoding error on attempt {attempt + 1}: {jde}")
                if attempt == max_retries:
                    return []
            except Exception as e:
                print(f"LLM API error on attempt {attempt + 1}: {e}")
                if attempt == max_retries:
                    return []

class ConsensusModel:
    def __init__(self, entity_types: List[str], weights: Dict[str, float]):
        self.entity_types = entity_types
        self.weights = weights

    def _calculate_confidence(self, weak_probs: np.ndarray, llm_annotations: List[EntityAnnotation]) -> Dict[str, float]:
        confidences = {ent: 0.0 for ent in self.entity_types}
        for i, ent in enumerate(self.entity_types):
            confidences[ent] += weak_probs[i] * self.weights.get('weak', 0.4)
        for ann in llm_annotations:
            if ann.entity in confidences:
                confidences[ann.entity] += self.weights.get('llm', 0.6)
        return confidences

    def resolve(self, text: str, weak_probs: np.ndarray, llm_annotations: List[EntityAnnotation]) -> List[EntityAnnotation]:
        final_annotations = []
        confidence = self._calculate_confidence(weak_probs, llm_annotations)
        for ann in llm_annotations:
            if confidence.get(ann.entity, 0) >= 0.7:
                final_annotations.append(ann)
        if not any(a.entity == "Years of experience" for a in llm_annotations):
            for match in re.finditer(r"\b(?:Less than 1 year|\d{1,2} years?)\b", text):
                final_annotations.append(EntityAnnotation(
                    entity="Years of experience", text=match.group(), start=match.start(), end=match.end()
                ))
        return final_annotations

class ResumeAnnotationPipeline:
    def __init__(self, config: dict):
        self.entity_types = config['entity_types']
        self.weak_engine = WeakSupervisionEngine(self.entity_types)
        self.llm_augmentor = LLMAugmentor(config['api_key'], self.entity_types)
        self.consensus = ConsensusModel(self.entity_types, config['weights'])

    def process(self, resumes: List[str]) -> List[Dict]:
        df = pd.DataFrame({"text": resumes})
        weak_probs = self.weak_engine.generate_labels(df)
        weak_anns = self.weak_engine.extract_weak_annotations(df)
        output = []

        for i, text in enumerate(resumes):
            llm_ann = self.llm_augmentor.get_annotations(text)
            final_ann = self.consensus.resolve(text, weak_probs[i], llm_ann)

            seen = {f"{a.entity}:{a.start}:{a.end}" for a in final_ann}
            for ann in weak_anns[i]:
                key = f"{ann.entity}:{ann.start}:{ann.end}"
                if key not in seen:
                    final_ann.append(ann)

            converted = {
                "content": text,
                "annotation": [
                    {
                        "label": [ann.entity],
                        "points": [{"start": ann.start, "end": ann.end, "text": ann.text}]
                    } for ann in final_ann
                ],
                "extras": None
            }
            output.append(converted)
        return output

def try_repair_json_array(raw: str):
    match = re.search(r"\[(.*)", raw, re.DOTALL)
    if not match:
        raise ValueError("No JSON array found")
    fixed_content = match.group(0).strip()
    if not fixed_content.endswith("]"):
        if fixed_content.endswith(","):
            fixed_content = fixed_content[:-1]
        fixed_content += "]"
    try:
        return json.loads(fixed_content)
    except json.JSONDecodeError:
        fixed_content = re.sub(r',\s*\{[^\}]*$', '', fixed_content) + "]"
        return json.loads(fixed_content)

if __name__ == "__main__":
    config = {
        "entity_types": [
            "Name", "Email Address", "Companies worked at", "Past roles", "Skills",
            "Years of experience", "Degree", "University", "Designation", "Location"
        ],
        "weights": {"weak": 0.4, "llm": 0.6},
        "api_key": ""
    }

    df = pd.read_csv("../../../data/processed/it_resumes_preprocessed.csv")
    resumes = df["cleaned_resume"].dropna().tolist()

    pipeline = ResumeAnnotationPipeline(config)
    annotations = pipeline.process(resumes)

    with open("augmented_resumes.jsonl", "w", encoding="utf-8") as f:
        for item in annotations:
            json.dump(item, f)
            f.write("\n")

    print(f"Parsed {len(resumes)} resumes and wrote to augmented_resumes.jsonl")
