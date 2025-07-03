from openai import OpenAI
import csv
import time
import random
import re
import json
from typing import Literal, Set, Tuple

api_key = ""
client = OpenAI(api_key=api_key)

ROLES = [
    "Frontend Developer", "Backend Developer", "DevOps Engineer", "Data Scientist",
    "Cybersecurity Specialist", "Mobile App Developer", "AI/ML Engineer", "QA Engineer",
    "Cloud Architect", "Database Administrator", "Full Stack Developer", "Software Engineer",
    "Application Developer", "Embedded Software Engineer", "Game Developer",
    "Desktop Application Developer", "Systems Software Engineer", "API Developer",
    "SDK Developer", "Firmware Engineer", "Web Developer", "Web3 Developer",
    "Blockchain Developer", "ERP Developer", "CRM Developer", "CMS Developer",
    "Data Analyst", "Machine Learning Engineer", "NLP Engineer", "Deep Learning Engineer",
    "Data Engineer", "Business Intelligence Developer", "Big Data Engineer",
    "Research Scientist (AI)", "Computer Vision Engineer", "AI Research Engineer",
    "Data Architect", "Data Quality Analyst", "MLOps Engineer", "Annotation Specialist",
    "Site Reliability Engineer (SRE)", "Cloud Engineer", "Platform Engineer",
    "Infrastructure Engineer", "Build and Release Engineer", "Automation Engineer",
    "System Administrator", "Network Engineer", "CI/CD Engineer", "Reliability Engineer",
    "Linux Systems Engineer", "Security Engineer", "Information Security Analyst",
    "Application Security Engineer", "Cloud Security Engineer", "SOC Analyst",
    "Ethical Hacker", "Penetration Tester", "Security Architect", "Threat Intelligence Analyst",
    "QA Automation Engineer", "Manual QA Tester", "Test Engineer", "Performance Tester",
    "Security Tester", "Software Test Architect", "Quality Analyst", "UAT Coordinator",
    "iOS Developer", "Android Developer", "Flutter Developer", "React Native Developer",
    "UX Designer", "UI Designer", "Interaction Designer", "Mobile UX Designer",
    "Visual Designer", "Product Designer", "Product Manager", "Technical Project Manager",
    "Scrum Master", "Agile Coach", "Program Manager", "Release Manager", "Delivery Manager",
    "Engineering Manager", "Technical Lead", "CTO", "VP of Engineering",
    "Solutions Architect", "Technical Architect", "Cloud Solutions Architect",
    "Enterprise Architect", "Kubernetes Engineer", "AWS Engineer", "Azure Engineer",
    "GCP Engineer", "SQL Developer", "NoSQL Developer", "ETL Developer",
    "Data Warehouse Engineer", "Database Developer", "Storage Engineer", "Technical Writer",
    "Developer Advocate", "Developer Evangelist", "Support Engineer", "Integration Engineer",
    "Robotics Engineer", "Hardware Engineer", "AR/VR Developer", "Simulation Engineer",
    "Game Designer", "GIS Developer", "Bioinformatics Engineer"
]


STYLES = [
    "formal corporate style",
    "friendly startup tone",
    "bullet-point concise format",
    "first-person voice",
    "third-person recruiter summary",
    "casual with slight humor",
    "very technical focus with jargon",
    "marketing-style pitch"
]

SYSTEM_PROMPT = """
You are a data generation assistant helping build a dataset for evaluating NLP similarity engines.
Each example contains:
- A realistic IT job description
- A matching or mismatching resume
Both should include technical skills, years of experience, degree, and designation (JD) or past roles (resume).
"""

BASE_MATCH_PROMPT = """
Generate a pair of job description and resume that MATCH well, using diverse wording and natural tone.

Each must:
- Be ≥800 characters (ideally 1000)
- Include relevant technical skills, years of experience, degree
- Use a different writing style each time (first/third person, tone, etc.)
- Be formatted as JSON: {
  "job_description": "...",
  "resume": "..."
}
"""

BASE_MISMATCH_PROMPT = """
Generate a pair of job description and resume that DO NOT MATCH. Make the mismatch clear through different roles, skills, or experience.

Same formatting rules as above apply. Output in JSON:
{
  "job_description": "...",
  "resume": "..."
}
"""

def safe_parse_json(text: str) -> dict:
    try:
        text = re.sub(r"```(?:json)?\s*|```", "", text)
        text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
        if text.strip().startswith("{'") or text.strip().startswith("{ '"):
            text = text.replace("'", '"')

        return json.loads(text)
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        print(f"Raw content: {text[:300]}...")
        raise

def generate_pair(label: Literal[0, 1], seen: Set[Tuple[str, str]]) -> dict:
    prompt_base = BASE_MATCH_PROMPT if label == 1 else BASE_MISMATCH_PROMPT
    role = random.choice(ROLES)
    style = random.choice(STYLES)

    prompt = (
        f"{prompt_base.strip()}\n\n"
        f"Role: {role}\n"
        f"Style: {style}\n"
    )

    for _ in range(3):
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT.strip()},
                    {"role": "user", "content": prompt.strip()}
                ],
                temperature=0.9
            )
            content = response.choices[0].message.content.strip()

            json_data = safe_parse_json(content)
            if not isinstance(json_data, dict):
                print("Skipped (not a dict)")
                continue

            jd_raw = json_data.get("job_description", "")
            resume_raw = json_data.get("resume", "")

            jd = str(jd_raw).strip()
            resume = str(resume_raw).strip()

            key = (str(jd[:250]), str(resume[:250]))

            if not isinstance(key, tuple):
                print(f"key is not a tuple: {type(key)}")
            if not all(isinstance(k, str) for k in key):
                print(f"key contains non-string elements: {[type(k) for k in key]}")

            try:
                seen.add(key)
            except Exception as e:
                print(f"Set insert failed: {e}")
                print(f"key = {key}")
                print(f"key types = {[type(k) for k in key]}")
                raise e

            return {
                "job_description": jd,
                "resume": resume,
                "label": label
            }

        except Exception as e:
            print(f"Error: {e} — Retrying in 5s...")
            time.sleep(5)

    raise RuntimeError("Failed to generate valid example after multiple retries.")

def generate_dataset(num_matches=200, num_mismatches=200, output_csv="jd_resume_dataset_v2.csv"):
    dataset = []
    seen_pairs = set()

    print("Generating MATCH pairs...")
    for i in range(num_matches):
        print(f"\nGenerating MATCH pair {i + 1}/{num_matches}")
        pair = generate_pair(1, seen_pairs)
        print(f"JD preview: {pair['job_description'][:120]}...")
        print(f"Resume preview: {pair['resume'][:120]}...")
        dataset.append(pair)
        time.sleep(1.5)

    print("\nGenerating MISMATCH pairs...")
    for i in range(num_mismatches):
        print(f"\nGenerating MISMATCH pair {i + 1}/{num_mismatches}")
        pair = generate_pair(0, seen_pairs)
        print(f"JD preview: {pair['job_description'][:120]}...")
        print(f"Resume preview: {pair['resume'][:120]}...")
        dataset.append(pair)
        time.sleep(1.5)

    print(f"\nWriting {len(dataset)} records to {output_csv}...")
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["job_description", "resume", "label"])
        writer.writeheader()
        writer.writerows(dataset)

    print(f"Done. Dataset saved to: {output_csv}")

if __name__ == "__main__":
    generate_dataset()
