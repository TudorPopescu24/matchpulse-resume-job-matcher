import re
from snorkel.labeling import labeling_function
import pandas as pd

def safe_read_csv(path, expected_columns):
    encodings_to_try = ['utf-8', 'cp1252', 'ISO-8859-1']

    for enc in encodings_to_try:
        try:
            df = pd.read_csv(
                path,
                encoding=enc,
                encoding_errors='ignore',
                on_bad_lines='skip'
            )
            df = df[[col for col in expected_columns if col in df.columns]]
            return df.dropna()
        except Exception as e:
            print(f"Failed to read {path} with encoding {enc}: {e}")

    print(f"Could not read {path} with tried encodings.")
    return pd.DataFrame(columns=expected_columns)


skills_df = safe_read_csv("../../../data/annotation/skills_dataset.csv", ["name:String"])
SKILLS = set(skills_df["name:String"].str.lower().unique())

jobtitles_df = safe_read_csv("../../../data/annotation/it_jobtitles_dataset.csv", ["Job Title"])
JOB_TITLES = set(jobtitles_df["Job Title"].str.lower().unique())

DEGREES = {
    "b.sc", "bachelor", "bachelors", "btech", "b.e", "m.sc", "mtech", "m.e",
    "mba", "phd", "ph.d", "master", "masters", "bsdc", "msdc"
}

LABELS = {
    "Name": 0,
    "Email Address": 1,
    "Companies worked at": 2,
    "Past roles": 3,
    "Skills": 4,
    "Years of experience": 5,
    "Degree": 6,
    "University": 7,
    "Designation": 8,
    "Location": 9
}

ABSTAIN = -1

def keyword_fuzzy_match(keywords, text):
    return any(kw in text.lower() for kw in keywords)

def keyword_exact_match(keywords, text):
    pattern = r"\b(" + "|".join(re.escape(kw) for kw in keywords) + r")\b"
    return bool(re.search(pattern, text.lower()))

@labeling_function()
def lf_email(x):
    if re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", x.text):
        return LABELS["Email Address"]
    return ABSTAIN

@labeling_function()
def lf_years_of_experience(x):
    if re.search(r"\b(?:less than\s*)?\d{1,2}(?:\+)?\s*(?:years?|yrs?)\b", x.text.lower()):
        return LABELS["Years of experience"]
    return ABSTAIN

@labeling_function()
def lf_graduation_year(x):
    if re.search(r"\b(19|20)\d{2}\b", x.text):
        return LABELS["Degree"]
    return ABSTAIN

@labeling_function()
def lf_skills(x):
    if keyword_exact_match(SKILLS, x.text):
        return LABELS["Skills"]
    return ABSTAIN

@labeling_function()
def lf_job_titles(x):
    if keyword_fuzzy_match(JOB_TITLES, x.text):
        return LABELS["Past roles"]
    return ABSTAIN

@labeling_function()
def lf_designation(x):
    if keyword_fuzzy_match(JOB_TITLES, x.text):
        return LABELS["Designation"]
    return ABSTAIN

@labeling_function()
def lf_location(x):
    if re.search(r"\b(Bangalore|Bengaluru|Hyderabad|Mumbai|Pune|Chennai|Delhi|Noida|Kolkata)\b", x.text, re.IGNORECASE):
        return LABELS["Location"]
    return ABSTAIN

@labeling_function()
def lf_degree_names(x):
    if keyword_fuzzy_match(DEGREES, x.text):
        return LABELS["Degree"]
    return ABSTAIN

@labeling_function()
def lf_name_like(x):
    if re.match(r"^[A-Z][a-z]+ [A-Z][a-z]+$", x.text.strip()):
        return LABELS["Name"]
    return ABSTAIN