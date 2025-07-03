import pandas as pd
import os

from src.data_preprocess.text_preprocessor import TextPreprocessor


def preprocess_resumes(input_path, output_path):
    if input_path.endswith(".jsonl"):
        df = pd.read_json(input_path, lines=True)
    else:
        df = pd.read_csv(input_path)

    preprocessor = TextPreprocessor()

    text_col = "Resume" if "Resume" in df.columns else "content"
    df["cleaned_resume"] = df[text_col].apply(preprocessor.preprocess)

    df.drop(columns=[text_col], inplace=True)

    df = df[df["cleaned_resume"].notnull()]
    df = df[df["cleaned_resume"].str.strip().astype(bool)]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if output_path.endswith(".jsonl"):
        df.to_json(output_path, orient="records", lines=True, force_ascii=False)
    else:
        df.to_csv(output_path, index=False)

    print(f"Resumes saved to {output_path}")


if __name__ == "__main__":
    preprocess_resumes(
        "../../../data/processed/v2/preprocessed_resumes_final.jsonl",
        "../../../data/cleaned/v2/cleaned_resumes_final.jsonl"
    )