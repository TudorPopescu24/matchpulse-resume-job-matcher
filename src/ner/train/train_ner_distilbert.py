import wandb
wandb.login()

import json
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset as TorchDataset, random_split
from transformers import (
    DistilBertTokenizerFast, DistilBertForTokenClassification,
    DistilBertConfig, Trainer, TrainingArguments,
    EarlyStoppingCallback
)
from transformers import PreTrainedModel, DistilBertModel, DistilBertConfig
from seqeval.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report

import os
os.environ["TRANSFORMERS_NO_TF"] = "1"

labels = ["O", "B-Skills", "I-Skills", "B-Years of experience", "I-Years of experience",
          "B-Designation", "I-Designation", "B-Degree", "I-Degree"]
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for label, i in label2id.items()}

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, ignore_index=-100):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logits, target):
        ce_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            target.view(-1),
            reduction='none',
            weight=self.alpha,
            ignore_index=self.ignore_index
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

class DistilBertForTokenClassificationWithFocalLoss(PreTrainedModel):
    config_class = DistilBertConfig

    def __init__(self, config, alpha=None, gamma=2.0):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.alpha = alpha
        self.gamma = gamma

        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased", config=config)

        self.dropout = nn.Dropout(config.dropout if hasattr(config, "dropout") else 0.1)
        self.classifier = nn.Linear(config.dim, config.num_labels)
        self.criterion = FocalLoss(alpha=self.alpha, gamma=self.gamma)

        self.post_init()

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs.last_hidden_state)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss = self.criterion(logits, labels)

        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}

def load_and_clean_jsonl(paths: List[str]) -> List[dict]:
    data = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                try:
                    item = json.loads(line)
                    text = item.get("cleaned_resume", "") or item.get("content", "") or item.get("text", "")
                    annotations = item.get("annotation", []) or item.get("annotations", [])
                    entities = []
                    for ann in annotations:
                        label = ann.get("label", [None])[0] if "label" in ann else ann.get("entity")
                        for point in ann.get("points", []) if "points" in ann else [ann]:
                            start, end = point.get("start"), point.get("end")
                            if isinstance(start, int) and isinstance(end, int) and isinstance(label, str):
                                entities.append((start, end, label))
                    if text and entities:
                        data.append({"text": text, "entities": entities})
                    else:
                        print(f"[SKIP] {path} line {i}: missing text or entities")
                except Exception as e:
                    print(f"[WARN] {path} line {i} skipped: {e}")
    print(f"[DONE] Loaded {len(data)} valid entries from {len(paths)} files.")
    return data

def encode(example: dict, max_length=512) -> dict:
    text = example["text"]
    entities = example["entities"]
    encoding = tokenizer(text, truncation=True, padding="max_length", max_length=max_length, return_offsets_mapping=True)
    labels = ["O"] * len(encoding["offset_mapping"])
    for start, end, label in entities:
        for idx, (token_start, token_end) in enumerate(encoding["offset_mapping"]):
            if token_end <= start:
                continue
            if token_start >= end:
                break
            if token_start >= start and token_end <= end:
                tag = f"B-{label}" if token_start == start else f"I-{label}"
                if tag in label2id:
                    labels[idx] = tag
    encoding.pop("offset_mapping")
    encoding["labels"] = [label2id[l] for l in labels]
    return {k: torch.tensor(v) for k, v in encoding.items()}

class ResumeNERDataset(TorchDataset):
    def __init__(self, data: List[dict]):
        self.samples = [encode(sample) for sample in data]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def compute_metrics(p):
    preds, labels = p
    preds = preds.argmax(-1)
    true_labels = [
        [id2label[l] for l in label if l != -100]
        for label in labels
    ]
    true_preds = [
        [id2label[p] for (p, l) in zip(pred, label) if l != -100]
        for pred, label in zip(preds, labels)
    ]

    report = classification_report(true_labels, true_preds, output_dict=True)

    print("\n📋 Classification report:")
    print(classification_report(true_labels, true_preds))

    metrics = {
        "precision": report["micro avg"]["precision"],
        "recall": report["micro avg"]["recall"],
        "f1": report["micro avg"]["f1-score"],
        "accuracy": accuracy_score(true_labels, true_preds),
    }

    for entity_label in report:
        if entity_label in ["micro avg", "macro avg", "weighted avg"]:
            continue
        metrics[f"{entity_label}_f1"] = report[entity_label]["f1-score"]

    return metrics

print("Loading data...")
data_paths = [
    "/cleaned_resumes_filtered_degrees_designations.jsonl",
    "/synthetic_split_1.jsonl",
    "/synthetic_split_2.jsonl",
    # "/synthetic_split_3.jsonl"
]
data = load_and_clean_jsonl(data_paths)
dataset = ResumeNERDataset(data)
train_size = int(0.8 * len(dataset))
eval_size = len(dataset) - train_size
train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])
print(f"📊 Training: {len(train_dataset)} samples | Eval: {len(eval_dataset)}")

config = DistilBertConfig.from_pretrained("distilbert-base-uncased")
config.num_labels = len(labels)
config.id2label = id2label
config.label2id = label2id

model = DistilBertForTokenClassificationWithFocalLoss(config=config, gamma=2.0)


args = TrainingArguments(
    output_dir="./ner_distilbert",
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    logging_dir="./logs_distilbert",
    report_to="wandb",
    run_name="ner-distilbert"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    compute_metrics=compute_metrics,
)

print("Starting training...")
trainer.train()

print("Saving model...")
model.save_pretrained("./ner_distilbert")
tokenizer.save_pretrained("./ner_distilbert")
print("Training complete.")