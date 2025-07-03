import json
from collections import Counter, defaultdict
import matplotlib.pyplot as plt

file_paths = [
    "../../data/cleaned/v2/cleaned_resumes_final.jsonl"
]

entity_types = ["Skills", "Years of experience", "Degree", "Designation"]

entity_counter = Counter()
entity_samples = defaultdict(list)

total_samples = 0
total_entities = 0

for file_path in file_paths:
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            total_samples += 1
            item = json.loads(line)
            for ann in item.get("annotation", []):
                labels = ann.get("label", [])
                text = item.get("content", "")
                span_text = text[ann["start"]:ann["end"]] if "start" in ann and "end" in ann else ""
                for label in labels:
                    if label in entity_types:
                        entity_counter[label] += 1
                        total_entities += 1
                        if len(entity_samples[label]) < 5:
                            entity_samples[label].append(span_text)

print(f"Total samples: {total_samples}")
print(f"Total entity mentions: {total_entities}\n")

print("Entity distribution (with percentage):")
for entity, count in entity_counter.items():
    percent = (count / total_entities) * 100 if total_entities > 0 else 0
    print(f"{entity}: {count} ({percent:.2f}%)")

print("\nSample spans per entity:")
for entity, samples in entity_samples.items():
    print(f"\n{entity}:")
    for i, sample in enumerate(samples, 1):
        print(f"  {i}. {sample}")

plt.figure(figsize=(10, 6))
bars = plt.bar(entity_counter.keys(), entity_counter.values(), color="skyblue")
plt.title("Distribution of Entity Types in Resume Annotations")
plt.xlabel("Entity Type")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.grid(axis='y')

for bar in bars:
    count = bar.get_height()
    percent = (count / total_entities) * 100 if total_entities > 0 else 0
    plt.text(bar.get_x() + bar.get_width() / 2, count + 5, f"{percent:.1f}%", ha='center')

plt.tight_layout()
plt.show()
