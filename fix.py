import json

file_path = "evaluation/results/ground_truth_labeled.json"

with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Keep only the first 25 properly evaluated queries
clean_data = data[:25]

with open(file_path, "w", encoding="utf-8") as f:
    json.dump(clean_data, f, indent=2, ensure_ascii=False)

print("Corrupted entries removed! Ready to resume from 26.")