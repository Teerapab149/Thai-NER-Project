import json

input_file = "thai_ner_project/data/labeled_news.jsonl"
output_file = "thai_ner_project/data/precheck_news.json"

data = []
with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        data.append(json.loads(line))

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"✅ แปลงไฟล์เรียบร้อย -> {output_file}")
