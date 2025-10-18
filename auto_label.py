from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import json
from pathlib import Path

# โหลดโมเดล WangchanBERTa สำหรับ NER
model_name = "wannaphong/wangchanberta-base-att-spm-ner"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)
ner = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple", truncation=True)

base_dir = Path(__file__).resolve().parent
input_file = base_dir / "thai_ner_project" / "data" / "news.jsonl"
output_file = base_dir / "thai_ner_project" / "data" / "labeled_news.jsonl"

with open(input_file, "r", encoding="utf-8") as f_in, open(output_file, "w", encoding="utf-8") as f_out:
    for line in f_in:
        record = json.loads(line)
        text = record["data"]["text"]

        try:
            labels = ner(text)
        except Exception as e:
            print(f"[skip] {e}")
            labels = []

        record["ner"] = labels
        f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

print("✅ ทำ NER labeling เสร็จ -> thai_ner_project/data/labeled_news.jsonl")
