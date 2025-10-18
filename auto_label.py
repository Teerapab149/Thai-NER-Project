from pythainlp.tag import NER
import json
from pathlib import Path

ner = NER(engine="thainer")

base_dir = Path(__file__).resolve().parent
input_file = base_dir / "thai_ner_project" / "data" / "news_cleaned.jsonl"
output_file = base_dir / "thai_ner_project" / "data" / "labeled_news.jsonl"

# 🔸 รวม entity ที่ต่อเนื่องกัน (เช่น B-PERSON + I-PERSON → สมชาย)
def merge_entities(ner_result):
    merged = []
    current = {"entity": "", "label": ""}

    valid_labels = {"PERSON", "LOCATION", "ORGANIZATION", "DATE", "TIME", "MONEY", "PERCENT", "LAW", "LEN"}

    for word, tag in ner_result:
        # ตัดช่องว่างหรือ \n ออก
        if not word.strip():
            continue

        # ข้าม tag แปลก ๆ
        if len(tag) < 2 or tag[-1] == "-":
            continue

        # แปลงให้แน่ใจว่ามี B-/I-
        if tag.startswith("B-") or tag.startswith("I-"):
            label = tag.split("-")[-1]
        else:
            label = tag

        if label not in valid_labels:
            continue

        # รวม BIO ตามปกติ
        if tag.startswith("B-"):
            if current["entity"]:
                merged.append(current)
            current = {"entity": word, "label": label}

        elif tag.startswith("I-") and current["label"] == label:
            current["entity"] += word
        else:
            if current["entity"]:
                merged.append(current)
            current = {"entity": "", "label": ""}

    if current["entity"]:
        merged.append(current)

    return merged

with open(input_file, "r", encoding="utf-8") as f_in, open(output_file, "w", encoding="utf-8") as f_out:
    for line in f_in:
        record = json.loads(line)
        text = record["data"]["text"]

        # 🔹 ทำ NER แล้วรวม entity ต่อเนื่อง
        raw_entities = ner.tag(text)
        entities = merge_entities(raw_entities)

        record["ner"] = entities
        f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

print("✅ NER labeling เสร็จ -> thai_ner_project/data/labeled_news.jsonl (engine='thainer')")
