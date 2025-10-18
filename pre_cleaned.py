import json
import re
from pathlib import Path

base_dir = Path(__file__).resolve().parent
input_file = base_dir / "thai_ner_project" / "data" / "news.jsonl"
output_file = base_dir / "thai_ner_project" / "data" / "news_cleaned.jsonl"

# pattern ที่เจอบ่อยในข่าวไทยรัฐ
junk_patterns = [
    r"ข่าว[\s\S]*?ไทยรัฐทีวี",  # “ข่าว\nวิดีโอ\nหนังสือพิมพ์\nไทยรัฐทีวี”
    r"ไลฟ์สไตล์[\s\S]*?THAIRATH\s*\+",  # footer ต่อท้าย
    r"\bMONEY\b", r"\bMIRROR\b",  # หมวดหมู่
    r"อ่านข่าว.*", r"\.{3,}"  # จุด … หรือ “อ่านข่าวเพิ่ม”
]

def clean_text(text):
    # ลบ pattern พวก footer ออก
    for p in junk_patterns:
        text = re.sub(p, "", text, flags=re.IGNORECASE)
    # ลบช่องว่างซ้ำ
    text = re.sub(r"\s+", " ", text).strip()
    return text

with open(input_file, "r", encoding="utf-8") as f_in, open(output_file, "w", encoding="utf-8") as f_out:
    for line in f_in:
        record = json.loads(line)
        raw_text = record["data"]["text"]
        record["data"]["text"] = clean_text(raw_text)
        f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

print("✅ Pre-clean เสร็จ -> thai_ner_project/data/news_cleaned.jsonl")