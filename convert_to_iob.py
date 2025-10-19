import json
from pathlib import Path
from pythainlp.tokenize import word_tokenize

input_file = "thai_ner_project/data/precheck_news.json"
output_file = "thai_ner_project/data/ner_dataset_iob.txt"

with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

with open(output_file, "w", encoding="utf-8") as f_out:
    for item in data:
        text = item["data"]["text"].strip()
        tokens = [t for t in word_tokenize(text, engine="newmm") if t.strip()]  # ❗ ตัดช่องว่างทิ้ง

        entities = item.get("ner", [])
        labels = ["O"] * len(tokens)

        for ent in entities:
            ent_text = ent["entity"].strip()
            ent_label = ent["label"].strip()
            ent_tokens = [t for t in word_tokenize(ent_text, engine="newmm") if t.strip()]

            for i in range(len(tokens)):
                if tokens[i:i+len(ent_tokens)] == ent_tokens:
                    labels[i] = f"B-{ent_label}"
                    for j in range(1, len(ent_tokens)):
                        labels[i+j] = f"I-{ent_label}"
                    break

        # ✅ เขียนเฉพาะบรรทัดที่มีคำจริง
        for token, label in zip(tokens, labels):
            if token.strip():
                f_out.write(f"{token}\t{label}\n")
        f_out.write("\n")  # แยกแต่ละข่าวด้วยบรรทัดว่างเดียว

print(f"✅ cleaned IOB file saved to: {output_file}")
