from pathlib import Path
import random

# ---------- CONFIG ----------
input_file = "thai_ner_project/data/ner_dataset_iob.txt"
train_file = "thai_ner_project/data/train.txt"
test_file = "thai_ner_project/data/test.txt"

split_ratio = 0.8  # 80% train / 20% test
random.seed(42)

# ---------- READ ----------
with open(input_file, "r", encoding="utf-8") as f:
    content = f.read().strip()

# แยกข่าวแต่ละชุดด้วยบรรทัดว่าง
samples = [s.strip() for s in content.split("\n\n") if s.strip()]

# สุ่มลำดับข่าว
random.shuffle(samples)

# ---------- SPLIT ----------
split_index = int(len(samples) * split_ratio)
train_samples = samples[:split_index]
test_samples = samples[split_index:]

# ---------- SAVE ----------
with open(train_file, "w", encoding="utf-8") as f:
    f.write("\n\n".join(train_samples))
with open(test_file, "w", encoding="utf-8") as f:
    f.write("\n\n".join(test_samples))

print(f"✅ แบ่งข้อมูลสำเร็จ: {len(train_samples)} train | {len(test_samples)} test")
print(f"📂 train -> {train_file}")
print(f"📂 test  -> {test_file}")
