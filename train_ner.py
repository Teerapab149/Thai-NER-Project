# train_ner.py
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
)
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
import numpy as np
import os

# ---------- CONFIG ----------
model_checkpoint = "airesearch/wangchanberta-base-att-spm-uncased" # โมเดลภาษาไทย (SentencePiece)
train_file = "thai_ner_project/data/train.txt"
test_file  = "thai_ner_project/data/test.txt"
output_dir = "./ner_model"


# ---------- LOAD IOB AS SENTENCES ----------
def read_iob2(path):
    """
    อ่านไฟล์ IOB2 แบบ:
        token LABEL
        token LABEL
        <บรรทัดว่าง> = จบประโยค
    คืนค่าเป็น list[{"tokens":[...], "ner_tags":[...]}]
    """
    sents = []
    tokens, labels = [], []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                # จบประโยค
                if tokens and labels and len(tokens) == len(labels):
                    sents.append({"tokens": tokens, "ner_tags": labels})
                tokens, labels = [], []
                continue

            # ข้ามบรรทัดแปลก ๆ
            parts = line.split()
            if len(parts) < 2:
                continue
            tok = parts[0]
            lab = parts[-1]  # เผื่อมีช่องว่างส่วนกลาง แต่อันสุดท้ายคือ label
            tokens.append(tok)
            labels.append(lab)

    # ประโยคสุดท้าย (ถ้าไฟล์ไม่ได้ลงท้ายบรรทัดว่าง)
    if tokens and labels and len(tokens) == len(labels):
        sents.append({"tokens": tokens, "ner_tags": labels})
    return sents


train_sents = read_iob2(train_file)
test_sents  = read_iob2(test_file)

# กันเคสว่าง/ไม่สมบูรณ์ (เผื่อไว้)
train_sents = [ex for ex in train_sents if ex["tokens"] and ex["ner_tags"] and len(ex["tokens"]) == len(ex["ner_tags"])]
test_sents  = [ex for ex in test_sents  if ex["tokens"] and ex["ner_tags"] and len(ex["tokens"]) == len(ex["ner_tags"])]

dataset = DatasetDict({
    "train": Dataset.from_list(train_sents),
    "test":  Dataset.from_list(test_sents),
})

# ---------- LABEL SPACE ----------
unique_tags = sorted({t for seq in dataset["train"]["ner_tags"] for t in seq})
tag2id = {tag: i for i, tag in enumerate(unique_tags)}
id2tag = {i: tag for tag, i in tag2id.items()}

# ---------- TOKENIZER ----------
# ใช้ slow tokenizer เพราะโมเดลนี้เป็น SentencePiece
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        padding="max_length",
        max_length=256,
    )

    # map label ตรง ๆ แบบ simple (ไม่พึ่ง word_ids)
    labels = []
    for seq_labels in examples["ner_tags"]:
        label_ids = [tag2id[tag] for tag in seq_labels][:256]
        label_ids += [-100] * (256 - len(label_ids))
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


encoded = dataset.map(tokenize_and_align_labels, batched=True)

# ---------- MODEL ----------
model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    num_labels=len(unique_tags),
    id2label=id2tag,
    label2id=tag2id,
)

# ---------- TRAIN ----------
args = TrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    report_to="none",  # ปิด wandb/tensorboard อัตโนมัติ
)

data_collator = DataCollatorForTokenClassification(tokenizer)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=encoded["train"],
    eval_dataset=encoded["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

trainer.train()

# ---------- EVALUATE ----------
preds_logits, labels, _ = trainer.predict(encoded["test"])
pred_ids = np.argmax(preds_logits, axis=2)

true_tags, pred_tags = [], []
for i in range(len(labels)):
    seq_true, seq_pred = [], []
    for j, lab_id in enumerate(labels[i]):
        if lab_id != -100:
            seq_true.append(id2tag[int(lab_id)])
            seq_pred.append(id2tag[int(pred_ids[i][j])])
    if seq_true:  # กันเคสว่าง
        true_tags.append(seq_true)
        pred_tags.append(seq_pred)

print("📊 Classification report:")
print(classification_report(true_tags, pred_tags))
print(f"F1-score: {f1_score(true_tags, pred_tags):.4f}")
