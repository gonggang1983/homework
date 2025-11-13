import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)

# ================== 配置 =====================
MODEL_NAME = "hfl/chinese-roberta-wwm-ext"  # 中文RoBERTa
MAX_LEN = 128
BATCH_SIZE = 8
EPOCHS = 3
LR = 2e-5
MODEL_SAVE_PATH = "bert_company_classifier.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# ================== 读取数据 =====================
train_df = pd.read_csv("training.csv")

# 第1列是标签，第2列是文本
texts = train_df.iloc[:, 1].astype(str).tolist()
labels_raw = train_df.iloc[:, 0].tolist()

# 把原始标签（如 1-11）映射到 0..num_labels-1
unique_labels = sorted(list(set(labels_raw)))
label2id = {label: i for i, label in enumerate(unique_labels)}
id2label = {i: label for label, i in label2id.items()}
num_labels = len(unique_labels)

print("标签映射 label2id:", label2id)

labels = [label2id[x] for x in labels_raw]

# 训练 / 验证划分
X_train, X_valid, y_train, y_valid = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

# ================== Dataset 定义 =====================
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

train_dataset = TextDataset(X_train, y_train, tokenizer, MAX_LEN)
valid_dataset = TextDataset(X_valid, y_valid, tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ================== 模型定义 =====================
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id
)
model.to(DEVICE)

# 类别不平衡：使用 class weights
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(num_labels),
    y=np.array(y_train)
)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
loss_fn = nn.CrossEntropyLoss(weight=class_weights)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)

# ================== 训练 & 验证函数 =====================
def train_one_epoch(epoch):
    model.train()
    total_loss = 0.0
    for batch in train_loader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        labels = batch.pop("labels")

        outputs = model(**batch)
        logits = outputs.logits
        loss = loss_fn(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{EPOCHS} - Train loss: {avg_loss:.4f}")


def evaluate():
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0

    with torch.no_grad():
        for batch in valid_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            labels = batch.pop("labels")

            outputs = model(**batch)
            logits = outputs.logits
            loss = loss_fn(logits, labels)

            total_loss += loss.item()

            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(valid_loader)
    acc = accuracy_score(all_labels, all_preds)

    print(f"Valid loss: {avg_loss:.4f}, accuracy: {acc:.4f}")
    print("分类报告（验证集）：")
    print(classification_report(
        all_labels,
        all_preds,
        digits=3,
        zero_division=0,
        target_names=[str(id2label[i]) for i in range(num_labels)]
    ))

    return acc

# ================== 训练主循环 =====================
best_acc = 0.0
for epoch in range(EPOCHS):
    train_one_epoch(epoch)
    acc = evaluate()

    # 保存 best model
    if acc > best_acc:
        best_acc = acc
        torch.save({
            "model_state_dict": model.state_dict(),
            "label2id": label2id,
            "id2label": id2label
        }, MODEL_SAVE_PATH)
        print(f"✨ 新的最佳模型已保存，验证集准确率: {best_acc:.4f}")

print("训练完成！最佳验证集准确率：", best_acc)
print("模型文件保存为：", MODEL_SAVE_PATH)
