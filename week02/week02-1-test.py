import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "hfl/chinese-roberta-wwm-ext"
MODEL_SAVE_PATH = "bert_company_classifier.pt"
MAX_LEN = 128
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# 读取 testing.csv
test_df = pd.read_csv("testing.csv")

# 第1列：序号，第2列：文本
ids = test_df.iloc[:, 0].tolist()
texts = test_df.iloc[:, 1].astype(str).tolist()

# 加载模型与 tokenizer
checkpoint = torch.load(MODEL_SAVE_PATH, map_location=DEVICE)
label2id = checkpoint["label2id"]
id2label = checkpoint["id2label"]
num_labels = len(label2id)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id
)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(DEVICE)
model.eval()

class TestDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len):
        self.texts = texts
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
        return {k: v.squeeze(0) for k, v in encoding.items()}

dataset = TestDataset(texts, tokenizer, MAX_LEN)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

all_preds = []
with torch.no_grad():
    for batch in loader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        outputs = model(**batch)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)
        all_preds.extend(preds.cpu().numpy())

# 把内部id映射回原始标签（1～11等）
pred_labels = [id2label[int(p)] for p in all_preds]

result_df = pd.DataFrame({
    "id": ids,
    "text": texts,
    "predicted_type": pred_labels
})

result_df.to_csv("pred_result_bert.csv", index=False, encoding="utf-8-sig")
print("预测完成，结果已保存为 pred_result_bert.csv")
