import pandas as pd
import numpy as np
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import joblib
import os

# ── 1. Load data ──────────────────────────────────────────────
train_df = pd.read_csv('train.csv')
test_df  = pd.read_csv('test.csv')

# ── 2. Encode labels ──────────────────────────────────────────
le = LabelEncoder()
train_df['encoded'] = le.fit_transform(train_df['label'])
test_df['encoded']  = le.transform(test_df['label'])

# Save label encoder
joblib.dump(le, 'model/label_encoder.pkl')
print("Labels:", list(le.classes_))

# ── 3. Tokenizer ──────────────────────────────────────────────
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
tokenizer.save_pretrained('model/')

MAX_LEN   = 256
BATCH     = 16
EPOCHS    = 3

# ── 4. Dataset class ──────────────────────────────────────────
class MedDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts  = list(texts)
        self.labels = list(labels)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = tokenizer(
            str(self.texts[idx]),
            max_length=MAX_LEN,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids':      enc['input_ids'].squeeze(),
            'attention_mask': enc['attention_mask'].squeeze(),
            'label':          torch.tensor(self.labels[idx], dtype=torch.long)
        }

train_loader = DataLoader(MedDataset(train_df['clean_text'], train_df['encoded']), batch_size=BATCH, shuffle=True)
test_loader  = DataLoader(MedDataset(test_df['clean_text'],  test_df['encoded']),  batch_size=BATCH)

# ── 5. Model ──────────────────────────────────────────────────
NUM_CLASSES = len(le.classes_)
device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Training on:", device)

model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased', num_labels=NUM_CLASSES
)
model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5)

# ── 6. Training loop ──────────────────────────────────────────
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for i, batch in enumerate(train_loader):
        input_ids      = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels         = batch['label'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss    = outputs.loss
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 20 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Batch {i}/{len(train_loader)} | Loss: {loss.item():.4f}")

    print(f"\nEpoch {epoch+1} complete. Avg Loss: {total_loss/len(train_loader):.4f}\n")

# ── 7. Evaluation ─────────────────────────────────────────────
model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for batch in test_loader:
        input_ids      = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels         = batch['label'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds   = torch.argmax(outputs.logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print("\n--- Classification Report ---")
print(classification_report(all_labels, all_preds, target_names=le.classes_))

# ── 8. Save model ─────────────────────────────────────────────
model.save_pretrained('model/')
print("\nModel saved to model/ folder successfully!")