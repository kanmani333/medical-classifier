import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib
import os

# Load data
df = pd.read_csv('mtsamples.csv')

# Keep ONLY 3 most distinct categories
categories = ['Radiology', 'Surgery', 'Neurology']
df = df[df['medical_specialty'].str.strip().isin(categories)]
df['label'] = df['medical_specialty'].str.strip()

# Use all text combined
df['text'] = (
    df['sample_name'].fillna('') + ' ' +
    df['description'].fillna('') + ' ' +
    df['transcription'].fillna('')
)

df = df[['text', 'label']]
df = df[df['text'].str.strip().str.len() > 50]

print("Dataset size:", len(df))
print("\nCategories:\n", df['label'].value_counts())

# Split
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df['label']
)
print("\nTrain:", len(train_df), "| Test:", len(test_df))

# Pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=100000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
        stop_words='english'
    )),
    ('svm', LinearSVC(
        class_weight='balanced',
        max_iter=5000,
        C=1.0
    ))
])

# Train
print("\nTraining model...")
pipeline.fit(train_df['text'], train_df['label'])
print("Training complete!")

# Evaluate
preds = pipeline.predict(test_df['text'])
print("\n--- Classification Report ---")
print(classification_report(test_df['label'], preds, zero_division=0))

correct = sum(p == t for p, t in zip(preds, test_df['label']))
print(f"Overall Accuracy: {correct}/{len(test_df)} = {correct/len(test_df)*100:.1f}%")

# Save
os.makedirs('model', exist_ok=True)
joblib.dump(pipeline, 'model/svm_pipeline.pkl')
joblib.dump(categories, 'model/labels.pkl')
print("\nModel saved successfully!")

# Quick test
print("\n--- Quick Test ---")
samples = [
    "MRI scan of the brain shows abnormal signal intensity in the frontal lobe",
    "Laparoscopic appendectomy performed, incision closed with sutures",
    "Patient presents with seizures, EEG shows abnormal brain wave activity",
    "X-ray chest shows bilateral infiltrates consistent with pneumonia",
    "Craniotomy performed for tumor resection under general anesthesia",
    "CT scan abdomen and pelvis with contrast shows no acute findings"
]
for s in samples:
    pred = pipeline.predict([s])[0]
    print(f"  Input: {s[:65]}")
    print(f"  Predicted: {pred}\n")