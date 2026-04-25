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

# Keep top 8 categories
top_categories = df['medical_specialty'].value_counts().head(8).index
df = df[df['medical_specialty'].isin(top_categories)]
df['label'] = df['medical_specialty'].str.strip()

# Combine description + sample_name — short specific text works better
df['text'] = df['description'].fillna('') + ' ' + df['sample_name'].fillna('')
df = df[['text', 'label']].dropna()
df = df[df['text'].str.strip() != '']

print("Dataset size:", len(df))
print("\nCategories:\n", df['label'].value_counts())

# Check sample text
print("\nSample texts:")
for i in range(3):
    print(f"  [{df['label'].iloc[i]}] {df['text'].iloc[i][:100]}")

# Split
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df['label']
)

print("\nTraining samples:", len(train_df))
print("Test samples:", len(test_df))

# Pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 3),
        sublinear_tf=True,
        min_df=1,
        analyzer='word',
        stop_words='english'
    )),
    ('svm', LinearSVC(
        class_weight='balanced',
        max_iter=5000,
        C=10.0
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

# Overall accuracy
correct = sum(p == t for p, t in zip(preds, test_df['label']))
print(f"Overall Accuracy: {correct}/{len(test_df)} = {correct/len(test_df)*100:.1f}%")

# Save
os.makedirs('model', exist_ok=True)
joblib.dump(pipeline, 'model/svm_pipeline.pkl')
joblib.dump(list(df['label'].unique()), 'model/labels.pkl')
print("\nModel saved successfully!")

# Test with a custom example
print("\n--- Quick Test ---")
test_samples = [
    "Patient presents with chest pain and shortness of breath",
    "MRI scan of the lumbar spine shows disc herniation",
    "Laparoscopic cholecystectomy performed successfully",
    "EEG shows abnormal brain wave activity seizure disorder"
]
for sample in test_samples:
    pred = pipeline.predict([sample])[0]
    print(f"  Input: {sample[:60]}")
    print(f"  Predicted: {pred}\n")