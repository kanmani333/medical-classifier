import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib
import os

# Load ORIGINAL data — not cleaned
df = pd.read_csv('mtsamples.csv')

# Keep top 8 categories
top_categories = df['medical_specialty'].value_counts().head(8).index
df = df[df['medical_specialty'].isin(top_categories)]
df = df[['transcription', 'medical_specialty']].dropna()
df.columns = ['text', 'label']
df['label'] = df['label'].str.strip()

print("Dataset size:", len(df))
print("Categories:\n", df['label'].value_counts())

# Split
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df['label']
)

print("\nTraining samples:", len(train_df))
print("Test samples:", len(test_df))

# Build pipeline — raw text works better for TF-IDF
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=80000,
        ngram_range=(1, 3),
        sublinear_tf=True,
        min_df=2,
        analyzer='word',
        stop_words='english'
    )),
    ('svm', LinearSVC(
        class_weight='balanced',
        max_iter=3000,
        C=5.0
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

# Save
os.makedirs('model', exist_ok=True)
joblib.dump(pipeline, 'model/svm_pipeline.pkl')

# Save label list
labels = list(df['label'].unique())
joblib.dump(labels, 'model/labels.pkl')

print("Model saved successfully!")