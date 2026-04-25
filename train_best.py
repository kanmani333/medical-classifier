import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Load data
df = pd.read_csv('mtsamples.csv')

# Keep top 8 categories
top_categories = df['medical_specialty'].value_counts().head(8).index
df = df[df['medical_specialty'].isin(top_categories)]
df['label'] = df['medical_specialty'].str.strip()

# Combine ALL text columns — more context = better accuracy
df['text'] = (
    df['sample_name'].fillna('') + ' ' +
    df['description'].fillna('') + ' ' +
    df['transcription'].fillna('')
)

df = df[['text', 'label']]
df = df[df['text'].str.strip().str.len() > 50]  # remove very short texts

print("Dataset size:", len(df))
print("\nCategories:\n", df['label'].value_counts())

# Split
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df['label']
)
print("\nTrain:", len(train_df), "| Test:", len(test_df))

# Pipeline with best settings
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=100000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
        stop_words='english',
        analyzer='word'
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

# Save model and labels
os.makedirs('model', exist_ok=True)
joblib.dump(pipeline, 'model/svm_pipeline.pkl')
joblib.dump(list(df['label'].unique()), 'model/labels.pkl')
print("\nModel saved to model/svm_pipeline.pkl")

# Quick test
print("\n--- Quick Test ---")
test_samples = [
    "Chest pain shortness of breath patient presents with cardiac symptoms ECG abnormal",
    "MRI scan lumbar spine disc herniation L4 L5 back pain",
    "Laparoscopic cholecystectomy gallbladder removal surgery performed",
    "EEG abnormal seizure disorder brain wave activity neurology",
    "Colonoscopy performed colon polyp gastroenterology endoscopy",
    "Knee replacement orthopedic surgery joint pain arthritis"
]
for sample in test_samples:
    pred = pipeline.predict([sample])[0]
    print(f"  Input: {sample[:65]}")
    print(f"  Predicted: {pred}\n")