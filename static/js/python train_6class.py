import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib
import os

# ── Load new dataset ──────────────────────────────────────────
df = pd.read_csv('train.dat', sep='\t', header=None, names=['label', 'text'])
df = df.dropna()
df = df[df['label'].apply(lambda x: str(x).strip().isdigit())]
df['label'] = df['label'].astype(int)

# Map all 5 categories from new dataset
label_map = {
    1: 'Gastroenterology',
    2: 'Cardiovascular / Pulmonary',
    3: 'General Medicine',
    4: 'Neurology',
    5: 'General Pathology'
}

df = df[df['label'].isin([1, 2, 3, 4, 5])]
df['label'] = df['label'].map(label_map)
df = df[['text', 'label']].dropna()

print("New dataset categories:")
print(df['label'].value_counts())

# ── Load mtsamples for Surgery and Radiology ──────────────────
df_mt = pd.read_csv('mtsamples.csv')
mt_categories = ['Radiology', 'Surgery']
df_mt = df_mt[df_mt['medical_specialty'].str.strip().isin(mt_categories)]
df_mt['label'] = df_mt['medical_specialty'].str.strip()
df_mt['text'] = (
    df_mt['sample_name'].fillna('') + ' ' +
    df_mt['description'].fillna('') + ' ' +
    df_mt['transcription'].fillna('')
)
df_mt = df_mt[['text', 'label']]
df_mt = df_mt[df_mt['text'].str.strip().str.len() > 50]

print("\nmtsamples categories:")
print(df_mt['label'].value_counts())

# ── Combine ───────────────────────────────────────────────────
df_combined = pd.concat([df, df_mt], ignore_index=True)
print("\nCombined:")
print(df_combined['label'].value_counts())

# ── 6 final categories ────────────────────────────────────────
categories = ['Radiology', 'Surgery', 'Neurology',
              'Gastroenterology', 'Cardiovascular / Pulmonary',
              'General Medicine']

df_combined = df_combined[df_combined['label'].isin(categories)]

# ── Oversample to balance at 1200 each ───────────────────────
balanced = []
for cat in categories:
    cat_df = df_combined[df_combined['label'] == cat].copy()
    print(f"{cat}: {len(cat_df)} samples")
    if len(cat_df) == 0:
        print(f"WARNING: No samples for {cat}!")
        continue
    if len(cat_df) < 1200:
        cat_df = cat_df.sample(1200, replace=True, random_state=42)
    else:
        cat_df = cat_df.sample(1200, replace=False, random_state=42)
    balanced.append(cat_df)

df_balanced = pd.concat(balanced).sample(frac=1, random_state=42)
print("\nBalanced size:", len(df_balanced))

# ── Split ─────────────────────────────────────────────────────
train_df, test_df = train_test_split(
    df_balanced, test_size=0.2, random_state=42, stratify=df_balanced['label']
)

# ── Pipeline ──────────────────────────────────────────────────
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=200000,
        ngram_range=(1, 3),
        sublinear_tf=True,
        min_df=1,
        stop_words='english',
        analyzer='word'
    )),
    ('svm', LinearSVC(
        class_weight='balanced',
        max_iter=10000,
        C=10.0
    ))
])

# ── Train ─────────────────────────────────────────────────────
print("\nTraining model...")
pipeline.fit(train_df['text'], train_df['label'])
print("Training complete!")

# ── Evaluate ──────────────────────────────────────────────────
preds = pipeline.predict(test_df['text'])
print("\n--- Classification Report ---")
print(classification_report(test_df['label'], preds, zero_division=0))

correct = sum(p == t for p, t in zip(preds, test_df['label']))
print(f"Overall Accuracy: {correct}/{len(test_df)} = {correct/len(test_df)*100:.1f}%")

# ── Save ──────────────────────────────────────────────────────
os.makedirs('model', exist_ok=True)
joblib.dump(pipeline, 'model/svm_pipeline.pkl')
joblib.dump(categories, 'model/labels.pkl')
print("\nModel saved successfully!")

# ── Quick test ────────────────────────────────────────────────
print("\n--- Quick Test ---")
print(f"{'Expected':<30} {'Predicted':<30} {'Result'}")
print("-" * 75)

samples = [
    ("Neurology", "MRI brain with contrast reveals ring enhancing lesion right temporal lobe EEG shows focal slowing patient presents with new onset seizures"),
    ("Gastroenterology", "Colonoscopy performed colon polyp removed endoscopy gastric biopsy taken patient had abdominal pain nausea vomiting"),
    ("Cardiovascular / Pulmonary", "12 lead ECG shows ST elevation V2 V5 anterior wall myocardial infarction troponin elevated cardiac catheterization heart"),
    ("Radiology", "PA lateral chest X-ray bilateral pulmonary infiltrates consolidation right lower lobe pleural effusion cardiac silhouette enlarged"),
    ("Surgery", "Laparoscopic appendectomy performed general anesthesia appendix inflamed incision closed absorbable sutures patient recovered"),
    ("General Medicine", "Patient presents fever 38.9 degrees productive cough shortness breath three days WBC elevated neutrophilia community acquired pneumonia amoxicillin"),
    ("Neurology", "Patient Parkinson disease tremor bradykinesia rigidity dopamine deficiency levodopa carbidopa treatment"),
    ("Gastroenterology", "Upper GI endoscopy reveals peptic ulcer stomach helicobacter pylori gastric acid reflux esophagitis"),
    ("Cardiovascular / Pulmonary", "Patient atrial fibrillation irregular heart rate blood pressure elevated hypertension coronary artery disease"),
    ("General Medicine", "Fever chills body ache headache fatigue viral infection influenza patient presents acute febrile illness treatment prescribed"),
]

correct_count = 0
for expected, text in samples:
    pred = pipeline.predict([text])[0]
    result = "CORRECT" if pred == expected else "WRONG"
    if pred == expected:
        correct_count += 1
    print(f"{expected:<30} {pred:<30} {result}")

print(f"\nQuick Test Score: {correct_count}/{len(samples)} = {correct_count/len(samples)*100:.0f}%")