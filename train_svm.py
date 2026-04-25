import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from sklearn.pipeline import Pipeline
import numpy as np
import joblib
import os

# Load data
train_df = pd.read_csv('train.csv')
test_df  = pd.read_csv('test.csv')

print("Training samples:", len(train_df))
print("Test samples:", len(test_df))
print("Categories:", train_df['label'].unique())

# Compute class weights to handle imbalance
classes = np.unique(train_df['label'])
weights = compute_class_weight('balanced', classes=classes, y=train_df['label'])
class_weight_dict = dict(zip(classes, weights))
print("\nClass weights computed to handle imbalance")

# Build pipeline — TF-IDF + SVM
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2
    )),
    ('svm', LinearSVC(
        class_weight='balanced',
        max_iter=2000,
        C=1.0
    ))
])

# Train
print("\nTraining TF-IDF + SVM model...")
pipeline.fit(train_df['clean_text'], train_df['label'])
print("Training complete!")

# Evaluate
preds = pipeline.predict(test_df['clean_text'])
print("\n--- Classification Report ---")
print(classification_report(test_df['label'], preds, zero_division=0))

# Save model
os.makedirs('model', exist_ok=True)
joblib.dump(pipeline, 'model/svm_pipeline.pkl')
print("Model saved to model/svm_pipeline.pkl successfully!")