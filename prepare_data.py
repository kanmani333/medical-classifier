import pandas as pd
import nltk
import re
from sklearn.model_selection import train_test_split

nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load dataset
df = pd.read_csv('mtsamples.csv')
print("Original shape:", df.shape)
print("\nAll categories:\n", df['medical_specialty'].value_counts())

# Keep only top 8 categories
top_categories = df['medical_specialty'].value_counts().head(8).index
df = df[df['medical_specialty'].isin(top_categories)]

# Drop nulls
df = df[['transcription', 'medical_specialty']].dropna()
df.columns = ['text', 'label']
df['label'] = df['label'].str.strip()

print("\nCleaned shape:", df.shape)
print("\nFinal categories:\n", df['label'].value_counts())

# Clean text function
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return ' '.join(tokens)

print("\nCleaning text... this takes 1-2 minutes...")
df['clean_text'] = df['text'].apply(clean_text)

# Save cleaned data
print("\nCleaning text... this takes 1-2 minutes...")
df['clean_text'] = df['text'].apply(clean_text)

# Save cleaned data
df.to_csv('cleaned_data.csv', index=False)
print("\nDone! cleaned_data.csv saved successfully!")

# Split and save
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
train_df.to_csv('train.csv', index=False)
test_df.to_csv('test.csv', index=False)
print(f"Done! Train size: {len(train_df)} | Test size: {len(test_df)}")