import pandas as pd

df = pd.read_csv('mtsamples.csv')

# Check what columns exist
print("Columns:", df.columns.tolist())
print("\nFirst row:\n")
print(df.iloc[0])
print("\n--- Sample transcription (first 300 chars) ---")
print(str(df['transcription'].iloc[0])[:300])
print("\n--- Sample transcription (second row) ---")
print(str(df['transcription'].iloc[1])[:300])

# Check for nulls
print("\nNull counts:")
print(df.isnull().sum())

# Check description column
print("\n--- Sample description ---")
print(df['description'].iloc[0])
print(df['description'].iloc[1])
print(df['description'].iloc[2])