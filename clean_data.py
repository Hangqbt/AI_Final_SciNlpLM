import pandas as pd
import matplotlib.pyplot as plt

# 1. Load Data
df = pd.read_csv("refilled_dataset_uncleaned.csv")
print(f"Original Count: {len(df)}")

# 2. Drop Duplicates
# It's common to accidentally scrape the same paper twice if it appears in multiple search pages.
df.drop_duplicates(subset=['text'], inplace=True)
print(f"Count after removing duplicates: {len(df)}")

# 3. Drop Bad Rows
# Remove rows where text is empty or too short (less than 50 characters is suspicious)
df.dropna(subset=['text'], inplace=True)
df = df[df['text'].str.len() > 50]
print(f"Final Valid Count: {len(df)}")

# 4. Check Class Balance
# For a fair comparison, you want roughly equal numbers (e.g., 280, 290, 300 is fine. 50 vs 300 is bad).
print("\nClass Distribution:")
print(df['label'].value_counts())

# 5. Label Encoding
# Computers need numbers, not words.
# Map: Neuroscience -> 0, Bioinformatics -> 1, Materials Science -> 2
label_map = {
    'Neuroscience': 0,
    'Bioinformatics': 1,
    'Materials Science': 2
}
df['label_id'] = df['label'].map(label_map)

# 6. Save the Clean Version
df.to_csv("clean_dataset.csv", index=False)
print("\nSaved 'clean_dataset.csv'. Ready for training!")