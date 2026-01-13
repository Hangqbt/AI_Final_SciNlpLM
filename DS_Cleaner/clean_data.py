import pandas as pd
import matplotlib.pyplot as plt

# We load the dataset
df = pd.read_csv("refilled_dataset_uncleaned.csv")
print(f"Original Count: {len(df)}")

# 2. We drop the duplicates here incase we accidentally have repetitions
df.drop_duplicates(subset=['text'], inplace=True)
print(f"Count after removing duplicates: {len(df)}")

# Here we scan and drop the bad rows (Rows with either no or suspiciously low word count
df.dropna(subset=['text'], inplace=True)
df = df[df['text'].str.len() > 50]
print(f"Final Valid Count: {len(df)}")

# We check for class balance here!
print("\nClass Distribution:")
print(df['label'].value_counts())

# We perform label encoding here so each topic class can be matched to an index
label_map = {
    'Neuroscience': 0,
    'Bioinformatics': 1,
    'Materials Science': 2
}
df['label_id'] = df['label'].map(label_map)

# We finally save the cleaned version
df.to_csv("clean_dataset.csv", index=False)
print("\nSaved 'clean_dataset.csv'. Ready for training!")