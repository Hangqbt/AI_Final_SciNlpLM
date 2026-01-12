import pandas as pd
import matplotlib.pyplot as plt
import re
from collections import Counter

# 1. Load the Datasets
# Assuming files are in the current directory
df_springer = pd.read_csv('clean_dataset.csv')
df_arxiv = pd.read_csv('arxiv_dataset.csv')

# Combine them into one "Corpus"
df_corpus = pd.concat([df_springer[['text', 'label']], df_arxiv[['text', 'label']]], ignore_index=True)

# 2. Text Cleaning Function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text) # Remove punctuation/numbers
    return text

# Define Stopwords (Generic English words + 'study', 'paper' boilerplate)
stopwords = set([
    'the', 'of', 'and', 'in', 'to', 'a', 'is', 'for', 'with', 'on', 'that', 'as',
    'are', 'by', 'this', 'an', 'be', 'from', 'which', 'at', 'it', 'can', 'we',
    'our', 'has', 'was', 'were', 'also', 'these', 'results', 'using', 'study',
    'paper', 'proposed', 'based', 'used', 'have', 'shown', 'such', 'new', 'between',
    'their', 'its', 'one', 'not', 'all', 'during', 'through', 'into', 'but', 'or'
])

# Apply Cleaning
df_corpus['clean_text'] = df_corpus['text'].apply(clean_text)

# 3. Analyze Vocabulary
class_counts = {}
labels = df_corpus['label'].unique()
all_words = []

# Count words for each class (Bio, Neuro, Material)
for label in labels:
    subset = df_corpus[df_corpus['label'] == label]
    words = []
    for text in subset['clean_text']:
        # Filter stopwords and short words
        tokens = [w for w in text.split() if w not in stopwords and len(w) > 2]
        words.extend(tokens)
    class_counts[label] = Counter(words)
    all_words.extend(words)

# Identify Top 20 Global Words
global_counter = Counter(all_words)
top_20_words = [w[0] for w in global_counter.most_common(20)]

# Prepare Data for Plotting
plot_data = {label: [] for label in labels}
for word in top_20_words:
    for label in labels:
        plot_data[label].append(class_counts[label][word])

# 4. Generate the Stacked Bar Chart
fig, ax = plt.subplots(figsize=(12, 8))
bottom = [0] * 20
colors = ['#1f77b4', '#ff7f0e', '#2ca02c'] # Blue, Orange, Green

for i, label in enumerate(labels):
    ax.bar(top_20_words, plot_data[label], bottom=bottom, label=label, color=colors[i])
    # Update bottom for stacking
    bottom = [sum(x) for x in zip(bottom, plot_data[label])]

plt.title('Vocabulary Overlap: Top 20 Frequent Words by Class', fontsize=16)
plt.xlabel('Top Keywords', fontsize=14)
plt.ylabel('Frequency Count', fontsize=14)
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.legend(title="Scientific Domain")
plt.tight_layout()

# Save the figure
plt.savefig('vocab_overlap.png')
print("Graph saved as 'vocab_overlap.png'")