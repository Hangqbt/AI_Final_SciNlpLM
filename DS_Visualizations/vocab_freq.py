import pandas as pd
import matplotlib.pyplot as plt
import re
from collections import Counter

# We load the dataset here
df_springer = pd.read_csv('clean_dataset.csv')
df_arxiv = pd.read_csv('arxiv_dataset.csv')

# We combine them into a single corpus
df_corpus = pd.concat([df_springer[['text', 'label']], df_arxiv[['text', 'label']]], ignore_index=True)

# 2. we call the text cleaning function here
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text) # Remove punctuation/numbers
    return text

# We define the stop words here to reduce noise in the output
stopwords = set([
    'the', 'of', 'and', 'in', 'to', 'a', 'is', 'for', 'with', 'on', 'that', 'as',
    'are', 'by', 'this', 'an', 'be', 'from', 'which', 'at', 'it', 'can', 'we',
    'our', 'has', 'was', 'were', 'also', 'these', 'results', 'using', 'study',
    'paper', 'proposed', 'based', 'used', 'have', 'shown', 'such', 'new', 'between',
    'their', 'its', 'one', 'not', 'all', 'during', 'through', 'into', 'but', 'or'
])

# We apply the cleaning here
df_corpus['clean_text'] = df_corpus['text'].apply(clean_text)

# 3. We analyze the vocabulary here
class_counts = {}
labels = df_corpus['label'].unique()
all_words = []

# We count the words for each class here
for label in labels:
    subset = df_corpus[df_corpus['label'] == label]
    words = []
    for text in subset['clean_text']:
        # Now we filter all the stopwords
        tokens = [w for w in text.split() if w not in stopwords and len(w) > 2]
        words.extend(tokens)
    class_counts[label] = Counter(words)
    all_words.extend(words)

# Here we identify the top 20 words
global_counter = Counter(all_words)
top_20_words = [w[0] for w in global_counter.most_common(20)]

# We then prepare to plot the output
plot_data = {label: [] for label in labels}
for word in top_20_words:
    for label in labels:
        plot_data[label].append(class_counts[label][word])

# Finally, we generate the graphs
fig, ax = plt.subplots(figsize=(12, 8))
bottom = [0] * 20
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

for i, label in enumerate(labels):
    ax.bar(top_20_words, plot_data[label], bottom=bottom, label=label, color=colors[i])
    bottom = [sum(x) for x in zip(bottom, plot_data[label])]

plt.title('Vocabulary Overlap: Top 20 Frequent Words by Class', fontsize=16)
plt.xlabel('Top Keywords', fontsize=14)
plt.ylabel('Frequency Count', fontsize=14)
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.legend(title="Scientific Domain")
plt.tight_layout()

# Then we save the output as an image
plt.savefig('vocab_overlap.png')
print("Graph saved as 'vocab_overlap.png'")