import torch
import torch.nn as nn
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from collections import Counter
import os

# we select the model type and its optimal checkpoint for testing
MODEL_TYPE = "Bi-LSTM"
# NOTE: The type can be changed to Either "Bi-LSTM" or "TextCNN"
MODEL_PATH = f"final_{MODEL_TYPE}_champion.pth"

# We use the springer dataset to build the vocabulary
TRAIN_FILE = "clean_dataset.csv"

# We select the testing dataset and the output image name here
TARGET_FILE = "arxiv_dataset.csv"  # <--- CHANGE THIS to plot different matrices
OUTPUT_IMAGE = f"confusion_matrix_{MODEL_TYPE}_{TARGET_FILE.split('_')[0]}.png"
MAX_LEN = 256


# This contains the same Baseline architectures found in the main code
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=100, num_classes=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv2d(1, 100, (k, embed_dim)) for k in [3, 4, 5]])
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(300, num_classes)

    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)
        x = [torch.relu(c(x)).squeeze(3) for c in self.convs]
        x = [torch.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        return self.fc(self.dropout(torch.cat(x, 1)))


class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=100, hidden=128, num_classes=3):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden * 2, num_classes)

    def forward(self, x):
        _, (h, _) = self.lstm(self.emb(x))
        return self.fc(torch.cat((h[-2], h[-1]), dim=1))


# This helper function, helps us build the vocab
def build_vocab(texts, max_words=20000):
    c = Counter()
    for t in texts: c.update(str(t).lower().split())
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for w, _ in c.most_common(max_words): vocab[w] = len(vocab)
    return vocab


def preprocess(text, vocab):
    token_ids = [vocab.get(w, vocab['<UNK>']) for w in str(text).lower().split()]
    if len(token_ids) < MAX_LEN:
        token_ids += [vocab['<PAD>']] * (MAX_LEN - len(token_ids))
    else:
        token_ids = token_ids[:MAX_LEN]
    return torch.tensor(token_ids).unsqueeze(0)


# This is the main function that runs the code and generates the matrix
def main():
    print(f"Generating Plots for {MODEL_TYPE} on {TARGET_FILE}...")

    # We build the vocab here
    print(f"Building Vocab from {TRAIN_FILE}...")
    df_train = pd.read_csv(TRAIN_FILE)
    vocab = build_vocab(df_train['text'].values)
    print(f"Vocab size: {len(vocab)}")

    # We load the target data (springer)
    print(f"Loading Target Data from {TARGET_FILE}...")
    df_target = pd.read_csv(TARGET_FILE)

    # We map the labels to indexes
    label_map = {'Neuroscience': 0, 'Bioinformatics': 1, 'Materials Science': 2}
    if df_target['label_id'].dtype == 'object':
        df_target['label_id'] = df_target['label'].map(label_map)

    X = df_target['text'].values
    y_true = df_target['label_id'].values

    # We load the model here
    model = TextCNN(len(vocab)) if MODEL_TYPE == "TextCNN" else BiLSTM(len(vocab))
    if not os.path.exists(MODEL_PATH):
        print(f"Error: {MODEL_PATH} not found.")
        return
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    # We run the inference heres
    y_pred = []
    print("Running Inference...")
    with torch.no_grad():
        for text in X:
            inputs = preprocess(text, vocab)
            outputs = model(inputs)
            pred = outputs.argmax(1).item()
            y_pred.append(pred)

    # We finally plot the matrix here
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=label_map.keys(), yticklabels=label_map.keys())
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'{MODEL_TYPE} Matrix on {TARGET_FILE}', fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE, dpi=300)
    print(f"[SUCCESS] Saved to {OUTPUT_IMAGE}")
    plt.show()


if __name__ == "__main__":
    main()