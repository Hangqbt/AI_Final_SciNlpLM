import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from collections import Counter
import random
import os
import time

# --- CONFIGURATION ---
TRAIN_FILE = "clean_dataset.csv"
TEST_FILE = "arxiv_dataset.csv"
OUTPUT_EXCEL = "results_baselines_professional.xlsx"
N_FOLDS = 5
SEED = 42
MAX_LEN = 256
BATCH_SIZE = 32
EPOCHS = 8


# --- 0. SEEDING ---
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(SEED)


# --- 1. DATA PREP ---
class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        token_ids = [self.vocab.get(w, self.vocab['<UNK>']) for w in str(self.texts[idx]).lower().split()]
        if len(token_ids) < self.max_len:
            token_ids += [self.vocab['<PAD>']] * (self.max_len - len(token_ids))
        else:
            token_ids = token_ids[:self.max_len]
        return torch.tensor(token_ids), torch.tensor(self.labels[idx])


def build_vocab(texts, max_words=20000):
    c = Counter()
    for t in texts: c.update(str(t).lower().split())
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for w, _ in c.most_common(max_words): vocab[w] = len(vocab)
    return vocab


# Load Data
df_springer = pd.read_csv(TRAIN_FILE)
df_arxiv = pd.read_csv(TEST_FILE)

# Ensure labels are ints
label_map = {'Neuroscience': 0, 'Bioinformatics': 1, 'Materials Science': 2}
if df_springer['label_id'].dtype == 'object': df_springer['label_id'] = df_springer['label'].map(label_map)
if df_arxiv['label_id'].dtype == 'object': df_arxiv['label_id'] = df_arxiv['label'].map(label_map)

X_springer, y_springer = df_springer['text'].values, df_springer['label_id'].values
X_arxiv, y_arxiv = df_arxiv['text'].values, df_arxiv['label_id'].values

# Build Vocab on Training Data ONLY
vocab = build_vocab(X_springer)


# --- 2. MODELS ---
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


# --- 3. TRAINING ENGINE ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_eval_loop(model, train_loader, val_loader):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    start_train = time.time()
    for epoch in range(EPOCHS):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
    train_time = time.time() - start_train

    start_eval = time.time()
    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            preds.extend(out.argmax(1).cpu().numpy())
            truths.extend(y.cpu().numpy())
    eval_time = time.time() - start_eval

    return accuracy_score(truths, preds), f1_score(truths, preds, average='macro'), train_time, eval_time


# --- 4. EXPERIMENT RUNNER ---
results = []
models_to_run = [("TextCNN", TextCNN), ("Bi-LSTM", BiLSTM)]

# K-FOLD CV
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

for model_name, ModelClass in models_to_run:
    print(f"\n>>> PROCESSING {model_name} <<<")
    fold_accs = []

    for fold, (t_idx, v_idx) in enumerate(skf.split(X_springer, y_springer)):
        print(f"  Fold {fold + 1}...")

        # Datasets
        train_ds = TextDataset(X_springer[t_idx], y_springer[t_idx], vocab, MAX_LEN)
        val_ds = TextDataset(X_springer[v_idx], y_springer[v_idx], vocab, MAX_LEN)

        # DataLoaders with Seeded Workers (LOCKDOWN)
        g = torch.Generator()
        g.manual_seed(SEED)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, generator=g)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

        # Run
        model = ModelClass(len(vocab))
        acc, f1, t_time, e_time = train_eval_loop(model, train_loader, val_loader)

        results.append({
            "Model": model_name, "Exp": f"Springer Fold {fold + 1}",
            "Accuracy": acc, "F1": f1, "Train Time (s)": t_time, "Eval Time (s)": e_time
        })
        fold_accs.append(acc)

        # Save Fold 1 Model for Generalization
        if fold == 0:
            torch.save(model.state_dict(), f"final_{model_name}_champion.pth")

    # GENERALIZATION (Train Full Springer -> Test arXiv)
    print(f"  Running Generalization Test on arXiv...")
    # Load Champion
    model = ModelClass(len(vocab))
    model.load_state_dict(torch.load(f"final_{model_name}_champion.pth"))

    # Test Set
    full_test = TextDataset(X_arxiv, y_arxiv, vocab, MAX_LEN)
    test_loader = DataLoader(full_test, batch_size=BATCH_SIZE)

    # Evaluate (No Training)
    start_eval = time.time()
    model = model.to(device)
    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            preds.extend(out.argmax(1).cpu().numpy())
            truths.extend(y.cpu().numpy())
    eval_time = time.time() - start_eval

    acc_g = accuracy_score(truths, preds)
    f1_g = f1_score(truths, preds, average='macro')

    results.append({
        "Model": model_name, "Exp": "Generalization (arXiv)",
        "Accuracy": acc_g, "F1": f1_g, "Train Time (s)": "-", "Eval Time (s)": eval_time
    })

    # Summary Row
    results.append({
        "Model": model_name, "Exp": "Mean ± Std",
        "Accuracy": f"{np.mean(fold_accs):.4f} ± {np.std(fold_accs):.4f}",
        "F1": "-", "Train Time (s)": "-"
    })

# Save
pd.DataFrame(results).to_excel(OUTPUT_EXCEL, index=False)
print(f"\n[DONE] Baselines saved to {OUTPUT_EXCEL}")