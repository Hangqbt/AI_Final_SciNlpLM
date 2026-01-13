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

##########################
# This is the main testing code for the baseline Models
##########################

# We fix the file paths for the datasets and the output file here
TRAIN_FILE = "clean_dataset.csv"
TEST_FILE = "arxiv_dataset.csv"
OUTPUT_EXCEL = "results_baselines_professional.xlsx"

# These variables determine the training settings of the models
N_FOLDS = 5  # This is the number of splits of the dataset during k-fold cross validation
SEED = 42  # A seed value is given here to be used for fixing randomness
MAX_LEN = 256  # This essentially determines the maximum sequence length of the tokens for the model during interpretation
BATCH_SIZE = 32  # Batch size determines the number of samples processed at one time
EPOCHS = 8  # This is the number of iterations we go through the dataset


# This function essentially takes the seed value and fixes all the randomness present in the code and the other supporting functions.
def set_seed(seed):
    """
Fixing these seeds will make the entire code fully deterministic ensuring reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # These variables force pytorch to be strictly deterministic throughout all the runs. Even though it slows down the process a bit, it is often worth the trade off for determinism.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# we run the function to lock everything before executing anything else
set_seed(SEED)


## here we are preparing the data to be loaded into the model
class TextDataset(Dataset):
    """
    This creates a custom pytorch dataset to handle the data
    This essentially converts raw string texts into a tensor form with integer ids based on the vocabulary of the data

    """

    def __init__(self, texts, labels, vocab, max_len):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # here the data gets to tokenized (ie, split but spaces or separators) then it is mapped to an id based on the vocabulary dictionary
        token_ids = [self.vocab.get(w, self.vocab['<UNK>']) for w in str(self.texts[idx]).lower().split()]

        # Here we either pad or truncate the tokens to match the Maximum sequence length
        if len(token_ids) < self.max_len:
            # here we pad by adding <PAD> for satisfying the sequence length
            token_ids += [self.vocab['<PAD>']] * (self.max_len - len(token_ids))
        else:
            # here we cut off the excess tokens to match the length (essentially truncate)
            token_ids = token_ids[:self.max_len]

        # now once the processing is done, we return the data as pytorch tensors.
        return torch.tensor(token_ids), torch.tensor(self.labels[idx])


def build_vocab(texts, max_words=20000):
    """
    This function essentially builds a vocabulary from strings which are mapped to unique numbers.
    """
    c = Counter()
    for t in texts: c.update(str(t).lower().split())

    # These are special characters which are mapped to numbers 0  and 1 for using for data parsing that we saw earlier
    vocab = {'<PAD>': 0, '<UNK>': 1}

    # This adds the most frequent words to the vocabulary
    for w, _ in c.most_common(max_words):
        vocab[w] = len(vocab)
    return vocab


# here we start to load and preprocess the texts
print("Loading datasets...")
df_springer = pd.read_csv(TRAIN_FILE)
df_arxiv = pd.read_csv(TEST_FILE)

# Here we map the string labels to integer numbers
label_map = {'Neuroscience': 0, 'Bioinformatics': 1, 'Materials Science': 2}
if df_springer['label_id'].dtype == 'object':
    df_springer['label_id'] = df_springer['label'].map(label_map)
if df_arxiv['label_id'].dtype == 'object':
    df_arxiv['label_id'] = df_arxiv['label'].map(label_map)

# Here we extract the raw numpy arrays from the datasets
X_springer, y_springer = df_springer['text'].values, df_springer['label_id'].values
X_arxiv, y_arxiv = df_arxiv['text'].values, df_arxiv['label_id'].values

# Here we build the vocabulary from the springer dataset alone
vocab = build_vocab(X_springer)
print(f"Vocabulary size: {len(vocab)}")


## Here we have the different baseline models that are being used for the study
# Namely, TextCNN and Bi-LSTM,
#While textCNN is based on convolution, which is generally used for image processing, it uses a one dimensional kernel to slide around the text which is essentially interpretted as an image by the model.
#Bi-LSTM on the other hand is a sequential processing model which essentially understands the data from both the directions, allowing it to temporally understand contexts

class TextCNN(nn.Module):
    """
    This model basically uses filters to capture n-gram features
    """

    def __init__(self, vocab_size, embed_dim=100, num_classes=3):
        super().__init__()

        #This is the embedding layer which is essentially a lookup table that converts words into vectors
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)


        #This is the convolutional layers with 3 parallel layers of different sizes, namely, 3,4, and 5
        # This layer has one input channel for texts and 100 output channels
        self.convs = nn.ModuleList([nn.Conv2d(1, 100, (k, embed_dim)) for k in [3, 4, 5]])

        self.dropout = nn.Dropout(0.5)
        # we regularize here
        self.fc = nn.Linear(300, num_classes)
        # Here is where the final prediction happens
        # Wtih 3 filters and 100 channels we get 300 features

    def forward(self, x):
        # we add the embedding dimensions here into the shape,
        # essentially changing [batch_size, seq_len] to [batch_size, seq_len, embed_dim]
        x = self.embedding(x)

        # Here we add the channel dimension
        # This basically tricks Conv2d to treat the text like its a grayscale image with height being the sequence length
        # and width being the embedding dimension
        x = x.unsqueeze(1)


        #Here we apply the convolution, use the activation functions then squeeze out the dimensions for inference
        x = [torch.relu(c(x)).squeeze(3) for c in self.convs]

        # Here we do max pooling to find the most important token
        x = [torch.max_pool1d(i, i.size(2)).squeeze(2) for i in x]

        # Here we concatenate all the features for the filters then we finally apply dropout
        x = torch.cat(x, 1)
        return self.fc(self.dropout(x))


class BiLSTM(nn.Module):
    """
    Like mentioned earlier, this model reads and processes texts from two directions to build up its context on the data
    """

    def __init__(self, vocab_size, embed_dim=100, hidden=128, num_classes=3):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # This is the long short term memory layer with the bidirectional variable enabled allowing it to processes it from both sides
        self.lstm = nn.LSTM(embed_dim, hidden, batch_first=True, bidirectional=True)

        # This is the FC layer and the input is given as "hidden*2" due to the bidirectional nature of the model
        self.fc = nn.Linear(hidden * 2, num_classes)

    def forward(self, x):
        x = self.emb(x)

        # When we run the LSTM Model it returns the output along with the hidden and cell states
        # here "h" contains all the hidden states for all layers and directions
        _, (h, _) = self.lstm(x)

        # Here we basically concatenate the final forward and  backward vector which gives us a summary vector as an output which holds the contexts of the current data that was just done processing
        hidden_cat = torch.cat((h[-2], h[-1]), dim=1)
        return self.fc(hidden_cat)


# Here we determine our training device that is to be used. The setup that was used to run the study used an Nvidia RTX 3060 (Laptop) GPU, which support CUDA, hence it is enabled and utilized.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def train_eval_loop(model, train_loader, val_loader):
    """
    We use a standard Pytorch training loop which runs for N number of Epochs and finally tests on the validiation set
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # This is the training phase of the model
    start_train = time.time()
    for epoch in range(EPOCHS):
        model.train()
        # Here we call the train functions which enables dropout and allows the model to learn
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            # Here we reset the gradient
            loss = criterion(model(x), y)
            # Here we calculate the loss
            loss.backward()
            # Here we perform backpropagation
            optimizer.step()
            # Here we update the weights accordingly
    train_time = time.time() - start_train

  # From here, we perform evaluation
    start_eval = time.time()
    model.eval()
    # This puts the model in evaluation mode disabling dropout
    preds, truths = [], []

    with torch.no_grad():
        # Here we disable gradient calculation to save memory
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            preds.extend(out.argmax(1).cpu().numpy())
            # Here we get the predicted class
            truths.extend(y.cpu().numpy())
    eval_time = time.time() - start_eval

    return accuracy_score(truths, preds), f1_score(truths, preds, average='macro'), train_time, eval_time


# The following is the main loop that runs all the models sequentially
results = []
models_to_run = [("TextCNN", TextCNN), ("Bi-LSTM", BiLSTM)]

# We initialize the statified Kfold to get balanced splits during the kfold process on the dataset
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

for model_name, ModelClass in models_to_run:
    print(f"\n>>> PROCESSING {model_name} <<<")
    fold_accs = []

    # Here we iterate through the 5 folds (because our k=5)
    for fold, (t_idx, v_idx) in enumerate(skf.split(X_springer, y_springer)):
        print(f"  Fold {fold + 1}...")

        # We create the dataset for this specific fold that is being performed
        train_ds = TextDataset(X_springer[t_idx], y_springer[t_idx], vocab, MAX_LEN)
        val_ds = TextDataset(X_springer[v_idx], y_springer[v_idx], vocab, MAX_LEN)

        # Here we create the dataloaders and fix the randomness to ensure reproducibility
        g = torch.Generator()
        g.manual_seed(SEED)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, generator=g)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

        # Here we initialize and train the models
        model = ModelClass(len(vocab))
        acc, f1, t_time, e_time = train_eval_loop(model, train_loader, val_loader)

        # We log the metrics here
        results.append({
            "Model": model_name, "Exp": f"Springer Fold {fold + 1}",
            "Accuracy": acc, "F1": f1, "Train Time (s)": t_time, "Eval Time (s)": e_time
        })
        fold_accs.append(acc)

        # here we save the best model from first test to be used for the generalization testing done below
        if fold == 0:
            torch.save(model.state_dict(), f"final_{model_name}_champion.pth")

   # Here we do generalization testing on the arXiv dataset to check the actual capability of the models on unseen data
    # Also to note that the models were trained only on the springer dataset and were never exposed to this data
    print(f"  Running Generalization Test on arXiv...")

    # We reload the model and its weights to be used here
    model = ModelClass(len(vocab))
    model.load_state_dict(torch.load(f"final_{model_name}_champion.pth"))

    # We create the dataloaders to load the datasets
    full_test = TextDataset(X_arxiv, y_arxiv, vocab, MAX_LEN)
    test_loader = DataLoader(full_test, batch_size=BATCH_SIZE)

    # We do the inference here on the dataset
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

    # We finally log the generalization results here
    results.append({
        "Model": model_name, "Exp": "Generalization (arXiv)",
        "Accuracy": acc_g, "F1": f1_g, "Train Time (s)": "-", "Eval Time (s)": eval_time
    })

    # We log the averaged K-fold score here
    results.append({
        "Model": model_name, "Exp": "Mean ± Std",
        "Accuracy": f"{np.mean(fold_accs):.4f} ± {np.std(fold_accs):.4f}",
        "F1": "-", "Train Time (s)": "-"
    })

# Finally we save and export all the information into an excel document
pd.DataFrame(results).to_excel(OUTPUT_EXCEL, index=False)
print(f"\n[DONE] Baselines saved to {OUTPUT_EXCEL}")