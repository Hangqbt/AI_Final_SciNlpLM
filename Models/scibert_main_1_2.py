import pandas as pd
import numpy as np
import torch
import random
import os
import time
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

# --- CONFIGURATION ---
MODEL_NAME = "allenai/scibert_scivocab_uncased"
TRAIN_FILE = "clean_dataset.csv"  # Springer (Gold Standard)
TEST_FILE = "arxiv_dataset.csv"  # arXiv (Generalization Test)
OUTPUT_EXCEL = "results_scibert_professional.xlsx"
N_FOLDS = 5
SEED = 42


# --- 0. ROBUST SEEDING (The "Lockdown") ---
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


set_seed(SEED)


# --- 1. DATA LOADING ---
def load_data(path):
    if not os.path.exists(path): raise FileNotFoundError(f"{path} not found!")
    df = pd.read_csv(path)
    # Ensure labels are integers
    if df['label_id'].dtype == 'object':
        label_map = {'Neuroscience': 0, 'Bioinformatics': 1, 'Materials Science': 2}
        df['label_id'] = df['label'].map(label_map)
    return df['text'].tolist(), df['label_id'].tolist()


X_springer, y_springer = load_data(TRAIN_FILE)
X_arxiv, y_arxiv = load_data(TEST_FILE)


# Custom Dataset
class ScienceDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self): return len(self.labels)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='macro')
    return {'accuracy': acc, 'f1': f1}


# --- 2. EXPERIMENT ENGINE ---
results_log = []

# PART A: 5-FOLD CV ON SPRINGER
print(f"\n>>> STARTING 5-FOLD CV ON SPRINGER DATA <<<")
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
fold_accuracies = []
fold_f1s = []

X_np = np.array(X_springer)
y_np = np.array(y_springer)

for fold, (train_idx, val_idx) in enumerate(skf.split(X_np, y_np)):
    print(f"\n--- FOLD {fold + 1}/{N_FOLDS} ---")

    # Split
    X_tr, X_val = X_np[train_idx], X_np[val_idx]
    y_tr, y_val = y_np[train_idx], y_np[val_idx]

    # Tokenize
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_enc = tokenizer(list(X_tr), truncation=True, padding=True, max_length=256)
    val_enc = tokenizer(list(X_val), truncation=True, padding=True, max_length=256)

    train_ds = ScienceDataset(train_enc, y_tr)
    val_ds = ScienceDataset(val_enc, y_val)

    # Model & Args
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)

    training_args = TrainingArguments(
        output_dir=f'./checkpoints/scibert_fold{fold}',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,  # Critical for valid results
        learning_rate=2e-5,
        seed=SEED,
        data_seed=SEED  # Lock DataLoader shuffling
    )

    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=train_ds, eval_dataset=val_ds,
        compute_metrics=compute_metrics
    )

    # Train
    train_res = trainer.train()
    eval_res = trainer.evaluate()

    results_log.append({
        "Model": "SciBERT",
        "Experiment": f"Springer Fold {fold + 1}",
        "Accuracy": eval_res['eval_accuracy'],
        "F1": eval_res['eval_f1'],
        "Train Time (s)": train_res.metrics['train_runtime'],
        "Eval Time (s)": eval_res['eval_runtime']
    })
    fold_accuracies.append(eval_res['eval_accuracy'])
    fold_f1s.append(eval_res['eval_f1'])

    # Save the BEST model from Fold 1 to use for Generalization Test
    if fold == 0:
        print("Saving Fold 1 model for Generalization Test...")
        model.save_pretrained("./final_scibert_champion")
        tokenizer.save_pretrained("./final_scibert_champion")

# PART B: GENERALIZATION (Train Springer -> Test arXiv)
print(f"\n>>> STARTING CROSS-DATASET GENERALIZATION (Springer -> arXiv) <<<")

# Load the Champion Model (Trained on Springer Fold 1)
tokenizer = AutoTokenizer.from_pretrained("./final_scibert_champion")
model = AutoModelForSequenceClassification.from_pretrained("./final_scibert_champion", num_labels=3)

# Tokenize arXiv
test_enc = tokenizer(X_arxiv, truncation=True, padding=True, max_length=256)
test_ds = ScienceDataset(test_enc, y_arxiv)

# Evaluator
trainer = Trainer(model=model, compute_metrics=compute_metrics)

print("Evaluating on full arXiv dataset...")
start_eval = time.time()
preds = trainer.predict(test_ds)
eval_time = time.time() - start_eval

acc = accuracy_score(y_arxiv, preds.predictions.argmax(-1))
f1 = f1_score(y_arxiv, preds.predictions.argmax(-1), average='macro')

results_log.append({
    "Model": "SciBERT",
    "Experiment": "Generalization (arXiv)",
    "Accuracy": acc,
    "F1": f1,
    "Train Time (s)": "-",  # Already trained
    "Eval Time (s)": eval_time
})

print(f"Generalization Accuracy: {acc:.4f}")

# --- SAVE REPORT ---
df_res = pd.DataFrame(results_log)
# Add Summary Row
summary = {
    "Model": "SciBERT",
    "Experiment": "K-Fold Mean ± Std",
    "Accuracy": f"{np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}",
    "F1": f"{np.mean(fold_f1s):.4f} ± {np.std(fold_f1s):.4f}",
    "Train Time (s)": "-",
    "Eval Time (s)": "-"
}
df_res = pd.concat([df_res, pd.DataFrame([summary])], ignore_index=True)
df_res.to_excel(OUTPUT_EXCEL, index=False)
print(f"\n[DONE] SciBERT Results saved to {OUTPUT_EXCEL}")