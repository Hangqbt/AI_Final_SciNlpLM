import pandas as pd
import numpy as np
import torch
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

# --- CONFIGURATION ---
MODEL_NAME = "allenai/scibert_scivocab_uncased"
FILE_PATH = "clean_dataset.csv"
SEEDS = [50, 100, 150]  # Professional reproducibility seeds


# --- 0. SEEDING ---
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# Load Data
df = pd.read_csv(FILE_PATH)
X_raw = df['text'].tolist()
y_raw = df['label_id'].tolist()


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='macro')
    return {'accuracy': acc, 'f1': f1}


class ScienceDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self): return len(self.labels)


# --- EXPERIMENT LOOP ---
acc_results = []
f1_results = []

print(f"\n>>> STARTING SCIBERT EXPERIMENT ({len(SEEDS)} runs) <<<")

for seed in SEEDS:
    print(f"\n--- Run Seed {seed} ---")
    set_seed(seed)

    # 1. Split
    X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.2, random_state=seed)

    # 2. Tokenize
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_enc = tokenizer(X_train, truncation=True, padding=True, max_length=256)
    test_enc = tokenizer(X_test, truncation=True, padding=True, max_length=256)

    train_ds = ScienceDataset(train_enc, y_train)
    test_ds = ScienceDataset(test_enc, y_test)

    # 3. Model
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)

    # 4. Train Args (FIXED HERE)
    training_args = TrainingArguments(
        output_dir=f'./results_scibert_seed{seed}',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        warmup_steps=50,
        weight_decay=0.01,
        logging_steps=50,
        eval_strategy="epoch",  # <--- CHANGED FROM evaluation_strategy
        save_strategy="no",
        seed=seed,
        data_seed=seed
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics
    )

    trainer.train()

    # 5. Evaluate
    res = trainer.evaluate()
    acc_results.append(res['eval_accuracy'])
    f1_results.append(res['eval_f1'])
    print(f"--> Seed {seed} Result: Acc={res['eval_accuracy']:.4f}, F1={res['eval_f1']:.4f}")

# --- FINAL REPORT ---
print("\n\n==============================================")
print("   SCIBERT FINAL PROFESSIONAL RESULTS")
print("==============================================")
print(f"Mean Accuracy: {np.mean(acc_results) * 100:.2f}% ± {np.std(acc_results) * 100:.2f}")
print(f"Mean F1-Score: {np.mean(f1_results):.4f} ± {np.std(f1_results):.4f}")