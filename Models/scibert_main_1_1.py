import pandas as pd
import numpy as np
import torch
import random
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

# --- CONFIGURATION ---
MODEL_NAME = "allenai/scibert_scivocab_uncased"
FILE_PATH = "clean_dataset.csv"
OUTPUT_EXCEL = "scibert_experiment_results_with_time.xlsx"
SEEDS = [50, 100, 150]


# --- 0. SEEDING ---
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# Load Data
if not os.path.exists(FILE_PATH):
    raise FileNotFoundError(f"Dataset {FILE_PATH} not found!")

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
results_data = []

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

    # 4. Train Args
    training_args = TrainingArguments(
        output_dir=f'./results_scibert_seed{seed}',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        warmup_steps=50,
        weight_decay=0.01,
        logging_steps=50,
        eval_strategy="epoch",
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

    # 5. Train & Capture Time
    train_result = trainer.train()
    train_time = train_result.metrics["train_runtime"]  # <--- CAPTURE TRAIN TIME

    # 6. Evaluate & Capture Time
    res = trainer.evaluate()
    eval_time = res["eval_runtime"]  # <--- CAPTURE EVAL TIME

    # Store in list
    results_data.append({
        "Seed": seed,
        "Accuracy": res['eval_accuracy'],
        "F1_Score": res['eval_f1'],
        "Train_Time(s)": train_time,
        "Eval_Time(s)": eval_time
    })

    print(f"--> Seed {seed}: Acc={res['eval_accuracy']:.4f}, TrainTime={train_time:.2f}s")

# --- FINAL REPORT & EXCEL EXPORT ---
print("\n\n==============================================")
print("   SCIBERT FINAL PROFESSIONAL RESULTS")
print("==============================================")

df_results = pd.DataFrame(results_data)

# Calculate Stats
mean_acc = df_results["Accuracy"].mean()
std_acc = df_results["Accuracy"].std()
mean_f1 = df_results["F1_Score"].mean()
std_f1 = df_results["F1_Score"].std()

# Calculate Time Stats
mean_train_time = df_results["Train_Time(s)"].mean()
std_train_time = df_results["Train_Time(s)"].std()
mean_eval_time = df_results["Eval_Time(s)"].mean()
std_eval_time = df_results["Eval_Time(s)"].std()

print(f"Mean Accuracy: {mean_acc * 100:.2f}% ± {std_acc * 100:.2f}")
print(f"Mean Train Time: {mean_train_time:.2f}s ± {std_train_time:.2f}")

# Add Summary Row
summary_row = {
    "Seed": "MEAN ± STD",
    "Accuracy": f"{mean_acc:.4f} ± {std_acc:.4f}",
    "F1_Score": f"{mean_f1:.4f} ± {std_f1:.4f}",
    "Train_Time(s)": f"{mean_train_time:.2f} ± {std_train_time:.2f}",
    "Eval_Time(s)": f"{mean_eval_time:.2f} ± {std_eval_time:.2f}"
}

# Append summary to the dataframe
df_final = pd.concat([df_results, pd.DataFrame([summary_row])], ignore_index=True)

# Save to Excel
df_final.to_excel(OUTPUT_EXCEL, index=False)
print(f"\n[SUCCESS] Results saved to {OUTPUT_EXCEL}")