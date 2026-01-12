import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import random
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import AutoModel, AutoTokenizer, Trainer, TrainingArguments
from transformers.modeling_outputs import SequenceClassifierOutput

# --- CONFIGURATION ---
MODEL_NAME = "allenai/scibert_scivocab_uncased"
FILE_PATH = "clean_dataset.csv"
OUTPUT_EXCEL = "hyperparameter_tuning_results.xlsx"
SEED = 42


# --- 1. THE CUSTOM MODEL (Allows Toggle Switching) ---
class TunableSciBERT(nn.Module):
    def __init__(self, model_name, num_classes=3, freeze_backbone=False, use_deep_head=False, dropout_rate=0.1):
        super(TunableSciBERT, self).__init__()

        self.bert = AutoModel.from_pretrained(model_name)

        # Toggle: Freeze Backbone
        if freeze_backbone:
            for param in self.bert.parameters():
                param.requires_grad = False

        self.use_deep_head = use_deep_head
        hidden_size = self.bert.config.hidden_size  # 768

        # Toggle: Deep Head vs Simple Head
        if use_deep_head:
            self.classifier = nn.Sequential(
                nn.Linear(hidden_size, 256),
                nn.ReLU(),
                nn.Dropout(dropout_rate),  # Tunable Dropout
                nn.Linear(256, num_classes)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_size, num_classes)
            )

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_token)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 3), labels.view(-1))

        return SequenceClassifierOutput(loss=loss, logits=logits)


# --- 2. SETUP ---
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


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
    return {'accuracy': acc}


# --- 3. EXPERIMENT LIST ---
EXPERIMENTS = [
    # 1. The Standard (Baseline)
    {"name": "Baseline (LR 2e-5)", "lr": 2e-5, "batch": 8, "freeze": False, "deep": False, "drop": 0.1},

    # 2. Hyperparameter Tuning (Learning Rate)
    {"name": "High LR (5e-5)", "lr": 5e-5, "batch": 8, "freeze": False, "deep": False, "drop": 0.1},
    {"name": "Low LR (1e-5)", "lr": 1e-5, "batch": 8, "freeze": False, "deep": False, "drop": 0.1},

    # 3. Hyperparameter Tuning (Batch Size)
    {"name": "Large Batch (16)", "lr": 2e-5, "batch": 16, "freeze": False, "deep": False, "drop": 0.1},

    # 4. Architecture Tuning (Deep Head)
    {"name": "Deep Head + Dropout", "lr": 2e-5, "batch": 8, "freeze": False, "deep": True, "drop": 0.3},

    # 5. The "Proof" (Frozen Backbone)
    {"name": "Frozen (Only Head)", "lr": 1e-3, "batch": 8, "freeze": True, "deep": False, "drop": 0.1},
]


# --- 4. THE RUNNER ---
def run_grid_search():
    set_seed(SEED)
    print(">>> LOADING DATA...")

    # Check if file exists relative to execution
    if not os.path.exists(FILE_PATH):
        raise FileNotFoundError(f"Could not find {FILE_PATH}. Make sure it is in the same folder as this script!")

    df = pd.read_csv(FILE_PATH)

    # Mapping
    if df['label_id'].dtype == 'object':
        label_map = {'Neuroscience': 0, 'Bioinformatics': 1, 'Materials Science': 2}
        df['label_id'] = df['label'].map(label_map)

    X = df['text'].tolist()
    y = df['label_id'].tolist()

    # 80/20 Split (Sufficient for Tuning)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_enc = tokenizer(X_train, truncation=True, padding=True, max_length=256)
    test_enc = tokenizer(X_test, truncation=True, padding=True, max_length=256)

    train_ds = ScienceDataset(train_enc, y_train)
    test_ds = ScienceDataset(test_enc, y_test)

    results_log = []

    print(f"\n>>> STARTING GRID SEARCH ({len(EXPERIMENTS)} Configs) <<<")

    for exp in EXPERIMENTS:
        print(f"\n--- Running: {exp['name']} ---")

        # Init Model with Config Settings
        model = TunableSciBERT(
            MODEL_NAME,
            num_classes=3,
            freeze_backbone=exp['freeze'],
            use_deep_head=exp['deep'],
            dropout_rate=exp['drop']
        )

        training_args = TrainingArguments(
            output_dir=f'./tuning_results/{exp["name"].replace(" ", "_")}',
            num_train_epochs=4,
            per_device_train_batch_size=exp['batch'],
            per_device_eval_batch_size=16,
            logging_steps=50,
            eval_strategy="epoch",  # <--- FIXED HERE (was evaluation_strategy)
            save_strategy="no",
            learning_rate=exp['lr'],
            seed=SEED
        )

        trainer = Trainer(
            model=model, args=training_args,
            train_dataset=train_ds, eval_dataset=test_ds,
            compute_metrics=compute_metrics
        )

        trainer.train()
        res = trainer.evaluate()

        print(f"--> Result: {res['eval_accuracy']:.4f}")

        results_log.append({
            "Experiment": exp['name'],
            "Accuracy": res['eval_accuracy'],
            "Learning Rate": exp['lr'],
            "Batch Size": exp['batch'],
            "Frozen": exp['freeze'],
            "Deep Head": exp['deep'],
            "Dropout": exp['drop']
        })

    # Save to Excel
    df_res = pd.DataFrame(results_log)
    df_res.to_excel(OUTPUT_EXCEL, index=False)
    print(f"\n[SUCCESS] Tuning results saved to {OUTPUT_EXCEL}")


if __name__ == "__main__":
    run_grid_search()