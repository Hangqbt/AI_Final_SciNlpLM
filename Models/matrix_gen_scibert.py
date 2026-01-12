import pandas as pd
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from torch.utils.data import Dataset

# --- CONFIGURATION ---
MODEL_PATH = "./final_scibert_champion"  # This is the saved model from run_scibert.py
DATA_FILE = "arxiv_dataset.csv"  # We test on the Springer data (or change to arxiv_dataset.csv)
OUTPUT_IMAGE = "confusion_matrix_scibert.png"


# --- DATA SETUP ---
class ScienceDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self): return len(self.labels)


def main():
    print("Loading Model and Data...")

    # 1. Load Data
    df = pd.read_csv(DATA_FILE)
    # Ensure ID mapping matches training
    label_map = {'Neuroscience': 0, 'Bioinformatics': 1, 'Materials Science': 2}
    if df['label_id'].dtype == 'object':
        df['label_id'] = df['label'].map(label_map)

    X = df['text'].tolist()
    y_true = df['label_id'].tolist()

    # 2. Load Saved Model
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    except OSError:
        print(f"Error: Could not find model at {MODEL_PATH}. Did you run training yet?")
        return

    # 3. Inference
    print("Running Inference...")
    encodings = tokenizer(X, truncation=True, padding=True, max_length=256)
    dataset = ScienceDataset(encodings, y_true)

    trainer = Trainer(model=model)
    preds = trainer.predict(dataset)
    y_pred = preds.predictions.argmax(-1)

    # 4. Generate Confusion Matrix
    print("Generating Plot...")
    cm = confusion_matrix(y_true, y_pred)

    # Plotting
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=label_map.keys(),
        yticklabels=label_map.keys()
    )
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix: SciBERT on Springer Dataset', fontsize=14)

    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE, dpi=300)
    print(f"[SUCCESS] Saved Confusion Matrix to {OUTPUT_IMAGE}")
    plt.show()


if __name__ == "__main__":
    main()