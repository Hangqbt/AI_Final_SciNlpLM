import pandas as pd
import numpy as np
import torch
import random
import os
import time
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

# Here we initialize variables to determine the model that is going to be used,
# the datasets used for training adn testing and the name of the output file

MODEL_NAME = "allenai/scibert_scivocab_uncased"
# This model is taken from huggingface

#Springer is used for training due to its peer reviewed nature of the data
TRAIN_FILE = "clean_dataset.csv"
# Since arXiv contains preprint copies of texts, we use it as noisy data for testing
TEST_FILE = "arxiv_dataset.csv"
OUTPUT_EXCEL = "results_scibert_professional.xlsx"


#here the seed is fixed for controlled-randomness and finally the number of folds for the cross validation
N_FOLDS = 5  # Number of splits for Cross-Validation
SEED = 42  # Fixed seed for reproducibility


#Here, Similar to the baselines, we fix all the randomness present in the code and other libraries to ensure reproducibility
def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # These variables force pytorch to be strictly deterministic throughout all the runs. Even though it slows down the process a bit, it is often worth the trade off for determinism.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Here we also fix the python hash seed for maintaining dictionary ordering consistency
    os.environ['PYTHONHASHSEED'] = str(seed)


# we run the function to lock everything before executing anything else
set_seed(SEED)


#here we do the data loading
def load_data(path):
    """
    Here we read the CSV Files and make sure that they are integer coded for the classes
    """
    if not os.path.exists(path): raise FileNotFoundError(f"{path} not found!")
    df = pd.read_csv(path)

    # Here we ensure that the labels given are in integers which map to its each corresponding class
    if df['label_id'].dtype == 'object':
        label_map = {'Neuroscience': 0, 'Bioinformatics': 1, 'Materials Science': 2}
        df['label_id'] = df['label'].map(label_map)

    # Here we return lists of text and integer label ids
    return df['text'].tolist(), df['label_id'].tolist()


#Here we load the datasets
X_springer, y_springer = load_data(TRAIN_FILE)
X_arxiv, y_arxiv = load_data(TEST_FILE)


class ScienceDataset(torch.utils.data.Dataset):
    """
    Custom Dataset compatible with HuggingFace Trainer.
    This is basically a custom dataset that is compatible with huggingface models and trainers
    This expects encodings and labels
    """

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        # We basically convert dictionary of lists into dictionary of tensors here
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def compute_metrics(pred):
    """
    This is a helper function that helps the trainer calculate the accuracy and the F1 score of the model during evaluation.
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    # We convert the predicted logits into class IDs here
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='macro')
    # we macro average it here to get a balanced score
    return {'accuracy': acc, 'f1': f1}


#The following part contains all the experimental setup and process
results_log = []

# here we do the 5 fold cross validation on the springer dataset
print(f"\n>>> STARTING 5-FOLD CV ON SPRINGER DATA <<<")
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
fold_accuracies = []
fold_f1s = []

# we convert the springer datasets into numpy arrays for easy processing
X_np = np.array(X_springer)
y_np = np.array(y_springer)

for fold, (train_idx, val_idx) in enumerate(skf.split(X_np, y_np)):
    print(f"\n--- FOLD {fold + 1}/{N_FOLDS} ---")

    # we split them into training and testing sets here
    X_tr, X_val = X_np[train_idx], X_np[val_idx]
    y_tr, y_val = y_np[train_idx], y_np[val_idx]

    # We tokenize the data here
    # Also to note that SciBERT Generally handles and processes scientific notations and jargons much better than Traditional BERT
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_enc = tokenizer(list(X_tr), truncation=True, padding=True, max_length=256)
    val_enc = tokenizer(list(X_val), truncation=True, padding=True, max_length=256)


    train_ds = ScienceDataset(train_enc, y_tr)
    val_ds = ScienceDataset(val_enc, y_val)

    # We initialize the model here
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)

    # We define the training arguements and the hyperparameters here
    training_args = TrainingArguments(
        output_dir=f'./checkpoints/scibert_fold{fold}',
        num_train_epochs=3,  # We configure it with the standard default for BERT of 3 epochs
        per_device_train_batch_size=8,  # A safe batch size was selected based on my GPU for training
        per_device_eval_batch_size=16,
        logging_steps=50,
        eval_strategy="epoch",  # we evaluate the model at the end of every epoch
        save_strategy="epoch",  # We also save the model as a checkpoint at every epoch
        load_best_model_at_end=True,  # This is enabled to ensure that we would always load the best optimal version and not just the latest
        learning_rate=2e-5,  # This is the standard default learning rate given in the SciBERT Configuration baseline
        seed=SEED,
        data_seed=SEED  # Here we lock the dataloader shuffing and the model seed with the seed value
    )

    # We initialize the trainer here
    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=train_ds, eval_dataset=val_ds,
        compute_metrics=compute_metrics
    )

    # Here we train and evaluate the model
    train_res = trainer.train()
    eval_res = trainer.evaluate()

    # Finally we log the results
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

    # Here we save the best model that will be used for generalization testing later on
    if fold == 0:
        print("Saving Fold 1 model for Generalization Test...")
        model.save_pretrained("./final_scibert_champion")
        tokenizer.save_pretrained("./final_scibert_champion")

# Similar to the baselines, we test the model that was trained in the peer reviewed springer data on the noisy ArXiv
# dataset which usaully contains preprint copies of texts with different writing styles
print(f"\n>>> STARTING CROSS-DATASET GENERALIZATION (Springer -> arXiv) <<<")

# we load the best model here
# and we do not retrain since we want to see the generalization performance directly on unseen data
tokenizer = AutoTokenizer.from_pretrained("./final_scibert_champion")
model = AutoModelForSequenceClassification.from_pretrained("./final_scibert_champion", num_labels=3)

# Here we tokenize the ArXiv dataset
test_enc = tokenizer(X_arxiv, truncation=True, padding=True, max_length=256)
test_ds = ScienceDataset(test_enc, y_arxiv)

# We directly evaluate the model here without training
trainer = Trainer(model=model, compute_metrics=compute_metrics)

print("Evaluating on full arXiv dataset...")
start_eval = time.time()
preds = trainer.predict(test_ds)
eval_time = time.time() - start_eval

# We finally calculate the metrics post inference here
acc = accuracy_score(y_arxiv, preds.predictions.argmax(-1))
f1 = f1_score(y_arxiv, preds.predictions.argmax(-1), average='macro')

results_log.append({
    "Model": "SciBERT",
    "Experiment": "Generalization (arXiv)",
    "Accuracy": acc,
    "F1": f1,
    "Train Time (s)": "-",
    "Eval Time (s)": eval_time
})

print(f"Generalization Accuracy: {acc:.4f}")

# Finally we save the results here
df_res = pd.DataFrame(results_log)

# here we add the summary rows for the k fold metrics
summary = {
    "Model": "SciBERT",
    "Experiment": "K-Fold Mean ± Std",
    "Accuracy": f"{np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}",
    "F1": f"{np.mean(fold_f1s):.4f} ± {np.std(fold_f1s):.4f}",
    "Train Time (s)": "-",
    "Eval Time (s)": "-"
}

# we push all the results into an excel file to save them
df_res = pd.concat([df_res, pd.DataFrame([summary])], ignore_index=True)
df_res.to_excel(OUTPUT_EXCEL, index=False)
print(f"\n[DONE] SciBERT Results saved to {OUTPUT_EXCEL}")