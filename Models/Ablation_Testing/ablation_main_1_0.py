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


## Here we perform ablation testing,
#This study is done to compare the sciBERT model's performance under various condition and configurations.

#Why?, It is to see whether if the SciBERT model is performing well based on its pre-trained weights and configurations
# or whether our datasets make an actual improvement in its prediction accuracy of scientific tests,
# Additionally, it helps us prove that features like self attention are essential for understanding contexts beyond just
# frequency (aka probability of the word appearing again in a similar format)



# Here we define the model, the file paths for the dataset and the output along with the seed for fixing randomness
MODEL_NAME = "allenai/scibert_scivocab_uncased"
FILE_PATH = "clean_dataset.csv"
OUTPUT_EXCEL = "hyperparameter_tuning_results.xlsx"
SEED = 42


#Here we build a class that we can use to fine tune the model under multiple configurations and run automated tests in the
# dataset to see the effects of its different features.

class TunableSciBERT(nn.Module):
    """
    This wrapper basically allows us to do architectural ablation

    So in this code, we currently do,

    - Backbone freezing (to test and compare feature extraction and finetuning)
    - Head complexity (whether having a deep layer would make a difference or if a simple linear layer would suffice)
    - Dropout rates (This is regularization tuning)
    """

    def __init__(self, model_name, num_classes=3, freeze_backbone=False, use_deep_head=False, dropout_rate=0.1):
        super(TunableSciBERT, self).__init__()

       # We load the pretrained scibert model
        self.bert = AutoModel.from_pretrained(model_name)

        #Here we freeze the backbone. Essentially this freezes the weight training making it static then it only trains the classification head
        # This allows us to see if the pretrained knowledge alone is enough to do the classification
        if freeze_backbone:
            for param in self.bert.parameters():
                param.requires_grad = False

        self.use_deep_head = use_deep_head
        hidden_size = self.bert.config.hidden_size

        # here we modify the classification head architecture
        if use_deep_head:
            #here we use a deeper head with a hidden layer, then an activation, then a high dropout layer then we have an output layer
            # Complex Head: Adds a hidden layer + non-linearity + higher dropout
            # Tests if more capacity is needed for classification
            self.classifier = nn.Sequential(
                nn.Linear(hidden_size, 256),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(256, num_classes)
            )
        else:
           # This is a simple linear layer configuration with a dropout layer
            self.classifier = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_size, num_classes)
            )

    def forward(self, input_ids, attention_mask=None, labels=None):
        # We pass the inputs into the model here
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # 2. Here we extract the [CLS] Token
        cls_token = outputs.last_hidden_state[:, 0, :]

        # 3. We pass our custom classifier here
        logits = self.classifier(cls_token)

        # 4. Here we calculate the loss
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 3), labels.view(-1))

        # Here we return the values in a HuggingFace format so that the model can provide it easily
        return SequenceClassifierOutput(loss=loss, logits=logits)


# Below are additional utility functions #

def set_seed(seed):
    # This function essentially takes the seed value and fixes all the randomness present in the code and the other supporting functions.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# This is a standard class for wrapping tokenized texts
class ScienceDataset(torch.utils.data.Dataset):


    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self): return len(self.labels)

# we compute the metrics for the trainer
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc}


#This is the heart of the testing code. We have the different configurations that were used to automatically test the model

# We go through this list to test out each of the conditions mentioned above
EXPERIMENTS = [
    # Condition 1: This is the baseline model which the main SciBERT used
    {"name": "Baseline (LR 2e-5)", "lr": 2e-5, "batch": 8, "freeze": False, "deep": False, "drop": 0.1},

    # Condition 2: Here we apply different learning rates to see what changes it makes to the models
    {"name": "High LR (5e-5)", "lr": 5e-5, "batch": 8, "freeze": False, "deep": False, "drop": 0.1},
    {"name": "Low LR (1e-5)", "lr": 1e-5, "batch": 8, "freeze": False, "deep": False, "drop": 0.1},

    # Condition 3: We test and see if having a bigger batch size makes any difference in performance.
    {"name": "Large Batch (16)", "lr": 2e-5, "batch": 16, "freeze": False, "deep": False, "drop": 0.1},

    # Condition 4: We see if having a deeper architecture aids in the separation and classification of the data
    {"name": "Deep Head + Dropout", "lr": 2e-5, "batch": 8, "freeze": False, "deep": True, "drop": 0.3},

    # Condition 5: We freeze the weights here to test and highlight the importance of finetuning
    # If the model performs poorly under this configuration, then it proves that fine-tuning plays a major part in classification.
    {"name": "Frozen (Only Head)", "lr": 1e-3, "batch": 8, "freeze": True, "deep": False, "drop": 0.1},
]


# This Function essentially runs all the experiments
def run_grid_search():
    set_seed(SEED)
    print(">>> LOADING DATA...")

    if not os.path.exists(FILE_PATH):
        raise FileNotFoundError(f"Could not find {FILE_PATH}. Make sure it is in the same folder as this script!")

    df = pd.read_csv(FILE_PATH)

    # We encode the labels and map them to Integer indexes here
    if df['label_id'].dtype == 'object':
        label_map = {'Neuroscience': 0, 'Bioinformatics': 1, 'Materials Science': 2}
        df['label_id'] = df['label'].map(label_map)

    X = df['text'].tolist()
    y = df['label_id'].tolist()


    # Here we don't perform 5 fold validation and instead opted to an 80/20 split validation approach to save time.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

    # We perform tokenization here
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_enc = tokenizer(X_train, truncation=True, padding=True, max_length=256)
    test_enc = tokenizer(X_test, truncation=True, padding=True, max_length=256)

    train_ds = ScienceDataset(train_enc, y_train)
    test_ds = ScienceDataset(test_enc, y_test)

    results_log = []

    print(f"\n>>> STARTING GRID SEARCH ({len(EXPERIMENTS)} Configs) <<<")

   # This is the main loop that runs the experiments automatically
    for exp in EXPERIMENTS:
        print(f"\n--- Running: {exp['name']} ---")

        # We initialize the custom model with our specific flags here
        model = TunableSciBERT(
            MODEL_NAME,
            num_classes=3,
            freeze_backbone=exp['freeze'],  # Toggle Freezing
            use_deep_head=exp['deep'],  # Toggle Architecture
            dropout_rate=exp['drop']  # Toggle Dropout
        )

        # Since the hyperparameters change every trial, we initialize them dynamically
        training_args = TrainingArguments(
            output_dir=f'./tuning_results/{exp["name"].replace(" ", "_")}',
            num_train_epochs=4,
            per_device_train_batch_size=exp['batch'],
            per_device_eval_batch_size=16,
            logging_steps=50,
            eval_strategy="epoch",
            save_strategy="no",
            learning_rate=exp['lr'],
            seed=SEED
        )

        trainer = Trainer(
            model=model, args=training_args,
            train_dataset=train_ds, eval_dataset=test_ds,
            compute_metrics=compute_metrics
        )

        # We finally train and evaluate the model under different conditions here
        trainer.train()
        res = trainer.evaluate()

        print(f"--> Result: {res['eval_accuracy']:.4f}")

        # Here we log the data
        results_log.append({
            "Experiment": exp['name'],
            "Accuracy": res['eval_accuracy'],
            "Learning Rate": exp['lr'],
            "Batch Size": exp['batch'],
            "Frozen": exp['freeze'],
            "Deep Head": exp['deep'],
            "Dropout": exp['drop']
        })

    # We export all the results of the ablation testing to an excel document
    df_res = pd.DataFrame(results_log)
    df_res.to_excel(OUTPUT_EXCEL, index=False)
    print(f"\n[SUCCESS] Tuning results saved to {OUTPUT_EXCEL}")


if __name__ == "__main__":
    run_grid_search()