
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification, AutoConfig, set_seed)
from torch.utils.data import DataLoader
import torch
import random
import evaluate
from tqdm.auto import tqdm
import numpy as np

# Set up

path_train = "en_ewt-ud-train.iob2"
path_dev = "en_ewt-ud-dev.iob2"

model_name = "model1"
batch_size = 12

# Dataset parsing 
def parse_iob2_file(filepath):
    sentences = []
    labels = []
    tokens = []
    tags = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                if tokens: #tokens is usually empty when the line starts with # 
                    sentences.append(tokens)
                    labels.append(tags)
                    tokens = []
                    tags = []
                continue
            splits = line.split("\t")
            tokens.append(splits[1])
            tags.append(splits[2])

    return sentences, labels

# Label to id conversion 
def labels_to_ids(labels, label2id): #labels is a nested list where labels[i][j] is the label for the j-th token in the i-th sentence
    L = []
    for seq in labels:
        seq_ids = []
        for tag in seq:
            if tag in label2id:
                seq_ids.append(label2id[tag])
            else:
                raise ValueError(f"{tag} not found")
        L.append(seq_ids)

    return L

print("Parsing datasets...")
# Parsing the datasets
dev_sentences, dev_labels = parse_iob2_file(path_dev)
train_sentences, train_labels = parse_iob2_file(path_train)

# Build label list and label2id/id2label mappings
all_labels = set(l for seq in train_labels + dev_labels for l in seq) # Collecting all unique labels
label_list = sorted(all_labels)
label2id = {l: i for i, l in enumerate(label_list)} # Converting labels to IDs, since pyTorch models work with integers
id2label = {i: l for l, i in label2id.items()} # Reverse mapping for ID to label, for evalutation later

dev_label_ids = labels_to_ids(dev_labels, label2id)
dev_dataset = Dataset.from_dict({"tokens": dev_sentences, "ner_tags": dev_label_ids})

# Tokenization and alignment

def tokenize_and_align_labels(examples): #parse some rows of the dataset above
    tokenized_inputs = tokenizer(
        examples["tokens"], # Lists of tokens for each sentence, e.g. [["EU", "rejects", "Germany", "call", "to", "boycott", "British", "lamb", "."]]
        max_length=128, #  Limits the total number of tokens (including special tokens) to 128. Longer sequences are truncated.
        padding=False, # All sentences keeps their original length, not making all sentences the same length
        truncation=True, # If sequence is longer than max_length, it will be truncated to fit the model's input size.
        is_split_into_words=True # Indicates that the input is already split into words
    )
    
    all_labels = []
    for i, labels in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i) #this maps each subword token back to the original word index, so we can align the labels correctly
        label_ids = []
        prev_word_id = None
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            elif word_id == prev_word_id:
                label_ids.append(-100)
            else:
                label_ids.append(labels[word_id])
            prev_word_id = word_id
        all_labels.append(label_ids)
    tokenized_inputs["labels"] = all_labels
    return tokenized_inputs

# Loading tokenizer, model configuration and data collator
print("Loading model and tokenizer...")
model = AutoModelForTokenClassification.from_pretrained("model1") #loads an NER model and automatically builds the correct architecture based on config.json
tokenizer = AutoTokenizer.from_pretrained("model1") #loads the tokenizer used during training and converts text to token IDs that the model can understand
data_collator = DataCollatorForTokenClassification(tokenizer) # used to dynamically pad the inputs and labels to the maximum length in a batch during training and evaluation, so all sequences in a batch have the same length 

# Move model to device (CPU/GPU)
if torch.cuda.is_available():
    device = "cuda"
    print('Moved model to GPU')
else:
    device = "cpu"
    print('Moved model to CPU')

model.to(device)

# Tokenizing datasets
print("Tokenizing datasets...")
tokenized_dev = dev_dataset.map(tokenize_and_align_labels, batched=True, remove_columns=dev_dataset.column_names)

# Creating DataLoader objects for training and development datasets, needed for pyTorch training loop
print("Creating DataLoader objects...")
dev_dataloader = DataLoader(tokenized_dev, collate_fn=data_collator, batch_size=batch_size)

# Evaluation on development set

metric = evaluate.load("seqeval")

def get_labels(predictions, references):
    true_predictions = []
    true_labels = []
    # For evaluation data needs to be in numpy format
    # PyTorch only allows conversion to numpy arrays when the tensor is on the CPU, so if used hpc before move to cpu now
    predictions = predictions.cpu().numpy() if hasattr(predictions, 'cpu') else np.array(predictions)
    # Ground truth
    references = references.cpu().numpy() if hasattr(references, 'cpu') else np.array(references)

    for pred, ref in zip(predictions, references):
        # Sentence-level predictions and labels
        pred_labels = []
        true_ref = []
        for p, r in zip(pred, ref):
        # Only consider non-subword tokens (those with label ID not equal to -100)
            if r != -100:
                # Convert prediction and labels back to human readable format using id2label mapping
                pred_labels.append(id2label[p])
                true_ref.append(id2label[r])
        true_predictions.append(pred_labels)
        true_labels.append(true_ref)
    return true_predictions, true_labels

def compute_metrics(preds, refs):
    results = metric.compute(predictions=preds, references=refs)
    return {
        "Precision": results["overall_precision"],
        "Recall": results["overall_recall"],
        "F1": results["overall_f1"],
        "Accuracy": results["overall_accuracy"],
    }

### Evaluation loop
print("Evaluating on development set...")
model.eval()
validation_progress_bar = tqdm(range(len(dev_dataloader)), desc="Evaluating")
# Storage
all_predictions = []
all_labels = []

for step, batch in enumerate(dev_dataloader):
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad(): # No gradient calculation during evaluation, for efficiency
        outputs = model(**batch)
    predictions = outputs.logits.argmax(dim=-1) # Get predicted label IDs by taking the index of the highest logit for each token
    labels = batch["labels"] # Ground truth labels
    predicted_labels, true_labels = get_labels(predictions, labels) # Convert predicted and true label IDs to human-readable labels, while filtering out subword tokens
    all_predictions.extend(predicted_labels)
    all_labels.extend(true_labels)
    validation_progress_bar.update(1)

validation_metrics = compute_metrics(all_predictions, all_labels)
print("Validation metrics:", validation_metrics)

# Saving metrics to a txt file
with open("dev_evaluation_metrics.txt", "w", encoding="utf-8") as f:
    for metric, value in validation_metrics.items():
        f.write(f"{metric}: {value}\n")
print("Val dev metrics saved to dev_evaluation_metrics.txt")