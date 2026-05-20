
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification, AutoConfig, set_seed)
from torch.utils.data import DataLoader
import torch
import argparse
# import random
# import evaluate
from tqdm.auto import tqdm
# import numpy as np

#######################################################
# Script contains: dataset preprocessing, tokenization and alignment, training loop and getting predictions on test set

only_trainning = True # train on training data and predict on test set

# Set up - parse command-line arguments for required values
parser = argparse.ArgumentParser(description="Train a token-classification model (NER)")
# Required positional args per user request
parser.add_argument("path_train", help="Path to training IOB2 file")
parser.add_argument("model_dir", help="Directory to save the trained model and tokenizer")
parser.add_argument("max_length", type=int, help="Maximum tokenization length (int)")
# Optional overrides with sensible defaults
parser.add_argument("--path_dev", default="data/FIN5_dev.txt", help="Path to development IOB2 file")
parser.add_argument("--path_test", default="data/FIN3_fixed.txt", help="Path to test IOB2 file")
parser.add_argument("--model_name", default="google-bert/bert-base-cased", help="Pretrained model name")
parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
parser.add_argument("--num_train_epochs", type=int, default=8, help="Number of training epochs")
parser.add_argument("--batch_size", type=int, default=15, help="Batch size")

args = parser.parse_args()

# Assign variables from parsed args
path_train = args.path_train
path_dev = args.path_dev
path_test = args.path_test

model_dir = args.model_dir
model_name = args.model_name

learning_rate = args.learning_rate
num_train_epochs = args.num_train_epochs
batch_size = args.batch_size
max_length = args.max_length

set_seed(42)

####################################################
# Dataset preprocessing

### Needed functions

# Dataset parsing function
def parse_iob2_file(filepath):
    sentences = []
    labels = []
    tokens = []
    tags = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                if tokens:
                    sentences.append(tokens)
                    labels.append(tags)
                    tokens = []
                    tags = []
                continue
            splits = line.split("\t")
            tokens.append(splits[1])
            tags.append(splits[2])

    return sentences, labels

# Label to id conversion function
def labels_to_ids(labels, label2id):
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

### Dataset loading
print("Parsing datasets...")
# Parsing the datasets
dev_sentences, dev_labels = parse_iob2_file(path_dev)
train_sentences, train_labels = parse_iob2_file(path_train)

# Build label list and label2id/id2label mappings
all_labels = set(l for seq in train_labels + dev_labels for l in seq) # Collecting all unique labels
label_list = sorted(all_labels)
label2id = {l: i for i, l in enumerate(label_list)} # Converting labels to IDs, since pyTorch models work with integers
id2label = {i: l for l, i in label2id.items()} # Reverse mapping for ID to label, for evalutation later

train_label_ids = labels_to_ids(train_labels, label2id)
dev_label_ids = labels_to_ids(dev_labels, label2id)

# Making dataset compatible with Hugging Face
train_dataset = Dataset.from_dict({"tokens": train_sentences, "ner_tags": train_label_ids})
dev_dataset = Dataset.from_dict({"tokens": dev_sentences, "ner_tags": dev_label_ids})

####################################################
# Tokenization and alignment

### Needed functions

# Function to tokenize dataset into subwords and align labels with those

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], # Lists of tokens for each sentence, e.g. [["EU", "rejects", "Germany", "call", "to", "boycott", "British", "lamb", "."]]
        max_length=max_length, #  Limits the total number of tokens (including special tokens) to 128. Longer sequences are truncated.
        padding=False, # All sentences keeps their original length, not making all sentences the same length
        truncation=True, # If sequence is longer than max_length, it will be truncated to fit the model's input size.
        is_split_into_words=True # Indicates that the input is already split into words
    )
    
    all_labels = []
    for i, labels in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
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

### Tokenization

# Loading tokenizer, model configuration and data collator
print(f"Loading {model_name} model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True) # Tool that prepares text for BERT, using faster implementation
config = AutoConfig.from_pretrained(
    model_name,
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id,
    # hidden_dropout_prob=0.2,  # Dropout in hidden layers
    # attention_probs_dropout_prob=0.2  # Dropout in attention layers
) # Loads configuration for pretrained model, setting how many NER labels there are, provided conext with rest
data_collator = DataCollatorForTokenClassification(tokenizer)

# Tokenizing datasets

tokenized_train = train_dataset.map(tokenize_and_align_labels, batched=True, remove_columns=train_dataset.column_names)
tokenized_dev = dev_dataset.map(tokenize_and_align_labels, batched=True, remove_columns=dev_dataset.column_names)

# Creating DataLoader objects for training and development datasets, needed for pyTorch training loop
train_dataloader = DataLoader(tokenized_train, shuffle=True, collate_fn=data_collator, batch_size=batch_size)
dev_dataloader = DataLoader(tokenized_dev, collate_fn=data_collator, batch_size=batch_size)


# Initialize the model

model = AutoModelForTokenClassification.from_pretrained(model_name, config=config)

# For hpc, move model to GPU if available, otherwise use CPU (computer)

if torch.cuda.is_available():
    device = "cuda"
    print('Moved model to GPU')
else:
    device = "cpu"
    print('Moved model to CPU')

model.to(device)

# Optimizer = Adamw (specialized for transformers)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

####################################################
# Training loop
print("Starting training...")

# Track losses for each epoch
epoch_losses = {}

model.train()
for epoch in range(num_train_epochs):
    total_loss = 0
    # Showing progress bar with tqdm
    pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Training Epoch {epoch+1}")
    for step, batch in pbar:
        # Move batch to device, for hpc
        batch = {k: v.to(device) for k, v in batch.items()}
        # Zero gradients each batch
        optimizer.zero_grad()
        # Forward pass
        outputs = model(**batch) # Predictions
        loss = outputs.loss # Cross entropy loss
        # Backward pass
        loss.backward() # Compute gradients
        # Update parameters
        optimizer.step() # Update model parameters with ADAM 
        # Track loss
        total_loss += loss.item()
        pbar.set_postfix({"loss": loss.item()})
    avg_loss = total_loss / len(train_dataloader)
    # Track average loss per epoch
    epoch_losses[epoch + 1] = avg_loss
    print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")


if only_trainning:
    # Save training losses to a text file
    with open("results/training_losses_FIN5full_256.txt", "w") as f:
        f.write(str(epoch_losses))
    # Save the trained model and tokenizer in model3 folder (gitignore since it is huge)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    print(f"Model and tokenizer saved to {model_dir} folder")
    print("Training complete, skipping predictions on test set since only_trainning is set to True")

##############################################################################
# Getting predictions on test set
if not only_trainning:
    print("Getting predictions on test set...")
    test_sentences, test_labels = parse_iob2_file(path_test)
    model.eval()
    predicted_labels = []

    for tokens in tqdm(test_sentences, desc="Predicting on test set"):

        # Had to tokenize each sentence to get predictions for each sentence, since test set is masked and has no labels, so cannot use DataLoader with collate function like before
        # Also not using defined function since it is in different designn
        # No batch processing
        inputs = tokenizer(tokens, is_split_into_words=True, return_tensors="pt", truncation=True, max_length=max_length)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        predictions = outputs.logits.argmax(dim=-1).squeeze().cpu().numpy()
        word_ids = inputs['input_ids'].squeeze().cpu().numpy()

        # Aligning predictions to original tokens
        word_ids_map = inputs['input_ids'].squeeze().tolist()
        token_predictions = []
        word_idx = 0
        for i, token_id in enumerate(inputs['input_ids'].squeeze()):
            if tokenizer.convert_ids_to_tokens(int(token_id)).startswith("##") or token_id in [tokenizer.cls_token_id, tokenizer.sep_token_id]:
                continue
            token_predictions.append(predictions[i])
            word_idx += 1
            if word_idx == len(tokens):
                break

        # Label ids to label names
        pred_labels = [id2label[pred] for pred in token_predictions]
        predicted_labels.append(pred_labels)

    # Save to txt
    output_file = "predictions/test_predictions.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        for tokens, labels in zip(test_sentences, predicted_labels):
            for i, (token, label) in enumerate(zip(tokens, labels)):
                f.write(f"{i+1}\t{token}\t{label}\n")
            f.write("\n")

    print(f"Predictions written to {output_file}")
