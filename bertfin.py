from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification, AutoConfig, set_seed)
from torch.utils.data import DataLoader
import torch
import random
import evaluate
from tqdm.auto import tqdm
import numpy as np

#######################################################
print("Loading trained model and tokenizer...")
model = AutoModelForTokenClassification.from_pretrained("model1")
tokenizer = AutoTokenizer.from_pretrained("model1")

path_train = "data/en_ewt-ud-train.iob2"
path_dev = "data/en_ewt-ud-dev.iob2"
path_test = "data/FIN3.txt"

####################################################

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


dev_sentences, dev_labels = parse_iob2_file(path_dev)
train_sentences, train_labels = parse_iob2_file(path_train)
all_labels = set(l for seq in train_labels + dev_labels for l in seq) # Collecting all unique labels
label_list = sorted(all_labels)
label2id = {l: i for i, l in enumerate(label_list)} # Converting labels to IDs, since pyTorch models work with integers
id2label = {i: l for l, i in label2id.items()} # Reverse mapping for ID to label, for evalutation later

# For hpc, move model to GPU if available, otherwise use CPU (computer)

if torch.cuda.is_available():
    device = "cuda"
    print('Moved model to GPU')
else:
    device = "cpu"
    print('Moved model to CPU')

model.to(device)

##############################################################################
# Getting predictions on test set
print("Getting predictions on test set...")
test_sentences, test_labels = parse_iob2_file(path_test)
model.eval()
predicted_labels = []

for tokens in tqdm(test_sentences, desc="Predicting on test set"):

    # Had to tokenize each sentence to get predictions for each sentence, since test set is masked and has no labels, so cannot use DataLoader with collate function like before
    # Also not using defined function since it is in different designn
    # No batch processing
    inputs = tokenizer(tokens, is_split_into_words=True, return_tensors="pt", truncation=True, max_length=128)
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
output_file = "predictions/test_predictions_fin3.txt"
with open(output_file, "w", encoding="utf-8") as f:
    for tokens, labels in zip(test_sentences, predicted_labels):
        for i, (token, label) in enumerate(zip(tokens, labels)):
            f.write(f"{i+1}\t{token}\t{label}\n")
        f.write("\n")

print(f"Predictions written to {output_file}")