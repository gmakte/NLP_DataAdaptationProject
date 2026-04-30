from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

input_path = "data/FIN3.txt"
output_path = "predictions/test_predictions_fin3_aligned.txt"
model_dir = "model1"

model = AutoModelForTokenClassification.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

def read_sentences(filepath):
    sentences = []
    sentence = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip() == "":
                if sentence:
                    sentences.append(sentence)
                    sentence = []
            else:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    sentence.append((parts[0], parts[1]))
        if sentence:
            sentences.append(sentence)
    return sentences

def predict_labels(sentences):
    all_preds = []
    for sentence in sentences:
        tokens = [tok for _, tok in sentence]
        encoding = tokenizer(tokens, is_split_into_words=True, return_tensors="pt", truncation=True, max_length=128)
        inputs = {k: v.to(model.device) for k, v in encoding.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        predictions = outputs.logits.argmax(dim=-1).squeeze().cpu().numpy()
        word_ids = encoding.word_ids(batch_index=0)
        pred_labels = []
        prev_word_idx = None
        for i, word_idx in enumerate(word_ids):
            if word_idx is None:
                continue
            if word_idx != prev_word_idx:
                pred_labels.append(model.config.id2label[int(predictions[i])])
            prev_word_idx = word_idx
        all_preds.append(pred_labels)
    return all_preds

sentences = read_sentences(input_path)
predicted_labels = predict_labels(sentences)

with open(output_path, "w", encoding="utf-8") as f:
    for sentence, labels in zip(sentences, predicted_labels):
        for (num, token), label in zip(sentence, labels):
            f.write(f"{num}\t{token}\t{label}\n")
        f.write("\n")

print(f"Predictions written to {output_path}")
