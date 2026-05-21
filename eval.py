# eval.py
# ├── load_from_memory()
# ├── load_from_file()
# ├── normalize()
# ├── compute_metrics()
# ├── evaluate()

import evaluate
from util.preprocessing import parse_iob2_file
from util.span_f1 import compute_span_f1
from util.entity_metrics import entity_metrics
from config.labels import id2label, label2id
from datasets import Dataset


# if predictions are saved to disk, load them in the correct format for evaluation
# if predictions are already in memory as Hugging Face Datasets, load them directly
def load_and_align(preds: Dataset, refs: Dataset, from_file=False):
    
    if from_file:
        print("Reading files...")
        pred_tokens, pred_labels = parse_iob2_file(preds)
        preds_ds = Dataset.from_dict({"ner_tags": pred_labels, "tokens": pred_tokens})

        ref_tokens, ref_labels = parse_iob2_file(refs)
        refs_ds = Dataset.from_dict({"ner_tags": ref_labels, "tokens": ref_tokens})
    
    else:
        print("Loading from memory...")
        assert isinstance(preds, Dataset) and isinstance(refs, Dataset), "Expected Hugging Face Dataset for in-memory loading"
        pred_labels, ref_labels = preds["ner_tags"], refs["ner_tags"]

    assert len(preds_ds) == len(refs_ds), "Predictions and references must match"

    # ensure nested structure
    if isinstance(pred_labels[0], str):
        raise ValueError("Expected list of lists, got flat list")
    
    # convert from label ids to label strings if necessary
    if isinstance(pred_labels[0][0], int):
        preds_ds["ner_tags"] = [[id2label[id] for id in seq] for seq in pred_labels]
    
    if isinstance(ref_labels[0][0], int):
        refs_ds["ner_tags"] = [[id2label[id] for id in seq] for seq in ref_labels]

    print("Aligning predictions and references...")
    tokens, preds, refs = align_labels(preds_ds, refs_ds)
    return tokens, preds, refs


def align_labels(preds_ds, refs_ds):
    preds = preds_ds["ner_tags"]
    refs = refs_ds["ner_tags"]
    pred_tokens = preds_ds["tokens"]
    ref_tokens = refs_ds["tokens"]

    tokens = []
    aligned_preds = []
    aligned_refs = []

    for i, (p_tok, r_tok, p_lab, r_lab) in enumerate(
        zip(pred_tokens, ref_tokens, preds, refs)
    ):
        if len(p_tok) == len(r_tok): # perfect match, append the entire list of labels for the sentence
            tokens.append(r_tok)
            aligned_preds.append(p_lab)
            aligned_refs.append(r_lab)

        elif len(p_tok) < len(r_tok):
            print(f"\nMismatch at sentence {i}")
            if p_tok == r_tok[:len(p_tok)]: # keep only the first k labels for the reference sentence, where k is the length of the predicted sentence
                print(f"Truncated reference tokens from {len(r_tok)} to {len(p_tok)}")
                aligned_preds.append(p_lab)
                aligned_refs.append(r_lab[:len(p_tok)])
                tokens.append(r_tok)
            else:
                raise ValueError(
                    f"Mismatch not at end for sentence {i}:\n"
                    f"Pred tokens: {p_tok}\nRef tokens: {r_tok}"
                )

        else:
            raise ValueError(
                f"Prediction longer than reference at sentence {i}:\n"
                f"Pred tokens: {p_tok}\nRef tokens: {r_tok}"
            )

    print("\nAlignment complete.")
    return tokens, aligned_preds, aligned_refs # preds and refs are lists of lists of label strings, aligned at the sentence level


def write_qualitative_predictions(tokens, gold_labels, pred_labels, output_path):
    with open(output_path, "w", encoding="utf-8") as f:

        for sent_tokens, sent_gold, sent_pred in zip(
            tokens, 
            gold_labels, 
            pred_labels
        ):

            f.write(f"\n")

            for token_idx, (token, gold, pred) in enumerate(
                zip(sent_tokens, sent_gold, sent_pred)
            ):

                f.write(
                    f"{token_idx}\t{token}\t{gold}\t{pred}\n"
                )

            f.write("\n")

        print(f"Aligned gold labels and predictions saved to {output_path}")


def compute_metrics(preds, refs):
    print("\nComputing metrics...")
    metric = evaluate.load("seqeval") #seqeval expects nested sentence structure, using sentence boundaries internally for entity span reconstruction
    results = metric.compute(predictions=preds, references=refs)
    print("Metrics computed.")
    return results
    

def evaluate_predictions(preds, refs, from_file=True, qualitative_output_path = None):

    # 1. Load data
    tokens, preds, refs = load_and_align(preds, refs, from_file=from_file)
    if qualitative_output_path is not None:
        write_qualitative_predictions(tokens, refs, preds, qualitative_output_path)

    # 2. Compute metrics
    seqeval_results = compute_metrics(preds, refs)
    span_results = compute_span_f1(gold_ners=refs, pred_ners=preds)
    cm = entity_metrics(all_gold_tags=refs, all_pred_tags=preds)

    results = {
        "seqeval": seqeval_results,
        "span_metrics": span_results,
        "entity_metrics": cm
    }

    return results