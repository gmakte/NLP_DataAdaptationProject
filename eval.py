# eval.py
# ├── load_from_memory()
# ├── load_from_file()
# ├── normalize()
# ├── compute_metrics()
# ├── evaluate()

import evaluate
from util.preprocessing import parse_iob2_file
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
    preds, refs = align_labels(preds_ds, refs_ds)
    return preds, refs


def align_labels(preds_ds, refs_ds):
    preds = preds_ds["ner_tags"]
    refs = refs_ds["ner_tags"]
    pred_tokens = preds_ds["tokens"]
    ref_tokens = refs_ds["tokens"]

    aligned_preds = []
    aligned_refs = []

    for i, (p_tok, r_tok, p_lab, r_lab) in enumerate(
        zip(pred_tokens, ref_tokens, preds, refs)
    ):
        if len(p_tok) == len(r_tok):
            aligned_preds.append(p_lab)
            aligned_refs.append(r_lab)

        elif len(p_tok) < len(r_tok):
            print(f"\nMismatch at sentence {i}")
            if p_tok == r_tok[:len(p_tok)]:
                print(f"Truncated reference tokens from {len(r_tok)} to {len(p_tok)}")
                aligned_preds.append(p_lab)
                aligned_refs.append(r_lab[:len(p_tok)])
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
    return aligned_preds, aligned_refs


def compute_metrics(preds, refs):
    print("\nComputing metrics...")
    metric = evaluate.load("seqeval")
    results = metric.compute(predictions=preds, references=refs)
    print("Metrics computed.")
    return results
    

def evaluate_predictions(preds, refs, from_file=True):

    # 1. Load data
    preds, refs = load_and_align(preds, refs, from_file=from_file)

    # 2. Compute metrics
    results = compute_metrics(preds, refs)

    print(f"Results: {results}")
    return results


evaluate_predictions(preds="./predictions/test_predictions_fin3.txt", refs="./data/FIN3.txt", from_file=True)