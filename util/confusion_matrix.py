import sys
from sklearn import metrics
from collections import Counter
from util.span_f1 import toSpans

############ GET PREDICTION PAIRS AND COMPUTE CONFUSION MATRIX ################

# function for loose overlap
def overlaps(span1, span2):

    beg1, end1, _ = span1
    beg2, end2, _ = span2

    return beg1 < end2 and beg2 < end1


#functio to measure the size of the overlap between two spans
def overlapSize(span1, span2):
    beg1, end1, _ = span1
    beg2, end2, _ = span2

    overlap_start = max(beg1, beg2)
    overlap_end = min(end1, end2)

    return max(0, overlap_end - overlap_start)


# find the best (strict or loose) match for a given golden span
def findBestMatch(gold_span, pred_spans, matched_predictions_strict, matched_predictions_loose):

    gold_beg, gold_end, _ = gold_span

    strict_match_idx = None

    best_loose_match_idx = None
    best_overlap = 0

    for pred_idx, pred_span in enumerate(pred_spans):

        pred_beg, pred_end, _ = pred_span

        # STRICT MATCH - 1st priority

        if (
            pred_idx not in matched_predictions_strict
            and gold_beg == pred_beg
            and gold_end == pred_end
        ):

            strict_match_idx = pred_idx
            break # an exact match for this span has been found, no need to keep searching

        # LOOSE MATCH - 2nd priority

        if pred_idx in matched_predictions_loose:
            continue # if the prediction is already used in another pair, ignore it

        overlap = overlapSize(gold_span, pred_span)

        if overlap > best_overlap:

            best_overlap = overlap
            best_loose_match_idx = pred_idx

    return strict_match_idx, best_loose_match_idx


# after assigning all gold labels to a pred label, check if there are remaining pred labels that are not associated with a gold label
def addFalsePositives(pred_spans, matched_predictions):

    all_predictions = set(range(len(pred_spans)))

    unmatched_predictions = all_predictions - matched_predictions

    fp_pairs = []

    for pred_idx in unmatched_predictions:

        _, _, pred_label = pred_spans[pred_idx]

        fp_pairs.append(("O", pred_label))

    return fp_pairs


# get pairs of (gold_label, pred_label) both in the strict and loose interpretation to compute the confusion matrix later
def getConfusions(gold_spans, pred_spans):

    strict_pairs = []
    loose_pairs = []

    matched_predictions_strict = set()
    matched_predictions_loose = set()

    for gold_span in gold_spans:

        _, _, gold_label = gold_span

        strict_match_idx, best_loose_match_idx = findBestMatch(gold_span, pred_spans, matched_predictions_strict, matched_predictions_loose)

        # after finding the best match for the current gold label
        if strict_match_idx is not None:
            # if there is a strict match, it counts for both strict and loose metrics

            # mark prediction label as used to avoid overcounting later
            matched_predictions_strict.add(strict_match_idx)
            matched_predictions_loose.add(strict_match_idx)

            _, _, pred_label = pred_spans[strict_match_idx]

            strict_pairs.append((gold_label, pred_label))
            loose_pairs.append((gold_label, pred_label))

        elif best_loose_match_idx is not None:
            matched_predictions_loose.add(best_loose_match_idx)
            _, _, pred_label = pred_spans[best_loose_match_idx]

            strict_pairs.append((gold_label, "O")) # for strict matches, this loose match counts as a miss
            loose_pairs.append((gold_label, pred_label))

        else: #count fn
            strict_pairs.append((gold_label, "O"))
            loose_pairs.append((gold_label, "O"))

    # FALSE POSITIVES
    fp_pairs = addFalsePositives(pred_spans, matched_predictions_loose) # counts both for strict and for loose because we want to catch real NER hallucinations
    strict_pairs.extend(fp_pairs)
    loose_pairs.extend(fp_pairs)

    return strict_pairs, loose_pairs


# takes a list of pairs [(gold_label1, pred_label1), (gold_label2, pred_label2), ...] and returns a confusion matrix
def pairstoConfusionMatrix(pairs): 
    y_true = []
    y_pred = []

    for gold, pred in pairs:
        y_true.append(gold)
        y_pred.append(pred)

    #compute metrics 
    labels = ["PER", "LOC", "ORG", "O"]
    raw_support = Counter(y_true)
    support = {label: raw_support.get(label, 0) for label in labels}

    confusion_matrix = metrics.confusion_matrix(y_true, y_pred, labels=labels, normalize="true")

    return {"confusion_matrix": confusion_matrix.tolist(), "support": support}


# create a confusion metrix both for strict and loose span
def confusion_matrix(all_gold_tags, all_pred_tags):

    all_strict_pairs = []
    all_loose_pairs = []

    for gold_tags, pred_tags in zip(all_gold_tags, all_pred_tags):

        #transform a list of BIO tags into a list of spans in the form (beg_idx, end_idx, label)
        gold_spans = toSpans(gold_tags)
        pred_spans = toSpans(pred_tags)
        
        strict_pairs, loose_pairs = getConfusions(gold_spans, pred_spans) #for each span, create a pair of gold label and pred label

        all_strict_pairs.extend(strict_pairs)
        all_loose_pairs.extend(loose_pairs)

    strict_results = pairstoConfusionMatrix(all_strict_pairs)
    loose_results = pairstoConfusionMatrix(all_loose_pairs)

    return {"strict": strict_results, "loose": loose_results}