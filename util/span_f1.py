import sys
from sklearn import metrics
from collections import Counter

def toSpans(tags):
    # Converts a list of tags (corresponding to one sentence) to a list of spans
    # in: ['B-PER', 'I-PER', 'O', 'O', 'O', 'O', 'O', 'B-ORG', 'I-ORG', 'O']
    # out: [(7, 9, 'ORG'), (0, 2, 'PER')] (end is exclusive)
    spans = []
    i = 0
    n = len(tags)

    while i < n:
        if tags[i].startswith("B-"): # if the first letter of the current tag is "B", this is the beginning of a new entity span
            beg = i
            label = tags[i][2:]

            end = beg
            end = beg + 1

            while end < n and tags[end].startswith("I-"):
                end += 1

            spans.append((beg, end, label))
            i+=1
        
        else:
            i+= 1

    return spans


########## OVERALL SPAN METRICS: EXACT, UNLABELED, LOOSE ##############

def getLooseOverlap(spans1, spans2): 
    # spans1 represents the set of ground truth spans, spans2 the set of predicted spans
    # returns the overlap of spans without taking the exact boundaries
    # into account. If entities overlap they also count as found.
    found = 0
    for span1 in spans1:
        spanBeg, spanEnd, label = span1
        match = False
        for span2 in spans2:
            span2Beg, span2End, label2 = span2
            if label == label2:
                if span2Beg >= spanBeg and span2Beg <= spanEnd: # right edge of span1 is inside span2
                    match = True
                    break
                if span2End <= spanEnd and span2End >= spanBeg: # left edge of span1 is inside span2
                    match = True
                    break
        if match:
            found += 1
    return found


def getUnlabeled(spans1, spans2): 
    # measures exact span boundary match ignoring labels completely, e.g. '7-9:ORG' and '7-9:PER' would count as a match
    # Counts the overlap in spans after removing the labels
    boundaries1 = {(beg, end) for beg, end, _ in spans1}
    boundaries2 = {(beg, end) for beg, end, _ in spans2}

    return len(boundaries1.intersection(boundaries2))


def calculate_metrics(tp, fp, fn):
    prec = 0.0 if tp+fp == 0 else tp/(tp+fp)
    rec = 0.0 if tp+fn == 0 else tp/(tp+fn)
    f1 = 0.0 if prec+rec == 0.0 else 2 * (prec * rec) / (prec + rec)
    return {"precision": prec, "recall": rec, "f1": f1}


def compute_span_f1(gold_ners, pred_ners):
    # these values count for the entire dataset, not per sentence
    tp = 0
    fp = 0
    fn = 0
    
    recall_loose_tp = 0
    recall_loose_fn = 0
    precision_loose_tp = 0
    precision_loose_fp = 0
    
    tp_ul = 0
    fp_ul = 0
    fn_ul = 0 
    
    for gold_ner, pred_ner in zip(gold_ners, pred_ners): #assuming nested structure
        gold_spans = set(toSpans(gold_ner))
        pred_spans = set(toSpans(pred_ner))

        overlap = len(gold_spans.intersection(pred_spans))  # counting exact matches between gold and predicted spans in a sentence
        tp += overlap
        fp += len(pred_spans) - overlap
        fn += len(gold_spans) - overlap

        overlap_ul = getUnlabeled(gold_spans, pred_spans)
        tp_ul += overlap_ul
        fp_ul += len(pred_spans) - overlap_ul # predicted spans that do NOT match any gold span
        fn_ul += len(gold_spans) - overlap_ul # gold spans that do NOT match any predicted span

        overlap_loose = getLooseOverlap(gold_spans, pred_spans)
        recall_loose_tp += overlap_loose
        recall_loose_fn += len(gold_spans) - overlap_loose

        overlap_loose = getLooseOverlap(pred_spans, gold_spans)
        precision_loose_tp += overlap_loose
        precision_loose_fp += len(pred_spans) - overlap_loose

    print('--- Exact match ---')
    exact_metrics = calculate_metrics(tp, fp, fn)
    print('recall:   ', exact_metrics["recall"])
    print('precision:', exact_metrics["precision"])
    print('slot-f1:  ', exact_metrics["f1"])

    print('\n--- Unlabeled (ignore entity type) ---')
    ul_metrics = calculate_metrics(tp_ul, fp_ul, fn_ul)
    print('ul_recall:   ', ul_metrics["recall"])
    print('ul_precision:', ul_metrics["precision"])
    print('ul_slot-f1:  ', ul_metrics["f1"])

    print('\n--- Loose (partial overlap with same label) ---')
    loose_prec = 0.0 if precision_loose_tp + precision_loose_fp == 0 else precision_loose_tp/(precision_loose_tp+precision_loose_fp)
    loose_rec = 0.0 if recall_loose_tp+recall_loose_fn == 0 else recall_loose_tp/(recall_loose_tp+recall_loose_fn)
    f1 = 0.0 if loose_prec+loose_rec == 0.0 else 2 * (loose_prec * loose_rec) / (loose_prec + loose_rec)
    loose_metrics = {"precision": loose_prec, "recall": loose_rec, "f1": f1}

    print('l_recall:   ', loose_rec)
    print('l_precision:', loose_prec)
    print('l_slot-f1:  ', f1)

    return {
        "exact_match": exact_metrics,
        "unlabeled_match": ul_metrics,
        "loose_match": loose_metrics
    }
