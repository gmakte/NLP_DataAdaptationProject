import sys

from sklearn import metrics

def readNlu(path):
    # reads labels from last column, assumes conll-like file
    # with 1 word per line, tab separation, and empty lines
    # for sentence splits. The BIO annotation is expected in the
    # third column (index 2), following universalNER.
    annotations = []
    cur_annotation = []
    for line in open(path, encoding='utf-8'):
        line = line.strip()
        if line == '':
            annotations.append(cur_annotation)
            cur_annotation = []
        elif line[0] == '#' and len(line.split('\t')) == 1:
            continue
        else:
            cur_annotation.append(line.split('\t')[2])
    return annotations

def toSpans(tags):
    # Converts a list of tags (corresponding to one sentence) to a list of spans
    # in: ['B-PER', 'I-PER', 'O', 'O', 'O', 'O', 'O', 'B-ORG', 'I-ORG', 'O']
    # out: {'7-9:ORG', '0-2:PER'} (end is exclusive)
    spans = set()
    for beg in range(len(tags)):
        if tags[beg][0] == 'B': # if the first letter of the current tag is "B", this is the beginning of a new entity span
            end = beg
            for end in range(beg+1, len(tags)):
                if tags[end][0] != 'I':
                    break
            spans.add(str(beg) + '-' + str(end) + ':' + tags[beg][2:])
    return spans


def getBegEnd(span): #e.g. '7-9:ORG' -> [7, 9]
    return [int(x) for x in span.split(':')[0].split('-')]


def getLooseOverlap(spans1, spans2): 
    # spans1 represents the set of ground truth spans, spans2 the set of predicted spans
    # returns the overlap of spans without taking the exact boundaries
    # into account. If entities overlap they also count as found.
    found = 0
    for span1 in spans1:
        spanBeg, spanEnd = getBegEnd(span1)
        label = span1.split(':')[1]
        match = False
        for span2 in spans2:
            span2Beg, span2End = getBegEnd(span2)
            label2 = span2.split(':')[1]
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
    return len(set([x.split(':')[0] for x in spans1]).intersection([x.split(':')[0] for x in spans2]))


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
        gold_spans = toSpans(gold_ner)
        pred_spans = toSpans(pred_ner)

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
