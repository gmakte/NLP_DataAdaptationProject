import numpy as np
import re
from collections import Counter, defaultdict
from util.span_f1 import toSpans

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

            token_line = line.split("\t")
            tokens.append(token_line[1])
            tags.append(token_line[2])

    # handle last sentence since there is no whitespace at the end of the file
    if tokens:
        sentences.append(tokens)
        labels.append(tags)

    return sentences, labels



# function for EDA count of word length per doc
def doc_token_count(nested_sentences):
    word_lens = []
    cur_word_count = 0

    for sent in nested_sentences:

        if sent[0] == "-DOCSTART-":
            word_lens.append(cur_word_count) # append document word count since we are starting a new doc
            cur_word_count = 0
            
            continue

        else:
            cur_word_count += len(sent)

    if cur_word_count != 0:
        word_lens.append(cur_word_count)
        cur_word_count = 0

    return word_lens[1:]



# compute a matrix of NE span counts per doc, for EDA. rows: documents, cols: PER, LOC, ORG
def doc_entity_count(nested_sentences, nested_labels):
    entity_matrix = []

    cur_doc_counter = Counter()

    for sent, sent_labels in zip(nested_sentences, nested_labels):

        # new document 
        if sent[0] == "-DOCSTART-":

            # save previous document counts
            if cur_doc_counter:

                entity_matrix.append([
                    cur_doc_counter["PER"],
                    cur_doc_counter["LOC"],
                    cur_doc_counter["ORG"]
                ])

            # reset counter for next document
            cur_doc_counter = Counter()

            continue

        # convert sentence bio tags to spans in the format (beg, end, entity)
        spans = toSpans(sent_labels)

        # count entities in sentence
        for _, _, label in spans:
            cur_doc_counter[label] += 1

    # append final document
    if cur_doc_counter:

        entity_matrix.append([
            cur_doc_counter["PER"],
            cur_doc_counter["LOC"],
            cur_doc_counter["ORG"]
        ])

    return np.array(entity_matrix)

def normalize_entity_text(text):

    text = text.casefold()

    # remove punctuation
    text = re.sub(r"[^\w\s]", "", text)

    # normalize whitespace
    text = " ".join(text.split())

    return text

# entity counter in dataset
def entityCounter(sentences, labels):

    entity_counts = defaultdict(Counter)

    for sent_tokens, sent_labels in zip(sentences, labels):

        spans = toSpans(sent_labels)

        for start, end, label in spans:

            entity_text = " ".join(sent_tokens[start:end])
            entity_text = normalize_entity_text(entity_text)

            entity_counts[label][entity_text] += 1

    return entity_counts


def entityDiversity(sentences, labels):

    entity_counts = entityCounter(sentences, labels)

    diversity = {}

    label_order = ["PER", "LOC", "ORG"]

    for label in label_order:

        counter = entity_counts[label]

        unique_entities = len(counter)
        total_mentions = sum(counter.values())

        diversity[label] = unique_entities / total_mentions

    return diversity