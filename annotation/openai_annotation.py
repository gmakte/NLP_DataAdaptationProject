all_pred_labels = []

chunk_tokens = []
chunk_sentence_lengths = []

current_chunk_len = 0

parsed = parse_iob2_file("FIN5_validation.txt")

contract_5_sentences = parsed[0]
contract_5_labels = parsed[1]
contract_5_lens = [len(sent) for sent in contract_5_sentences]

def annotate_chunk(tokens):

    response = client.responses.create(
        model="gpt-5.4",
        temperature=0,
        input=(
            "You are a strict NER tagger.\n\n"

            "Task:\n"
            "Assign a BIO tag to EACH token.\n\n"

            "Allowed labels:\n"
            "B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC, O\n\n"

            "Rules:\n"
            "- EXACTLY one label per token\n"
            "- SAME number of labels as tokens\n"
            "- SAME order as tokens\n"
            "- Do not modify tokens\n"
            "- Output MUST be valid JSON\n"
            "- Output ONLY a JSON list of labels\n"
            "- No explanations\n\n"

            "Example input:\n"
            + json.dumps(["John", "Smith", "works", "at", "OpenAI"])
            + "\n\n"

            "Example output:\n"
            + json.dumps(["B-PER", "I-PER", "O", "O", "B-ORG"])
            + "\n\n"

            "Now annotate this input:\n"
            + json.dumps(tokens)
        )
    )

    raw_output = response.output[0].content[0].text

    try:
        pred_labels = json.loads(raw_output)

    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON returned:\n{raw_output}")

    # STRICT VALIDATION

    if not isinstance(pred_labels, list):
        raise ValueError("Model output is not a list.")

    if len(pred_labels) != len(tokens):
        raise ValueError(
            f"Label count mismatch.\n"
            f"Expected {len(tokens)} labels.\n"
            f"Got {len(pred_labels)} labels."
        )

    return pred_labels


def reconstruct_sentences(flat_labels, sentence_lengths):

    nested_labels = []

    start = 0

    for sent_len in sentence_lengths:

        end = start + sent_len

        nested_labels.append(flat_labels[start:end])

        start = end

    return nested_labels


for sent_tokens, sent_labels in zip(contract_5_sentences, contract_5_labels):

    sent_len = len(sent_tokens)

    # sentence fits into current chunk
    if current_chunk_len + sent_len <= MAX_CHUNK_SIZE:

        chunk_tokens.extend(sent_tokens)
        chunk_sentence_lengths.append(sent_len)

        current_chunk_len += sent_len

    # chunk full -> annotate current chunk first
    else:

        pred_flat_labels = annotate_chunk(chunk_tokens)

        pred_nested_labels = reconstruct_sentences(
            pred_flat_labels,
            chunk_sentence_lengths
        )

        all_pred_labels.extend(pred_nested_labels)

        # START NEW CHUNK with current sentence
        chunk_tokens = sent_tokens.copy()

        chunk_sentence_lengths = [sent_len]

        current_chunk_len = sent_len


# FINAL FLUSH
if chunk_tokens:

    pred_flat_labels = annotate_chunk(chunk_tokens)

    pred_nested_labels = reconstruct_sentences(
        pred_flat_labels,
        chunk_sentence_lengths
    )

    all_pred_labels.extend(pred_nested_labels)


print("Annotation complete.")
print(f"Total annotated sentences: {len(all_pred_labels)}")