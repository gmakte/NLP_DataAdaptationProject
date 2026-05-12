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

