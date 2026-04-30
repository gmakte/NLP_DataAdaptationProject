input_path = "../data/FIN5.txt"
output_path = "../data/FIN5_fixed.txt"

with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:
    prev_tag = "O"
    prev_type = ""
    for line in infile:
        if line.strip() == "":
            outfile.write("\n")
            prev_tag = "O"
            prev_type = ""
            continue
        parts = line.strip().split("\t")
        if len(parts) < 3:
            outfile.write(line)
            continue
        tag = parts[2]
        if tag.startswith("I-"):
            tag_type = tag[2:]
            if prev_tag == "O" or prev_type != tag_type:
                tag = f"B-{tag_type}"
            else:
                tag = f"I-{tag_type}"
            prev_type = tag_type
        else:
            prev_type = ""
        outfile.write(f"{parts[0]}\t{parts[1]}\t{tag}\n")
        prev_tag = tag