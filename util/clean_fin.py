fin5 = '../data/FIN5_old.txt'
fin5_clean = '../data/FIN5.txt'

fin3 = '../data/FIN3_old.txt'
fin3_clean = '../data/FIN3.txt'

with open(fin3, 'r', encoding="utf-8") as infile, \
    open(fin3_clean, 'w', encoding="utf-8") as outfile:

    word_count = 0
    
    for line in infile:
        line = line.strip()

        if line == "":
            outfile.write("\n")
            word_count = 0
            continue
        
        word_count += 1
        line_array = line.split()
        token, ner_tag = line_array[0], line_array[3]
        outfile.write(f"{word_count}\t{token}\t{ner_tag}\n")
