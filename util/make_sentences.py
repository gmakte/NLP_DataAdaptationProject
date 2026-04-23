from sacremoses import MosesDetokenizer
detokenizer = MosesDetokenizer(lang='en')

fin5 = './data/FIN5.txt'
fin5_sentences = './data/FIN5_sentences.txt'


with open(fin5, 'r', encoding="utf-8") as infile, \
    open(fin5_sentences, 'w', encoding="utf-8") as outfile:
    
    sentence = []

    for line in infile:
        line = line.strip()

        if line == "":
            if sentence:
                outfile.write(f"{detokenizer.detokenize(sentence)}\n")
                sentence = []
            continue
            
        token = line.strip().split('\t')[1]
        sentence.append(token)