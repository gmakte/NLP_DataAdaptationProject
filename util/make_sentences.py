fin5 = './data/FIN5.txt'
fin5_sentences = './data/FIN5_sentences.txt'


with open(fin5, 'r', encoding="utf-8") as infile:

    
    for line in infile:
        print(line.split('\t'))
        break