# these are some one time-transformations we ran to clean the data before using


##### 1. REMOVE UNNECESSARY COLUMNS FROM THE .TXT FILES #####
# fin5_old = '../data/FIN5_old.txt'
# fin5 = '../data/FIN5.txt'

# fin3_old = '../data/FIN3_old.txt'
# fin3 = '../data/FIN3.txt'

# with open(fin3_old, 'r', encoding="utf-8") as infile, \
#     open(fin3_old, 'w', encoding="utf-8") as outfile:

#     word_count = 0
    
#     for line in infile:
#         line = line.strip()

#         if line == "":
#             outfile.write("\n")
#             word_count = 0
#             continue
        
#         word_count += 1
#         line_array = line.split()
#         token, ner_tag = line_array[0], line_array[3]
#         outfile.write(f"{word_count}\t{token}\t{ner_tag}\n")



###### 2. MAKE SENTENCES FOR FEEDING THE FEW-SHOT EXAMPLES TO THE LLM FOR SYNTHETIC DATA GENERATION #####
# from sacremoses import MosesDetokenizer
# detokenizer = MosesDetokenizer(lang='en')

# fin5 = './data/FIN5.txt'
# fin5_sentences = './data/FIN5_sentences.txt'


# with open(fin5, 'r', encoding="utf-8") as infile, \
#     open(fin5_sentences, 'w', encoding="utf-8") as outfile:
    
#     sentence = []

#     for line in infile:
#         line = line.strip()

#         if line == "":
#             if sentence:
#                 outfile.write(f"{detokenizer.detokenize(sentence)}\n")
#                 sentence = []
#             continue
            
#         token = line.strip().split('\t')[1]
#         sentence.append(token)


##### FIND DIFFERENCES BETWEEN TEST TXT AND PREDICTIONS TXT (THEY ARE NOT THE SAME LENGTH) ######
def load_sentences(path):
    sentences = []
    current = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line == "":
                if current:
                    sentences.append(current)
                    current = []
            else:
                token = line.split()[1]  
                current.append(token)

    if current:
        sentences.append(current)

    return sentences


real = load_sentences("../data/FIN3.txt")
pred = load_sentences("../predictions/test_predictions_fin3.txt")

print(len(real), len(pred))

for i, (r, p) in enumerate(zip(real, pred)):
    if len(r) != len(p):
        print(f"\nSentence {i} mismatch:")
        print(f"REAL ({len(r)}):", r)
        print(f"PRED ({len(p)}):", p)