from tqdm import tqdm
import numpy as np
import pickle
import re


#word_as = [['he', 'man', 'his', 'boy', 'son'], ['she', 'woman', 'her', 'girl', 'daughter'], ['they', 'person', 'their', 'child', 'kid']]
word_as = [['he', 'man', 'his', 'boy', 'son'], ['she', 'woman', 'her', 'girl', 'daughter']]
all_wordas = [item for sublist in word_as for item in sublist]
word_b = ['nurse', 'doctor', 'baker', 'lawyer']

men_biased = ['janitor', 'cook', 'mover', 'laborer', 'constructor', 'chief', 'developer', 'carpenter', 'lawyer', 'farmer', 'driver', 'salesperson', 'physician', 'guard', 'analyst', 'sheriff']
women_biased = ['cashier', 'nurse', 'secretary', 'auditor', 'cleaner', 'receptionist', 'counselor', 'designer', 'hairdresser', 'writer', 'housekeeper', 'baker', 'accountant', 'editor', 'librarian', 'sewer']

word_b = men_biased + women_biased

matrix = np.zeros((len(word_b), len(word_as)))

with open('english-adjectives.txt', 'r') as f:
    lines = f.readlines()
adjectives = [adj.rstrip() for adj in lines]
adj_to_count = {}
for adj in adjectives:
    adj_to_count[adj] = 0

files = ["/n/fs/visualai-scr/Data/WikiText/wikitext-103/wiki.train.tokens", "/n/fs/visualai-scr/Data/WikiText103/wikitext-103/wiki.test.tokens", "/n/fs/visualai-scr/Data/WikiText103/wikitext-103/wiki.valid.tokens"]
files = [files[0]]
for file_name in files:
    f = open(file_name, 'r')
    for index, line in enumerate(tqdm(f)):
        if '=' in line:
            continue
        else:
            for sent in line.split('.'):
                intersect_adj = set(re.findall(r'\w+', sent))&set(adjectives)
                for adj in intersect_adj:
                    adj_to_count[adj] += 1

                index_i = None
                for i, occu in enumerate(word_b):
                    if occu in sent:
                        if index_i is not None:
                            index_i = None
                            continue
                        index_i = i
                if index_i is None:
                    continue
                index_j = None
                for j, gend in enumerate(word_as):
                    if len(set(gend)&set(sent.split(' '))) > 0:
                        if index_j is not None:
                            if j != 2:
                                index_j = None
                            continue
                        index_j = j
                if index_j is None:
                    continue
                matrix[index_i][index_j] += 1

pickle.dump(matrix, open('wiki2_train_matrix.pkl', 'wb'))
pickle.dump(adj_to_count, open('wiki2_adj_count.pkl', 'wb'))


