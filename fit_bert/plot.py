import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pickle

men_biased = ['janitor', 'cook', 'mover', 'laborer', 'constructor', 'chief', 'developer', 'carpenter', 'lawyer', 'farmer', 'driver', 'salesperson', 'physician', 'guard', 'analyst', 'sheriff']
women_biased = ['cashier', 'nurse', 'secretary', 'auditor', 'cleaner', 'receptionist', 'counselor', 'designer', 'hairdresser', 'writer', 'housekeeper', 'baker', 'accountant', 'editor', 'librarian', 'sewer']

# taking some out
men_biased = ['janitor', 'cook', '', 'laborer', 'constructor', '', '', '', '', '', '', '', '', 'guard', '', '']
women_biased = ['', 'nurse', 'secretary', '', 'cleaner', '', 'counselor', '', 'hairdresser', '', 'housekeeper', '', 'accountant', '', '', 'sewer']
word_b = men_biased + women_biased

plt.figure(figsize=(6, 3))
bog_tilde = pickle.load(open('v1bogtilde.pkl', 'rb'))
ta_by_occ = pickle.load(open('ta_by_occ.pkl', 'rb'))

ta_by_occ = np.array(ta_by_occ)
bert_matrix = np.mean(ta_by_occ, axis=1) 
plt.scatter(bog_tilde[:, 0], bert_matrix, s=8)
for n in range(len(bog_tilde)):
    plt.annotate(word_b[n], (bog_tilde[n, 0]+.01, bert_matrix[n]+.01), fontsize=12)

fontsize = 15
plt.plot([.5, .5], [0, 1], c='tab:gray')
plt.plot(np.arange(0, 1.1, .1), np.arange(0, 1.1, .1), c='tab:gray')
plt.xlabel('2016 US Labor Force (WinoBias)', fontsize=fontsize)
plt.ylabel('FitBERT', fontsize=fontsize)
plt.title("Probability an Occupation is 'he' (compared to 'she') ", fontsize=fontsize)

plt.tight_layout()
plt.savefig('occu_scatter.png', dpi=300, bbox_inches='tight')

