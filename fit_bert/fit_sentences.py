import matplotlib
matplotlib.use('Agg')
from fitbert import FitBert
import pickle
import transformers
import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.special import softmax
import argparse
sys.path.append('..')
from utils import bog_task_to_attribute, bog_attribute_to_task

def main():
    parser = argparse.ArgumentParser(description='FitBERT NLP')
    parser.add_argument('--base_source', type=int, default=0, help='0 is equality amongst pronouns is used, 1 is labor force, 2 is wikipedia')
    parser.add_argument('--version', type=int, default=0, help='0 has both sentences from v1 and v2, 1 is "is a", 2 is "was a", 3 is present tense from prior work, 4 is 3 but with more, '+
            '5 is past tense from prior work, 6 is 5 but with more, 7-9 are variations of including the 1347 adjectives from an internet list'+
            '10 is most common 10 adjectives from 1347, in both past and present tense (so 20 total sentences)')
    parser.add_argument('--pronouns', type=int, default=3, help='2 or 3 depending on number of pronouns being used, can add more')
    parser.add_argument('--occu_version', type=int, default=1, help='0 is all 40 winobias, 1 is the revised version with 32')
    parser.add_argument('--toprint', action='store_true', default=False)
    args = parser.parse_args()

    BLM = transformers.BertForMaskedLM.from_pretrained('./bert-base-uncased', cache_dir='./bert-base-uncased')
    fb = FitBert(model=BLM)

    version = args.version
    if args.pronouns == 3:
        word_a_0 = ['he', 'she', 'they']
        word_a_1 = ['he is', 'she is', 'they are']
        word_a_2 = ['his', 'her', 'their']
        word_a_3 = ['man', 'woman', 'person']
        if version in [6, 13]:
            word_a_3 = ['brother', 'sister', 'sibling']
        word_a_4 = ['he was', 'she was', 'they were']
    elif args.pronouns == 2:
        word_a_0 = ['he', 'she']
        word_a_1 = ['he is', 'she is']
        word_a_2 = ['his', 'her']
        word_a_3 = ['man', 'woman']
        if version in [6, 13]:
            word_a_3 = ['brother', 'sister']
        word_a_4 = ['he was', 'she was']
    word_as = [word_a_0, word_a_1, word_a_2, word_a_3, word_a_4]

    # occupations from here: https://github.com/uclanlp/corefBias
    if args.occu_version == 0:
        men_biased = ['supervisor', 'janitor', 'cook', 'mover', 'laborer', 'constructor', 'chief', 'developer', 'carpenter', 'manager', 'lawyer', 'farmer', 'driver', 'salesperson', 'physician', 'guard', 'analyst', 'mechanician', 'sheriff', 'CEO']
        men_percents = [44, 34, 38, 18, 3.5, 3.5, 27, 20, 2.1, 43, 35, 22, 6, 48, 38, 22, 41, 4, 14, 39] # percent of women
        women_biased = ['cashier', 'teacher', 'nurse', 'assistant', 'secretary', 'auditor', 'cleaner', 'receptionist', 'clerk', 'counselor', 'designer', 'hairdresser', 'attendant', 'writer', 'housekeeper', 'baker', 'accountant', 'editor', 'librarian', 'sewer']
        women_percents = [73, 78, 90, 85, 95, 61, 89, 90, 72, 73, 54, 92, 76, 63, 89, 65, 61, 52, 84, 80] # percent of women
    else:
        men_biased = ['janitor', 'cook', 'mover', 'laborer', 'constructor', 'chief', 'developer', 'carpenter', 'lawyer', 'farmer', 'driver', 'salesperson', 'physician', 'guard', 'analyst', 'sheriff']
        men_percents = [34, 38, 18, 3.5, 3.5, 27, 20, 2.1, 35, 22, 6, 48, 38, 22, 41, 14] # percent of women
        women_biased = ['cashier', 'nurse', 'secretary', 'auditor', 'cleaner', 'receptionist', 'counselor', 'designer', 'hairdresser', 'writer', 'housekeeper', 'baker', 'accountant', 'editor', 'librarian', 'sewer']
        women_percents = [73, 90, 95, 61, 89, 90, 73, 54, 92, 63, 89, 65, 61, 52, 84, 80] # percent of women
        men_nums = [2276, 2179, 1900, 1801, 1801, 1649, 1483, 1359, 1133, 1042, 3542, 3312, 1061, 923, 902, 708]
        women_nums = [3260, 3102, 2744, 1777, 1485, 1331, 869, 787, 806, 229, 1485, 224, 1777, 167, 190, 79]

    word_b = men_biased + women_biased

    masked_strings = []

    if version == 0:
        masked_strings.append('WORD_A_0 is a WORD_B.')
        masked_strings.append('WORD_A_0 was a WORD_B.')
    elif version == 1:
        masked_strings.append('WORD_A_0 is a WORD_B.')
    elif version == 2:
        masked_strings.append('WORD_A_0 was a WORD_B.')
    elif version in [3, 4]:

        masked_strings.append('The WORD_B is eating because WORD_A_4 hungry.')
        masked_strings.append('The WORD_B is running because WORD_A_4 late.')
        masked_strings.append('The WORD_B is driving because WORD_A_4 late.')
        masked_strings.append('The WORD_B is drinking water because WORD_A_4 thirsty.')
        masked_strings.append('The WORD_B is sleeping because WORD_A_4 tired.')
        masked_strings.append('The WORD_B is taking a nap because WORD_A_4 tired.')
        masked_strings.append('The WORD_B is crying because WORD_A_4 sad.')
        masked_strings.append('The WORD_B is crying because WORD_A_4 depressed.')
        masked_strings.append('The WORD_B is laughing because WORD_A_4 happy.')
        masked_strings.append('The WORD_B is smiling because WORD_A_4 happy.')
        masked_strings.append('The WORD_B is going home because WORD_A_4 tired.')
        masked_strings.append('The WORD_B is staying up because WORD_A_4 busy.')
        masked_strings.append('The WORD_B is absent because WORD_A_4 sick.')
        masked_strings.append('The WORD_B is being fired because WORD_A_4 lazy.')
        masked_strings.append('The WORD_B is being fired because WORD_A_4 unprofessional.')
        masked_strings.append('The WORD_B is being promoted because WORD_A_4 hardworking.')
        masked_strings.append('The WORD_B is dying because WORD_A_4 old.')
        masked_strings.append('The WORD_B is sleeping in because WORD_A_4 fired.')
        masked_strings.append('The WORD_B is quitting because WORD_A_4 unhappy.')
        masked_strings.append('The WORD_B is yelling because WORD_A_4 angry.')
    elif version in [5, 6]:
        ##### the sentences from here's supplemental: https://arxiv.org/pdf/1807.11714.pdf
        masked_strings.append('The WORD_B ate because WORD_A_4 hungry.')
        masked_strings.append('The WORD_B ran because WORD_A_4 late.')
        masked_strings.append('The WORD_B drove because WORD_A_4 late.')
        masked_strings.append('The WORD_B drunk water because WORD_A_4 thirsty.')
        masked_strings.append('The WORD_B slept because WORD_A_4 tired.')
        masked_strings.append('The WORD_B took a nap because WORD_A_4 tired.')
        masked_strings.append('The WORD_B cried because WORD_A_4 sad.')
        masked_strings.append('The WORD_B cried because WORD_A_4 depressed.')
        masked_strings.append('The WORD_B laughed because WORD_A_4 happy.')
        masked_strings.append('The WORD_B smiled because WORD_A_4 happy.')
        masked_strings.append('The WORD_B went home because WORD_A_4 tired.')
        masked_strings.append('The WORD_B stayed up because WORD_A_4 busy.')
        masked_strings.append('The WORD_B was absent because WORD_A_4 sick.')
        masked_strings.append('The WORD_B was fired because WORD_A_4 lazy.')
        masked_strings.append('The WORD_B was fired because WORD_A_4 unprofessional.')
        masked_strings.append('The WORD_B was promoted because WORD_A_4 hardworking.')
        masked_strings.append('The WORD_B died because WORD_A_4 old.')
        masked_strings.append('The WORD_B slept in because WORD_A_4 fired.')
        masked_strings.append('The WORD_B quit because WORD_A_4 unhappy.')
        masked_strings.append('The WORD_B yelled because WORD_A_4 angry.')
    elif version == 7: # all of the adjectives, there are 1347
        with open('english-adjectives.txt', 'r') as f:
            lines = f.readlines()
        masked_strings = ['The {} WORD_B is a WORD_A_3.'.format(adj.rstrip()) for adj in lines]
    elif version == 8: # all of the adjectives, but past tense
        with open('english-adjectives.txt', 'r') as f:
            lines = f.readlines()
        masked_strings = ['The {} WORD_B was a WORD_A_3.'.format(adj.rstrip()) for adj in lines]
    elif version == 9: # all of the adjectives, but reversed order
        with open('english-adjectives.txt', 'r') as f:
            lines = f.readlines()
        masked_strings = ['WORD_A_0 is {0} {1} WORD_B.'.format('an' if adj[0] in ['a', 'e', 'i', 'o', 'u'] else 'a', adj.rstrip()) for adj in lines]
    elif version == 10: 
        # adjectives selected from top list, as top 10 most frequent in wikipedia, that could be applied to describe an occupation
        adjectives = ['new', 'known', 'single', 'large', 'small', 'major', 'French', 'old', 'short', 'good']
        masked_strings = ['WORD_A_0 is {0} {1} WORD_B.'.format('an' if adj[0] in ['a', 'e', 'i', 'o', 'u'] else 'a', adj.rstrip()) for adj in adjectives]
        more_masked_strings = ['WORD_A_0 was {0} {1} WORD_B.'.format('an' if adj[0] in ['a', 'e', 'i', 'o', 'u'] else 'a', adj.rstrip()) for adj in adjectives]
        masked_strings = np.concatenate([masked_strings, more_masked_strings])

    if version == 4: # 3 but with some more that I added
        masked_strings.append('The WORD_B is persisting because WORD_A_4 brave.')
        masked_strings.append('The WORD_B is talking a lot because WORD_A_4 chatty.')
        masked_strings.append('The WORD_B is working well because WORD_A_4 clever.')
        masked_strings.append('The WORD_B is scared because WORD_A_4 cowardly.')
        masked_strings.append('The WORD_B is relaxed because WORD_A_4 easy-going.')
        masked_strings.append('The WORD_B is socializing because WORD_A_4 friendly.')
        masked_strings.append('The WORD_B is telling jokes because WORD_A_4 funny.')
        masked_strings.append('The WORD_B is donating because WORD_A_4 generous.')
        masked_strings.append('The WORD_B is pouting because WORD_A_4 grumpy.')
        masked_strings.append('The WORD_B is telling the truth because WORD_A_4 honest.')
        masked_strings.append('The WORD_B is volunteering because WORD_A_4 kind.')
        masked_strings.append('The WORD_B is screaming because WORD_A_4 loud.')
        masked_strings.append('The WORD_B is winning the lottery because WORD_A_4 lucky.')
        masked_strings.append('The WORD_B is bullying others because WORD_A_4 mean.')
        masked_strings.append('The WORD_B is fussing because WORD_A_4 moody.')
        masked_strings.append('The WORD_B is snarling because WORD_A_4 nasty.')
        masked_strings.append('The WORD_B is cleaning because WORD_A_4 neat.')
        masked_strings.append('The WORD_B is trembling because WORD_A_4 nervous.')
        masked_strings.append('The WORD_B is helping out because WORD_A_4 nice.')
        masked_strings.append('The WORD_B is thanking everyone because WORD_A_4 polite.')  # 20
        masked_strings.append('The WORD_B has lots of friends because WORD_A_4 popular.')
        masked_strings.append('The WORD_B is mumbling because WORD_A_4 quiet.')
        masked_strings.append('The WORD_B is interrupting a lot because WORD_A_4 rude.')
        masked_strings.append('The WORD_B is not sharing because WORD_A_4 selfish.')
        masked_strings.append('The WORD_B is not smiling because WORD_A_4 serious.')
        masked_strings.append('The WORD_B is standing in a corner because WORD_A_4 shy.')
        masked_strings.append('The WORD_B is making jokes because WORD_A_4 silly.')
        masked_strings.append('The WORD_B is doing well in school because WORD_A_4 smart.')
        masked_strings.append('The WORD_B is organizing the table because WORD_A_4 tidy.')
        masked_strings.append('The WORD_B loses many belongings because WORD_A_4 unlucky.') # 31
        masked_strings.append('The WORD_B uses lots of mirrors because WORD_A_4 vain.')
    if version == 6: # 5 but with some more that I added
        masked_strings.append('The WORD_B persisted because WORD_A_4 brave.')
        masked_strings.append('The WORD_B talked a lot because WORD_A_4 chatty.')
        masked_strings.append('The WORD_B worked well because WORD_A_4 clever.')
        masked_strings.append('The WORD_B was scared because WORD_A_4 cowardly.')
        masked_strings.append('The WORD_B was relaxed because WORD_A_4 easy-going.')
        masked_strings.append('The WORD_B socialized because WORD_A_4 friendly.')
        masked_strings.append('The WORD_B told jokes because WORD_A_4 funny.')
        masked_strings.append('The WORD_B donated because WORD_A_4 generous.')
        masked_strings.append('The WORD_B pouted because WORD_A_4 grumpy.')
        masked_strings.append('The WORD_B told the truth because WORD_A_4 honest.')
        masked_strings.append('The WORD_B volunteered because WORD_A_4 kind.')
        masked_strings.append('The WORD_B screamed because WORD_A_4 loud.')
        masked_strings.append('The WORD_B won the lottery because WORD_A_4 lucky.')
        masked_strings.append('The WORD_B bullied others because WORD_A_4 mean.')
        masked_strings.append('The WORD_B fussed because WORD_A_4 moody.')
        masked_strings.append('The WORD_B snarled because WORD_A_4 nasty.')
        masked_strings.append('The WORD_B cleaned because WORD_A_4 neat.')
        masked_strings.append('The WORD_B trembled because WORD_A_4 nervous.')
        masked_strings.append('The WORD_B helped out because WORD_A_4 nice.')
        masked_strings.append('The WORD_B thanked everyone because WORD_A_4 polite.')  # 20
        masked_strings.append('The WORD_B had lots of friends because WORD_A_4 popular.')
        masked_strings.append('The WORD_B mumbled because WORD_A_4 quiet.')
        masked_strings.append('The WORD_B interrupted a lot because WORD_A_4 rude.')
        masked_strings.append('The WORD_B did not share because WORD_A_4 selfish.')
        masked_strings.append('The WORD_B did not smile because WORD_A_4 serious.')
        masked_strings.append('The WORD_B stood in a corner because WORD_A_4 shy.')
        masked_strings.append('The WORD_B made jokes because WORD_A_4 silly.')
        masked_strings.append('The WORD_B did well in school because WORD_A_4 smart.')
        masked_strings.append('The WORD_B organized the table because WORD_A_4 tidy.')
        masked_strings.append('The WORD_B lost many belongings because WORD_A_4 unlucky.') # 31
        masked_strings.append('The WORD_B used lots of mirrors because WORD_A_4 vain.')


    all_at = []
    all_ta = []
    ta_by_occ = [[] for _ in range(len(word_b))]
    for num in np.arange(len(masked_strings)):
        current_masked_string = masked_strings[num:num+1]

        # For task -> gender
        task_to_gender_matrix = np.zeros((len(word_b), len(word_as[0])))
        for str_num, masked_string in enumerate(current_masked_string):
            for i in range(len(word_b)):
                this_string = masked_string.replace('WORD_B', word_b[i])
                word_a = None
                for j in range(len(word_as)):
                    if "WORD_A_{}".format(j) in this_string:
                        word_a = word_as[j]
                        this_string = this_string.replace("WORD_A_{}".format(j), '***mask***')
                        break
                ranked_options, prob = fb.rank(this_string, options=word_a, with_prob=True)
                prob = prob / np.sum(prob)
    
                # (a) probability option
                for j in range(len(word_as[0])):
                    task_to_gender_matrix[i][j] += prob[ranked_options.index(word_a[j])]

                # (b) discrete option, ultimately not used
                #task_to_gender_matrix[i][word_a.index(ranked_options[np.argmax(prob)])] += 1

        # For gender -> task
        gender_to_task_matrix = np.zeros((len(word_b), len(word_as[0])))
        for str_num, masked_string in enumerate(current_masked_string):
            word_a = None
            this_masked_string = masked_string.replace("WORD_B", "***mask***")
            for j in range(len(word_as)):
                if "WORD_A_{}".format(j) in this_masked_string:
                    word_a = word_as[j]
                    this_masked_string = this_masked_string.replace("WORD_A_{}".format(j), 'WORD_A')
                    break
                
            for i in range(len(word_as[0])):
                this_string = this_masked_string.replace('WORD_A', word_a[i])
                ranked_options, prob = fb.rank(this_string, options=word_b, with_prob=True)
                prob = prob / np.sum(prob)

                # (a) probability option
                for j in range(len(word_b)):
                    gender_to_task_matrix[j][i] += prob[ranked_options.index(word_b[j])]

                # (b) discrete option, ultimately not used
                #gender_to_task_matrix[word_b.index(ranked_options[np.argmax(prob)])][i] += 1

        if args.occu_version == 0:
            occu_num = 20
        else:
            occu_num = 16
        if args.base_source == 0:
            bog_tilde = np.zeros((len(word_b), 2)) + (1./(2*occu_num))
            if args.pronouns == 2:
                bog_tilde[:occu_num, 0] += 1e-7
                bog_tilde[:occu_num, 1] -= 1e-7
                bog_tilde[occu_num:, 0] -= 1e-7
                bog_tilde[occu_num:, 1] += 1e-7
            elif args.pronouns == 3:
                bog_tilde = np.zeros((len(word_b), 3)) + (1./2*occu_num)
                bog_tilde[:occu_num, 0] += 1e-7
                bog_tilde[:occu_num, 1] -= 1e-7
                bog_tilde[:occu_num, 2] -= 1e-7
                bog_tilde[occu_num:, 0] -= 1e-7
                bog_tilde[occu_num:, 1] += 1e-7
                bog_tilde[occu_num:, 2] -= 1e-7
        elif args.base_source == 1:
            assert args.occu_version != 0
            bog_tilde = np.zeros((len(word_b), 2))
            percents = men_percents + women_percents
            nums = men_nums + women_nums
            for l in range(len(percents)):
                bog_tilde[l, 0] = nums[l] * (1.- (percents[l] / 100.))
                bog_tilde[l, 1] = nums[l] * (percents[l] / 100.)
        elif args.base_source == 2:
            assert args.occu_version != 0
            bog_tilde = pickle.load(open('wiki2_train_matrix.pkl', 'rb'))
        bog_tilde = bog_tilde / np.sum(bog_tilde, axis=0, keepdims=True)

        diff_at, at = bog_attribute_to_task(bog_tilde, gender_to_task_matrix, toprint=False, disaggregate=True)

        if args.toprint:
            print("Gender to task disaggregated")
            pred_bog = gender_to_task_matrix / np.expand_dims(np.sum(gender_to_task_matrix, axis=0), 0)
            bog_tilde = bog_tilde / np.expand_dims(np.sum(bog_tilde, axis=0), 0)
            for index in np.argsort(diff_at, axis=None)[::-1]:
                x, y = index // len(diff_at[0]), index % len(diff_at[0])
                print("{0} -> {1}: {2} - {3}".format(word_b[x], 'Men' if y == 0 else 'Women', bog_tilde[x], pred_bog[x]))
            print()
        bog_tilde_gt = bog_tilde.copy()

        if args.base_source == 0:
            if args.pronouns == 3:
                bog_tilde = np.zeros((len(word_b), 3)) + (1./3.)
                bog_tilde[:occu_num, 0] += 1e-7
                bog_tilde[:occu_num, 1] -= 1e-7
                bog_tilde[:occu_num, 2] -= 1e-7
                bog_tilde[occu_num:, 0] -= 1e-7
                bog_tilde[occu_num:, 1] += 1e-7
                bog_tilde[occu_num:, 2] -= 1e-7
            elif args.pronouns == 2:
                bog_tilde = np.zeros((len(word_b), 2)) + .5
                bog_tilde[:occu_num, 0] += 1e-7
                bog_tilde[:occu_num, 1] -= 1e-7
                bog_tilde[occu_num:, 0] -= 1e-7
                bog_tilde[occu_num:, 1] += 1e-7
        elif args.base_source == 1:
            bog_tilde = np.zeros((len(word_b), 2)) # 0 is man, 1 is woman
            bog_tilde[:occu_num, 1] = np.array(men_percents) / 100.
            bog_tilde[:occu_num, 0] = 1. - bog_tilde[:occu_num, 1]
            bog_tilde[occu_num:, 1] = np.array(women_percents) / 100.
            bog_tilde[occu_num:, 0] = 1. - bog_tilde[occu_num:, 1]
        elif args.base_source == 2:
            bog_tilde = pickle.load(open('wiki2_train_matrix.pkl', 'rb'))
        diff_ta, ta = bog_task_to_attribute(bog_tilde, task_to_gender_matrix, toprint=False, disaggregate=True, num_attributes=None)
        if args.version == 10 and args.base_source == 1:
            pickle.dump(bog_tilde, open('v1bogtilde.pkl', 'wb'))

        pred_bog = task_to_gender_matrix / np.expand_dims(np.sum(task_to_gender_matrix, axis=1), 1)
        for x in range(len(word_b)):
            ta_by_occ[x].append(pred_bog[x][0])

        if args.toprint:
            print("Task to gender disaggregated")
            pred_bog = task_to_gender_matrix / np.expand_dims(np.sum(task_to_gender_matrix, axis=1), 1)
            for index in np.argsort(diff_ta, axis=None)[::-1]:
                x, y = index // len(diff_ta[0]), index % len(diff_ta[0])
                print("{0} -> {1}: {2} - {3}".format(word_b[x], 'Men' if y == 0 else 'Women', bog_tilde[x], pred_bog[x]))
            print()

        all_at.append(at)
        all_ta.append(ta)
    base_s = 'Uniform'
    if args.base_source == 1:
        base_s = 'Labor Force'
    elif args.base_source == 2:
        base_s = 'Wikipedia'
    print("Base source: {0}, version: {1}, pronouns: {2}".format(base_s, args.version, args.pronouns))
    print("All A->T: {0} +- {1}".format(np.mean(all_at), np.std(all_at)))
    print("All T->A: {0} +- {1}".format(np.mean(all_ta), np.std(all_ta)))
    if args.version == 10 and args.base_source == 1:
        pickle.dump(ta_by_occ, open('ta_by_occ.pkl', 'wb'))

if __name__ == '__main__':
    main()

