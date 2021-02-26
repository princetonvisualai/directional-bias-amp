import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import pickle
import numpy as np
from tqdm import tqdm
from classifier import get_bogs
from sklearn.metrics import average_precision_score, f1_score
from sklearn import metrics
import time
import sys
sys.path.append('..')
from utils import bog_task_to_attribute, bog_attribute_to_task

def get_bogs(all_preds, all_labels):
    total_labels = len(all_labels[0]) - 1
    bog_tilde = np.zeros((total_labels, 2)) 
    bog_gt_g = np.zeros((total_labels, 2)) 
    bog_gt_o = np.zeros((total_labels, 2)) 
    bog_preds = np.zeros((total_labels, 2))

    for i, objs in enumerate([all_preds, all_labels]):
        female = np.where(objs[:, -1] == 0)[0]
        male = np.where(objs[:, -1] == 1)[0]
        for j in range(total_labels):
            if i == 0:
                bog_preds[j][0] = np.sum(objs[male][:, j])
                bog_preds[j][1] = np.sum(objs[female][:, j])

                bog_gt_o[j][0] = np.sum(all_labels[male][:, j])
                bog_gt_o[j][1] = np.sum(all_labels[female][:, j]) 
            elif i == 1:
                bog_tilde[j][0] = np.sum(objs[male][:, j])
                bog_tilde[j][1] = np.sum(objs[female][:, j]) 

    female = np.where(all_labels[:, -1] == 0)[0]
    male = np.where(all_labels[:, -1] == 1)[0]
    all_preds, all_labels = all_preds[:, :-1], all_labels[:, :-1]
    for i, objs in enumerate([all_preds]):
        for j in range(total_labels):
            bog_gt_g[j][0] = np.sum(objs[male][:, j])
            bog_gt_g[j][1] = np.sum(objs[female][:, j])
    return bog_tilde, bog_gt_g, bog_gt_o, bog_preds

all_masks = [0, 9, 7]

ats = [[] for _ in range(len(all_masks))]
tas = [[] for _ in range(len(all_masks))]

for i, model in enumerate(all_masks):
    for j in range(5):
        try:
            loss_info = pickle.load(open('models/mask{0}_v{1}/loss_info.pkl'.format(model, j), 'rb'))
        except:
            print("File not found: {}".format('models/mask{0}_v{1}/loss_info.pkl'.format(model, j)))

        epoch = np.argmin(loss_info['val_loss'])
        test_probs, test_labels = loss_info['test_probs'][epoch], loss_info['test_labels'][epoch]
        val_probs, val_labels = loss_info['val_probs'][epoch], loss_info['val_labels'][epoch]

        num_men = np.sum(test_labels[:, -1])
        num_attributes = [num_men, len(test_labels) - num_men]

        thresholds = []
        all_ats = []
        ta = None
        test_preds = test_probs.copy()
        for l in range(len(val_labels[0])):

            # calibration is used to pick threshold
            this_thresholds = np.sort(val_probs[:, l].flatten())
            calib = this_thresholds[-int(np.sum(val_labels[:, l]))-1]
            pred_num = int(np.sum(val_probs[:, l] > calib))
            actual_num = int(np.sum(val_labels[:, l]))
            if pred_num != actual_num:
                uniques = np.sort(np.unique(val_probs[:, l].flatten()))
                next_calib = uniques[list(uniques).index(calib)+1]
                next_num = int(np.sum(val_probs[:, l] > next_calib))
                if np.absolute(next_num - actual_num) < np.absolute(pred_num - actual_num):
                    calib = next_calib
            thresholds.append(calib)
            test_preds[:, l] = test_probs[:, l] > thresholds[-1]

        these_bog_tilde, these_bog_gt_g, these_bog_gt_o, these_bog_preds = get_bogs(test_preds, test_labels)
        ats[i].append(bog_task_to_attribute(these_bog_tilde, these_bog_gt_g, toprint=False, num_attributes=num_attributes))
        tas[i].append(bog_task_to_attribute(these_bog_tilde, these_bog_gt_o, toprint=False, num_attributes=num_attributes))

names = {0: 'Baseline', 1: 'Partial Person', 2: 'No Person'}
for i in range(len(names)):
    print("-------------{}----------".format(names[i]))
    print("A->T: {0} +- {1}, ci: {2}".format(round(np.mean(ats[i]), 5), round(np.std(ats[i]), 5), round(1.96*np.std(ats[i])/np.sqrt(len(ats[i])), 5)))
    print("T->A: {0} +- {1}, ci: {2}".format(round(np.mean(tas[i]), 5), round(np.std(tas[i]), 5), round(1.96*np.std(tas[i])/np.sqrt(len(tas[i])), 5)))
    print()


         
