import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import math
import numpy as np
import string
import csv
from sklearn.metrics import average_precision_score
import sklearn.metrics as metrics
import numpy as np
import pickle
from tqdm import tqdm
import argparse
import os
import scipy.stats
from sklearn.metrics import f1_score, average_precision_score
import sys
sys.path.append('..')
from utils import bog_task_to_attribute, bog_attribute_to_task


parser = argparse.ArgumentParser(description='Making graphs for celebA')
parser.add_argument('--attribute', type=int, default=0)
parser.add_argument('--num_runs', type=int, default=5, help='number of runs per model')
parser.add_argument('--ratios_not_models', action='store_true', default=False,
                    help='When false, compares across models. When true, compares across ratios')
args = parser.parse_args()
print(args)

if args.attribute not in [7, 39]:
    print("---Axes names may not be correct for the attributes entered---")

def get_at(running_labels, running_preds):
    bog_tilde = np.zeros((2, 2)) 
    bog_gt_a = np.zeros((2, 2)) 
    gt_woman = np.where(running_labels[:, 1] == 0)[0]
    gt_man = np.where(running_labels[:, 1] == 1)[0]
    gt_att = np.where(running_labels[:, 0] == 0)[0]
    gt_noatt = np.where(running_labels[:, 0] == 1)[0]
    for i, objs in enumerate([running_labels, running_preds]):
        woman = np.where(objs[:, 1] == 0)[0]
        man = np.where(objs[:, 1] == 1)[0]
        att = np.where(objs[:, 0] == 0)[0]
        noatt = np.where(objs[:, 0] == 1)[0]
        if i == 0:
            bog_tilde[0][0] = len(set(att)&set(woman))
            bog_tilde[0][1] = len(set(att)&set(man))
            bog_tilde[1][0] = len(set(noatt)&set(woman))
            bog_tilde[1][1] = len(set(noatt)&set(man))

        elif i == 1:
            bog_gt_a[0][0] = len(set(att)&set(gt_woman))
            bog_gt_a[0][1] = len(set(att)&set(gt_man))
            bog_gt_a[1][0] = len(set(noatt)&set(gt_woman))
            bog_gt_a[1][1] = len(set(noatt)&set(gt_man))
    at = bog_attribute_to_task(bog_tilde, bog_gt_a, toprint=False)
    return at

def scale(arr, i):
    return (arr - scale_per_metric[i][0]) / (scale_per_metric[i][1] - scale_per_metric[i][0])

width = 2.7

if args.ratios_not_models:
    weight_names = {0: ['1.5', 'a'], 1: ['1.75', 'b'], 2: ['2.0', 'c'], 3: ['2.25', 'd'], 4: ['2.5', 'e']}
    plt.figure(figsize=(3.1, 1.5))
    att = args.attribute
    at_means = []
    at_intervals = []
    for i in range(len(weight_names)):
        this_at = []
        for j in range(args.num_runs):
            loss_dict = pickle.load(open('models_celeba/resnet_{0}/{1}_{2}/loss_dict.pkl'.format(weight_names[i][1], att, j), 'rb'))
            val_loss = loss_dict['val_loss']
            epoch = np.argmin(val_loss) 
            test_labels, test_probs = loss_dict['test_labels'][epoch], loss_dict['test_probs'][epoch]
            val_labels, val_probs = loss_dict['val_labels'][epoch], loss_dict['val_probs'][epoch]

            actual = np.sum(val_labels[:, 0])
            threshold = np.sort(val_probs[:, 0])[-actual-1]
            now_test_preds = test_probs.copy()
            now_test_preds[:, 0] = now_test_preds[:, 0] > threshold
            this_at.append(get_at(test_labels, now_test_preds))

        at_means.append(np.mean(this_at))
        at_intervals.append(1.96*np.std(this_at)/np.sqrt(len(this_at)))
    name = 'Big Nose' if att == 7 else 'Young'
    color = 'C0' if att == 7 else 'C1'
    (_, caps, _) = plt.errorbar(np.arange(len(weight_names)), at_means, yerr=at_intervals, marker='o', markersize=1, capsize=width+1, linestyle='None', linewidth=width, label=name, c=color)
    for cap in caps:
        cap.set_markeredgewidth(width)
    plt.xticks(np.arange(len(weight_names)), [weight_names[i][0] for i in range(len(weight_names))])
    plt.xlabel('Majority to Minority Groups Ratio')
    plt.ylabel('A->T Bias\nAmplification')
    plt.tight_layout(pad=.14)
    plt.savefig('view/graph_ratio_{}.png'.format(args.attribute), dpi=300)
    plt.close()
else:
    model_names = ['AlexNet', 'ResNet18', 'VGG16']
    att = args.attribute

    at_means = []
    at_intervals = []
    ap_means = []
    ap_intervals = []
    fp_means = []
    fp_intervals = []

    for i in range(len(model_names)):
        this_at = []
        this_ap = []
        this_fp = []
        for j in range(args.num_runs):
            loss_dict = pickle.load(open('models_celeba/{0}/{1}_{2}/loss_dict.pkl'.format(model_names[i], att, j), 'rb'))
            val_loss = loss_dict['val_loss']
            epoch = np.argmin(val_loss) 
            test_labels, test_probs = loss_dict['test_labels'][epoch], loss_dict['test_probs'][epoch]
            val_labels, val_probs = loss_dict['val_labels'][epoch], loss_dict['val_probs'][epoch]

            # bias amp
            actual = np.sum(val_labels[:, 0])
            threshold = np.sort(val_probs[:, 0])[-actual-1]
            now_test_preds = test_probs.copy()
            now_test_preds[:, 0] = now_test_preds[:, 0] > threshold
            this_at.append(get_at(test_labels, now_test_preds))

            # ap
            this_ap.append(average_precision_score(test_labels[:, 0], test_probs[:, 0]))

            # fp
            woman = np.where(test_labels[:, 1] == 0)[0]
            man = np.where(test_labels[:, 1] == 1)[0]
            nowith_att = np.where(test_labels[:, 0] == 0)[0]
            with_att = np.where(test_labels[:, 0] == 1)[0]
            keeps_test = [list(set(nowith_att)&set(woman)), list(set(nowith_att)&set(man)), list(set(with_att)&set(woman)), list(set(with_att)&set(man))]
            this_equals = np.equal(test_labels[:, 0], now_test_preds[:, 0])
            fpr_diff = (1. - np.mean(this_equals[keeps_test[1]])) - (1. - np.mean(this_equals[keeps_test[0]]))
            this_fp.append(fpr_diff)


        at_means.append(np.mean(this_at))
        at_intervals.append(1.96*np.std(this_at)/np.sqrt(len(this_at)))
        ap_means.append(np.mean(this_ap))
        ap_intervals.append(1.96*np.std(this_ap)/np.sqrt(len(this_ap)))
        fp_means.append(np.mean(this_fp))
        fp_intervals.append(1.96*np.std(this_fp)/np.sqrt(len(this_fp)))
    name = 'Big Nose' if att == 7 else 'Young'
    color = 'C0' if att == 7 else 'C1'

    yaxes = ['BiasAmp', 'AP (%)', 'FPR Diff']
    mean_inters = [[at_means, at_intervals], [ap_means, ap_intervals], [fp_means, fp_intervals]]
    plot_names = ['at', 'ap', 'fp']

    for k in range(len(yaxes)):
        plt.figure(figsize=(2.2, 1.5))
        mult = 1.
        if k == 1:
            mult = 100.
        (_, caps, _) = plt.errorbar(np.arange(len(model_names))+.5, mult*np.array(mean_inters[k][0]), yerr=mult*np.array(mean_inters[k][1]), marker='o', markersize=1, capsize=width+1, elinewidth=width, linestyle='None', linewidth=width, label=name, c=color)
        for cap in caps:
            cap.set_markeredgewidth(width)
        if k == 1 and att == 39:
            plt.yticks([81, 82, 83, 84], ['81.0', '82.0', '83.0', '84.0'])
        plt.xticks(np.arange(len(model_names)+1), ['']*(len(model_names)+1))
        plt.xlabel('Model Architecture')
        plt.ylabel(yaxes[k])
        plt.tight_layout(pad=.14)
        plt.savefig('view/graph_model_{0}_{1}.png'.format(att, plot_names[k]), dpi=300)
        plt.close()











