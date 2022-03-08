import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn import svm
import pickle
from sklearn import tree
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score
from sklearn.ensemble import BaggingClassifier
from responsibly.dataset import COMPASDataset
from sklearn.metrics import confusion_matrix
import sys
sys.path.append('..')
from utils import bog_task_to_attribute, bog_attribute_to_task

compas_ds = COMPASDataset()

# two_year_recid as gt, and decile_score is what judges get
df = compas_ds.df

thresholds = []
fps = []
bogs = []
aa_fps = []
c_fps = []
aa_acc = []
c_acc = []

this_cat = 'race'

#for category in ['race', 'sex']:
for category in [this_cat]:
    for threshold in np.arange(0, 11):
        thresholds.append(threshold)
        print("----{0}----{1}------".format(category, threshold))
        categories = list(df[category].unique())
        aa_fp = None
        c_fp = None
        for cat in categories:
            indices = np.where(np.array(df[category]) == cat)[0]
            y_true, y_pred = np.array(df['two_year_recid'])[indices], np.array(df['decile_score'])[indices] > threshold
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            if cat in ['African-American', 'Caucasian'] and category == 'race':
                print("{0}: TN - {1}, FP - {2}, FN - {3}, TP - {4}".format(cat, tn, fp, fn, tp))
                print("TPR - {0}, FPR - {1}, FNR - {2}".format(tp/(tp+fn), fp/(fp+tn), fn/(fn+tp)))
                if cat == 'African-American':
                    aa_fp = fp/(fp+tn)
                    aa_fps.append(aa_fp)
                    aa_acc.append(np.mean(y_true == y_pred))
                else:
                    c_fp = fp/(fp+tn)
                    c_fps.append(c_fp)
                    c_acc.append(np.mean(y_true == y_pred))
                print("Accuracy: {}:".format(np.mean(y_true == y_pred)))
            elif category == 'sex' and cat == 'Female':
                aa_fp = fp/(fp+tn)
                aa_fps.append(aa_fp)
                aa_acc.append(np.mean(y_true == y_pred))
            elif category == 'sex' and cat == 'Male':
                c_fp = fp/(fp+tn)
                c_fps.append(c_fp)
                c_acc.append(np.mean(y_true == y_pred))
        if category == 'race':
            fps.append(aa_fp-c_fp)
        elif category == 'sex':
            fps.append(aa_fp-c_fp)

        bog_tilde = np.zeros((2, len(categories)))
        bog_gt_g = np.zeros((2, len(categories)))
        for i in df['race'].keys():
            this_cat_now = categories.index(df[category][i])
            gt_o = df['two_year_recid'][i]
            pred_o = int(df['decile_score'][i] > threshold)
            bog_tilde[gt_o][this_cat_now] += 1
            bog_gt_g[pred_o][this_cat_now] += 1

        if category == 'race':
            bog_tilde, bog_gt_g = bog_tilde[:, 1:3], bog_gt_g[:, 1:3]
        this_bog = bog_attribute_to_task(bog_tilde, bog_gt_g)
        bogs.append(this_bog)

minority = ''
majority = ''
print("thiscat: {}".format(this_cat))
if this_cat == 'race':
    minority = 'Group 1'
    majority = 'Group 2'
elif this_cat == 'sex':
    minority = 'Female'
    majority = 'Male'

fig = plt.figure(figsize=(6, 3))
ax1 = fig.add_subplot(111)
ax1.plot(thresholds, aa_fps, label='{} FPR'.format(minority), c='C0')
ax1.plot(thresholds, c_fps, label='{} FPR'.format(majority), c='C1')
ax1.set_ylabel('FPR')
ax1.set_xlabel('Threshold')

ax2 = ax1.twinx()
ax2.plot(thresholds, np.array(bogs), label='BiasAmp  ', c='C2')
ax2.plot(thresholds, [0]*len(thresholds), '--', c='k')
ax2.set_ylabel('Bias Amplification', color='k')
fig.legend(bbox_to_anchor=(1.15, 1.1))
plt.tight_layout()
plt.savefig('recidivism_thresholds.png', dpi=400)
plt.close()
