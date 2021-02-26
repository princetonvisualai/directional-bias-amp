import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('..')
from utils import bog_task_to_attribute, bog_attribute_to_task, bog_mals

def fpr_tpr_differences(bog_tilde, bog_gt_a):
    women_paint_diff = bog_gt_a[0][0] - bog_tilde[0][0]
    women_fp = max(women_paint_diff, 0)
    women_fpr = women_fp / bog_tilde[1][0]

    men_paint_diff = bog_gt_a[0][1] - bog_tilde[0][1]
    men_fp = max(men_paint_diff, 0)
    men_fpr = men_fp / bog_tilde[1][1]

    women_tp = min(bog_gt_a[0][0], bog_tilde[0][0])
    women_tpr = women_tp / bog_tilde[0][0]

    men_tp = min(bog_gt_a[0][1], bog_tilde[0][1])
    men_tpr = men_tp / bog_tilde[1][0]

    return men_fpr - women_fpr, men_tpr - women_tpr

def acc_diff(bog_tilde, bog_gt_a):
    women_acc = np.absolute(bog_gt_a[0][0] - bog_tilde[0][0]) / (bog_tilde[0][0]+bog_tilde[1][0])
    men_acc = np.absolute(bog_gt_a[0][1] - bog_tilde[0][1]) / (bog_tilde[0][1]+bog_tilde[1][1])
    return (1.-men_acc) - (1.-women_acc)

def mean_acc_subgroups(bog_tilde, bog_gt_a):
    total = 0.
    for i in range(2):
        for j in range(2):
            total += min(bog_gt_a[i][j], bog_tilde[i][j]) / bog_tilde[i][j]
    return total / 4.

def mean_diff_pred_vs_actual(bog_tilde, bog_gt_a):
    women_pva = (bog_gt_a[0][0] - bog_tilde[0][0]) / sum(bog_tilde[:, 0])
    men_pva = (bog_gt_a[0][1] - bog_tilde[0][1]) / sum(bog_tilde[:, 1])
    return (np.absolute(women_pva)+np.absolute(men_pva))/2.

wc, mn = 30, 30
wn, mc = 10, 10
bog_tilde = np.array([[wc, mc], [wn, mn]])

metric1 = []
metric2 = []
metric3 = []
metric4 = []
metric5 = []
metric6 = []
all_ps = np.linspace(0, (wc+wn)/wc, wc+wn+1) 
for p in all_ps:
    bog_gt_a = np.array([[p*wc, mc], [wn+(1.-p)*wc, mn]])

    metric1.append(fpr_tpr_differences(bog_tilde, bog_gt_a))
    metric2.append(acc_diff(bog_tilde, bog_gt_a))
    metric3.append(mean_acc_subgroups(bog_tilde, bog_gt_a))
    metric4.append(mean_diff_pred_vs_actual(bog_tilde, bog_gt_a))
    metric5.append(bog_mals(bog_tilde, bog_gt_a, toprint=False))
    metric6.append(bog_attribute_to_task(bog_tilde, bog_gt_a, toprint=False))
    
sizea, sizeb = 3, 3
fontsize = 15
metric1_a = [chunk[0] for chunk in metric1]
metric1_b = [chunk[1] for chunk in metric1]

all_metrics = [metric1_a, metric1_b, metric2, metric3, metric4, metric5, metric6]
all_names = ['FPR Difference', 'TPR Difference', 'Accuracy Diff', 'Mean Accuracy', 'Pred vs Actual Ratios', 'BiasAmp_MALS', 'A-> T Bias Amp']

for i in range(len(all_metrics)):
    plt.figure(figsize=(sizea, sizeb))
    color = 'C0'
    if i in [5, 6]:
        color = 'C1'
    plt.plot(all_ps, all_metrics[i], c=color)
    plt.title(all_names[i], fontsize=fontsize, pad=15)
    plt.xticks([0, 1/3, 2/3, 1, 4/3], ['0', '1/4', '2/4', '3/4', '1'], fontsize=fontsize)
    if i == 5:
        yticks = [-.6, -.4, -.2, 0, .2] 
    elif i == 6:
        yticks = [-.36, -.24, -.12, 0, .12]
    else:
        yticks = [round(chunk, 2) for chunk in np.linspace(min(all_metrics[i]), max(all_metrics[i]), num=5)]
    plt.yticks(yticks, yticks, fontsize=fontsize)
    plt.tight_layout()
    plt.savefig('view/all_metrics_comparison_{}.png'.format(i), dpi=300)
    plt.close()

