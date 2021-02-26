import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import numpy as np
import csv
import inference_debias as cocoutils
import sys
from sklearn.metrics import average_precision_score, f1_score
from tqdm import tqdm
from scipy.special import softmax
import sklearn.metrics as metrics
from inference_debias import id2object
sys.path.append('..')
from utils import bog_task_to_attribute, bog_attribute_to_task, bog_mals

def get_bogs(all_preds, all_labels):
    TOTAL_LABELS = len(all_preds[0]) - 1
    bog_tilde = np.zeros((TOTAL_LABELS, 2)) 
    bog_gt_g = np.zeros((TOTAL_LABELS, 2)) 
    bog_gt_o = np.zeros((TOTAL_LABELS, 2)) 
    bog_preds = np.zeros((TOTAL_LABELS, 2))

    for i, objs in enumerate([all_preds, all_labels]):
        att1 = np.where(objs[:, -1] == 0)[0]
        att2 = np.where(objs[:, -1] == 1)[0]
        for j in range(TOTAL_LABELS):
            if i == 0:
                bog_preds[j][0] = np.sum(objs[att2][:, j])
                bog_preds[j][1] = np.sum(objs[att1][:, j])

                #bog_gt_o[j][0] = np.sum(all_labels[att2][:, j])
                #bog_gt_o[j][1] = np.sum(all_labels[att1][:, j]) 
            elif i == 1:
                bog_tilde[j][0] = np.sum(objs[att2][:, j])
                bog_tilde[j][1] = np.sum(objs[att1][:, j]) 

                bog_gt_o[j][0] = np.sum(all_preds[att2][:, j])
                bog_gt_o[j][1] = np.sum(all_preds[att1][:, j]) 

    att1 = np.where(all_labels[:, -1] == 0)[0]
    att2 = np.where(all_labels[:, -1] == 1)[0]
    all_preds, all_labels = all_preds[:, :-1], all_labels[:, :-1]
    for i, objs in enumerate([all_preds]):
        for j in range(TOTAL_LABELS):
            bog_gt_g[j][0] = np.sum(objs[att2][:, j])
            bog_gt_g[j][1] = np.sum(objs[att1][:, j])
    return bog_tilde, bog_gt_g, bog_gt_o, bog_preds

version = 'train'
train_data = pickle.load(open('data/{}.data'.format(version), 'rb'))

train_labels = []
for i in range(len(train_data)):
    top1 = cocoutils.my_fast_inference([train_data[i]['annotation']])
    this_label = np.zeros(81)
    this_label[-1] = 1 if top1[0][0] == 0 else 0

    for ind in top1[0][1]:
        this_label[ind] = 1
    train_labels.append(this_label)
train_labels = np.array(train_labels)
num_attributes = len(np.where(train_labels[:, -1] == 1)[0]), len(np.where(train_labels[:, -1] == 0)[0])
object2id = {v: k for k, v in id2object.items()}

# original paper only kept 66 of 80 coco objects
kept_objs = open('data/objs').readlines()
kept_objs = [obj.rstrip() for obj in kept_objs]
keep_labels = np.array([object2id[obj] for obj in kept_objs])
remove_labels = list(set(np.arange(80)).difference(set(keep_labels))) 

#remove_labels = []
bog_tilde_train, _, _, _ = get_bogs(np.ones_like(train_labels), train_labels)

train_labels = np.delete(train_labels, remove_labels, axis=1)
bog_tilde_train, _, _, _ = get_bogs(np.ones_like(train_labels), train_labels)
val_thresholds = []
test_thresholds = []
calib_thresholds = []
calib_val_thresholds = []
calib_test_thresholds = []
temperatures = []

for version in ['dev', 'test']:
    print("--------version: {}--------".format(version))
    test_data = pickle.load(open('data/{}.data'.format(version), 'rb'))
    test_potentials = pickle.load(open('data/potentials_{}'.format(version), 'rb'), encoding='latin1')

    # first convert their 322 format into a) object prediction and b) gender prediction
    preds = []
    probs = []
    labels = []
    logits = []
    for i in range(len(test_potentials)):
        #top1 = cocoutils.my_inference(test_potentials[i]['output'], threshold=2.795) # for thresholding results
        top1 = cocoutils.my_inference(test_potentials[i]['output'])
        for j in range(len(top1)):
            this_top1 = top1[j]
            this_pred = np.zeros(81)
            this_prob = np.zeros(81)
            this_pred[-1] = 1 if this_top1[0] == 0 else 0
            this_prob[-1] = this_top1[3]
            this_prob[:-1] = this_top1[2]
            for ind in this_top1[1]:
                this_pred[ind] = 1
            this_logits = this_top1[4]
            preds.append(this_pred)
            probs.append(this_prob)
            logits.append(this_logits)
    preds = np.array(preds)
    preds = np.delete(preds, remove_labels, axis=1)
    probs = np.array(probs)
    probs = np.delete(probs, remove_labels, axis=1)
    logits = np.array(logits)
    logits = np.delete(logits, remove_labels, axis=1)

    for i in range(len(test_data)):
        top1 = cocoutils.my_fast_inference([test_data[i]['annotation']])
        this_pred = np.zeros(81)
        this_pred[-1] = 1 if top1[0][0] == 0 else 0
        for ind in top1[0][1]:
            this_pred[ind] = 1
        labels.append(this_pred)
    labels = np.array(labels)
    labels = np.delete(labels, remove_labels, axis=1)

    for i in range(len(preds[0])):
        this_thresholds = np.sort(probs[:, i].flatten())
        calib_thresholds.append(this_thresholds[-int(np.sum(labels[:, i]))-1])
        if version == 'dev':
            calib_val_thresholds.append(this_thresholds[-int(np.sum(labels[:, i]))-1])
        elif version == 'test':
            calib_test_thresholds.append(this_thresholds[-int(np.sum(labels[:, i]))-1])

    if version == 'test':
        new_preds = probs.copy()
        for i in range(len(preds[0])):
            new_preds[:, i] = new_preds[:, i] > calib_val_thresholds[i]

        bog_tilde, bog_gt_g, bog_gt_o, bog_preds = get_bogs(new_preds, labels)
        num_attributes = len(np.where(labels[:, -1] == 1)[0]), len(np.where(labels[:, -1] == 0)[0])

        menshop_values = bog_mals(bog_tilde_train, bog_preds, bog_tilde_train=bog_tilde_train, bog_gt_g=bog_gt_g, bog_gt_o=bog_gt_o, toprint=False)
        gt = bog_attribute_to_task(bog_tilde, bog_gt_g, bog_tilde_train=bog_tilde_train, toprint=False, num_attributes=num_attributes)
        tg = bog_task_to_attribute(bog_tilde, bog_gt_o, bog_tilde_train=bog_tilde_train, toprint=False, num_attributes=num_attributes)

        new_preds = probs.copy()
        new_preds[:, -1] = new_preds[:, -1] > calib_val_thresholds[-1]
        new_preds[:, :-1] = new_preds[:, :-1] > .5
        bog_tilde, bog_gt_g, bog_gt_o, bog_preds = get_bogs(new_preds, labels)
        menshop_values = bog_mals(bog_tilde_train, bog_preds, bog_tilde_train=bog_tilde_train, bog_gt_g=bog_gt_g, bog_gt_o=bog_gt_o, toprint=False)
        print("MALS when gender threshold is calibrated: {}".format(menshop_values[0]))

        props = []
        this_menshops = []
        this_tgs = []
        for thresh in np.unique(probs[:, -1]):
            new_preds = probs.copy()
            new_preds[:, -1] = new_preds[:, -1] > thresh
            bog_tilde, bog_gt_g, bog_gt_o, bog_preds = get_bogs(new_preds, labels)
            props.append(np.mean(new_preds[:, -1]))
            tg = bog_task_to_attribute(bog_tilde, bog_gt_o, bog_tilde_train=bog_tilde_train, toprint=False, num_attributes=num_attributes)
            this_tgs.append(tg)
            
            new_preds[:, :-1] = new_preds[:, :-1] > .5
            bog_tilde, bog_gt_g, bog_gt_o, bog_preds = get_bogs(new_preds, labels)
            menshop_values = bog_mals(bog_tilde_train, bog_preds, bog_tilde_train=bog_tilde_train, bog_gt_g=bog_gt_g, bog_gt_o=bog_gt_o, toprint=False)
            this_menshops.append(menshop_values[0])

        plt.figure(figsize=(4, 2))
        amount = 1800
        props, this_menshops, this_tgs = props[:amount], this_menshops[:amount], this_tgs[:amount]
        plt.plot(props, this_menshops, label='MALS')
        plt.plot(props, this_tgs, label='T->A')
        calibrated_prop = np.mean(probs[:, -1] > calib_val_thresholds[-1])
        default_prop = np.mean(probs[:, -1] > .5)
        y_min = np.amin([np.amin(this_menshops), np.amin(this_tgs)])
        y_max = np.amax([np.amax(this_menshops), np.amax(this_tgs)])
        plt.plot([calibrated_prop, calibrated_prop], [y_min, y_max], ':', label='Calibrated Threshold')
        plt.plot([default_prop, default_prop], [y_min, y_max], ':', label='Default Threshold')
        plt.plot([np.amin(props), 1], [0, 0], ':', c='k')
        plt.legend(prop={'size': 8})
        plt.xlabel('Proportion Classified to be Men')
        plt.ylabel('Bias Amplification')
        plt.tight_layout()
        plt.savefig('view/biasamp_by_prop.png', dpi=400)
        plt.close()

    bog_tilde, bog_gt_g, bog_gt_o, bog_preds = get_bogs(preds, labels)

    if version == 'dev':
        with open('coco_probs.tsv', 'wt') as out_file:
            tsv_writer = csv.writer(out_file, delimiter='\t')
            tsv_writer.writerow(['object', 'p(a-hat | t-hat)', 'p(a | t)', 'p(a-hat | t)', 'p(t-hat | a)', 'p(t | a)', 'p_train(t)', 'p_train(a, t)'])
            counter = 0
            for i in range(len(id2object)):
                if i in remove_labels:
                    continue
                obj = id2object[counter]
                p_ahat_that = bog_preds[counter][0] / np.sum(bog_preds[counter])
                p_a_t = bog_tilde_train[counter][0] / np.sum(bog_tilde_train[counter])
                p_ahat_t = bog_gt_o[counter][0] / np.sum(bog_gt_o[counter])
                p_that_a = bog_gt_g[counter][0] / np.sum(bog_gt_g[:, 0])
                p_t_a = bog_tilde_train[counter][0] / np.sum(bog_tilde_train[:, 0])
                p_t = np.sum(bog_tilde_train[counter]) / len(train_data)
                p_ta = np.sum(bog_tilde_train[counter][0]) / len(train_data)
                tsv_writer.writerow([obj, p_ahat_that, p_a_t, p_ahat_t, p_that_a, p_t_a, p_t, p_ta])
                counter += 1

    num_attributes = len(np.where(labels[:, -1] == 1)[0]), len(np.where(labels[:, -1] == 0)[0])

    menshop_values = bog_mals(bog_tilde_train, bog_preds, bog_tilde_train=bog_tilde_train, bog_gt_g=bog_gt_g, bog_gt_o=bog_gt_o, toprint=False)
    gt = bog_attribute_to_task(bog_tilde, bog_gt_g, bog_tilde_train=bog_tilde_train, toprint=False, num_attributes=num_attributes, total_images=len(labels), total_images_train=len(train_labels), num_attributes_train=np.array([np.sum(1-train_labels[:, -1]), np.sum(train_labels[:, -1])]))
    tg = bog_task_to_attribute(bog_tilde, bog_gt_o, bog_tilde_train=bog_tilde_train, toprint=False, num_attributes=num_attributes, total_images=len(labels), total_images_train=len(train_labels), num_attributes_train=np.array([np.sum(1-train_labels[:, -1]), np.sum(train_labels[:, -1])]))
    #print("A->T: {0}, T->A: {1}".format(gt, tg))
    print("MALS: {}".format(menshop_values[0]))

