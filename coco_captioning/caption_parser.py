import pickle
import re
from tqdm import tqdm
import json
import matplotlib.patches as mpatches
import seaborn as sns
from pycocotools.coco import COCO
import json
import numpy as np
import matplotlib.pyplot as plt
import string
import sys
import os
from scipy.spatial import distance

sys.path.append('..')
from coco_mask.datasets import *
from mals.inference_debias import id2object
from utils import bog_task_to_attribute, bog_attribute_to_task, bog_mals

val_dataset = CoCoDataset(None, version='val')
categories = val_dataset.categories
labels_to_names = val_dataset.labels_to_names

man_words = ['man', 'boy', 'gentleman', 'male', 'men']
woman_words = ['woman', 'girl', 'lady', 'female', 'women']
def caption_to_array(captions, categories):
    this_categories = []
    gender = None
    # iterate through caption and append to this_cat
    for caption in captions:
        caption = caption.lower().translate(str.maketrans('', '', string.punctuation))
        for i in range(len(categories)):
            if labels_to_names[categories[i]].replace(' ', '') in caption.replace(' ', ''):
                if labels_to_names[categories[i]] == 'car':
                    this_caption = caption.replace(' ', '')
                    is_car = False
                    while 'car' in this_caption:
                        if this_caption[this_caption.find('car'):this_caption.find('car')+len('carrot')] == 'carrot':
                            this_caption = this_caption[this_caption.find('car')+2:]
                        else:
                            is_car = True
                            break
                    if not is_car:
                        continue
                if labels_to_names[categories[i]] == 'dog':
                    this_caption = caption.replace(' ', '')
                    is_dog = False
                    while 'dog' in this_caption:
                        if this_caption[max(0, this_caption.find('dog')-3):this_caption.find('dog')+3] == 'hotdog':
                            this_caption = this_caption[this_caption.find('dog')+2:]
                        else:
                            is_dog = True
                            break
                    if not is_dog:
                        continue
                this_categories.append(i)
        if gender == -1:
            continue
        for man_word in man_words:
            if man_word in caption.split():
                if gender == 0:
                    gender = -1
                else:
                    gender = 1
        for woman_word in woman_words:
            if woman_word in caption.split():
                if gender == 1:
                    gender = -1
                else:
                    gender = 0
    if gender == -1:
        gender = None
    return list(set(this_categories)), gender

########### generate bog_tilde_train for the captions #######
if os.path.isfile('coco_captions_bog_tilde_train.pkl'):
    bog_tilde_train, num_attributes_train, num_images_train = pickle.load(open('coco_captions_bog_tilde_train.pkl', 'rb'))
else:
    
    train_dataset = CoCoDataset(None, version='train')
    categories = train_dataset.categories
    labels_to_names = train_dataset.labels_to_names
    version = 'train'
    coco = COCO('/n/fs/visualai-scr/Data/Coco/2014data/annotations/captions_{}2014.json'.format(version))
    gender_data = pickle.load(open('/n/fs/visualai-scr/Data/Coco/2014data/bias_splits/{}.data'.format(version), 'rb'))
    gender_info = {int(chunk['img'][10+len(version):22+len(version)]): chunk['annotation'][0] for chunk in gender_data}
    num_labels = len(categories)
    bog_tilde_train = np.zeros((num_labels, 2)) 
    num_attributes_train = [0, 0]
    for image_id in train_dataset.image_ids:
        if int(image_id) not in gender_info.keys():
            continue
        gender = gender_info[image_id]
        annIds = coco.getAnnIds(imgIds=image_id)
        anns = coco.loadAnns(annIds)
        captions = [chunk['caption'] for chunk in anns]
    
        gt_categories, gt_gender = caption_to_array(captions, categories)
        if gt_gender is None:
            continue
        num_attributes_train[gt_gender] += 1
        for gt_cat in gt_categories:
            if gt_cat == 0:
                continue
            bog_tilde_train[gt_cat][gt_gender] += 1
    pickle.dump([bog_tilde_train, num_attributes_train, sum(num_attributes_train)], open('coco_captions_bog_tilde_train.pkl', 'wb'))

# getting an image caption given an id
version = 'val'

all_image_ids = {}

imageid_to_captions = {}

coco = COCO('/n/fs/visualai-scr/Data/Coco/2014data/annotations/captions_{}2014.json'.format(version))
gender_data = pickle.load(open('/n/fs/visualai-scr/Data/Coco/2014data/bias_splits/{}.data'.format(version), 'rb'))
gender_info = {int(chunk['img'][10+len(version):22+len(version)]): chunk['annotation'][0] for chunk in gender_data}
women_snowboard = ['baseline_ft', 'equalizer', 'confident', 'upweight', 'balanced', 'confusion']
model_names = ['baseline_ft', 'equalizer']
for ind, model_name in enumerate(model_names):
    print("Model name: {}".format(model_name))

    if model_name in women_snowboard:
        with open('final_captions_eccv2018/{}.json'.format(model_name)) as f:
            results = json.load(f)
    else:
        results = pickle.load(open('ImageCaptioning.pytorch/vis/{}_results.pkl'.format(model_name), 'rb'))

    num_labels = len(categories)
    bog_tilde = np.zeros((num_labels, 2)) 
    bog_gt_g = np.zeros((num_labels, 2)) 
    bog_gt_o = np.zeros((num_labels, 2)) 
    bog_preds = np.zeros((num_labels, 2))

    # for outcome divergence between genders measure
    gender_accs = [[0, 0, 0], [0, 0, 0]]

    actual_nums_of_gender = [0, 0]
    predict_nums_of_gender = [0, 0]

    num_samples = 0
    for i in tqdm(range(len(results))):
        # figure out labels and stuff based on captions
        if model_name in women_snowboard:
            eval_id = results[i]['image_id']
        else:
            eval_id = results[i]['file_name'].split('/')[-1].split('_')[-1][:-4]
        if int(eval_id) not in gender_info.keys():
            continue
        gender = gender_info[int(eval_id)]
        annIds = coco.getAnnIds(imgIds=int(eval_id))
        anns = coco.loadAnns(annIds)
        captions = [chunk['caption'] for chunk in anns]

        gt_categories, gt_gender = caption_to_array(captions, categories)
        if gt_gender is None:
            continue

        pred_caption = [results[i]['caption']]
        pred_categories, pred_gender = caption_to_array(pred_caption, categories)

        if ind == 0 and gt_gender != pred_gender:
            all_image_ids[eval_id] = set(pred_categories).intersection(set(gt_categories))
            imageid_to_captions[eval_id] = [pred_caption, None]
        if ind == 1:
            if eval_id in all_image_ids.keys():
                if gt_gender != pred_gender:
                    del all_image_ids[eval_id]
                else:
                    imageid_to_captions[eval_id][1] = pred_caption
                    wrong_cats = set(pred_categories).symmetric_difference(set(gt_categories))
                    all_image_ids[eval_id] = all_image_ids[eval_id].intersection(wrong_cats)


        if pred_gender is None: # if not predict gender, skip
            gender_accs[gt_gender][2] += 1
            continue

        if gt_gender != pred_gender and pred_gender is not None:
            gender_accs[gt_gender][1] += 1
        else:
            gender_accs[gt_gender][0] += 1

        num_samples += 1
        actual_nums_of_gender[gt_gender] += 1
        predict_nums_of_gender[pred_gender] += 1



        for gt_cat in gt_categories:
            if gt_cat == 0:
                continue
            bog_tilde[gt_cat][gt_gender] += 1
            bog_gt_o[gt_cat][pred_gender] += 1
            
        for pred_cat in pred_categories:
            if pred_cat == 0:
                continue
            bog_gt_g[pred_cat][gt_gender] += 1
            bog_preds[pred_cat][pred_gender] += 1

    print("Numbers of gender, ACTUAL: {0}, PRED: {1}".format(actual_nums_of_gender, predict_nums_of_gender))
    num_attributes = actual_nums_of_gender
    diff_ta, t_to_a_value = bog_task_to_attribute(bog_tilde, bog_gt_o, num_attributes=num_attributes, disaggregate=True, num_attributes_train=num_attributes_train, bog_tilde_train=bog_tilde_train)
    diff_at, a_to_t_value = bog_attribute_to_task(bog_tilde, bog_gt_g, num_attributes=num_attributes, disaggregate=True, num_attributes_train=num_attributes_train, bog_tilde_train=bog_tilde_train)
    bog_mals(bog_tilde_train, bog_preds)

    if ind == 0:
        base = [diff_ta, diff_at, bog_tilde, bog_gt_o, bog_gt_g]
    elif ind == 1:
        equalize = [diff_ta, diff_at, bog_tilde, bog_gt_o, bog_gt_g]

    # this is gender neutral version from paper
    #print(gender_accs)
    #gender_accs[0] = 100*(gender_accs[0] / np.sum(gender_accs[0][:2]))
    #gender_accs[1] = 100*(gender_accs[1] / np.sum(gender_accs[1][:2]))
    #print("Outcome divergence: {}".format(distance.jensenshannon(gender_accs[0][:2], gender_accs[1][:2])))
    #gender_accs[0] = 100*(gender_accs[0] / np.sum(gender_accs[0]))
    #gender_accs[1] = 100*(gender_accs[1] / np.sum(gender_accs[1]))
    #print("Outcome divergence: {}".format(distance.jensenshannon(gender_accs[0], gender_accs[1])))

    print("---")


print("--------- Comparing between baseline and equalizer --------")
for i in range(len(diff_at)):
    for j in range(len(diff_at[0])):
        if base[0][i][j] > equalize[0][i][j] and base[1][i][j] < equalize[1][i][j] and equalize[1][i][j] > 0: # t->a goes down, a->t goes up
            print("{0} ({9}) - {1}\nbase T->A: {2}, A->T: {3}\nequalizer T->A: {4}, A->T: {5}\nnumbers for A->T: {6} to base: {7}, equalizer: {8}\n\n".format(labels_to_names[categories[i]], 'woman' if j == 0 else 'man', base[0][i][j], base[1][i][j], equalize[0][i][j], equalize[1][i][j], base[2][i], base[4][i], equalize[4][i], i))

###### qualitative image examples ######
for image_id in all_image_ids.keys():
    if len(all_image_ids[image_id]) > 0:
        print("{0}:\nbaseline: {1}\nequalizer: {2}\n\n".format(image_id, imageid_to_captions[image_id][0], imageid_to_captions[image_id][1]))
