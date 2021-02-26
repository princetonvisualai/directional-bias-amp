import itertools
import numpy as np
import pdb
import pickle
import os
import copy
from scipy.special import softmax

id2object = {0: 'toilet', 1: 'teddy_bear', 2: 'sports_ball', 3: 'bicycle', 4: 'kite', 5: 'skis', 6: 'tennis_racket', 7: 'donut', 8: 'snowboard', 9: 'sandwich', 10: 'motorcycle', 11: 'oven', 12: 'keyboard', 13: 'scissors', 14: 'chair', 15: 'couch', 16: 'mouse', 17: 'clock', 18: 'boat', 19: 'apple', 20: 'sheep', 21: 'horse', 22: 'giraffe', 23: 'person', 24: 'tv', 25: 'stop_sign', 26: 'toaster', 27: 'bowl', 28: 'microwave', 29: 'bench', 30: 'fire_hydrant', 31: 'book', 32: 'elephant', 33: 'orange', 34: 'tie', 35: 'banana', 36: 'knife', 37: 'pizza', 38: 'fork', 39: 'hair_drier', 40: 'frisbee', 41: 'umbrella', 42: 'bottle', 43: 'bus', 44: 'zebra', 45: 'bear', 46: 'vase', 47: 'toothbrush', 48: 'spoon', 49: 'train', 50: 'airplane', 51: 'potted_plant', 52: 'handbag', 53: 'cell_phone', 54: 'traffic_light', 55: 'bird', 56: 'broccoli', 57: 'refrigerator', 58: 'laptop', 59: 'remote', 60: 'surfboard', 61: 'cow', 62: 'dining_table', 63: 'hot_dog', 64: 'car', 65: 'cup', 66: 'skateboard', 67: 'dog', 68: 'bed', 69: 'cat', 70: 'baseball_glove', 71: 'carrot', 72: 'truck', 73: 'parking_meter', 74: 'suitcase', 75: 'cake', 76: 'wine_glass', 77: 'baseball_bat', 78: 'backpack', 79: 'sink'}

def compute_man_female_per_object_322(samples):
    count = dict()
    for i in range(80):
        count[i] = [0,0]
    for sample in samples:
        sample = sample['annotation']
        if sample[0] == 1: #man
            objs = sample[2:162]
            for j in range(80):
                if objs[2*j] == 1:
                    count[j][0] += 1
        else:
            objs = sample[162:]
            for j in range(80):
                if objs[2*j] == 1:
                    count[j][1] += 1
    return count

def compute_man_female_per_object_81(samples): #for the predicted results
    count = dict()
    for i in range(80):
        count[i] = [0,0]
    for sample in samples:
        if sample[0] == 0: #man
            for j in sample[1]:
                count[(j-2)/2][0] += 1
        else:#woman
            for j in sample[1]:
                count[(j - 162)/2][1] += 1
    return count
def my_fast_inference(output):
    """outputshould be list, num_sample*322"""
    results = list()
    top1 = list()
    for i in range(len(output)):
        output_one = output[i]
        man_score = output_one[0]
        woman_score = output_one[1]
        man_objects = output_one[2:162]
        woman_objects = output_one[162:]
        man_index = list()
        woman_index = list()

        for j in range(80):
            if man_objects[j*2] > man_objects[j*2+1]:
                man_index.append(j)
                man_score += man_objects[j*2]
            else:
                man_score += man_objects[j*2+1]

        for j in range(80):
            if woman_objects[j*2] > woman_objects[j*2+1]:
                woman_index.append(j)
                woman_score += woman_objects[j*2]
            else:
                woman_score += woman_objects[j*2+1]

        result = list()
        result_num = [0]*81
        if man_score > woman_score:
            result.append("man")
            tmp = []
            for elem in man_index:
                tmp.append(elem)
            top1.append((0, tmp))
        else:
            result.append("woman")
            tmp = []
            for elem in woman_index:
                result.append(id2object[elem])
                tmp.append(elem)
            top1.append((1, tmp))
        results.append(result)
    return top1

def my_inference(output, threshold=0.):
    """outputshould be list, num_sample*322"""
    results = list()
    top1 = list()
    for i in range(len(output)):
        output_one = output[i]
        man_score = output_one[0]
        woman_score = output_one[1]
        man_objects = output_one[2:162]
        woman_objects = output_one[162:]
        man_index = list()
        woman_index = list()
        man_probs = list()
        woman_probs = list()

        man_logits = []
        woman_logits = []

        for j in range(80):
            man_probs.append(softmax([man_objects[j*2], man_objects[j*2+1]])[0])
            man_logits.append(man_objects[j*2:j*2+2])
            if man_objects[j*2] > man_objects[j*2+1]:
                man_index.append(j)
                man_score += man_objects[j*2]
            else:
                man_score += man_objects[j*2+1]

        for j in range(80):
            woman_logits.append(woman_objects[j*2:j*2+2])
            woman_probs.append(softmax([woman_objects[j*2], woman_objects[j*2+1]])[0])
            if woman_objects[j*2] > woman_objects[j*2+1]:
                woman_index.append(j)
                woman_score += woman_objects[j*2]
            else:
                woman_score += woman_objects[j*2+1]

        result = list()
        result_num = [0]*81
        man_prob = softmax([man_score, woman_score])[0]
        gender_logit = [man_score, woman_score]
        if man_score > woman_score + threshold: 
            result.append("man")
            tmp = []
            for elem in man_index:
                tmp.append(elem)
            top1.append((0, tmp, man_probs, man_prob, np.concatenate([man_logits, [gender_logit]], axis=0)))
        else:
            result.append("woman")
            tmp = []
            for elem in woman_index:
                result.append(id2object[elem])
                tmp.append(elem)
            top1.append((1, tmp, woman_probs, man_prob, np.concatenate([woman_logits, [gender_logit]], axis=0)))
        results.append(result)
    return top1


