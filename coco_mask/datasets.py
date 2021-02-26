import torch.utils.data as data
from pycocotools.coco import COCO
from PIL import Image
import csv
import numpy as np
import os
import pickle
import torch
import xml.etree.ElementTree as ET
import re
from lxml import etree
import cv2
import json
import os

TOTAL_LABELS = 66

class CoCoDataset(data.Dataset):

    def __init__(self, transform, version='train', mask_person=0): # version is train, val, or test
        self.transform = transform
        self.version = version
        self.mask_person = mask_person

        self.img_folder = '/n/fs/visualai-scr/Data/Coco/2014data/{}2014'.format(version)
        self.coco = COCO('/n/fs/visualai-scr/Data/Coco/2014data/annotations/instances_{}2014.json'.format(version))
        gender_data = pickle.load(open('/n/fs/visualai-scr/Data/Coco/2014data/bias_splits/{}.data'.format(version), 'rb'))
        self.gender_info = {int(chunk['img'][10+len(version):22+len(version)]): chunk['annotation'][0] for chunk in gender_data}

        ids = list(self.coco.anns.keys())
        self.image_ids = list(set([self.coco.anns[this_id]['image_id'] for this_id in ids]))


        # dataset will only include gendered people:
        self.image_ids = list(set(self.gender_info.keys()) & set(self.image_ids))
        print("version: {0}, num: {1}".format(version, len(self.image_ids)))

        kept_objs = open('/n/fs/biasamp-scr/biasamp-arrow/mals/data/objs').readlines()
        kept_objs = [obj.rstrip() for obj in kept_objs]

        cats = self.coco.loadCats(self.coco.getCatIds())
        self.labels_to_names = {}
        for cat in cats:
            if cat['name'].replace(' ', '_') in kept_objs:
                self.labels_to_names[cat['id']] = cat['name']

        self.categories = list(self.labels_to_names.keys())
        #self.categories = list(pickle.load(open('/n/fs/biasamp-scr/biasamp-arrow/coco_mask/coco_labels_to_keep.pkl', 'rb')))

        
    def __getitem__(self, index):
        image_id = self.image_ids[index]
        path = self.coco.loadImgs(int(image_id))[0]["file_name"]

        file_path = os.path.join(self.img_folder, path)
        return self.from_path(file_path)

    def __len__(self):
        return len(self.image_ids)


    def from_path(self, file_path):
        image_id = int(os.path.basename(file_path)[-16:-4])

        image = np.array(Image.open(file_path).convert("RGB"))

        annIds = self.coco.getAnnIds(imgIds=image_id);
        coco_anns = self.coco.loadAnns(annIds) # coco is [x, y, width, height]
        formatted_anns = []
        target = np.zeros(TOTAL_LABELS+1)
        image.setflags(write=1)

        all_people_masks = np.ones((image.shape[0], image.shape[1]))
        all_other_masks = np.ones((image.shape[0], image.shape[1]))
        
        for ann in coco_anns:
            if ann['category_id'] not in self.categories:
                continue
            target[self.categories.index(ann['category_id'])] = 1

            if ann['category_id'] == 1:
                if self.mask_person == 1:
                    x, y, w, h = ann['bbox']
                    x, y, w, h = int(x), int(y), int(w), int(h)
                    image[y:y+h, x:x+w, :] = 0

                elif self.mask_person == 2:
                    labelMask = np.expand_dims(1 - self.coco.annToMask(ann), 2)
                    image = image * labelMask
                elif self.mask_person == 3:
                    labelMask = self.coco.annToMask(ann)

                    # randomly put half of the mask back
                    random_half = np.ones_like(labelMask).flatten()
                    random_half[:len(random_half)//2] = 0
                    np.random.shuffle(random_half)
                    random_half = random_half.reshape(labelMask.shape)
                    
                    labelMask = labelMask * random_half
                    labelMask = np.expand_dims(1 - labelMask, 2)
                    
                    image = image * labelMask
                elif self.mask_person == 4: # just person
                    labelMask = np.expand_dims(self.coco.annToMask(ann), 2)
                    image = image * labelMask
                elif self.mask_person == 5:
                    labelMask = np.expand_dims(self.coco.annToMask(ann), 2)
                    new_mask = np.zeros_like(labelMask)
                    possible_mid = int(np.sqrt(np.sum(labelMask)))
                    possible_ws = np.arange(10) + (possible_mid - 5)
                    w = possible_ws[np.argmin([np.sum(labelMask) % pos if pos > 1 else float('inf') for pos in possible_ws])]
                    h = np.sum(labelMask) // w
                    h, w = int(np.clip(h, 1, len(labelMask[0]) - 1)), int(np.clip(w, 1, len(labelMask) - 1))
                    y = np.random.randint(len(labelMask[0]) - h)
                    x = np.random.randint(len(labelMask) - w)
                    new_mask[x:x+w, y:y+h] = 1.
                    new_mask = 1 - new_mask
                    image = image * new_mask

                elif self.mask_person == 6:
                    labelMask = np.expand_dims(self.coco.annToMask(ann), 2)
                    new_mask = np.zeros_like(labelMask)
                    possible_mid = int(np.sqrt(np.sum(labelMask)))
                    possible_ws = np.arange(10) + (possible_mid - 5)
                    w = possible_ws[np.argmin([np.sum(labelMask) % pos if pos > 1 else float('inf') for pos in possible_ws])]
                    h = np.sum(labelMask) // w
                    h, w = int(np.clip(h, 1, len(labelMask[0]) - 1)), int(np.clip(w, 1, len(labelMask) - 1))
                    y = np.random.randint(len(labelMask[0]) - h)
                    x = np.random.randint(len(labelMask) - w)
                    new_mask[x:x+w, y:y+h] = 1.
                    image = image * new_mask
                elif self.mask_person == 7: # masking the person, but adds back in any objects
                    all_people_masks = all_people_masks * (1. - self.coco.annToMask(ann))
                elif self.mask_person == 8: # masking the objects, but adds back in the person
                    all_people_masks = all_people_masks * (1. - self.coco.annToMask(ann))
                elif self.mask_person == 9: # like 7 except now only masking half of the person
                    all_people_masks = all_people_masks * (1. - self.coco.annToMask(ann))
            else:
                if self.mask_person == 7:
                    all_other_masks = all_other_masks * (1. - self.coco.annToMask(ann))
                elif self.mask_person == 8:
                    all_other_masks = all_other_masks * (1. - self.coco.annToMask(ann))
                if self.mask_person == 9:
                    all_other_masks = all_other_masks * (1. - self.coco.annToMask(ann))

        if self.mask_person == 7:
            all_other_masks, all_people_masks = all_other_masks.astype(int), all_people_masks.astype(int)
            just_person_mask = np.expand_dims(1. - np.bitwise_and(all_other_masks, 1 - all_people_masks), 2).astype(np.uint8)
            image = image * just_person_mask
        elif self.mask_person == 8:
            all_other_masks, all_people_masks = all_other_masks.astype(int), all_people_masks.astype(int)
            just_objects_mask = np.expand_dims(1. - np.bitwise_and(all_people_masks, 1 - all_other_masks), 2).astype(np.uint8)
            image = image * just_objects_mask
        elif self.mask_person == 9:
            #from PIL import Image as pilimage
            #im = pilimage.fromarray(image)
            #im.save('outfile.jpeg')

            all_other_masks, all_people_masks = all_other_masks.astype(int), all_people_masks.astype(int)

            #just_person_mask = np.expand_dims(1. - np.bitwise_and(all_other_masks, 1 - all_people_masks), 2).astype(np.uint8)
            #image1 = image * just_person_mask
            #im = pilimage.fromarray(image1)
            #im.save('outfile1.jpeg')

            # randomly put half of the mask back
            all_people_masks = 1 - all_people_masks
            random_half = np.ones_like(all_people_masks).flatten()
            random_half[:len(random_half)//2] = 0
            np.random.shuffle(random_half)
            random_half = random_half.reshape(all_people_masks.shape)
            all_people_masks = all_people_masks * random_half
            all_people_masks = 1 - all_people_masks
            just_person_mask = np.expand_dims(1. - np.bitwise_and(all_other_masks, 1 - all_people_masks), 2).astype(np.uint8)
            image = image * just_person_mask

            #im = pilimage.fromarray(image)
            #im.save('outfile2.jpeg')

        image = Image.fromarray(image)
        target[TOTAL_LABELS] = self.gender_info[image_id]

        
        image = self.transform(image)


        return image, target

