import matplotlib
matplotlib.use('Agg')
from celeba_dset import CelebA
import os
import copy
import random
import pickle
import numpy as np
import argparse
import matplotlib.pyplot as plt
from torchvision import transforms
import torch.nn.functional as F
from torchvision import models
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.optim import lr_scheduler
from torch import optim
from torchvision.utils import make_grid, save_image
import torchvision
from tqdm import tqdm
import cv2
import time
from sklearn.linear_model import LogisticRegression
import itertools
import sys
sys.path.append('..')
from utils import bog_task_to_attribute, bog_attribute_to_task

def get_attribute_set(attr, attribute_set):
    if attribute_set < 0:
        if attribute_set == -20:
            attribute_set = 0
        keep_noatt = np.where(np.array(attr[:, -attribute_set]) == 0)[0]
        keep_woman = np.where(np.array(attr[:, 20]) == 0)[0]
        keep_att = np.where(np.array(attr[:, -attribute_set]) == 1)[0]
        keep_man = np.where(np.array(attr[:, 20]) == 1)[0]
    return keep_noatt, keep_woman, keep_att, keep_man

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

def train(args, model, device, train_loader, optimizer, criterion, epoch, twoheads, file, loss_dict):
    if twoheads is not None:
        head1, head2, optimizer1, optimizer2 = twoheads
        head1.train()
        head2.train()
    model.train()
    running_loss = []
    running_labels = []
    running_preds = []
    running_probs = []

    for batch_idx, (data, target) in (enumerate(tqdm(train_loader)) if args.interact else enumerate(train_loader)):
        if args.criterion == 0:
            data, target = data.to(device), target.to(device).long()
        elif args.criterion == 2:
            data, target = data.to(device), target.to(device).float()

        optimizer.zero_grad()
        output = model.forward(data)

        if args.twohead:
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            output = torch.cat([head1(output), head2(output)], 1)

        if args.criterion == 0:
            softmax = nn.Softmax(dim=1)
            a_preds = torch.round(torch.cat([softmax(output[:, :2]), softmax(output[:, 2:])], 1)).data.cpu().numpy()
            a_preds = np.vstack([np.argmax(a_preds[:, :2], axis=1), np.argmax(a_preds[:, 2:], axis=1)]).T
            b_preds = torch.cat([softmax(output[:, :2]), softmax(output[:, 2:])], 1).data.cpu().numpy()
            b_preds = np.vstack([b_preds[:, 1], b_preds[:, 3]]).T
            running_probs.extend(b_preds)
            running_preds.extend(a_preds)
        elif args.criterion == 2:
            running_preds.extend(torch.round(output).data.cpu().numpy())
            running_probs.extend(output.data.cpu().numpy())
        running_labels.extend(target.data.cpu().numpy())

        if args.criterion == 0:
            if args.training_version == 1:
                if args.twohead:
                    if epoch % args.adv_interval == 0:
                        loss = criterion(output[:, 2:], target[:, 1])
                    else:
                        confusion = - torch.clamp(criterion(output[:, 2:], target[:, 1]), 0., args.random_gen_ce)
                        loss = criterion(output[:, :2], target[:, 0]) + confusion
                else:
                    assert NotImplementedError
            else:
                loss = criterion(output[:, :2], target[:, 0])# + criterion(output[:, 2:], target[:, 1])
        elif args.criterion == 2:
            loss = criterion(output, target).mean()

        if args.weighting_version in [1, 2]:
            these_labels = target.data.cpu().numpy()
            if args.criterion == 0:
                keep_noatt = np.where(np.array(these_labels[:, 0]) == 1)[0]
                keep_man = np.where(np.array(these_labels[:, 1]) == 1)[0]
                keep_att = np.where(np.array(these_labels[:, 0]) == 0)[0]
                keep_woman = np.where(np.array(these_labels[:, 1]) == 0)[0]
            else:
                raise NotImplementedError

            keeps_a = list(set(keep_noatt) & set(keep_man)) 
            keeps_b = list(set(keep_att) & set(keep_woman)) 
            keeps_c = list(set(keep_noatt) & set(keep_woman)) 
            keeps_d = list(set(keep_att) & set(keep_man)) 

            weights = torch.ones(len(loss))
            weights[keeps_a] = args.bal_weights[0]
            weights[keeps_b] = args.bal_weights[1]
            weights[keeps_c] = args.bal_weights[2]
            weights[keeps_d] = args.bal_weights[3]
            weights = weights.to(device)
            loss = loss * weights
            loss = loss.mean()

        loss.backward()
        #if args.training_version == 1:
        #    torch.nn.utils.clip_grad_norm_(model.parameters(), .25)
        #    torch.nn.utils.clip_grad_norm_(head1.parameters(), .25)
        #    torch.nn.utils.clip_grad_norm_(head2.parameters(), .25)
        if args.training_version == 1:
            if epoch % args.adv_interval == 0:
                optimizer2.step()
            else:
                optimizer.step()
                optimizer1.step()
        else:
            optimizer.step()
            if args.twohead:
                optimizer1.step()
                optimizer2.step()

        running_loss.append(loss.item() * len(target))

        if batch_idx % 1000 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    running_labels, running_probs, running_preds = np.array(running_labels), np.array(running_probs), np.array(running_preds)
    total_batch_loss = np.sum(running_loss) / len(running_preds)
    total_att_acc = np.mean(np.equal(np.array(running_labels[:, 0]), np.array(running_preds[:, 0])))
    total_gen_acc = np.mean(np.equal(np.array(running_labels[:, 0]), np.array(running_preds[:, 0])))
    
    print("Train Epoch {}".format(epoch))
    print("Loss is: {}".format(total_batch_loss))
    loss_dict['train_loss'].append(total_batch_loss)
    loss_dict['train_att_accs'].append(total_att_acc)
    loss_dict['train_gen_accs'].append(total_gen_acc)
    loss_dict['train_labels'].append(running_labels)
    loss_dict['train_probs'].append(running_probs)
    print("------")


    if file is not None:
        file.write("\nEpoch: {0}\n".format(epoch))
        file.write("Loss: {}\n".format(total_batch_loss))
        file.write("Train Att Accuracy: {}\n".format(total_att_acc))

    if args.save_model:
        pickle.dump(loss_dict, open("models_celeba/{}/loss_dict.pkl".format(args.model_name), 'wb'))
    #if epoch % 2 == 0:
    #    torch.save(model.state_dict(),"models_celeba/{0}/model_{1}.pt".format(args.model_name, epoch))
    #    if args.twohead:
    #        torch.save(head1.state_dict(), "models_celeba/{0}/head1_{1}.pt".format(args.model_name, epoch))
    #        torch.save(head2.state_dict(), "models_celeba/{0}/head2_{1}.pt".format(args.model_name, epoch))
    return loss_dict


def test(args, model, device, criterion, test_loader, epoch, optimizer, twoheads, file, loss_dict):
    test_loader, valid_loader = test_loader # reversed

    if twoheads is not None:
        head1, head2, optimizer1, optimizer2 = twoheads
        head1.eval()
        head2.eval()
    running_loss = []
    model.eval()
    running_preds = []
    running_probs = []
    running_labels = []

    torch.set_grad_enabled(False)
    for batch_idx, (data, target) in (enumerate(tqdm(test_loader)) if args.interact else enumerate(test_loader)):
        optimizer.zero_grad()

        if args.twohead:
            optimizer1.zero_grad()
            optimizer2.zero_grad()
        if args.criterion == 0:
            data, target = data.to(device), target.to(device).long()
        elif args.criterion == 2:
            data, target = data.to(device), target.to(device).float()
        
        output = model.forward(data)#.float()
        if args.twohead:
            output = torch.cat([head1(output), head2(output)], 1).detach()
        
        if args.criterion == 0:
            loss = criterion(output[:, :2], target[:, 0])
        elif args.criterion == 2:
            loss = criterion(output, target).mean()
        if args.weighting_version in [1, 2]:
            these_labels = target.data.cpu().numpy()
            if args.criterion == 0:
                keep_noatt = np.where(np.array(these_labels[:, 0]) == 1)[0]
                keep_man = np.where(np.array(these_labels[:, 1]) == 1)[0]
                keep_att = np.where(np.array(these_labels[:, 0]) == 0)[0]
                keep_woman = np.where(np.array(these_labels[:, 1]) == 0)[0]

            else:
                raise NotImplementedError

            keeps_a = list(set(keep_noatt) & set(keep_man)) 
            keeps_b = list(set(keep_att) & set(keep_woman)) 
            keeps_c = list(set(keep_noatt) & set(keep_woman)) 
            keeps_d = list(set(keep_att) & set(keep_man)) 

            weights = torch.ones(len(target))
            weights[keeps_a] = args.bal_weights[0]
            weights[keeps_b] = args.bal_weights[1]
            weights[keeps_c] = args.bal_weights[2]
            weights[keeps_d] = args.bal_weights[3]
            weights = weights.to(device)
            loss = loss * weights
            loss = loss.mean()

        if args.criterion == 0:
            softmax = nn.Softmax(dim=1)
            a_preds = torch.round(torch.cat([softmax(output[:, :2]), softmax(output[:, 2:])], 1)).data.cpu().numpy()
            a_preds = np.vstack([np.argmax(a_preds[:, :2], axis=1), np.argmax(a_preds[:, 2:], axis=1)]).T
            b_preds = torch.cat([softmax(output[:, :2]), softmax(output[:, 2:])], 1).data.cpu().numpy()
            b_preds = np.vstack([b_preds[:, 1], b_preds[:, 3]]).T
            running_probs.extend(b_preds)
            running_preds.extend(a_preds)
        elif args.criterion == 2:
            running_preds.extend(torch.round(output).data.cpu().numpy())
            running_probs.extend(output.data.cpu().numpy())
        running_labels.extend(target.data.cpu().numpy())
        running_loss.append(loss.mean().item() * len(target))
    torch.set_grad_enabled(True)

    total_batch_loss = np.sum(running_loss) / len(running_labels)
    running_labels, running_preds, running_probs = np.array(running_labels), np.array(running_preds), np.array(running_probs)
  
    if args.attribute_set < 0 and args.dataset_version == 1:
        att_acc = np.mean(running_labels[:, 0] == running_preds[:, 0])
        gen_acc = np.mean(running_labels[:, 1] == running_preds[:, 1])

        loss_dict['val_att_accs'].append(att_acc)
        loss_dict['val_gen_accs'].append(gen_acc)
    
        test_labels, test_preds, test_probs = [], [], []
        with torch.no_grad():
            for batch_idx, (data, target) in (enumerate(tqdm(valid_loader)) if args.interact else enumerate(valid_loader)):
                if args.criterion == 0:
                    data, target = data.to(device), target.to(device).long()
                elif args.criterion in [1, 2]:
                    data, target = data.to(device), target.to(device).float()
            
                output = model.forward(data).detach()
                if args.twohead:
                    output = torch.cat([head1(output), head2(output)], 1).detach()

                test_labels.extend(target.data.cpu().numpy())

                if args.criterion == 0:
                    softmax = nn.Softmax(dim=1)
                    a_preds = torch.round(torch.cat([softmax(output[:, :2]), softmax(output[:, 2:])], 1)).data.cpu().numpy()
                    a_preds = np.vstack([np.argmax(a_preds[:, :2], axis=1), np.argmax(a_preds[:, 2:], axis=1)]).T
                    b_preds = torch.cat([softmax(output[:, :2]), softmax(output[:, 2:])], 1).data.cpu().numpy()
                    b_preds = np.vstack([b_preds[:, 1], b_preds[:, 3]]).T
                    test_probs.extend(b_preds)
                    test_preds.extend(a_preds)
                elif args.criterion == 2:
                    test_preds.extend(torch.round(output).data.cpu().numpy())
                    test_probs.extend(output.data.cpu().numpy())
        test_labels, test_probs, test_preds = np.array(test_labels), np.array(test_probs), np.array(test_preds)

        loss_dict['val_labels'].append(running_labels)
        loss_dict['val_probs'].append(running_probs)
        loss_dict['test_labels'].append(test_labels)
        loss_dict['test_probs'].append(test_probs)
    print("Test Epoch {}".format(epoch))
    print("Loss is: {}".format(total_batch_loss))
    loss_dict['val_loss'].append(total_batch_loss)

    if file is not None and epoch != -1:
        file.write("Test Loss: {}\n".format(total_batch_loss))
    return loss_dict

def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--model_name', type=str, default="model", 
                        help='path to model to load, or model to save')
    parser.add_argument('--optimizer', type=str, default="sgd", 
                        help='sgd adam etc')
    parser.add_argument('--criterion', type=int, default=0, 
                        help='what loss criterion, 0 is the softmax pre task so crossentropyloss, 2 is 0 but no reduce')
    parser.add_argument('--dataset_version', type=float, default=1,
                        help='dataset version: 0 is regular with all 40 attributes, '+
                        '1 is whatever the data comes with for attribute_set specified')
    parser.add_argument('--attribute_set', type=int, default=0,
                        help='all performed on dataset version 3 only: ' +
                        'negatives is the actual attribute, so 0 is changed so that 20 of gender is actually 0 of shadow')
    parser.add_argument('--training_version', type=int, default=0,
                        help='training version: 0 is regular not predicting gender just 1 attribute' +
                        '1 is adversarial so features do not know gender')
    parser.add_argument('--model_type', type=str, default="multi", 
                        help='model type, like vgg')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--load-model', action='store_true', default=False,
                        help='if loading a model')
    parser.add_argument('--continue-model', action='store_true', default=False,
                        help='if loading a model, and this is true, then continue training from here')
    parser.add_argument('--interact', action='store_true', default=False,
                        help='if running interactively use tqdm, else do not')
    parser.add_argument('--weighting_version', type=int, default=0,
                        help='for dataset version 3, only done on training set, 0 is nothing'+
                        '1 is makes equal the 4 subgroups, '+
                        '2 is sets the ratio between majority and minority groups to be what weighting_ratio is')
    parser.add_argument('--weighting_ratio', type=float, default=.0,
                        help='the ratio for when 2 is set in above')
    parser.add_argument('--adv_interval', type=int, default=3,
                        help='interval that training version 1 does adversarial training')
    args = parser.parse_args()

    print("args: {}".format(args))
    use_cuda = torch.cuda.is_available()

    args.seed = random.randint(1, 10000)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    print("Device: {}".format(device))
    dataroot = '/n/fs/visualai-scr/Data/CelebA'
    image_size = (224, 224)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    args.twohead = False
    workers = 1

    train_dataset = CelebA(dataroot, split="train", download=True, transform=transforms.Compose([
                                   transforms.Resize(image_size),
                                   # transforms.CenterCrop(image_size),
                                   transforms.ToTensor(),
                                   normalize,
                               ]))
    valid_dataset = CelebA(dataroot, split="valid", download=True, transform=transforms.Compose([
                                   transforms.Resize(image_size),
                                   # transforms.CenterCrop(image_size),
                                   transforms.ToTensor(),
                                   normalize,
                               ]))
    test_dataset = CelebA(dataroot, split="test", download=True, transform=transforms.Compose([
                                   transforms.Resize(image_size),
                                   # transforms.CenterCrop(image_size),
                                   transforms.ToTensor(),
                                   normalize,
                               ]))

    out_features = 40

    if args.dataset_version == 1:
        assert args.criterion == 0
        assert args.attribute_set < 0
        out_features = 4

        for j, dset in enumerate([train_dataset, valid_dataset, test_dataset]):
            attr = dset.attr
            keep_noatt, keep_man, keep_att, keep_woman = get_attribute_set(attr, args.attribute_set)
     
            keeps_a = list(set(keep_noatt) & set(keep_man)) 
            keeps_b = list(set(keep_att) & set(keep_woman)) 
            keeps_c = list(set(keep_noatt) & set(keep_woman)) 
            keeps_d = list(set(keep_att) & set(keep_man)) 

            keeps_a_labels = np.zeros((len(keeps_a), 2))
            keeps_a_labels[:, 0] = 1
            keeps_a_labels[:, 1] = 1
            keeps_b_labels = np.zeros((len(keeps_b), 2))
            keeps_c_labels = np.zeros((len(keeps_c), 2))
            keeps_c_labels[:, 0] = 1
            keeps_d_labels = np.zeros((len(keeps_d), 2))
            keeps_d_labels[:, 1] = 1
            dset.attr = np.concatenate([keeps_a_labels, keeps_b_labels, keeps_c_labels, keeps_d_labels], axis=0)
            dset.filename = np.concatenate([np.array(dset.filename)[keeps_a], np.array(dset.filename)[keeps_b], np.array(dset.filename)[keeps_c], np.array(dset.filename)[keeps_d]], axis=0)
            

    ### tiny for debugging
    #train_dataset.attr = train_dataset.attr[:100]
    #train_dataset.filename = train_dataset.filename[:100]
    #valid_dataset.attr = valid_dataset.attr[:100]
    #valid_dataset.filename = valid_dataset.filename[:100]
    #test_dataset.attr = test_dataset.attr[:100]
    #test_dataset.filename = test_dataset.filename[:100]

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                             shuffle=True, num_workers=workers)
    shuffle_test = True
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size,
                                             shuffle=shuffle_test, num_workers=workers)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                             shuffle=shuffle_test, num_workers=workers)

    print("Train loader: {0}, valid loader: {1}, test loader: {2}".format(len(train_loader), len(valid_loader), len(test_loader)))

    test_loader = [valid_loader, test_loader]
    
    if args.model_type == 'vgg_orig': # grad cam 2
        model = models.vgg16(pretrained=True)
        model.classifier[6] = nn.Linear(4096, out_features)
        if args.criterion == 2:
            model.classifier.add_module('last_sigmoid', nn.Sigmoid())
        else:
            if args.training_version == 1:
                model.classifier = model.classifier[:-1]
                args.twohead = True
                head1 = nn.Linear(4096, out_features // 2).to(device)
                head2 = nn.Linear(4096, out_features // 2).to(device)
        model.to(device)
    elif args.model_type == 'alexnet': # grad cam 2
        model = models.alexnet(pretrained=True)
        model.classifier[6] = nn.Linear(4096, out_features)
        if args.criterion == 2:
            model.classifier.add_module('last_sigmoid', nn.Sigmoid())
        else:
            if args.training_version == 1:
                model.classifier = model.classifier[:-1]
                args.twohead = True
                head1 = nn.Linear(4096, out_features // 2).to(device)
                head2 = nn.Linear(4096, out_features // 2).to(device)
        model.to(device)
    elif args.model_type == 'resnet101': # grad cam 2
        model = models.resnet101(pretrained=True)
        if args.criterion == 2:
            model.fc = nn.Sequential(nn.Linear(2048, out_features), nn.Sigmoid())
        else:
            if args.training_version == 1:
                model.fc = Identity()
                args.twohead = True
                head1 = nn.Linear(2048, out_features // 2).to(device)
                head2 = nn.Linear(2048, out_features // 2).to(device)
            else:
                model.fc = nn.Linear(2048, out_features)
        model.to(device)
    elif args.model_type == 'resnet18':
        model = models.resnet18(pretrained=True)
        if args.criterion == 2:
            model.fc = nn.Sequential(nn.Linear(512, out_features), nn.Sigmoid())
        else:
            if args.training_version == 1:
                model.fc = Identity()
                args.twohead = True
                head1 = nn.Linear(512, out_features // 2).to(device)
                head2 = nn.Linear(512, out_features // 2).to(device)
            else:
                model.fc = nn.Linear(512, out_features)
        model.to(device)
    elif args.model_type == 'resnet34':
        model = models.resnet34(pretrained=True)
        if args.criterion == 2:
            model.fc = nn.Sequential(nn.Linear(512, out_features), nn.Sigmoid())
        else:
            if args.training_version == 1:
                model.fc = Identity()
                args.twohead = True
                head1 = nn.Linear(512, out_features // 2).to(device)
                head2 = nn.Linear(512, out_features // 2).to(device)
            else:
                model.fc = nn.Linear(512, out_features)
        model.to(device)
    elif args.model_type == 'resnet50':
        model = models.resnet50(pretrained=True)
        if args.criterion == 2:
            model.fc = nn.Sequential(nn.Linear(2048, out_features), nn.Sigmoid())
        else:
            if args.training_version == 1:
                model.fc = Identity()
                args.twohead = True
                head1 = nn.Linear(2048, out_features // 2).to(device)
                head2 = nn.Linear(2048, out_features // 2).to(device)
            else:
                model.fc = nn.Linear(2048, out_features)
        model.to(device)
    if args.criterion == 0:
        criterion = nn.CrossEntropyLoss()
        if args.weighting_version in [1, 2]:
            criterion = nn.CrossEntropyLoss(reduction='none')
    elif args.criterion == 2:
        criterion = nn.BCELoss(reduction='none')

    if args.training_version == 1:
        baserate = LogisticRegression()
        X = train_dataset.attr[:, 0].reshape(-1, 1)
        gender = train_dataset.attr[:, 1]
        baserate.fit(X, gender)
        p = baserate.score(X, gender)
        print("Logistic Regression p: {}".format(p))
        args.random_gen_ce = -np.log(max(p, 1.-p))

    if args.weighting_version in [1, 2]:
        keep_noatt = np.where(np.array(train_dataset.attr[:, 0]) == 1)[0]
        keep_man = np.where(np.array(train_dataset.attr[:, 1]) == 1)[0]
        keep_att = np.where(np.array(train_dataset.attr[:, 0]) == 0)[0]
        keep_woman = np.where(np.array(train_dataset.attr[:, 1]) == 0)[0]

        keeps_a = list(set(keep_noatt) & set(keep_man)) 
        keeps_b = list(set(keep_att) & set(keep_woman)) 
        keeps_c = list(set(keep_noatt) & set(keep_woman)) 
        keeps_d = list(set(keep_att) & set(keep_man)) 

        amounts = np.array([len(keeps_a), len(keeps_b), len(keeps_c), len(keeps_d)])
        print("amounts: {}".format(amounts))

        if args.weighting_version == 2:
            first = None
            if len(keeps_a) / (len(keeps_a)+len(keeps_d)) > len(keeps_c)/(len(keeps_c)+len(keeps_b)): # this means biased towards men aka keepsa and keepsb are majorities
                first = True
                major = len(keeps_a)+len(keeps_b)
                minor = len(keeps_c)+len(keeps_d)
            else:
                first = False
                minor = len(keeps_a)+len(keeps_b)
                major = len(keeps_c)+len(keeps_d)
            weight_major = 1
            weight_minor = major / minor

            weight_major = weight_major * args.weighting_ratio

            # now normalize
            normalizing = ((weight_major*major)+(weight_minor*minor)) / (major+minor)
            if first:
                weights = np.array([weight_major, weight_major, weight_minor, weight_minor])
            else:
                weights = np.array([weight_minor, weight_minor, weight_major, weight_major])
            weights /= normalizing
            print("weights: {}".format(weights))
            args.bal_weights = weights
        elif args.weighting_version == 1:
            weights = np.prod(amounts) / amounts
            weights /= np.sum(weights)
            weights *= 4.
            print("weights: {}".format(weights))
            args.bal_weights = weights

    if args.load_model or args.continue_model:
        #model_loc = 'models_celeba/{0}/model_{1}.pt'.format(args.model_name, args.epochs)
        model_loc = 'models_celeba/{0}/model.pt'.format(args.model_name)
        model.load_state_dict(torch.load(model_loc))
        if args.twohead:
            model_loc = 'models_celeba/{0}/head1_{1}.pt'.format(args.model_name, args.epochs)
            head1.load_state_dict(torch.load(model_loc))
            model_loc = 'models_celeba/{0}/head2_{1}.pt'.format(args.model_name, args.epochs)
            head2.load_state_dict(torch.load(model_loc))

    optimizer = None
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        if args.twohead:
            optimizer1 = optim.SGD(head1.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
            optimizer2 = optim.SGD(head2.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    elif args.optimizer == 'adam':
        beta1 = .5
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(beta1, 0.999))
        if args.twohead:
            optimizer1 = optim.Adam(head1.parameters(), lr=args.lr, betas=(beta1, 0.999))
            optimizer2 = optim.Adam(head2.parameters(), lr=args.lr, betas=(beta1, 0.999))
    if args.continue_model:
        file = open("models_celeba/{}/training.txt".format(args.model_name),"a") 
        loss_dict = pickle.load(open("models_celeba/{}/loss_dict.pkl".format(args.model_name), 'rb'))
    elif args.save_model:
        if not os.path.exists("models_celeba/" + args.model_name):
            os.makedirs("models_celeba/" + args.model_name)
        file = open("models_celeba/{}/training.txt".format(args.model_name),"w+") 
        loss_dict = {}
        loss_dict['train_loss'] = []
        loss_dict['train_att_accs'] = []
        loss_dict['train_gen_accs'] = []
        loss_dict['val_loss'] = []
        loss_dict['val_att_accs'] = []
        loss_dict['val_gen_accs'] = []

        loss_dict['train_labels'] = []
        loss_dict['train_probs'] = []
        loss_dict['val_labels'] = []
        loss_dict['val_probs'] = []
        loss_dict['test_labels'] = []
        loss_dict['test_probs'] = []
    else:
        file = None
        loss_dict = None
    twoheads = None
    if args.twohead:
        twoheads = [head1, head2, optimizer1, optimizer2]

    if args.load_model:
        test(args, model, device, criterion, test_loader, -1, optimizer, twoheads, file, loss_dict)
        exit()

    for epoch in range(1, args.epochs + 1):
        if args.continue_model:
            loss_dict = train(args, model, device, train_loader, optimizer, criterion, args.epochs+epoch, twoheads, file, loss_dict)
            loss_dict = test(args, model, device, criterion, test_loader, args.epochs+epoch, optimizer, twoheads, file, loss_dict)
        else:
            loss_dict = train(args, model, device, train_loader, optimizer, criterion, epoch, twoheads, file, loss_dict)
            loss_dict = test(args, model, device, criterion, test_loader, epoch, optimizer, twoheads, file, loss_dict)

    if args.save_model:
        torch.save(model.state_dict(),"models_celeba/{}/model.pt".format(args.model_name))
        if args.twohead:
            torch.save(head1.state_dict(), "models_celeba/{}/head1.pt".format(args.model_name))
            torch.save(head2.state_dict(), "models_celeba/{}/head2.pt".format(args.model_name))
        file.close()
        pickle.dump(loss_dict, open("models_celeba/{}/loss_dict.pkl".format(args.model_name), 'wb'))
        if args.dataset_version == 1:
            plt.plot(np.arange(len(loss_dict['train_loss'])), loss_dict['train_loss'], label='train loss')
            plt.plot(np.arange(len(loss_dict['val_loss'])), loss_dict['val_loss'], label='val loss')
            plt.legend(loc='best')
            plt.savefig("models_celeba/{0}/losses.png".format(args.model_name))
            plt.close()


if __name__ == '__main__':
    main()
