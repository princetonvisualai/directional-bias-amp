from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
import os
import cv2
import pickle
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
import numpy as np
from scipy import ndimage
from datasets import CoCoDataset
from sklearn.metrics import average_precision_score
import sklearn.metrics as metrics

TOTAL_LABELS = 66

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

def get_bogs(all_preds, all_labels):
    bog_tilde = np.zeros((TOTAL_LABELS, 2)) 
    bog_gt_g = np.zeros((TOTAL_LABELS, 2)) 
    bog_gt_o = np.zeros((TOTAL_LABELS, 2)) 
    bog_preds = np.zeros((TOTAL_LABELS, 2))

    for i, objs in enumerate([all_preds, all_labels]):
        female = np.where(objs[:, -1] == 0)[0]
        male = np.where(objs[:, -1] == 1)[0]
        for j in range(TOTAL_LABELS):
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
        for j in range(TOTAL_LABELS):
            bog_gt_g[j][0] = np.sum(objs[male][:, j])
            bog_gt_g[j][1] = np.sum(objs[female][:, j])
    return bog_tilde, bog_gt_g, bog_gt_o, bog_preds

def train(args, model, device, train_loader, optimizer, epoch, criterion, loss_info, file):
    model.train()
    all_preds = []
    all_probs = []
    all_labels = []
    
    running_loss = []
    for batch_idx, (data, target) in (enumerate(tqdm(train_loader)) if args.interact else enumerate(train_loader)):
        target = target.float()

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model.forward(data)

        loss = criterion(output, target)
        weight = target*(args.pos_weight - 1.) + 1.
        if args.training_version == 1: # will cancel out positive weight if that is an argument
            num_a = torch.sum(target[:, -1])
            num_b = len(target) - num_a
            alpha = ((len(target) // 2) - num_a) / (len(target) // 2)
            weight = (target[:, -1] * 2) - 1
            weight = weight * alpha
            weight = torch.ones_like(weight) + weight
            weight = weight.unsqueeze(1)
        loss = loss * weight
        loss = loss.mean()
        loss.backward()
        
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
        running_loss.append(loss.item()*len(data))

        all_labels.extend(target.data.cpu().numpy())
        all_probs.extend(output.data.cpu().numpy())
        preds = torch.round(output).data.cpu().numpy()
        all_preds.extend(preds)
    all_probs, all_preds, all_labels = np.array(all_probs), np.array(all_preds), np.array(all_labels)
    corrects = np.equal(all_preds, all_labels)
    running_batch_loss = np.sum(running_loss) / len(all_labels)
    if len(all_preds.shape) == 1:
        all_preds = np.expand_dims(all_preds, 1)
    probs_ap_score = average_precision_score(all_labels, all_probs)
    print("Train Epoch: {0}, AP: {1}, Loss: {2} ".format(epoch, probs_ap_score, running_batch_loss))

    female = np.where(all_labels[:, -1] == 0)[0]
    male = np.where(all_labels[:, -1] == 1)[0]

    for indices in [male, female]:
        accuracies = np.mean(corrects[indices], axis=0)
        this_ap_score = average_precision_score(all_labels[indices, :-1], all_probs[indices, :-1])
        sep_acc = [this_ap_score, accuracies[-1]]
        print(corrects[indices].shape)
        print(sep_acc)

    if file is not None:
        file.write("Train Epoch: {0}, AP: {1}, Loss: {2}\n\n".format(epoch, probs_ap_score, running_batch_loss))
        loss_info['train_map'].append(probs_ap_score)
        loss_info['train_gen'].append(np.mean(all_labels[:, -1] == all_preds[:, -1]))
        loss_info['train_loss'].append(running_batch_loss)
    return loss_info


def test(args, model, device, test_loader, epoch, criterion, loss_info, file):
    valid_loader, test_loader = test_loader
    model.eval()
    test_loss = 0
    all_preds = []
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for batch_idx, (data, target) in (enumerate(tqdm(test_loader)) if args.interact else enumerate(test_loader)):
            target = target.float()
            data, target = data.to(device), target.to(device)
            output = model.forward(data)
            loss = criterion(output, target).mean() # sum up batch loss
            test_loss += loss.item() * len(target)

            all_labels.extend(target.data.cpu().numpy())
            all_probs.extend(output.data.cpu().numpy())
            preds = torch.round(output).data.cpu().numpy()
            all_preds.extend(preds)

    test_loss /= len(all_labels)
    all_preds, all_labels, all_probs = np.array(all_preds), np.array(all_labels), np.array(all_probs)
    if len(all_preds.shape) == 1:
        all_preds = np.expand_dims(all_preds, 1)
    corrects = np.equal(all_preds, all_labels)
    probs_ap_score = average_precision_score(all_labels, all_probs)
    loss_info['test_loss'].append(test_loss)
    loss_info['test_probs'].append(all_probs)
    loss_info['test_labels'].append(all_labels)
    per_group = []
    labels_to_names = test_loader.dataset.labels_to_names
    categories = test_loader.dataset.categories

    test_map = average_precision_score(all_labels[:, :-1], all_probs[:, :-1])
    test_gender = np.mean(corrects, axis=0)[-1]
    print("Total mAP: {0}, Total gender: {1}".format(test_map, test_gender))
    loss_info['test_map'].append(test_map)
    loss_info['test_gen'].append(test_gender)

    # validation set
    valid_probs = []
    valid_preds = []
    valid_labels = []
    val_loss = 0
    with torch.no_grad():
        for batch_idx, (data, target) in (enumerate(tqdm(valid_loader)) if args.interact else enumerate(valid_loader)):
            target = target.float()
            data, target = data.to(device), target.to(device)
            output = model.forward(data)
            loss = criterion(output, target).mean() # sum up batch loss
            val_loss += loss.item() * len(target)

            valid_labels.extend(target.data.cpu().numpy())
            valid_probs.extend(output.data.cpu().numpy())
            preds = torch.round(output).data.cpu().numpy()
            valid_preds.extend(preds)
    val_loss /= len(valid_labels)
    valid_preds, valid_probs, valid_labels = np.array(valid_preds), np.array(valid_probs), np.array(valid_labels)
    loss_info['val_labels'].append(valid_labels)
    loss_info['val_probs'].append(valid_probs)
    loss_info['val_loss'].append(val_loss)


    if file is not None:
        file.write("Test Epoch: {0}, mAP: {1}, Loss: {2}\n\n".format(epoch, probs_ap_score, test_loss))
    print("\n--------\n\n")
    return loss_info


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')

    parser.add_argument('--model_name', type=str, default="", 
                        help='path to model to load, or model to save')
    parser.add_argument('--model_type', type=str, default="vgg16", 
                        help='model type')
    parser.add_argument('--pos_weight', type=float, default=1.,
                        help='how much to weigh the positive examples')
    parser.add_argument('--maskloss_weight', type=float, default=0.,
                        help='weight for mask norm on optimal mask')
    parser.add_argument('--interact', action='store_true', default=False,
                        help='For if showing tqdm')
    parser.add_argument('--pretrained', action='store_false', default=True,
                        help='if the model weights taken should be pretrained, defaulrt is false')
    parser.add_argument('--test_is_train', action='store_true', default=False,
                        help='test dataset is train')
    parser.add_argument('--mask_person', type=int, default=0,
                        help='0 is not masked, 1 is masked with box, 2 is masked with segmentation map'+
                        '3 is noisy half mask of segmentation map, 4 is opposite of 2 where all is covered except person' +
                        '5 is random mask blocking the size of the person, 6 is random mask blocking the size of the background'+
                        '7 is better version of 2 since masks person, but adds back any background objects'+
                        '8 is opposite of 7 where objects are all removed but person is added back')
    parser.add_argument('--training_version', type=int, default=0,
                        help='0 is regular, 1 is equalize genders where losses equal between two genders, 2 is adversarial loss at no gender')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--load-model', action='store_true', default=False,
                        help='if loading a model')
    parser.add_argument('--continue-model', action='store_true', default=False,
                        help='if loading a model but want to continue training, not testing')
    args = parser.parse_args()
    print(args)
    use_cuda = torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")

    print("Device: {}".format(device))

    workers = 1

    folder = "models/" + args.model_name
    if not os.path.exists(folder):
        os.makedirs(folder)

    transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    global TOTAL_LABELS
    train_dataset = CoCoDataset(transform, mask_person=args.mask_person)
    test_dataset = CoCoDataset(transform, version='val', mask_person=args.mask_person)

    val_dataset = CoCoDataset(transform, mask_person=args.mask_person)
    train_indices, val_indices = pickle.load(open('coco_validation_indices.pkl', 'rb'))
    print("len train: {0}, len val: {1}".format(len(train_indices), len(val_indices)))
    train_dataset.image_ids = list(np.array(train_dataset.image_ids)[train_indices])
    val_dataset.image_ids = list(np.array(val_dataset.image_ids)[val_indices])
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
            shuffle=True, num_workers=workers)

    ##tiny
    #train_dataset.image_ids = train_dataset.image_ids[:100]
    #test_dataset.image_ids = test_dataset.image_ids[:100]

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
            shuffle=True, num_workers=workers)
    if args.test_is_train:
        test_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size//2,
                shuffle=False, num_workers=workers)
    else:
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size//2,
                shuffle=False, num_workers=workers)

    out_features = TOTAL_LABELS
    out_features += 1
    
    if args.model_type == 'vgg16':
        model = models.vgg16(pretrained=args.pretrained)
        model.classifier[6] = nn.Linear(4096, out_features)
        model.classifier.add_module('last_sigmoid', nn.Sigmoid())
        model.to(device)
    elif args.model_type == 'alexnet':
        model = models.alexnet(pretrained=args.pretrained)
        model.classifier[6] = nn.Linear(4096, out_features)
        model.classifier.add_module('last_sigmoid', nn.Sigmoid())
        model.to(device)
    elif args.model_type == 'resnet101':
        model = models.resnet101(pretrained=args.pretrained)
        model.fc = nn.Sequential(nn.Linear(2048, out_features), nn.Sigmoid())
        model.to(device)
    elif args.model_type == 'resnet18':
        model = models.resnet18(pretrained=args.pretrained)
        if args.training_version == 2:
            model.fc = Identity()
            gen_classifier = nn.Sequential(nn.Linear(512, 1), nn.Sigmoid())
            label_classifier = nn.Sequential(nn.Linear(512, out_features - 1), nn.Sigmoid())
        else:
            model.fc = nn.Sequential(nn.Linear(512, out_features), nn.Sigmoid())
        model.to(device)
    elif args.model_type == 'resnet34':
        model = models.resnet34(pretrained=args.pretrained)
        model.fc = nn.Sequential(nn.Linear(512, out_features), nn.Sigmoid())
        model.to(device)
    elif args.model_type == 'resnet50':
        model = models.resnet50(pretrained=args.pretrained)
        model.fc = nn.Sequential(nn.Linear(2048, out_features), nn.Sigmoid())
        model.to(device)


    criterion = nn.BCELoss(reduction='none')
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    loss_info = None
    if args.load_model:
        model.load_state_dict(torch.load("{0}/model.pt".format(folder)))
        loss_info = pickle.load(open("{0}/loss_info.pkl".format(folder), 'rb'))
        if args.continue_model:
            keys = ['test_menalsoshop', 'test_all_gts', 'test_all_tgs', 'test_gen']
            for key in keys:
                if key not in loss_info.keys():
                    loss_info[key] = []
            file = open("{0}/training.txt".format(folder),"a") 
            for epoch in range(1+len(loss_info['test_map']), args.epochs + 1):
                loss_info = train(args, model, device, train_loader, optimizer, epoch, criterion, loss_info, file)

                loss_info = test(args, model, device, [val_loader, test_loader], epoch, criterion, loss_info, file)
        else:
            loss_info = test(args, model, device, [val_loader, test_loader], args.epochs, criterion, loss_info, None)
    else:
        if args.save_model:
            file = open("{0}/training.txt".format(folder),"a") 
            loss_info = {}
            loss_info['train_map'] = [] 
            loss_info['train_loss'] = []
            loss_info['train_gen'] = []
            loss_info['test_map'] = []
            loss_info['test_loss'] = []
            loss_info['test_gen'] = []
            loss_info['val_loss'] = []

            loss_info['val_labels'] = []
            loss_info['val_probs'] = []
            loss_info['test_labels'] = []
            loss_info['test_probs'] = []
        else:
            file = None
        for epoch in range(1, args.epochs + 1):
            loss_info = train(args, model, device, train_loader, optimizer, epoch, criterion, loss_info, file)
            loss_info = test(args, model, device, [val_loader, test_loader], epoch, criterion, loss_info, file)
        if args.save_model:
            file.close()

    if (args.save_model):
        torch.save(model.state_dict(),"{0}/model.pt".format(folder))
        pickle.dump(loss_info, open("{0}/loss_info.pkl".format(folder), 'wb'))

        plt.plot(np.arange(len(loss_info['test_map'])), loss_info['test_map'], label='test')
        plt.plot(np.arange(len(loss_info['train_map'])), loss_info['train_map'], label='train')
        plt.xlabel('Epochs')
        plt.ylabel('mAP or Accuracy')
        plt.legend(loc='best')
        plt.savefig("{0}/training_curve.png".format(folder))
        
if __name__ == '__main__':
    main()
