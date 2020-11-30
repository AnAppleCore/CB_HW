import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch_size', type = int, default = 256)
parser.add_argument('-e', '--epoch', type = int, default = 200)
parser.add_argument('-s', '--seed', type = int, default = 1234)
parser.add_argument('-r', '--learning_rate', type = float, default = 1e-5)
parser.add_argument('-d', '--device_index', type = str, default = '1')
parser.add_argument('-ds', '--data_set', type = str, default = 'GIL')
opt = parser.parse_args()

import os
import sys
import time
import random
import timeit
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from collections import defaultdict
from sklearn.metrics import roc_auc_score, precision_score, recall_score

'''
    Initialization
'''

L = 40
L_DICT = 29
RATIO = 0.8 # 5-fold cross-validation
amino_acid = defaultdict(lambda: len(amino_acid)) # amino acid dict

FOLDER = './data/'
DIRIO = './classification/'
POS_SET = FOLDER + opt.data_set + '_positive.txt'
NEG_SET = FOLDER + opt.data_set + '_negative.txt'
FILE_AUCs = DIRIO + opt.data_set + '_AUC.txt'
FILE_model = DIRIO + opt.data_set + '_model.pth'


def load_data(path):
    '''
        Load the traning data, and return a list
    '''
    data = []
    with open(path, 'r') as f:
        data = f.read().strip().split('\n')
    return data


def seq_process(seq):
    '''
        preprocess each amino acid sequence 
        return a tensor with size 29 X 40
    '''
    seq = '-' + seq + '='
    len_seq = len(seq)
    # pad each seq to the same length
    pad_len = (L - len_seq) // 2

    # one-hot encoding
    x = torch.zeros(L_DICT, L)
    for i in range(len_seq):
        x[amino_acid[seq[i]]][i+pad_len] = 1
    return x


def data_process(path, label = 'positive'):
    '''
        prepare the traning data from 
        positive and negative samples,
        return a list containing tensors
        of seq vec and label
    '''
    data = load_data(path)
    len_data = len(data)
    print('Finish loading ', path)

    output_data = []
    y = torch.LongTensor(np.array([float(1) if label == 'positive' else float(0)]))
    for no, seq in enumerate(data):
        x = seq_process(seq)
        output_data.append([x, y])
        if (no+1)% 1000 == 0:
            print(no+1, '/', len_data)
    print('Finish processing ', path)

    return output_data


def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset


def split_dataset(dataset, ratio):
    n = int(ratio * len(dataset))
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2


class TCR_CLS(nn.Module):
    def __init__(self, condfig = None):
        super(TCR_CLS, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(L_DICT, 256, kernel_size = 2*1+1, stride = 2, padding = 1),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size = 2*1+1, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv1d(256, 512, kernel_size = 2*1+1, stride = 2, padding = 1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(512 * L // 4, 512 * L // 4),
            nn.ReLU(),
            nn.Linear(512 * L // 4, 2)
        )
    
    def forward(self, x):
        y = x
        y = torch.unsqueeze(y, 0)
        y = self.encoder(y)
        y = torch.squeeze(y, 0)

        y = y.view(-1, 512 * L // 4)
        y = self.decoder(y)

        return y

    def __call__(self, data, train = True):
        x, y_true = data
        x, y_true = x.to(device), y_true.to(device)
        y_pred = self.forward(x)

        if train:
            loss = F.cross_entropy(y_pred, y_true)
            return loss
        else:
            correct_labels = y_true.to('cpu').data.numpy()
            ys = F.softmax(y_pred, 1).to('cpu').data.numpy()
            pred_labels = list(map(lambda x: np.argmax(x), ys))
            pred_scores = list(map(lambda x: x[1], ys))
            return correct_labels, pred_labels, pred_scores

class Trainer(object):
    def __init__(self, model):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr = opt.learning_rate)

    def train(self, dataset):
        np.random.shuffle(dataset)
        loss_total = 0
        for data in dataset:
            loss = self.model(data)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_total += loss.to('cpu').data.numpy()
        return loss_total

class Tester(object):
    def __init__(self, model):
        self.model = model

    def test(self, dataset):
        T, Y, S = [], [], []
        for data in dataset:
            (correct_labels, pred_labels, pred_scores) = self.model(data, train=False)
            T.append(correct_labels)
            Y.append(pred_labels)
            S.append(pred_scores)
        AUC = roc_auc_score(T, S)
        precision = precision_score(T, Y)
        recall = recall_score(T, Y)
        return AUC, precision, recall
    
    def save_AUCs(self, AUCs, filename):
        with open(filename, 'a') as f:
            f.write('\t'.join(map(str, AUCs)) + '\n')
    
    def save_model(self, model, filename):
        torch.save(model.state_dict(), filename)

def train():
    '''
        Training part
    '''

    # create the datasets
    train_data = []
    pos_data = data_process(POS_SET, 'positive')
    neg_data = data_process(NEG_SET, 'negative')
    train_data.extend(pos_data)
    train_data.extend(neg_data)

    train_data = shuffle_dataset(train_data, opt.seed)
    train_data, test_data = split_dataset(train_data, RATIO)
    train_data, valid_data = split_dataset(train_data, RATIO)

    # set the model
    torch.manual_seed(opt.seed)
    model = TCR_CLS().to(device)
    trainer = Trainer(model)
    tester = Tester(model)

    # output statistics
    AUCs = ('Epoch\tTime(sec)\tLoss_train\tAUC_dev\tAUC_test\tPrecision_test\tRecall_test')
    with open(FILE_AUCs, 'w') as f:
        f.write(AUCs + '\n')

    # start training
    print('Training ...')
    print(AUCs)
    start = timeit.default_timer()
    best_auc_test = 0.5
    best_auc_dev = 0.5

    for epoch in range(opt.epoch):

        loss_train = trainer.train(train_data)
        AUC_dev = tester.test(valid_data)[0]
        AUC_test, precision_test, recall_test = tester.test(test_data)

        end = timeit.default_timer()
        time = end - start

        AUCs = [epoch, time, loss_train, AUC_dev, AUC_test, precision_test, recall_test]
        tester.save_AUCs(AUCs, FILE_AUCs)

        if best_auc_test <= AUC_test:
            best_auc_test = AUC_test
            best_auc_dev = AUC_dev
            tester.save_model(model, FILE_model)

        print('\t'.join(map(str, AUCs)))
    
    print('Training complete!')  
    print('BSET AUC_dev:', best_auc_dev, '\tBEST AUC_test:', best_auc_test)


'''
    PROGRAM START
'''
# CPU or GPU
if torch.cuda.is_available():
    device = torch.device('cuda:'+opt.device_index)
    print('The code uses GPU...')
else:
    device = torch.device('cpu')
    print('The code uses CPU!!!')

train()