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
np.random.seed(opt.seed) # set the random seed
amino_acid = defaultdict(lambda: len(amino_acid)) # amino acid dict

FOLDER = './data/'
DIRIO = './classification/'
POS_SET = FOLDER + opt.data_set + '_positive.txt'
NEG_SET = FOLDER + opt.data_set + '_negative.txt'
FILE_AUCs = DIRIO + opt.data_set + '_AUC_v4.txt'
FILE_model = DIRIO + opt.data_set + '_model_v4.pth'


def load_data(path):
    '''
        Load the traning data, and return a list
    '''
    data = []
    with open(path, 'r') as f:
        data = f.read().strip().split('\n')
    return data


def seq_process(seq, randn = False):
    '''
        preprocess each amino acid sequence 
        return a tensor with size 29 X 40
    '''
    seq = '-' + seq + '='
    len_seq = len(seq)

    # one-hot encoding
    x = torch.zeros(L_DICT, L)
    for i in range(len_seq):
        # one-hot encode all amino acids, and keep the missing ones as 0
        x[amino_acid[seq[i]]][i] = 1 + (np.random.standard_normal()/20 if randn else 0)
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
    for no, seq in enumerate(data):
        len_seq = len(seq) + 2
        if label == 'positive':
            y = torch.LongTensor(np.array([float(1)]))
            x = seq_process(seq)
            output_data.append([x, y])
            x = seq_process(seq, randn = True)
            output_data.append([x, y])
        else:
            y = torch.LongTensor(np.array([float(0)]))
            x = seq_process(seq)
            output_data.append([x, y])
        if (no+1)% 1000 == 0:
            print(no+1, '/', len_data)
    print('Finish processing ', path)

    return output_data


def shuffle_dataset(dataset):
    np.random.shuffle(dataset)
    return dataset


def split_dataset(dataset, ratio):
    n = int(ratio * len(dataset))
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2


class TCR_dataset(Dataset):
    def __init__(self, data):
        self.TCR_records = data
        super(TCR_dataset, self).__init__()

        self.len = len(self.TCR_records)

    def __getitem__(self, index):
        x, y = self.TCR_records[index]
        return x, y

    def __len__(self):
        return self.len


class TCR_CLS(nn.Module):
    def __init__(self, condfig = None):
        super(TCR_CLS, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(L_DICT, 256, kernel_size = 2*1+1, stride = 2, padding = 1),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size = 2*1+1, stride = 1, padding = 1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(256 * L //2, 256 * L //2),
            nn.ReLU(),
            nn.Linear(256 * L //2, 1)
        )
    
    def forward(self, x):
        y = x
        y = self.encoder(y)

        y = y.view(-1, 256 * L //2)
        y = self.decoder(y)

        return y


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

    train_data = shuffle_dataset(train_data)
    train_data, test_data = split_dataset(train_data, RATIO)
    train_data, valid_data = split_dataset(train_data, RATIO)

    train_src = TCR_dataset(train_data)
    valid_src = TCR_dataset(valid_data)
    test_src = TCR_dataset(test_data)
    len_train = len(train_data)
    len_valid = len(valid_data)
    len_test = len(test_data)

    train_loader = DataLoader(dataset = train_src, batch_size = opt.batch_size, shuffle = True, pin_memory = True)
    valid_loader = DataLoader(dataset = valid_src, batch_size = len_valid, shuffle = False, pin_memory = True)
    test_loader = DataLoader(dataset = test_src, batch_size = len_test, shuffle = False, pin_memory = True)

    print('# train_src: %d' % len_train)
    print('# valid_src: %d' % len_valid)
    print('# test_src: %d' % len_test)

    # set the model
    torch.manual_seed(opt.seed)
    model = TCR_CLS().to(device)
    optim = torch.optim.Adam(model.parameters(), lr = opt.learning_rate)

    # start training
    print('Training ...')
    start = timeit.default_timer()
    auc_best_valid = 0.5
    auc_best_test = 0.5

    for epoch in range(opt.epoch):
        for _, src in enumerate(train_loader):
            data, target = src
            data, target = data.to(device), target.to(device)

            pred = model(data)
            loss = ((target - pred) ** 2).sum()

            optim.zero_grad()
            loss.backward()
            optim.step()

        if epoch < 3 or not (epoch + 1) % 10:
            print('epoch %04d:' % (epoch + 1))
            print('\ttrain: %04f' % loss.item())

            with torch.no_grad():
                for _, src in enumerate(valid_loader):
                    data, target = src
                    data, target = data.to(device), target.to(device)
                    pred = model(data)

                target = target.squeeze().cpu()
                pred = pred.squeeze().cpu()
                # AUC score from sklearn
                target_numpy = target.numpy()
                pred_numpy = pred.numpy()
                auc_valid = roc_auc_score(target_numpy, pred_numpy)
                print('\tvalid: %04f' % auc_valid)

                for _, src in enumerate(test_loader):
                    data, target = src
                    data, target = data.to(device), target.to(device)
                    pred = model(data)

                target = target.squeeze().cpu()
                pred = pred.squeeze().cpu()
                # AUC score from sklearn
                target_numpy = target.numpy()
                pred_numpy = pred.numpy()
                auc_test = roc_auc_score(target_numpy, pred_numpy)
                print('\ttest: %04f' % auc_test)

                sys.stdout.flush()

                if auc_valid > auc_best_valid:
                    auc_best_valid = auc_valid
                    auc_best_test = auc_test
                    torch.save(model.state_dict(), FILE_model)

    print('best valid: %04f' % auc_best_valid)
    print('best test: %04f' % auc_best_test)
    sys.stdout.flush()


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