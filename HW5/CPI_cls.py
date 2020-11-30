import argparse

parser = argparse.ArgumentParser()
# data preprocessing
parser.add_argument('--t', '--train', action = 'store_true')
# predict
parser.add_argument('--p', '--predict', action = 'store_true')
# wether to split the interactions by protein types
parser.add_argument('-i', '--is_pro', action = 'store_true')
# choose the GPU id (0,1,2,3)
parser.add_argument('-d', '--device_index', type = str, default = '1')
# choose the random seed
parser.add_argument('-r', '--ran_seed', type = int, default = 1234)
opt = parser.parse_args()

import os
import sys
import csv
import copy
import time
import timeit
import pickle
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from rdkit import Chem
from rdkit.Chem import AllChem
from collections import defaultdict
from sklearn.metrics import roc_auc_score, precision_score, recall_score, average_precision_score, precision_recall_curve, auc

'''
    Initialization
'''
FOLDER = './CPI-data/'
DIRIO = './CPI-data/CLS/'
WORDDICT = 'word_dict.pickle'
SOURCE_NAME = 'CPI_cls_data.tsv'
PRED_NAME = 'CPI_cls_pred_data.tsv'
SOURCE_PATH = FOLDER + SOURCE_NAME
PRED_PATH = FOLDER + PRED_NAME
FILE_AUCs = DIRIO + ('AUC_AUPR_ran.txt' if not opt.is_pro else 'AUC_AUPR_pro.txt')
FILE_model = DIRIO + ('CLS_ran.pth' if not opt.is_pro else 'CLS_pro.pth')
FILE_result =DIRIO + 'pred.txt'

CPI_NUM = 17756
RATIO = 0.1 # 10-fold cross-validation
PRO_DICT_LEN = 127
WORD_DICT_LEN = 7844

word_dict = defaultdict(lambda: len(word_dict)) # protein 3-words-dict
protein_dict = defaultdict(lambda: len(protein_dict)) # proteins dict


def dump_dictionary(dictionary, filename):
    """Store the dictionary produced by data 
    preprocessing"""
    with open(filename, 'wb') as f:
        pickle.dump(dict(dictionary), f)


def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)

'''
    Load the CPI data from the tsv file
'''
def load_data(path):
    data = []
    with open(path, 'r') as tsv_in:
        reader = csv.reader(tsv_in, delimiter = '\t')
        for record in reader:
            data.append(record)
    return data

'''
    Pre-process the compound and protein data
    Encoding and Feature extration
'''
def com_pre(inchi_str):
    mol = Chem.MolFromInchi(inchi_str)
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius = 2, nBits = 256)

    x_com = [i for i in fingerprint]
    x_com = torch.unsqueeze(torch.Tensor(x_com), 0)
    
    return x_com

def pro_pre(pro_seq):
    seq = '-' + pro_seq + '='
    words = [word_dict[seq[i:i+3]] for i in range(len(seq)-3+1)]
    return torch.LongTensor(np.array(words))

def record_process(cpi_record):
    com_id = cpi_record[0]
    pro_id = cpi_record[1]
    inchi_str = cpi_record[2]
    pro_seq = cpi_record[3]

    y = torch.LongTensor(np.array([float(cpi_record[4])]))
    x_com = com_pre(inchi_str).to(device)
    x_pro = pro_pre(pro_seq).to(device) # notice that here x_pro is a np.ndarray

    return com_id, pro_id, x_com, x_pro, y.to(device)

def data_process(path, is_pro = False, is_pred = False):
    if is_pro:
        '''
        Split interactions into train and validation according to proteins
        '''
        l = (int)(PRO_DICT_LEN*RATIO)
        ID = np.array([])
        for i in range(9):
            ID = np.append(ID, i*np.ones(l))
        ID = np.append(ID, 9*np.ones(PRO_DICT_LEN-9*l))
        np.random.shuffle(ID)
    else:
        '''
        Split interactions into train and validation randomly
        '''
        l = (int)(CPI_NUM*RATIO)
        ID = np.array([])
        for i in range(9):
            ID = np.append(ID, i*np.ones(l))
        ID = np.append(ID, 9*np.ones(CPI_NUM-9*l))
        np.random.shuffle(ID)
    
    cpi_data = load_data(path)
    print('Finishing loading:', path)

    if not is_pred:

        train_data = [[] for x in range(10)]
        start = time.time()
        print('Start splitting')
        
        for cnt, cpi_record in enumerate(cpi_data):
            CPI_record = record_process(cpi_record)
            com_id, pro_id, x_com, x_pro, y = CPI_record
            if is_pro:
                piece_id = int (ID[protein_dict[pro_id]])
            elif not is_pro:
                piece_id = int (ID[cnt])
            train_data[piece_id].append(CPI_record)

        print('Data processing completed in ', time.time()-start, 's\n')
        return train_data
    
    if is_pred:

        pred_data = []
        start = time.time()

        for cnt, cpi_record in enumerate(cpi_data):
            CPI_record = record_process(cpi_record)
            pred_data.append(CPI_record)
        
        print('Data processing completed in ', time.time()-start, 's\n')
        return pred_data


'''
    Network architecture
'''
class CPI_CLS(nn.Module):
    def __init__(self):
        super(CPI_CLS, self).__init__()
        self.embed_com = nn.Sequential(
            # input 1 x 256 tensor
            nn.Conv1d(1, 1, kernel_size =2*8+1, stride =2, padding = 8),
            nn.ReLU(),
            nn.Conv1d(1, 1, kernel_size =2*8+1, stride =2, padding = 8),
            nn.ReLU()
            # output 1 x 64 tensor
        )
        self.embed_pro = nn.ModuleList(
            [nn.Embedding(WORD_DICT_LEN, 64),
            # 1 x 1 x L x 64
            nn.Conv2d(1, 1, kernel_size=2*10+1, stride=1, padding=10),
            nn.Conv2d(1, 1, kernel_size=2*10+1, stride=1, padding=10),
            nn.Conv2d(1, 1, kernel_size=2*10+1, stride=1, padding=10)]
        )
        self.out = nn.Sequential(
            nn.Linear(2*64, 2*64), nn.ReLU(),
            nn.Linear(2*64, 2)
        )

    def forward(self, inputs):

        x_com, x_pro = inputs

        x_com = torch.unsqueeze(x_com,0)
        x_com = self.embed_com(x_com) 
        x_com = torch.squeeze(x_com,0) # 1 x 64

        x_pro = self.embed_pro[0](x_pro)
        x_pro = torch.unsqueeze(torch.unsqueeze(x_pro, 0), 0)
        for i in range(3):
            x_pro = torch.relu(self.embed_pro[i+1](x_pro))
        x_pro = torch.squeeze(torch.squeeze(x_pro, 0), 0) # L x 64
        x_pro = torch.unsqueeze(torch.mean(x_pro, 0), 0) # 1 x 64

        x = torch.cat((x_com, x_pro), 1)
        y = self.out(x)
        return y

    def __call__(self, data, train = True):
        inputs, correct_interaction = data[2:-1], data[-1]
        pred_interaction = self.forward(inputs)

        if train:
            loss = F.cross_entropy(pred_interaction, correct_interaction)
            return loss
        else:
            correct_labels = correct_interaction.to('cpu').data.numpy()
            ys = F.softmax(pred_interaction, 1).to('cpu').data.numpy()
            pred_labels = list(map(lambda x: np.argmax(x), ys))
            pred_scores = list(map(lambda x: x[1], ys))
            return correct_labels, pred_labels, pred_scores


class Trainer(object):
    def __init__(self, model):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(),lr=1e-3, weight_decay=1e-6)
    
    def train(self, dataset):
        np.random.shuffle(dataset)
        # N = len(dataset)
        loss_total = 0
        for data in dataset:
            data = data
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
        # N = len(dataset)
        T, Y, S = [], [], []
        for data in dataset:
            (correct_labels, pred_labels, pred_scores) = self.model(data, train=False)
            T.append(correct_labels)
            Y.append(pred_labels)
            S.append(pred_scores)
        AUC = roc_auc_score(T, S)
        precision = precision_score(T, Y)
        recall = recall_score(T, Y)
        AP = average_precision_score(T, S)
        precisions, recalls, _ = precision_recall_curve(T, S)
        AUPR = auc(recalls, precisions)
        return AUC, precision, recall, AP, AUPR

    def save_AUCs(self, AUCs, filename):
        with open(filename, 'a') as f:
            f.write('\t'.join(map(str, AUCs)) + '\n')

    def save_model(self, model, filename):
        torch.save(model.state_dict(), filename)


def train():
    '''
        Traning part start
    '''

    train_data = data_process(SOURCE_PATH, is_pro = opt.is_pro)
    pred_set = data_process(PRED_PATH, is_pred = True)

    for epoch in range(10):
        traindata = []
        for i in range(10):
            if not i==(epoch%10):
                traindata.extend(train_data[i])
        train_set = traindata
        valid_set = train_data[epoch%10]

    # set the model
    torch.manual_seed(opt.ran_seed)
    model = CPI_CLS().to(device)
    trainer = Trainer(model)
    tester = Tester(model)

    AUCs = ('Epoch\tTime(sec)\tLoss_train\tAUC\tPrecision\tRecall\tAP\tAUPR')
    with open(FILE_AUCs, 'w') as f:
        f.write(AUCs + '\n')
    print(AUCs)
    
    start = timeit.default_timer()
    print('Start training!')
    best_auc = 0.5

    for epoch in range(1, 10):

        # the following 10 denotes the decay interval
        if epoch % 10 == 0:
            trainer.optimizer.param_groups[0]['lr'] *=lr_decay

        loss_train = trainer.train(train_set)
        AUC, precision, recall, AP, AUPR = tester.test(valid_set)

        end = timeit.default_timer()
        time = end - start

        AUCs = [epoch, time, loss_train, AUC,
                precision, recall, AP, AUPR]
        tester.save_AUCs(AUCs, FILE_AUCs)

        if best_auc <= AUC:
            best_auc = AUC
            tester.save_model(model, FILE_model)

        print('\t'.join(map(str, AUCs)))

    print('Training complete!')    
    print('BSET AUC:', best_auc)
    

def predict():
    '''
        Make prediction using the pretrained model
    '''

    dataset = data_process(PRED_PATH, is_pred = True)

    """Set and load a model."""
    torch.manual_seed(opt.ran_seed)
    model = CPI_CLS().to(device)
    model.load_state_dict(torch.load(FILE_model))

    N = len(dataset)
    results = ''
    for i, data in enumerate(dataset):
        com_id = data[0]
        pro_id = data[1]
        inputs = data[2:-1]
        pred = model.forward(inputs)
        pred = F.softmax(pred, 1).to('cpu').data.numpy()
        pred = pred[0][1]

        results += '\t'.join([com_id,pro_id,str(pred)])+'\n'
        if (i+1)%1000 == 0:
            print('/'.join(map(str, [i+1, N])))

    with open(FILE_result, 'w') as f:
        f.write(results)
    
    print('Prediction complete!')

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

if opt.t:
    # train the model using training data
    train()
    dump_dictionary(word_dict, DIRIO + WORDDICT)
if opt.p:
    # pred CPI using the model stored
    word_dict = load_pickle(DIRIO + WORDDICT)
    predict()