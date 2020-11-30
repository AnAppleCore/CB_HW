import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--load', action = 'store_true')
parser.add_argument('--save', action = 'store_true')
parser.add_argument('--predict', action = 'store_true')
parser.add_argument('--use_gpu', action = 'store_true')
parser.add_argument('-i', '--is_pro', type = int, default = 1)
parser.add_argument('-b', '--batch_size', type = int, default = 512)
parser.add_argument('-e', '--epoch_num', type = int, default = 100)
parser.add_argument('-lr', '--learning_rate', type = float, default = 1e-5)
parser.add_argument('-d', '--device_index', type = str, default = '3')
opt = parser.parse_args()

import os

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = opt.device_index

import sys
import csv
import copy
import time
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from rdkit import Chem
from sklearn.metrics import roc_auc_score

'''
    Initialization
'''
RATIO = 0.1
# EPSILON = 0.005

def mkdir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

path_weights = 'CPI-data/output'
CLS_weights_pth = os.path.join(path_weights, 'CLS_old_pro.pth' if opt.is_pro else 'CLS_old_ran.pth')
CLS_result_pth = os.path.join(path_weights, 'old_restults.txt')
mkdir(path_weights)

folder = './CPI-data/'
file_name = 'CPI_cls_data.tsv'
pred_name = 'CPI_cls_pred_data.tsv'
file_path = folder + file_name
pred_path = folder + pred_name

R_PRO = 21
R_COM = 44
PRO_NUM = 127
CPI_NUM = 17756
VEC_LEN = 256
LEN_PRO = 200
STRIDE_PRO = 50
LEN_SMI = 56
STRIDE_SMI = 8
L = 64

aa_alphabet = dict()
smi_alphabet = dict()
pro_alphabet = dict()
amino_acids = ['P','Q','I','T','L','W','R','F','V','K','E','G','A','D','M','N','Y','C','S','H','X']
smi_elements = ['C', '(', ')', 'N', '[', '@', 'H', ']', 'O', 'c', '1', '=', '2', 'S', 'n', 'i', '3', '#', 'F', '-', '4', '5', '6', 's', 'P', '/', 'l', 'o', 'B', 'r', '7', '\\', 'I', '.', '+', '8', '9', '%', '0', 'a', 'M', 'g', 'K', 'e']
proteins = ['Q9YQ12', 'Q72874', 'P89582', 'Q72547', 'Q76353', 'Q5L478', 'A3EZI9', 'Q9IP65', 'P03470', 'P03469', 'P03433', 'Q4U254', 'Q9WKE8', 'P89449', 'P03211', 'Q8JXU8', 'P27958', 'P06479', 'A0A0K1CY61', 'P03369', 'Q7ZJM1', 'O92972', 'P06935', 'P26662', 'P03452', 'P03367', 'P26663', 'P08546', 'Q9WMX2', 'Q76270', 'P26664', 'P08543', 'Q91H74', 'Q6L709', 'Q9YQ30', 'P03474', 'P04293', 'P00521', 'P03468', 'Q82323', 'Q6QLK5', 'Q82122', 'P16753', 'A3EZJ3', 'B4URF0', 'P04585', 'P05778', 'P03313', 'Q3Y5H1', 'P10236', 'P03234', 'Q9QNF7', 'P10274', 'A5Z252', 'P24433', 'Q81258', 'P35961', 'P19550', 'P04578', 'P04618', 'P03428', 'Q77YF8', 'P06856', 'P69332', 'P04014', 'Q2Q167', 'P29990', 'Q20MD5', 'SARS-3CLpro', 'Q6Q793', 'P0DOF8', 'O41156', 'Q20MD3', 'P20536', 'P28857', 'SARS-helicase', 'P03374', 'Q91RS4', 'P04326', 'P10210', 'P09250', 'SARS-PLpro', 'D2K2A8', 'P09252', 'Q49PX0', 'A3RLP8', 'Q9IH62', 'P03126', 'P04303', 'P04292', 'Q3ZDS5', 'Q9IQ47', 'P03421', 'P03438', 'P04608', 'C4LRQ6', 'P03359', 'P03101', 'P03418', 'P03437', 'P88142', 'Q2LFS1', 'D5F1R0', 'Q9IGQ6', 'P15682', 'Q75VQ4', 'P03300', 'P05866', 'Q194T1', 'P16788', 'P03354', 'P0DOF9', 'P04545', 'P03118', 'P00530', 'Q06347', 'P03120', 'P03471', 'Q06992', 'P13159', 'P0C6U6', 'Q03732', 'Q05320', 'P24740', 'P03206', 'K7N5L2', 'P03377']
for k, aa in enumerate([x for x in amino_acids]):
    aa_alphabet[aa] = k
for j, se in enumerate([x for x in smi_elements]):
    smi_alphabet[se] = j
for i, pro in enumerate([x for x in proteins]):
    pro_alphabet[pro] = i

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
    Using One-Hot encoding
'''
def com_pre(inchi_str):
    mol = Chem.MolFromInchi(inchi_str)
    smiles_string = Chem.MolToSmiles(mol)
    len_smi = len(smiles_string)
    new_smi = ''
    if(len_smi<STRIDE_SMI):
        for i in np.random.randint(len_smi, size = LEN_SMI):
            new_smi += smiles_string[i]
    else:
        for i in np.random.randint(0, high = len_smi-STRIDE_SMI+1, size = LEN_SMI // STRIDE_SMI):
            new_smi += smiles_string[i:i+STRIDE_SMI]
    x_com = torch.zeros(R_COM, LEN_SMI)
    for i in range(LEN_SMI):
        if not smi_alphabet.__contains__(new_smi[i]):
            new_smi = new_smi.replace(new_smi[i], '.')
        x_com[smi_alphabet[new_smi[i]], i] = 1
    return x_com

def pro_pre(pro_seq):
    len_pro = len(pro_seq)
    new_pro = ''
    for i in np.random.randint(0, high = len_pro-STRIDE_PRO+1, size = LEN_PRO // STRIDE_PRO):
        new_pro += pro_seq[i:i+STRIDE_PRO]
    x_pro = torch.zeros(R_COM, LEN_PRO)
    for i in range(LEN_PRO):
        x_pro[aa_alphabet[new_pro[i]], i] = 1
    return x_pro

def record_process(cpi_record):
    com_id = cpi_record[0]
    pro_id = cpi_record[1]
    inchi_str = cpi_record[2]
    pro_seq = cpi_record[3]

    y = torch.zeros(1)
    y[:] = float(1) if cpi_record[4]=='1' else float(0)
    x_com = com_pre(inchi_str)
    x_pro = pro_pre(pro_seq)

    return com_id, pro_id, x_com, x_pro, y

def data_process(path, is_pro = True):
    if is_pro:
        '''
        Split interactions into train and validation according to proteins
        '''
        l = (int)(PRO_NUM*RATIO)
        ID = np.array([])
        for i in range(9):
            ID = np.append(ID, i*np.ones(l))
        ID = np.append(ID, 9*np.ones(PRO_NUM-9*l))
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

    train_data = [[] for x in range(10)]
    len_pieces = [[] for x in range(10)]
    start = time.time()
    print('Start splitting')
    
    cnt = 0
    for cpi_record in cpi_data:
        CPI_record = record_process(cpi_record)
        com_id, pro_id, x_com, x_pro, y = CPI_record
        if is_pro:
            piece_id = int (ID[pro_alphabet[pro_id]])
        elif not is_pro:
            piece_id = int (ID[cnt])
        train_data[piece_id].append(CPI_record)
        cnt += 1
    
    for i in range(10):
        len_pieces[i] = len(train_data[i])

    print('Data processing completed in ', time.time()-start, 's\n')
    return train_data, len_pieces

'''
    CPI record dataset
'''
class CPI_dataset(Dataset):
    def __init__(self, cpi_data):
        self.CPI_records = cpi_data
        # self.rand = rand
        super(CPI_dataset, self).__init__()

        self.len = len(self.CPI_records)

    def __getitem__(self, index):
        com_id, pro_id, x_com, x_pro, y = self.CPI_records[index]
        # x_com = x_com + (torch.randn_like(x_com)*EPSILON if self.rand else 0)
        # x_pro = x_pro + (torch.randn_like(x_pro)*EPSILON if self.rand else 0)
        x = torch.cat((x_com, x_pro), 1)
        return x, y

    def __len__(self):
        return self.len


'''
    Network architecture
'''
class network_vanilla(nn.Module):
    def __init__(self, config = None):
        super(network_vanilla, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(R_COM, 256, kernel_size = 3, stride = 2, padding = 1), nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size = 3, stride = 1, padding = 1), nn.ReLU(),
            nn.Conv1d(256, 512, kernel_size = 3, stride = 2, padding = 1), nn.ReLU(),
            nn.Conv1d(512, 512, kernel_size = 3, stride = 1, padding = 1), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(512 * L, 1), nn.Sigmoid()
        )

    def forward(self, x):
        y = x
        y = self.encoder(y)
        y = y.view(-1, 512 * L)
        y = self.decoder(y)
        return y


if opt.use_gpu:
    network = network_vanilla().cuda()
    network_best = network_vanilla().cuda()
else:
    network = network_vanilla()
    network_best = network_vanilla()
optim = torch.optim.Adam(network.parameters(), lr = opt.learning_rate)

if not opt.predict:
    '''
        Traning part start
    '''
    print('Start training!')
    auc_best_valid = 0.5
    auc_best_test = 0.5

    train_data, len_pieces = data_process(file_path, is_pro = opt.is_pro)

    for epoch in range(opt.epoch_num):
        traindata = []
        len_train = 0
        for i in range(10):
            if not i==(epoch%10):
                traindata.extend(train_data[i])
                len_train += len_pieces[i]
        train_set = CPI_dataset(traindata)
        valid_set = CPI_dataset(train_data[epoch%10])
        if opt.use_gpu:
            train_loader = DataLoader(dataset = train_set, batch_size = opt.batch_size,pin_memory = True)
            valid_loader = DataLoader(dataset = valid_set, batch_size = len_pieces[epoch%10], pin_memory = True)
        else:
            train_loader = DataLoader(dataset = train_set, batch_size = opt.batch_size)
            valid_loader = DataLoader(dataset = valid_set, batch_size = len_pieces[epoch%10])

        for _, src in enumerate(train_loader):
            data, target = src
            if opt.use_gpu:
                data, target = data.cuda(), target.cuda()

            pred = network(data)
            loss = ((target - pred)**2).sum()

            optim.zero_grad()
            loss.backward()
            optim.step()

        if epoch < 3 or epoch%10==9:
            print('epoch %04d:' %(epoch + 1))
            print('\t train: %04f' %loss.item())

            with torch.no_grad():
                for _, src in enumerate(valid_loader):
                    data, target = src
                    if opt.use_gpu:
                        data, target = data.cuda(), target.cuda()
                    pred = network(data)
                
                target = target.squeeze().cpu()
                pred = pred.squeeze().cpu()
                target_numpy = target.numpy()
                pred_numpy = pred.numpy()
                auc_valid = roc_auc_score(target_numpy, pred_numpy)
                print('\t valid: %04f' % auc_valid)
                sys.stdout.flush()

                if auc_valid > auc_best_valid:
                    auc_best_valid = auc_valid
                    network_best = copy.deepcopy(network)
                    if opt.save:
                        torch.save(network_best.state_dict(), CLS_weights_pth)

    print('best valid: %04f' % auc_best_valid)
    if opt.save:
        print('Model saved in', CLS_weights_pth)
    sys.stdout.flush()

else:
    '''
        Prediction
    '''
    # load parameters
    if opt.load and os.path.exists(CLS_weights_pth):
        print('Load model from', CLS_weights_pth)
        network_best.load_state_dict(torch.load(CLS_weights_pth))

    pred_data = load_data(pred_path)
    print('Finishing loading:', pred_path)

    results = ''

    cnt = 0
    for pred_record in pred_data:
        cnt += 1
        com_id, pro_id, x_com, x_pro, y = record_process(pred_record)
        x = torch.cat((x_com, x_pro), 1).unsqueeze(0).cuda()
        pred = network_best(x)
        results += '\t'.join([com_id, pro_id, str(max(pred.item(), 0))])
        results += '\n'

        if cnt % 1000 == 0 :
            print(cnt)
    
    with open(CLS_result_pth, 'w') as f:
        f.write(results)
    print('Results stored in', CLS_result_pth)