import argparse

parser = argparse.ArgumentParser()
# data preprocessing
parser.add_argument('--pp', '--preprocess', action = 'store_true')
# model training
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
import timeit
import pickle
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from rdkit import Chem
from collections import defaultdict
from sklearn.metrics import roc_auc_score, precision_score, recall_score, average_precision_score, precision_recall_curve, auc

'''
    Path Initialization
'''
FOLDER = './CPI-data/'
DIR_INPUT_TRAIN = './CPI-data/input/train/'
DIR_INPUT_PRED = './CPI-data/input/pred/'
SOURCE_NAME = 'CPI_cls_data.tsv'
PRED_NAME = 'CPI_cls_pred_data.tsv'
WORDDICT = 'word_dict.pickle'
FINGERDICT = 'fingerprint_dict.pickle'
SOURCE_PATH = FOLDER + SOURCE_NAME
PRED_PATH = FOLDER + PRED_NAME
FILE_AUCs = FOLDER + ('output/AUC_AUPR_ran.txt' if not opt.is_pro else 'output/AUC_AUPR_pro.txt')
FILE_model = FOLDER + ('output/CLS_new_ran.pth' if not opt.is_pro else 'output/CLS_new_pro.pth')
FILE_result =FOLDER + 'output/pred_new.txt'
n_fingerprint = 235336
n_word = 7844
PRO_NUM = 127


'''
    Dictionary Initialization
'''
atom_dict = defaultdict(lambda: len(atom_dict))
bond_dict = defaultdict(lambda: len(bond_dict))
fingerprint_dict = defaultdict(lambda: len(fingerprint_dict))
edge_dict = defaultdict(lambda: len(edge_dict))
word_dict = defaultdict(lambda: len(word_dict))
protein_dict = defaultdict(lambda: len(protein_dict))


'''
    Functions for encoding and feature extraction
'''
def create_atoms(mol):
    """Create a list of atom (e.g., hydrogen and oxygen) IDs
    considering the aromaticity."""
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    # denote the aromatic atoms in the mol
    for a in mol.GetAromaticAtoms():
        i = a.GetIdx()
        atoms[i] = (atoms[i], 'aromatic')
    atoms = [atom_dict[a] for a in atoms]
    return np.array(atoms)


def create_ijbonddict(mol):
    """Create a dictionary, which each key is a node ID
    and each value is the tuples of its neighboring node
    and bond (e.g., single and double) IDs. Namely, i 
    stands for the node id, and jbond denotes the 
    neighboring node and bond. """
    i_jbond_dict = defaultdict(lambda: [])
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bond = bond_dict[str(b.GetBondType())]
        i_jbond_dict[i].append((j, bond))
        i_jbond_dict[j].append((i, bond))
    return i_jbond_dict


def extract_fingerprints(atoms, i_jbond_dict, radius):
    """Extract the r-radius subgraphs (i.e., fingerprints)
    from a molecular graph using Weisfeiler-Lehman algorithm."""

    # only one atom or 0-radius
    if (len(atoms) == 1) or (radius == 0):
        fingerprints = [fingerprint_dict[a] for a in atoms]

    else:
        nodes = atoms
        i_jedge_dict = i_jbond_dict

        for _ in range(radius):

            """Update each node ID considering its neighboring nodes and edges
            (i.e., r-radius subgraphs or fingerprints)."""
            fingerprints = []
            for i, j_edge in i_jedge_dict.items():
                neighbors = [(nodes[j], edge) for j, edge in j_edge]
                fingerprint = (nodes[i], tuple(sorted(neighbors)))
                fingerprints.append(fingerprint_dict[fingerprint])
            nodes = fingerprints

            """Also update each edge ID considering two nodes
            on its both sides."""
            _i_jedge_dict = defaultdict(lambda: [])
            for i, j_edge in i_jedge_dict.items():
                for j, edge in j_edge:
                    both_side = tuple(sorted((nodes[i], nodes[j])))
                    edge = edge_dict[(both_side, edge)]
                    _i_jedge_dict[i].append((j, edge))
            i_jedge_dict = _i_jedge_dict

    return np.array(fingerprints)


def create_adjacency(mol):
    """create the adjacency matrix of mol"""
    adjacency = Chem.GetAdjacencyMatrix(mol)
    return np.array(adjacency)


def split_sequence(sequence, ngram):
    """Obtain ngram word vector of seq"""
    # '-' and '=' denotes the beginning and 
    # ending position
    sequence = '-' + sequence + '='
    words = [word_dict[sequence[i:i+ngram]] for i in range(len(sequence)-ngram+1)]
    return np.array(words)


def dump_dictionary(dictionary, filename):
    """Store the dictionary produced by data 
    preprocessing"""
    with open(filename, 'wb') as f:
        pickle.dump(dict(dictionary), f)


def preprocess(path, dir_input):
    """preprocess the raw .tsv data for both
    the training and predicting data, specified 
    by the argument path"""

    # extract fingerprint with radius 1, 2
    # split the seq in words with len <= 3
    radius, ngram = 2, 3

    # load the raw data
    raw_data = []
    with open(path, 'r') as tsv_in:
        reader = csv.reader(tsv_in, delimiter = '\t')
        raw_data = [record for record in reader]

    N = len(raw_data)

    comid, proid, Smiles, compounds, adjacencies, proteins, interactions = '', '', '', [], [], [], []
    
    for no, record in enumerate(raw_data):

        com_id = record[0]
        pro_id = record[1]
        inchi_str = record[2]
        pro_seq = record[3]
        interaction = record[4]

        if (no+1)%1000 == 0:
            print('/'.join(map(str, [no+1, N])))

        mol = Chem.MolFromInchi(inchi_str)
        smiles = Chem.MolToSmiles(mol)

        # exclude the record with '.' in the
        # Smiles format since it denotes no-bound.
        # The two parts of the mol is not bonded
        if '.' in smiles:
            continue

        # store the compoud and protein id, smiles
        comid += com_id + '\n'
        proid += pro_id + '\n'
        Smiles += smiles + '\n'
            
        # store the compounds by fingerprints
        # consider the molecule with Hs
        mol = Chem.AddHs(mol)
        atoms = create_atoms(mol)
        i_jbond_dict = create_ijbonddict(mol)
        fingerprints = extract_fingerprints(atoms, i_jbond_dict, radius)
        compounds.append(fingerprints)

        # store the adjacency matrix of compounds
        adjacency = create_adjacency(mol)
        adjacencies.append(adjacency)

        # store the proteins by words
        words = split_sequence(pro_seq, ngram)
        proteins.append(words)

        # store the interactions
        interactions.append(np.array([float(interaction)]))

    print('The encoding and feature extraction has finished!')
    print('Total valid records #: %d'%len(interactions))

    # store the processed data
    os.makedirs(dir_input, exist_ok=True)

    with open(dir_input + 'Smiles.txt', 'w') as f:
        f.write(Smiles)
    with open(dir_input + 'comid.txt', 'w') as f:
        f.write(comid)
    with open(dir_input + 'proid.txt', 'w') as f:
        f.write(proid)
    np.save(dir_input + 'compounds', compounds)
    np.save(dir_input + 'adjacencies', adjacencies)
    np.save(dir_input + 'proteins', proteins)
    np.save(dir_input + 'interactions', interactions)


'''
    Classes for training and testing
'''
class CPI_CLS(nn.Module):
    def __init__(self):
        super(CPI_CLS, self).__init__()
        # embed each key in both dicts into a 10d vector
        self.embed_fingerprint = nn.Embedding(n_fingerprint, 10)
        self.embed_word = nn.Embedding(n_word, 10)
        # 3-layer GNN for molecule
        self.W_gnn = nn.ModuleList([nn.Linear(10, 10) for _ in range(3)])
        # 3-layer CNN for proteins
        self.W_cnn = nn.ModuleList([nn.Conv2d(
            in_channels=1, out_channels=1, 
            kernel_size=2*11+1, stride=1, padding=11) for _ in range(3)])
        self.W_attention = nn.Linear(10, 10)
        self.W_out = nn.ModuleList([nn.Linear(2*10, 2*10) for _ in range(3)])
        self.W_interaction = nn.Linear(2*10, 2)

    def gnn(self, xs, A, layer):
        for i in range(layer):
            hs = torch.relu(self.W_gnn[i](xs))
            xs = xs + torch.matmul(A, hs)
        return torch.unsqueeze(torch.mean(xs, 0), 0)
    
    def attention_cnn(self, x, xs, layer):
        """The attention mechanism is applied to the last 
        layer of CNN"""

        # cnn part
        xs = torch.unsqueeze(torch.unsqueeze(xs, 0), 0)
        for i in range(layer):
            xs = torch.relu(self.W_cnn[i](xs))
        xs = torch.squeeze(torch.squeeze(xs, 0), 0)

        # attention
        h = torch.relu(self.W_attention(x))
        hs = torch.relu(self.W_attention(xs))
        weights = torch.tanh(F.linear(h, hs))
        ys = torch.t(weights) * hs

        return torch.unsqueeze(torch.mean(ys, 0), 0)

    def forward(self, inputs):

        fingerprints, adjacency, words = inputs

        """Compound vector with GNN."""
        fingerprint_vectors = self.embed_fingerprint(fingerprints)
        compound_vector = self.gnn(fingerprint_vectors, adjacency, 3)

        """Protein vector with attention-CNN."""
        word_vectors = self.embed_word(words)
        protein_vector = self.attention_cnn(compound_vector, word_vectors, 3)

        """Concatenate the above two vectors and output the interaction."""
        cat_vector = torch.cat((compound_vector, protein_vector), 1)
        for j in range(3):
            cat_vector = torch.relu(self.W_out[j](cat_vector))
        interaction = self.W_interaction(cat_vector)

        return interaction

    def __call__(self, data, train=True):

        # fingerprints, adjacency, words = inputs
        inputs, correct_interaction = data[:-1], data[-1]
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


'''
    Functions to load preprocessed data and 
    train the model, make predictions
'''
def load_tensor(file_name, dtype):
    return [dtype(d).to(device) for d in np.load(file_name + '.npy', allow_pickle=True)]


def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)


def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset


def split_dataset(dataset, proid, is_pro = False):
    """splite the dataset according to is_pro"""
    dataset_out = [[] for i in range(10)]
    if not is_pro:
        dataset = shuffle_dataset(dataset, opt.ran_seed)
        n = int(0.1 * len(dataset))
        for i in range(10):
            dataset_out[i] = dataset[(i*n):(i*n+n)]
    elif is_pro:
        l = (int)(PRO_NUM * 0.1)

        ID = np.array([])
        for i in range(9):
            ID = np.append(ID, i*np.ones(l))
        ID = np.append(ID, 9*np.ones(PRO_NUM-9*l))
        np.random.shuffle(ID)

        for i, data in enumerate(dataset):
            pro_id = proid[i]
            piece_id = int(ID[protein_dict[pro_id]])
            dataset_out[piece_id].append(data)
    for i in range(10):
        dataset_out[i] = shuffle_dataset(dataset_out[i], opt.ran_seed)
        # print(len(dataset_out[i]))
    return dataset_out


def train():
    '''
        Train the model using pre-processed data
    '''

    """Load preprocessed training data"""
    dir_input = DIR_INPUT_TRAIN
    proid = []
    with open(dir_input + 'proid.txt', 'r') as f:
        proid = f.read().strip().split('\n')
    compounds = load_tensor(dir_input + 'compounds', torch.LongTensor)
    adjacencies = load_tensor(dir_input + 'adjacencies', torch.FloatTensor)
    proteins = load_tensor(dir_input + 'proteins', torch.LongTensor)
    interactions = load_tensor(dir_input + 'interactions', torch.LongTensor)


    """Create a dataset and split it into train/dev/test."""
    dataset = list(zip(compounds, adjacencies, proteins, interactions))
    print('len dataset:', len(dataset))
    # we need a tuple of 10 split dataset according to is_pro
    dataset = split_dataset(dataset, proid, is_pro=opt.is_pro)


    """Set a model."""
    torch.manual_seed(opt.ran_seed)
    model = CPI_CLS().to(device)
    trainer = Trainer(model)
    tester = Tester(model)
    
    AUCs = ('Epoch\tTime(sec)\tLoss_train\tAUC_dev\t'
            'AUC_test\tPrecision_test\tRecall_test\tAP_test\tAUPR_test')
    with open(FILE_AUCs, 'w') as f:
        f.write(AUCs + '\n')

    """Start training."""
    print('Training...')
    print(AUCs)
    start = timeit.default_timer()
    best_auc_test = 0.5
    best_auc_dev = 0.5

    for epoch in range(1, 10):

        j = epoch%10
        # the following 10 denotes the decay interval
        if epoch % 10 == 0:
            trainer.optimizer.param_groups[0]['lr'] *=lr_decay
        
        dataset_train = []
        for i in range(10):
            if i != j:
                if i != j+1:
                    dataset_train.extend(dataset[i])
                else:
                    dataset_dev = dataset[i]
        dataset_test = dataset[j]
        # print(len(dataset_train))
        # print(len(dataset_dev))
        # print(len(dataset_test))

        loss_train = trainer.train(dataset_train)
        AUC_dev = tester.test(dataset_dev)[0]
        AUC_test, precision_test, recall_test, AP_test, AUPR_test = tester.test(dataset_test)

        end = timeit.default_timer()
        time = end - start

        AUCs = [epoch, time, loss_train, AUC_dev,
                AUC_test, precision_test, recall_test, AP_test, AUPR_test]
        tester.save_AUCs(AUCs, FILE_AUCs)

        if best_auc_test <= AUC_test:
            best_auc_test = AUC_test
            best_auc_dev = AUC_dev
            tester.save_model(model, FILE_model)

        print('\t'.join(map(str, AUCs)))

    print('Training complete!')    
    print('BSET AUC_dev:', best_auc_dev, '\tBEST AUC_test:', best_auc_test)

def predict():
    '''
        Make prediction using the pretrained model
    '''

    """Load preprocessed predicting data"""
    dir_input = DIR_INPUT_PRED
    comid = []
    proid = []
    with open(dir_input + 'comid.txt', 'r') as f:
        comid = f.read().strip().split('\n')
    with open(dir_input + 'proid.txt', 'r') as f:
        proid = f.read().strip().split('\n')
    compounds = load_tensor(dir_input + 'compounds', torch.LongTensor)
    adjacencies = load_tensor(dir_input + 'adjacencies', torch.FloatTensor)
    proteins = load_tensor(dir_input + 'proteins', torch.LongTensor)
    interactions = load_tensor(dir_input + 'interactions', torch.LongTensor)

    """Create a dataset and split it into train/dev/test."""
    dataset = list(zip(compounds, adjacencies, proteins, interactions))

    """Set and load a model."""
    torch.manual_seed(opt.ran_seed)
    model = CPI_CLS().to(device)
    model.load_state_dict(torch.load(FILE_model))

    N = len(dataset)
    results = ''
    for i, data in enumerate(dataset):
        inputs = data[:-1]
        pred = model.forward(inputs)
        pred = F.softmax(pred, 1).to('cpu').data.numpy()
        pred = pred[0][1]

        results += '\t'.join([comid[i],proid[i],str(pred)])+'\n'
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

if opt.pp:
    # preprocess the train and pred data
    preprocess(SOURCE_PATH, DIR_INPUT_TRAIN)
    preprocess(PRED_PATH, DIR_INPUT_PRED)
    dump_dictionary(fingerprint_dict, FOLDER + FINGERDICT)
    dump_dictionary(word_dict, FOLDER + WORDDICT)
if opt.t:
    # train the model using training data
    fingerprint_dict = load_pickle(FOLDER + FINGERDICT)
    word_dict = load_pickle(FOLDER + WORDDICT)
    train()
if opt.p:
    # pred CPI using the model stored
    predict()