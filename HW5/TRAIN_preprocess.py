'''
    This code helps to report the statistics of the input 
    training data ./CPI-data/CPI_cls_data.tsv
    and write the processed data for new model
    into file ./CPI-data/CPI_cls_data.txt
'''

import csv
import numpy as np
from rdkit import Chem

folder = './CPI-data/'
source_name = 'CPI_cls_data.tsv'
dest_name = 'CPI_cls_data.txt'
source_path = folder + source_name
dest_path = folder + dest_name

def load_data(path):
    data = []
    with open(path, 'r') as tsv_in:
        reader = csv.reader(tsv_in, delimiter = '\t')
        for record in reader:
            data.append(record)
    return data

len_max_pro = 0
len_min_pro = 512
len_max_smi = 0
len_min_smi = 512

cnt = 0
pos_cnt = 0
ave_len_smi = 0.0
ave_len_pro = 0.0

aa_alphabet = dict()
smi_alphabet = dict()
com_alphabet = dict()
pro_alphabet = dict()

processed_data = ''

CPI_data = load_data(source_path)

for cpi_record in CPI_data:
    cnt = cnt + 1

    com_id = cpi_record[0]
    pro_id = cpi_record[1]
    inchi_str = cpi_record[2]
    pro_seq = cpi_record[3]
    interaction = cpi_record[4]

    pos_cnt = pos_cnt + (1 if interaction=='1' else 0)
    mol = Chem.MolFromInchi(inchi_str)
    smiles_string = Chem.MolToSmiles(mol)

    len_seq = len(pro_seq)
    len_smi = len(smiles_string)
    ave_len_pro += len_seq / 17756
    ave_len_smi += len_smi / 17756
    len_max_pro = max(len_max_pro, len_seq)
    len_min_pro = min(len_min_pro, len_seq)
    len_max_smi = max(len_max_smi, len_smi)
    len_min_smi = min(len_min_smi, len_smi)

    for i in range(len_smi):
        if not smi_alphabet.__contains__(smiles_string[i]):
            smi_alphabet[smiles_string[i]] = 1
    for i in range(len_seq):
        if not aa_alphabet.__contains__(pro_seq[i]):
            aa_alphabet[pro_seq[i]] = 1
    if not com_alphabet.__contains__(com_id):
            com_alphabet[com_id] = 1
    if not pro_alphabet.__contains__(pro_id):
            pro_alphabet[pro_id] = 1

    processed_data += ' '.join([smiles_string, pro_seq, interaction])
    processed_data += '\n'

    if cnt % 1000 == 0:
        print(cnt)

with open(dest_path, 'w') as f:
    f.write(processed_data)

print('\n', cnt, '\n')
print(pos_cnt, '\n')
print('len_max_pro=', len_max_pro, '\n')
print('len_min_pro=', len_min_pro, '\n')
print('len_max_smi=', len_max_smi, '\n')
print('len_min_smi=', len_min_smi, '\n')
print('ave_len_pro=', ave_len_pro, '\n')
print('ave_len_smi=', ave_len_smi, '\n')
print('smi keys num:', len(smi_alphabet))
print('smi keys:', smi_alphabet.keys(), '\n')
print('aa keys num:', len(aa_alphabet))
print('aa keys:', aa_alphabet.keys(), '\n')
print('com keys num:', len(com_alphabet), '\n')
# print('com keys:', com_alphabet.keys(), '\n')
print('pro keys num:', len(pro_alphabet), '\n')
# print('pro keys:', pro_alphabet.keys(), '\n')