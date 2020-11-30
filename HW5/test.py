import rdkit
import csv
import torch
from torch import nn
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from collections import defaultdict

y = [[0,1]]
x = torch.zeros(1,2)

y = torch.LongTensor(np.array(y))

print(y.shape)
print(y)

print(x.shape)
print(x)
