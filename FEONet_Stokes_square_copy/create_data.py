import torch
import os
import argparse
import re
import pickle
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from scipy.sparse import diags
from pprint import pprint


# ARGS
parser = argparse.ArgumentParser("SEM")
## Data
parser.add_argument("--bc", type=str)
parser.add_argument("--num_data", type=int, default=3000)
parser.add_argument("--file", type=str, help='Example: --file 3000N18')
parser.add_argument("--kind", type=str, default='train', choices=['train', 'validate'])

args = parser.parse_args()
gparams = args.__dict__

NUM_DATA = gparams['num_data']
FILE = gparams['file']
KIND = gparams['kind']
BC = gparams['bc']

# Seed
if KIND=='train':
    np.random.seed(5)
elif KIND=='validate':
    np.random.seed(10)
else:
    print('error!')

match = re.search(r'N(\d+)', FILE)
if match:
    ne_val = match.group(1)   # e.g. "72"
else:
    raise ValueError(f"Could not parse 'N' value from FILE: {FILE}")

# Base path
base = f"data_ordered/P2x1_ne{ne_val}_stokes"

# Add bc + dt if bc is set
if gparams["bc"] is not None:
    mesh_path = f"{base}_{gparams['bc']}_BC.npz"
else:
    mesh_path = f"{base}.npz"

# Load mesh data
mesh = np.load(mesh_path, allow_pickle=True)
num_element, num_pts, p = mesh['ne'], mesh['ng'], mesh['p']
idx_sol = mesh['idx_sol']

matrix=mesh['matrix']
    
def f(x,y,coeff):
    m0, m1, n0, n1, n2, n3=coeff
    return m0*np.sin(n0*x+n1*y) + m1*np.cos(n2*x+n3*y), coeff

def standard(n,matrix, kind):
    b=mesh[f'{kind}_load_vectors'][n]
    S=matrix
    coeff_u=np.linalg.solve(S, b)
    return coeff_u

def create_data(num_data, p, matrix, kind):
    data = []
    for n in tqdm(range(num_data)):
        f_value, coeff_f=f(p[:,0],p[:,1],mesh[f'{kind}_coeff_fs'][n])
        coeff_u= standard(n, matrix, kind)
        data.append([coeff_u, f_value, coeff_f])
    return np.array(data, dtype=object)

def save_obj(data, name, bc, kind):
    cwd = os.getcwd()
    path = os.path.join(cwd, 'data_ordered', bc, kind)
    if os.path.isdir(path) == False:
        os.makedirs(path)
    with open(path + '/' + name + '.pkl', 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
         
data = create_data(NUM_DATA, p[idx_sol[0]], matrix, KIND)
save_obj(data, FILE, BC, KIND)