import torch
import os
import argparse
import pickle
import re
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from scipy.sparse import diags
from scipy.spatial.distance import cdist
from scipy.linalg import cholesky
from scipy.interpolate import griddata
from pprint import pprint


# ARGS
parser = argparse.ArgumentParser("SEM")
## Data
parser.add_argument("--bc", type=str, choices=['lower', 'channel_flow'])
parser.add_argument("--forcing_term", type=str, choices=['sincos', 'grf'])
parser.add_argument("--resol_in", type=int, default=None)
parser.add_argument("--num_data", type=int, default=3000)
parser.add_argument("--file", type=str, help='Example: --file 3000N18')
parser.add_argument("--kind", type=str, default='train', choices=['train', 'validate'])

args = parser.parse_args()
gparams = args.__dict__

NUM_DATA = gparams['num_data']
FILE = gparams['file']
KIND = gparams['kind']
BC = gparams['bc']
FORCE = gparams['forcing_term']
RESOL_IN = gparams['resol_in']


# Seed
if KIND=='train':
    np.random.seed(5)
elif KIND=='validate':
    np.random.seed(10)
else:
    print('error!')

match = re.search(r'N(\d+)', FILE)
if match:
    ne_val = match.group(1)
else:
    raise ValueError(f"Could not parse 'N' value from FILE: {FILE}")

# Base path
base = f"data_ordered/P2x1_ne{ne_val}_stokes"

# Add bc + dt if bc is set
if gparams["bc"] is not None:
    mesh_path = f"{base}_{gparams['bc']}_BC_{FORCE}.npz"
else:
    mesh_path = f"{base}.npz"

mesh=np.load(mesh_path, allow_pickle=True)
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

def create_data(num_data, p, matrix, force, kind):
    data = []
    if force == "sincos":
        for n in tqdm(range(num_data)):
            f_value, coeff_f=f(p[:,0],p[:,1],mesh[f'{kind}_coeff_fs'][n])
            coeff_u = standard(n, matrix, kind)

            data.append([coeff_u, f_value, coeff_f])
    elif force == "grf":
        for n in tqdm(range(num_data)):
            u_x = mesh[f'{kind}_forcing_term'][n][:len(idx_sol[0])]
            u_y = mesh[f'{kind}_forcing_term'][n][len(idx_sol[0]):]

            grid_x = np.linspace(-1, 1, RESOL_IN)
            xx, yy = np.meshgrid(grid_x, grid_x, indexing="ij")
            input_grid = np.column_stack([xx.ravel(), yy.ravel()])

            u_x_new = griddata(p, u_x, input_grid, method='cubic')
            u_y_new = griddata(p, u_y, input_grid, method='cubic')
            f_value = np.stack([u_x_new, u_y_new], axis=0)

            coeff_u= standard(n, matrix, kind)

            data.append([coeff_u, f_value, mesh[f'{kind}_coeff_fs'][n]])
    return np.array(data, dtype=object)

def save_obj(data, name, bc, force, kind):
    cwd = os.getcwd()
    path = os.path.join(cwd, 'data_ordered', bc, force, kind)
    if os.path.isdir(path) == False:
        os.makedirs(path)
    with open(path + '/' + name + '.pkl', 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

data = create_data(NUM_DATA, p[idx_sol[0]], matrix, FORCE, KIND)
save_obj(data, FILE, BC, FORCE, KIND)