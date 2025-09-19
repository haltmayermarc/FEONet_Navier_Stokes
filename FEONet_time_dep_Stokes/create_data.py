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
parser.add_argument("--num_data", type=int, default=1000)
parser.add_argument("--file", type=str, help='Example: --file 3000N18')
parser.add_argument("--kind", type=str, default='train', choices=['train', 'validate'])
parser.add_argument("--dt", type=float, default=0.01)
parser.add_argument("--bc", type=str, choices=['lower', 'zero', 'channel_flow'])
parser.add_argument("--forcing_term", type=str, choices=['five', 'zero'])


args = parser.parse_args()
gparams = args.__dict__

NUM_DATA = gparams['num_data']
FILE = gparams['file']
KIND = gparams['kind']
DT = gparams['dt']
BC = gparams['bc']
FORCE = gparams['forcing_term']

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

# Format dt safely for filenames: replace "." with "_"
dt_str = str(DT).replace(".", "_")

# Base path
base = f"data_ordered/P2x1_ne{ne_val}_stokes"

# Add bc + dt if bc is set
if gparams["bc"] is not None:
    mesh_path = f"{base}_{gparams['bc']}_BC_{FORCE}_dt_{dt_str}.npz"
else:
    mesh_path = f"{base}.npz"

# Load mesh data
#mesh_path = f"data_ordered/P2x1_ne{ne_val}_stokes.npz"
mesh = np.load(mesh_path, allow_pickle=True)
num_element, num_pts, p = mesh['ne'], mesh['ng'], mesh['p']
idx_sol = mesh['idx_sol']
load_vector = mesh['load_vector']
S = mesh['S']
A = mesh['A']
T=[0,1]

#S=torch.tensor(S, dtype=torch.float64)
#A=torch.tensor(A, dtype=torch.float64)
#load_vector=torch.tensor(load_vector, dtype=torch.float64)


def standard(n, S, A, dt, kind):
    grid_t = np.arange(T[0], T[1] + dt, dt)

    u_solve_linalg = np.zeros(len(idx_sol[0]) + len(idx_sol[1]) + len(idx_sol[2]), dtype=np.float64)
    u_solve_linalg[idx_sol[0]] = mesh[f'{kind}_coeffs_init'][n][0]
    u_solve_linalg[idx_sol[1]] = mesh[f'{kind}_coeffs_init'][n][1]

    u_linalg = [u_solve_linalg]

    M = S + A * dt

    for _ in grid_t[1:]:
        rhs = S @ u_linalg[-1].reshape(-1, 1) + dt * load_vector.reshape(-1, 1)
        predict = np.linalg.solve(M, rhs)
        u_linalg.append(predict.reshape(-1))

    return np.array(u_linalg)

def create_data(num_data, p, S, A, dt, kind):
    data = []
    for n in tqdm(range(num_data)):
        init_value_x = mesh[f'{kind}_values_init'][n][0]
        init_value_y = mesh[f'{kind}_values_init'][n][1]
        coeffs_u= standard(n, S , A, dt, kind)
        coeffs_init = mesh[f'{kind}_coeffs_init'][n]
        data.append([coeffs_u, init_value_x, init_value_y, coeffs_init])
    return np.array(data, dtype=object)

def save_obj(data, name, kind, bc, force, dt):
    cwd = os.getcwd()
    dt_str = str(dt).replace(".", "_")
    # build path like: data_ordered/lower/five/0_01/train
    path = os.path.join(cwd, 'data_ordered', bc, force, dt_str, kind)
    os.makedirs(path, exist_ok=True)

    filepath = os.path.join(path, name + '.pkl')
    with open(filepath, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    print(f"Saved {filepath}")


data = create_data(NUM_DATA, p, S, A, DT, KIND)
save_obj(data, FILE, KIND, BC, FORCE, DT)