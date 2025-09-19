import random
import torch
import time
import datetime
import subprocess
import re
import os
import argparse
import gc
import sys
import pickle
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.sparse import diags
from scipy.sparse import identity
from scipy.sparse.linalg import onenormest
from pprint import pprint
import matplotlib.pyplot as plt
from network import *


# ARGS
parser = argparse.ArgumentParser("SEM")
## Data
#parser.add_argument("--equation", type=str, default='Stokes', choices=['Stokes'])
parser.add_argument("--bc", type=str)
parser.add_argument("--file", type=str, help='Example: --file 3000N18')
parser.add_argument("--domain", type=str, default='dolfin', choices=['dolfin' ,'square'])

## Train parameters
parser.add_argument("--pretrained", type=str, default=None)
parser.add_argument("--model", type=str, default='Net2D', choices=['Net2D', 'FCNN', 'UNetWithHead'])
parser.add_argument("--optimizer", type=str)
parser.add_argument("--do_precond", type=int) 
parser.add_argument("--batch_size", type=int, default=None)
parser.add_argument("--resol_in", type=int, default=20)
parser.add_argument("--blocks", type=int, default=0)
parser.add_argument("--ks", type=int, default=5)
parser.add_argument("--filters", type=int, default=32, choices=[8, 16, 32, 64])
parser.add_argument("--loss", type=str, default='MSE', choices=['MAE', 'MSE', 'RMSE', 'RelMSE'])
parser.add_argument("--epochs", type=int, default=80000)
parser.add_argument("--pre_epochs", type=int, default=0)

args = parser.parse_args()
gparams = args.__dict__

#Equation
#EQUATION = gparams['equation']
FILE = gparams['file']
NUM_DATA = int(FILE.split('N')[0])
DOMAIN=gparams['domain']
BC = gparams['bc']

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
#mesh_path = f"/home/haltmayermarc/FEONet_Stokes/FEONet_Stokes_square/data_ordered/P2x1_ne{ne_val}_stokes.npz"
mesh = np.load(mesh_path, allow_pickle=True)
NUM_ELEMENT, NUM_PTS, p = mesh['ne'], mesh['ng'], mesh['p']
IDX_SOL = mesh['idx_sol']
MATRIX=mesh['matrix']
cond_number = np.linalg.cond(MATRIX)

#Model
models = {
          'Net2D': Net2D,
          'FCNN': FCNN,
          'UNetWithHead': UNetWithHead,
          }
MODEL = models[gparams['model']]
BLOCKS = int(gparams['blocks'])
KERNEL_SIZE = int(gparams['ks'])
FILTERS = int(gparams['filters'])
PADDING = (KERNEL_SIZE - 1)//2
RESOL_IN=gparams['resol_in']

#Train
EPOCHS = int(gparams['epochs'])
pre_EPOCHS = int(gparams['pre_epochs'])
LOSS=gparams['loss']
D_in = 2
D_out = NUM_PTS
if gparams['batch_size']==None:
    BATCH_SIZE = NUM_DATA
else:
    BATCH_SIZE = gparams['batch_size']

def spai(A, m):
    """Perform m step of the SPAI iteration."""

    n = A.shape[0]

    ident = identity(n, format='csr')
    alpha = 2 / onenormest(A @ A.T)
    M = alpha * A

    for _ in tqdm(range(m)):
        C = A @ M
        G = ident - C
        AG = A @ G
        trace = (G.T @ AG).diagonal().sum()
        alpha = trace / np.linalg.norm(AG.data)**2
        M = M + alpha * G

    return M

if gparams['do_precond'] > 0:
    print("Old condition number: ", cond_number)
    print("Preconditioning...")
    DO_PRECOND = True
    m = gparams['do_precond']

    precond_path = f'precond_{NUM_ELEMENT}_{BC}.npy'

    if os.path.exists(precond_path):
        PRECOND = np.load(precond_path)
        print("Loaded existing preconditioner.")
    else:
        PRECOND = spai(MATRIX, m)
        np.save(precond_path, PRECOND)
        print("Preconditioner generated and saved.")

    new_cond_number = np.linalg.cond(MATRIX @ PRECOND)
    print("Done! Condition number: ", new_cond_number)
else:
    DO_PRECOND = False
    PRECOND = np.zeros_like(MATRIX)

    
#Save file
cur_time = str(datetime.datetime.now()).replace(' ', 'T')
cur_time = cur_time.replace(':','').split('.')[0].replace('-','')
FOLDER = f'{gparams["model"]}_epochs{EPOCHS}_{cur_time}'
PATH = os.path.join('../../train', 'results', FILE, FOLDER)


if gparams['model'] == "Net2D":
    model_FEONet = MODEL(RESOL_IN, D_in, FILTERS, D_out, kernel_size=KERNEL_SIZE, padding=PADDING, blocks=BLOCKS)
elif gparams['model'] == "FCNN":
    model_FEONet = MODEL(6, D_out, hidden_dims = [16, 32, 64, 128, 256])
elif gparams['model'] == "UNetWithHead":
    model_FEONet = MODEL(resol_in=RESOL_IN, 
                        in_ch=D_in,
                        base_ch = 32,
                        latent_ch = 64,
                        d_out = D_out, 
                        head_filters = FILTERS,
                        head_blocks=BLOCKS,
                        head_kernel_size=KERNEL_SIZE, 
                        head_padding=PADDING)
    
device = torch.device(f"cuda:2" if torch.cuda.is_available() else "cpu")
model_FEONet = model_FEONet.to(device)
    

# KAIMING INITIALIZATION
def weights_init(m):
    if isinstance(m, nn.Conv1d):
        # torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.kaiming_normal_(m.weight.data)
        torch.nn.init.zeros_(m.bias)

        
model_FEONet.apply(weights_init)


class Dataset(Dataset):
    def __init__(self, gparams, mesh, kind='train'):
        self.pickle_file = gparams['file']
        with open(f'/home/haltmayermarc/FEONet_Stokes/FEONet_Stokes_square_copy/data_ordered/{BC}/{kind}/' + self.pickle_file + '.pkl', 'rb') as f:
            self.data = pickle.load(f)
        self.load_vector = mesh[f'{kind}_load_vectors']
        self.fenics_u1 = mesh[f'{kind}_fenics_u1']
        self.fenics_u2 = mesh[f'{kind}_fenics_u2']
        self.fenics_p = mesh[f'{kind}_fenics_p']
    def __getitem__(self, idx):
        coeff_u = torch.FloatTensor(self.data[idx,0]).unsqueeze(0)
        f_value = torch.FloatTensor(self.data[idx,1]).unsqueeze(0)
        coeff_f = torch.FloatTensor(self.data[idx,2])
        load_vec_f = torch.FloatTensor(self.load_vector[idx])
        fenics_u1 = torch.FloatTensor(self.fenics_u1[idx])
        fenics_u2 = torch.FloatTensor(self.fenics_u2[idx])
        fenics_p = torch.FloatTensor(self.fenics_p[idx])
        return {'coeff_u': coeff_u, 'f_value': f_value, 'coeff_f': coeff_f, 'load_vec_f' : load_vec_f, 'fenics_u1': fenics_u1, 'fenics_u2': fenics_u2, 'fenics_p': fenics_p}

    def __len__(self):
        return len(self.data)

lg_dataset = Dataset(gparams, mesh, kind='train')
trainloader = DataLoader(lg_dataset, batch_size=BATCH_SIZE, shuffle=True)
lg_dataset = Dataset(gparams, mesh, kind='validate')
validateloader = DataLoader(lg_dataset, batch_size=BATCH_SIZE, shuffle=False)

def init_optim_lbfgs(model):
    params = {'history_size': 10,
              'max_iter': 20,
              'tolerance_grad': 1e-15,
              'tolerance_change': 1e-15,
              'max_eval': 10,
                }
    return torch.optim.LBFGS(model.parameters(), **params)

def init_optim_adam(model, lr=1e-3, weight_decay=0):
    params = {
        'lr': lr,
        'weight_decay': weight_decay,
        'betas': (0.9, 0.999),
        'eps': 1e-8,
    }
    return torch.optim.Adam(model.parameters(), **params)

def init_optim_sgd(model, lr=1e-2, momentum=0.9, weight_decay=0):
    return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

def init_optim_adamw(model, lr=1e-3, weight_decay=1e-2):
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

def init_optim_adagrad(model, lr=1e-2, weight_decay=0):
    params = {
        'lr': lr,
        'weight_decay': weight_decay,
        'lr_decay': 0,     # learning rate decay (optional, default 0)
        'eps': 1e-10       # term for numerical stability
    }
    return torch.optim.Adagrad(model.parameters(), **params)


if gparams['optimizer'] == "LBFGS":
    optimizer = init_optim_lbfgs(model_FEONet)
elif gparams['optimizer'] == "Adam":
    optimizer = init_optim_adam(model_FEONet)
elif gparams['optimizer'] == "SGD":
    optimizer = init_optim_sgd(model_FEONet)
elif gparams['optimizer'] == "AdamW":
    optimizer = init_optim_adamw(model_FEONet)
elif gparams['optimizer'] == "Adagrad":
    optimizer = init_optim_adagrad(model_FEONet)

    
MATRIX = torch.tensor(MATRIX).to(device).float()
PRECOND = torch.tensor(PRECOND).to(device).float()

criterion_wf = torch.nn.MSELoss(reduction="sum")

def weak_form(coeff_u, load_vec_f, matrix, precond):

    if DO_PRECOND==True:
       LHS = torch.stack([(matrix@precond).mm(coeff_u_one) for coeff_u_one in coeff_u.transpose(1,2)])
    else:
        LHS = torch.stack([matrix.mm(coeff_u_one) for coeff_u_one in coeff_u.transpose(1,2)])
    LHS=torch.sum(LHS,dim=-1)

    ## RHS
    RHS = load_vec_f.to(device)#.float()
    return LHS, RHS



def closure(model, coeff_f, load_vec_f, matrix, precond, resol_in):
    if gparams['model'] == "Net2D" or gparams['model'] == "UNetWithHead":
        def f(x,y,coeff):
            m0, m1, n0, n1, n2, n3=coeff[:,[0]], coeff[:,[1]], coeff[:,[2]], coeff[:,[3]], coeff[:,[4]], coeff[:,[5]]
            return torch.stack([m0*torch.sin(n0*x+n1*y), m1*torch.cos(n2*x+n3*y)],dim=1)
        grid_x=torch.linspace(-1,1,resol_in)
        input_grid=torch.cartesian_prod(grid_x,grid_x)
        input_grid=input_grid.to(device)
        value_f=f(input_grid[:,0],input_grid[:,1], coeff_f).reshape(-1,2,resol_in,resol_in)
        pred_coeff_u = model(value_f)
    else:
        pred_coeff_u = model(coeff_f).unsqueeze(1)
    LHS, RHS = weak_form(pred_coeff_u, load_vec_f, matrix, precond)
    
    ## Loss
    loss_wf=torch.zeros((NUM_PTS,))
    for ii in range(NUM_PTS):
        # criterion_wf => summation on basis functions
        loss_wf[ii]=criterion_wf(LHS[:,ii], RHS[:,ii])

    # torch.sum => summation on funcions f_i
    loss = torch.sum(loss_wf)

    if DO_PRECOND==True:
        return  loss, (precond@pred_coeff_u.transpose(1,2)).transpose(1,2)
    else:
        return  loss, pred_coeff_u


def rel_L2_error(pred, true):
    return (torch.sum((true-pred)**2, dim=-1)/torch.sum((true)**2, dim=-1))**0.5

def log_gparams(gparams):
    cwd = os.getcwd()
    os.chdir(PATH)
    with open('parameters.txt', 'w') as f:
        for k, v in gparams.items():
            if k == 'losses':
                df = pd.DataFrame(gparams['losses'])
                df.to_csv('losses.csv')
            else:
                entry = f"{k}:{v}\n"
                f.write(entry)
    os.chdir(cwd)


def log_path(path):
    with open("../../paths.txt", "a") as f:
        f.write(str(path) + '\n')
        f.close()

#log_path(PATH)
#log_gparams(gparams)
path = os.path.join(os.getcwd(), 'model', BC, gparams["model"])
#path = os.path.join(os.getcwd(), 'model')
if not os.path.exists(path):
    os.makedirs(path)

log_dir = os.path.join(os.getcwd(), "log", BC, gparams["model"])
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(log_dir, f"training_log_Stokes_unit_square{timestamp}.txt")

nparams = sum(p.numel() for p in model_FEONet.parameters() if p.requires_grad)

# Write some basic information into log file
if not os.path.exists(log_file):
    with open(log_file, "w") as f:
        f.write(f"Number of parameters: {nparams}\n")
        f.write("=" * 60 + "\n")
        f.write(f"Old condition number: {cond_number}\n")
        f.write(f"New condition number: {new_cond_number}\n")
        f.write("=" * 60 + "\n")
        f.write("Trained with ordering!")
        f.write("=" * 60 + "\n")
        f.write(f"Model Architecture:\n{str(model_FEONet)}\n")
        f.write("=" * 60 + "\n")
        f.write(f"Optimizer:\n{str(optimizer)}\n")
        f.write("=" * 60 + "\n")
################################################
time0 = time.time()
losses=[]
train_rel_L2_errors=[]
train_u1_errors=[]
train_u2_errors=[]
train_p_errors=[]
test_rel_L2_errors=[]
test_u1_errors=[]
test_u2_errors=[]
test_p_errors=[]

print("#####################################################################")
print("Start training! #####################################################")
print("#####################################################################")

epoch_durations = []
start_time_100 = time.time()

#for epoch in tqdm(range(1, EPOCHS+1)):
for epoch in range(1, EPOCHS+1):
    model_FEONet.train()
    loss_total = 0
    num_samples=0
    train_rel_L2_error = 0
    train_u1_error = 0
    train_u2_error = 0
    train_p_error = 0

    for batch_idx, sample_batch in enumerate(trainloader):
        optimizer.zero_grad()
        sample_batch = {k: v.to(device) for k, v in sample_batch.items()}
        #coeff_u = sample_batch['coeff_u']
        #fenics_u1 = sample_batch['fenics_u1']
        #fenics_u2 = sample_batch['fenics_u2']
        #fenics_p = sample_batch['fenics_p']
        coeff_f = sample_batch['coeff_f']
        load_vec_f = sample_batch['load_vec_f']

        #loss, u_pred = closure(model_FEONet, coeff_f, load_vec_f, MATRIX, RESOL_IN)
        loss, u_pred = closure(model_FEONet, coeff_f, load_vec_f, MATRIX, PRECOND, RESOL_IN)
        loss.backward()
        optimizer.step(loss.item)

        loss_total += np.round(float(loss.item()), 4)
        num_samples += coeff_f.shape[0]

        u_pred=u_pred.squeeze().detach().cpu()

    if epoch%100==0:
        # Measure time for last 100 epochs
        elapsed_time = time.time() - start_time_100
        epoch_durations.append(elapsed_time)
        moving_avg = sum(epoch_durations[-5:]) / min(len(epoch_durations), 5)  # moving avg over last 5
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        avg_minutes = int(moving_avg // 60)
        avg_seconds = int(moving_avg % 60)
        ## Test
        num_samples=0
        test_rel_L2_error = 0
        test_u1_error = 0
        test_u2_error = 0
        test_p_error = 0
        for batch_idx, sample_batch in enumerate(validateloader):
            with torch.no_grad():
                model_FEONet.eval()
                coeff_u = sample_batch['coeff_u']
                fenics_u1 = sample_batch['fenics_u1']
                fenics_u2 = sample_batch['fenics_u2']
                fenics_p = sample_batch['fenics_p']
                coeff_f = sample_batch['coeff_f'].to(device)
                load_vec_f = sample_batch['load_vec_f'].to(device)
                _,u_pred = closure(model_FEONet, coeff_f, load_vec_f, MATRIX, PRECOND, RESOL_IN)
                u_pred=u_pred.squeeze().detach().cpu()
                coeff_u=coeff_u.squeeze()
                fenics_u1=fenics_u1.squeeze()
                fenics_u2=fenics_u2.squeeze()
                fenics_p=fenics_p.squeeze()
                test_rel_L2_error += torch.sum(rel_L2_error(u_pred, coeff_u))
                test_u1_error += torch.sum(rel_L2_error(u_pred[:,IDX_SOL[0]], fenics_u1))
                test_u2_error += torch.sum(rel_L2_error(u_pred[:,IDX_SOL[1]], fenics_u2))
                test_p_error += torch.sum(rel_L2_error(u_pred[:,IDX_SOL[2]], fenics_p))
                num_samples += coeff_u.shape[0]

        test_rel_L2_error /= num_samples
        test_u1_error /= num_samples
        test_u2_error /= num_samples
        test_p_error /= num_samples
        
        ##Save and print
        losses.append(loss_total)
        test_rel_L2_errors.append(test_rel_L2_error)
        test_u1_errors.append(test_u1_error)
        test_u2_errors.append(test_u2_error)
        test_p_errors.append(test_p_error)
        torch.save({'model_state_dict': model_FEONet.state_dict(),
                    'losses': losses,
                    'train_rel_L2_errors': train_rel_L2_errors,
                    'test_rel_L2_errors': test_rel_L2_errors
        }, path + '/model.pt')

        log_str = (
            f"Epoch {epoch:4d}: loss {loss_total:.4f} | "
            f"Time for last 100 epochs: {minutes}m {seconds}s | "
            f"Moving avg: {avg_minutes}m {avg_seconds}s\n"
            #f"  Train: rel_error={train_rel_L2_error:.5f}, u1={train_u1_error:.5f}, "
            #f"u2={train_u2_error:.5f}, p={train_p_error:.5f}\n"
            f"  Test:  rel_error={test_rel_L2_error:.5f}, u1={test_u1_error:.5f}, "
            f"u2={test_u2_error:.5f}, p={test_p_error:.5f}"
        )

        print(log_str)

        with open(log_file, "a") as f:
            f.write(log_str + "\n")

        start_time_100 = time.time()

train_t=time.time()-time0
nparams = sum(p.numel() for p in model_FEONet.parameters() if p.requires_grad)
with open(log_file, "a") as f:
    f.write("=" * 60 + "\n")
    f.write(f"Training completed in {train_t:.2f} seconds.\n")
    f.write(f"Number of parameters: {nparams}\n")
    f.write("=" * 60 + "\n")

print("Saving model to: ")
print(path + '/model.pth')
torch.save(model_FEONet.state_dict(), path + '/model.pth')