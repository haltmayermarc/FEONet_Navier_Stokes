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
from torch.nn.utils import clip_grad_norm_
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.sparse import diags
from scipy.sparse import identity
from scipy.sparse.linalg import onenormest
from pprint import pprint
import matplotlib.pyplot as plt
from network import UNetWithHead, UNetWithHead1D, VectorToSequenceRNN, UNetWithTemporalHead



# ARGS
parser = argparse.ArgumentParser("SEM")

## Data
parser.add_argument("--file", type=str, help='Example: --file 3000N18')

## Train parameters
parser.add_argument("--pretrained", type=str, default=None)
parser.add_argument("--optimizer", type=str)
parser.add_argument("--gpu", type=int)
parser.add_argument("--do_precond", type=int, default=0)
parser.add_argument("--dt", type=float)
parser.add_argument("--bc", type=str, choices=['lower', 'zero', 'channel_flow'])
parser.add_argument("--forcing_term", type=str, choices=['five', 'zero'])
parser.add_argument("--loss", type=str, default='MSE', choices=['MAE', 'MSE', 'RMSE', 'RelMSE'])
parser.add_argument("--epochs", type=int, default=150000)
parser.add_argument("--pre_epochs", type=int, default=0)
parser.add_argument("--batch_size", type=int, default=None)
parser.add_argument("--model", type=str, default='RNN',
                    choices=['UNet2D', 'UNet1D', 'RNN', 'UNetTemporal'])


#UNetWithHead2D parameters
parser.add_argument("--latent_ch", type=int, default=32)
parser.add_argument("--base_ch", type=int, default=32)

# RNN parameters
parser.add_argument("--hidden_dim", type=int, default=128)
parser.add_argument("--rnn_type", type=str, default=None, choices=['gru', 'lstm'])
parser.add_argument("--seq_len", type=int, default=None)
parser.add_argument("--num_layers", type=int, default=1)


args = parser.parse_args()

if args.seq_len is None:
    parser.error("--seq_len is required for time-dep models")

if args.model in ["UNet2D", "UNet1D"]:
    if args.latent_ch is None or args.base_ch is None:
        parser.error("--base_ch, --latent_ch are required when model is UNetWithHead!")

elif args.model in ["RNN", "UNetTemporal"]:
    if args.rnn_type is None:
        parser.error("--rnn_type is required when model is RNN")

gparams = args.__dict__

#Equation
FILE = gparams['file']
NUM_DATA = int(FILE.split('N')[0])
DT = gparams['dt']
BC = gparams['bc']
FORCE = gparams['forcing_term']


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
#mesh_path = f"/home/haltmayermarc/FEONet_Stokes/FEONet_time_dep_Stokes/data_ordered/P2x1_ne{ne_val}_stokes.npz"
mesh = np.load(mesh_path, allow_pickle=True)
NUM_ELEMENT, NUM_PTS, P = mesh['ne'], mesh['ng'], mesh['p']
IDX_SOL = mesh['idx_sol']
S = mesh['S']
A = mesh['A']
SYSTEM_MATRIX = S+A*DT
cond_number = np.linalg.cond(SYSTEM_MATRIX)


#Model
models = {
          'UNet2D': UNetWithHead,
          'UNet1D': UNetWithHead1D,
          'RNN': VectorToSequenceRNN,
          'UNetTemporal': UNetWithTemporalHead
          }
MODEL = models[gparams['model']]
model_str = gparams['model']

if gparams['model'] in ["UNet1D", "UNet2D", "UNetTemporal"]:
    BASE_CHANNEL = gparams["base_ch"]
    LATENT_CHANNEL = gparams["latent_ch"]
    D_in = 2
    HIDDEN_DIM = gparams['hidden_dim']
    RNN_TYPE = gparams['rnn_type']
    SEQ_LEN = int(gparams['seq_len'])
    NUM_LAYERS = int(gparams['num_layers'])
elif gparams['model'] == "RNN":
    D_in = NUM_PTS
    HIDDEN_DIM = gparams['hidden_dim']
    RNN_TYPE = gparams['rnn_type']
    SEQ_LEN = int(gparams['seq_len'])
    NUM_LAYERS = int(gparams['num_layers'])


#Train
EPOCHS = int(gparams['epochs'])
pre_EPOCHS = int(gparams['pre_epochs'])
LOSS=gparams['loss']
D_out = NUM_PTS

if gparams['batch_size']==None:
    BATCH_SIZE = NUM_DATA
else:
    BATCH_SIZE = gparams['batch_size']

def spai(A, m):
    n = A.shape[0]

    ident = identity(n, format='csr')
    alpha = 2 / onenormest(A @ A.T)
    M = alpha * A

    for index in tqdm(range(m)):
        C = A @ M
        G = ident - C
        AG = A @ G
        trace = (G.T @ AG).diagonal().sum()
        alpha = trace / np.linalg.norm(AG.data)**2
        M = M + alpha * G

    return M

if gparams['do_precond'] > 0:
    DO_PRECOND = True
    m = gparams['do_precond']
    print("Old Condition number of (S+dt*A):", cond_number)
    print("Preconditioning...")

    if os.path.exists('precond.npy'):
        PRECOND = np.load('precond.npy')
        print("Loaded existing preconditioner.")
    else:
        PRECOND = spai(SYSTEM_MATRIX, m)
        np.save('precond.npy', PRECOND)
        print("Preconditioner generated and saved.")

    new_cond_number = np.linalg.cond(SYSTEM_MATRIX @ PRECOND)
    print("Done! Condition number: ", new_cond_number)
else:
    DO_PRECOND = False
    PRECOND = np.zeros_like(SYSTEM_MATRIX)


if gparams['model'] == 'UNet2D':
    model_FEONet = MODEL(in_ch=D_in, 
                         base_ch=BASE_CHANNEL,
                         latent_ch=LATENT_CHANNEL,
                         d_out=D_out,
                         hidden=HIDDEN_DIM)
elif gparams['model'] == 'UNetTemporal':
    model_FEONet = MODEL(in_ch=D_in, 
                         base_ch=BASE_CHANNEL,
                         latent_ch=LATENT_CHANNEL,
                         d_out=D_out,
                         hidden=HIDDEN_DIM,
                         rnn_type=RNN_TYPE,
                         num_layers=NUM_LAYERS)
elif gparams['model'] == "RNN":
    model_FEONet = MODEL(input_dim=D_in,
                         hidden_dim = HIDDEN_DIM,
                         output_dim = D_out,
                         rnn_type = RNN_TYPE,
                         num_layers = NUM_LAYERS)
elif gparams['model'] == 'UNet1D':
    model_FEONet = MODEL(in_ch=3, 
                         base_ch=BASE_CHANNEL,
                         latent_ch=LATENT_CHANNEL)


# SEND TO GPU (or CPU)
gpu_no = gparams['gpu']
device = torch.device(f"cuda:{gpu_no}" if torch.cuda.is_available() else "cpu")
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
        with open(f'/home/haltmayermarc/FEONet_Stokes/FEONet_time_dep_Stokes/data_ordered/{BC}/{FORCE}/{dt_str}/{kind}/' + self.pickle_file + '.pkl', 'rb') as f:
            self.data = pickle.load(f)
        self.load_vector = mesh['load_vector']
        self.fenics_u1 = mesh[f'{kind}_fenics_u1']
        self.fenics_u2 = mesh[f'{kind}_fenics_u2']
        self.fenics_p = mesh[f'{kind}_fenics_p']
    def __getitem__(self, idx):
        coeffs_u    = torch.as_tensor(self.data[idx,0], dtype=torch.float32)
        init_value_x = torch.as_tensor(self.data[idx,1], dtype=torch.float32).unsqueeze(0)
        init_value_y = torch.as_tensor(self.data[idx,2], dtype=torch.float32).unsqueeze(0)
        coeffs_init  = torch.as_tensor(self.data[idx,3], dtype=torch.float32)
        load_vec_f = torch.as_tensor(self.load_vector, dtype=torch.float32)
        fenics_u1  = torch.as_tensor(self.fenics_u1[idx], dtype=torch.float32)
        fenics_u2  = torch.as_tensor(self.fenics_u2[idx], dtype=torch.float32)
        fenics_p   = torch.as_tensor(self.fenics_p[idx], dtype=torch.float32)

        return {
            'coeffs_u': coeffs_u,
            'init_value_x': init_value_x,
            'init_value_y': init_value_y,
            'coeffs_init': coeffs_init,
            'load_vec_f': load_vec_f,
            'fenics_u1': fenics_u1,
            'fenics_u2': fenics_u2,
            'fenics_p': fenics_p,
            }


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

    
SYSTEM_MATRIX = torch.tensor(SYSTEM_MATRIX).to(device).float()
PRECOND = torch.tensor(PRECOND).to(device).float()
S = torch.tensor(S).to(device).float() 
A = torch.tensor(A).to(device).float()
P = torch.tensor(P).to(device).float()

criterion_wf = torch.nn.MSELoss(reduction="sum")

def assemble_u_init(init_x, init_y, idx_sol, num_pts, device):
    if init_x.dim() == 3:
        init_x = init_x.squeeze(1)
    if init_y.dim() == 3:
        init_y = init_y.squeeze(1)

    B = init_x.size(0)
    u0 = torch.zeros(B, num_pts, device=device)

    idx_u1, idx_u2, _ = IDX_SOL
    u0[:, torch.as_tensor(idx_u1, device=device, dtype=torch.long)] = init_x
    u0[:, torch.as_tensor(idx_u2, device=device, dtype=torch.long)] = init_y
    return u0

def lower_init(x,y,m0,n0,m1,n1):
    return torch.stack([-5.0+m0*torch.sin(n0*x)*torch.sin(y), 0.0+m1*torch.cos(n1*x)*torch.sin(y)],dim=1)

def flow_init(x,y,m0,m1):
    return torch.stack([0.1*m0*(1 - y)*x[1], 0.1*m1*torch.sin(torch.pi*x[0])*(1-y)*y],dim=1)

def weak_form_sequence(pred_seq, load_vec_f, S_mat, A_mat, precond, dt, u_init, do_precond):
    B, T, N = pred_seq.shape
    system_mat = S_mat + dt * A_mat
    if do_precond:
        M = system_mat @ precond          
    else:
        M = system_mat                    

    # LHS_t = (M @ u_t)^T  == pred_seq @ M^T
    LHS = pred_seq @ M.T                  

    # RHS_t = S @ u_{t-1} + dt * f, with u_{-1} = u_init
    RHS_list = []
    u_prev = u_init                       
    for _t in range(T):
        RHS_t = u_prev @ S_mat.T + dt * load_vec_f   
        RHS_list.append(RHS_t)
        u_prev = pred_seq[:, _t, :]       # next previous = current pred
    RHS = torch.stack(RHS_list, dim=1)   
    return LHS, RHS

def closure(model, coeffs_init, init_value_x, init_value_y,
                load_vec_f, S_mat, A_mat, p, precond, dt, seq_len):
    
    u_init = assemble_u_init(init_value_x, init_value_y, IDX_SOL, NUM_PTS, device)

    if gparams['model'] == "RNN":
        pred_seq = model(u_init, seq_len=seq_len)
    elif gparams['model'] == "UNet1D":
        u_init = u_init.unsqueeze(1)
        p = P.T.unsqueeze(0).repeat(1000, 1, 1)
        input_tensor = torch.cat([u_init, p], dim=1)
        pred_seq = model(input_tensor, seq_len=seq_len)
    elif gparams['model'] == "UNet2D" or gparams['model'] == 'UNetTemporal':
        resol_in = 64
        grid_x=torch.linspace(-1,1,resol_in)
        input_grid=torch.cartesian_prod(grid_x,grid_x)
        input_grid=input_grid.to(device)
        if BC == "lower":
            init_values = lower_init(input_grid[:,0],input_grid[:,1],coeffs_init[:,[0]],coeffs_init[:,[1]],coeffs_init[:,[2]],coeffs_init[:,[3]]).reshape(-1,2,resol_in,resol_in)
        elif BC == "flow":
            init_values = flow_init(input_grid[:,0],input_grid[:,1],coeffs_init[:,[0]],coeffs_init[:,[1]]).reshape(-1,2,resol_in,resol_in)
        pred_seq = model(init_values, seq_len=seq_len)

    LHS, RHS = weak_form_sequence(
        pred_seq=pred_seq,
        load_vec_f=load_vec_f,
        S_mat=S_mat,
        A_mat=A_mat,
        precond=precond,
        dt=DT,
        u_init=u_init,
        do_precond=DO_PRECOND
    )

    resid = LHS - RHS                 
    per_t = (resid**2).sum(dim=(0, 2))
    loss = per_t.mean()

    if DO_PRECOND:
        pred_seq_pc = pred_seq @ precond.T
        return loss, pred_seq_pc
    else:
        return loss, pred_seq


def rel_L2_error(pred, true):
    return (torch.sum((true-pred)**2, dim=-1)/torch.sum((true)**2, dim=-1))**0.5

def relative_L2(pred, ref):
    num = torch.norm(pred - ref, dim=-1)   
    den = torch.norm(ref, dim=-1) + 1e-12 
    rel = num / den
    return rel.mean(dim=1).mean()

def log_path(path):
    with open("../../paths.txt", "a") as f:
        f.write(str(path) + '\n')
        f.close()

path = os.path.join(os.getcwd(), f'model/{BC}/{FORCE}/{dt_str}', gparams["model"])
#path = os.path.join(os.getcwd(), 'model')
if not os.path.exists(path):
    os.makedirs(path)

log_dir = f'log/{BC}/{FORCE}/{dt_str}/{model_str}'
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
        if DO_PRECOND:
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
epoch_durations = []

print("#####################################################################")
print("Start training! #####################################################")
print("#####################################################################")

start_time_100 = time.time()
CLIP_VALUE = 5.0

for epoch in range(1, EPOCHS+1):
    model_FEONet.train()
    loss_total = 0
    num_samples=0
    train_rel_L2_error = 0
    train_u1_error = 0
    train_u2_error = 0
    train_p_error = 0

    for batch_idx, sample_batch in enumerate(trainloader):
        sample_batch = {k: v.to(device) for k, v in sample_batch.items()}
        coeffs_u     = sample_batch['coeffs_u'   ].to(device)          
        init_value_x = sample_batch['init_value_x'].to(device)
        init_value_y = sample_batch['init_value_y'].to(device)
        coeffs_init  = sample_batch['coeffs_init'].to(device)          
        load_vec_f   = sample_batch['load_vec_f' ].to(device).float()
        

        loss, _ = closure(
            model=model_FEONet,
            coeffs_init=coeffs_init,
            init_value_x=init_value_x,
            init_value_y=init_value_y,
            load_vec_f=load_vec_f,
            S_mat=S,
            A_mat=A,
            p = P,
            precond=PRECOND,
            dt=DT,
            seq_len=SEQ_LEN
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        #clip_grad_norm_(model_FEONet.parameters(), max_norm=CLIP_VALUE)

        optimizer.step(loss.item)

        loss_total += np.round(float(loss.item()), 4)
        num_samples += load_vec_f.shape[0]

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
        test_u1_error_linalg = 0
        test_u2_error_linalg = 0
        test_p_error_linalg = 0
        for batch_idx, sample_batch in enumerate(validateloader):
            with torch.no_grad():
                model_FEONet.eval()
                coeffs_u     = sample_batch['coeffs_u'   ].to(device)          
                init_value_x = sample_batch['init_value_x'].to(device)
                init_value_y = sample_batch['init_value_y'].to(device)
                coeffs_init  = sample_batch['coeffs_init'].to(device)          
                load_vec_f   = sample_batch['load_vec_f' ].to(device).float()
                fenics_u1   = sample_batch['fenics_u1'].to(device).float()
                fenics_u2   = sample_batch['fenics_u2'].to(device).float()
                fenics_p   = sample_batch['fenics_p'].to(device).float()

                loss, pred_seq = closure(
                    model=model_FEONet,
                    coeffs_init=coeffs_init,
                    init_value_x=init_value_x,
                    init_value_y=init_value_y,
                    load_vec_f=load_vec_f,
                    S_mat=S,
                    A_mat=A,
                    p=P,
                    precond=PRECOND,
                    dt=DT,
                    seq_len=SEQ_LEN
                )

                pred_u1 = pred_seq[:, :, IDX_SOL[0]]  
                pred_u2 = pred_seq[:, :, IDX_SOL[1]]  
                pred_p  = pred_seq[:, :, IDX_SOL[2]]  

                # Cut Fenics reference to last 10 timesteps (first one is just t=0)
                coeffs_u_cut = coeffs_u[:, 1:, :]
                linalg_u1_cut = coeffs_u_cut[:, :, IDX_SOL[0]]
                linalg_u2_cut = coeffs_u_cut[:, :, IDX_SOL[1]]
                linalg_p_cut = coeffs_u_cut[:, :, IDX_SOL[2]]

                fenics_u1_cut = fenics_u1[:, 1:, :]   
                fenics_u2_cut = fenics_u2[:, 1:, :]   
                fenics_p_cut  = fenics_p[:, 1:, :]

                # Compute errors
                batch_error = relative_L2(pred_seq, coeffs_u_cut)
                batch_u1_error_linalg = relative_L2(pred_u1, linalg_u1_cut)
                batch_u2_error_linalg = relative_L2(pred_u2, linalg_u2_cut)
                batch_p_error_linalg  = relative_L2(pred_p,  linalg_p_cut)
                batch_u1_error = relative_L2(pred_u1, fenics_u1_cut)
                batch_u2_error = relative_L2(pred_u2, fenics_u2_cut)
                batch_p_error  = relative_L2(pred_p,  fenics_p_cut)

                test_rel_L2_error += batch_error.item()
                test_u1_error_linalg += batch_u1_error_linalg.item()
                test_u2_error_linalg += batch_u2_error_linalg.item()
                test_p_error_linalg  += batch_p_error_linalg.item()
                test_u1_error += batch_u1_error.item()
                test_u2_error += batch_u2_error.item()
                test_p_error  += batch_p_error.item()
                num_samples   += 1

                pred_u1 = pred_seq[:, :, IDX_SOL[0]]  
                pred_u2 = pred_seq[:, :, IDX_SOL[1]]  
                pred_p  = pred_seq[:, :, IDX_SOL[2]]  

                # Cut Fenics reference to last 10 timesteps (first one is just t=0)
                coeffs_u_cut = coeffs_u[:, 1:, :]
                linalg_u1_cut = coeffs_u_cut[:, :, IDX_SOL[0]]
                linalg_u2_cut = coeffs_u_cut[:, :, IDX_SOL[1]]
                linalg_p_cut = coeffs_u_cut[:, :, IDX_SOL[2]]

                fenics_u1_cut = fenics_u1[:, 1:, :]   
                fenics_u2_cut = fenics_u2[:, 1:, :]   
                fenics_p_cut  = fenics_p[:, 1:, :]

                # Compute errors
                batch_error = relative_L2(pred_seq, coeffs_u_cut)
                batch_u1_error = relative_L2(pred_u1, linalg_u1_cut)
                batch_u2_error = relative_L2(pred_u2, linalg_u2_cut)
                batch_p_error  = relative_L2(pred_p,  linalg_p_cut)

                test_rel_L2_error += batch_error.item()
                test_u1_error += batch_u1_error.item()
                test_u2_error += batch_u2_error.item()
                test_p_error  += batch_p_error.item()
                num_samples   += 1

        test_rel_L2_error /= num_samples
        test_u1_error /= num_samples
        test_u2_error /= num_samples
        test_p_error /= num_samples
        test_u1_error_linalg /= num_samples
        test_u2_error_linalg /= num_samples
        test_p_error_linalg /= num_samples
        
        ##Save and print
        losses.append(loss_total)
        test_u1_errors.append(test_u1_error)
        test_u2_errors.append(test_u2_error)
        test_p_errors.append(test_p_error)
        torch.save({'model_state_dict': model_FEONet.state_dict(),
                    'losses': losses,
                    'train_rel_L2_errors': train_rel_L2_errors,
                    'test_rel_L2_errors': test_rel_L2_errors
        }, path + f'/model_{timestamp}.pt')

        log_str = (
            f"Epoch {epoch:4d}: loss {loss_total:.4f} | "
            f"Time for last 100 epochs: {minutes}m {seconds}s | "
            f"Moving avg: {avg_minutes}m {avg_seconds}s\n"
            f"Test: linalg={test_rel_L2_error:.5f}"
            f"  Test: u1={test_u1_error_linalg:.5f}, "
            f"u2={test_u2_error_linalg:.5f}, p={test_p_error_linalg:.5f}\n"
            f"Test fenics: u1={test_u1_error:.5f}, "
            f"u2={test_u2_error:.5f}, p={test_p_error:.5f}\n"
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