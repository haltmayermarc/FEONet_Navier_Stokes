import numpy as np
import pandas as pd
import scipy
from scipy import io
from scipy.spatial.distance import cdist
from scipy.linalg import cholesky
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import time
from datetime import datetime
import os
import random
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dolfin import *
from mshr import *

# ARGS
parser = argparse.ArgumentParser("SEM")
parser.add_argument("--bc", type=str, choices=['lower',  'channel_flow'])
parser.add_argument("--forcing_term", type=str, choices=['grf',  'sincos'])
args = parser.parse_args()
gparams = args.__dict__

BC = gparams['bc']
FORCE = gparams['forcing_term']

# Generate GRF
def rbf_kernel(X, length_scale, variance):
    dists = cdist(X, X, metric='euclidean')
    return variance * np.exp(-0.5 * (dists / length_scale)**2)

def generate_grf_sample(coords, length_scale=2.0, variance=1.0):
    # Compute RBF kernel
    K = rbf_kernel(coords, length_scale, variance)
    # Add small jitter for numerical stability
    K += 1e-8 * np.eye(K.shape[0])
    # Sample two independent GRFs for x and y components
    L = cholesky(K, lower=True)
    u_x = L @ np.random.randn(coords.shape[0])
    u_y = L @ np.random.randn(coords.shape[0])
    return u_x, u_y

# Domain parameters
num = 15
deg_f = 5
Le = 1.0
He = 1.0
#domain = Rectangle(Point(0,0), Point(1,1)) # - Circle(Point(0,0), 0.5)
#mesh = generate_mesh(domain, num)
mesh = RectangleMesh ( Point(0.0, 0.0), Point(Le, He), num, num)

def create_data(mesh, num_input , num, deg_f, ordering=False):
    
    V = VectorElement('CG', triangle, 2)
    Q = FiniteElement('CG', triangle, 1)
    TH = V * Q
    W = FunctionSpace(mesh, TH)

    # Define the Dirichlet condition
    if BC == "lower":
        SlipRate = Expression ( ( "(3.0 + 1.7 * sin ( 2.0 * pi * x[0] ) )", "0.0" ), degree=3 )
        def LowerBoundary ( x, on_boundary ):
            return x[1] < DOLFIN_EPS and on_boundary
        bc = DirichletBC(W.sub ( 0 ), SlipRate, LowerBoundary)
        bcs = [bc]
    elif BC == "channel_flow":
        inflow  = 'near(x[0], 0)'
        outflow = 'near(x[0], 1)'
        walls   = 'near(x[1], 0) || near(x[1], 1)'

        bcu_noslip  = DirichletBC(W.sub(0), Constant((0, 0)), walls)
        bcp_inflow  = DirichletBC(W.sub(1), Constant(8), inflow)
        bcp_outflow = DirichletBC(W.sub(1), Constant(0), outflow)
        bcs = [bcp_inflow, bcp_outflow, bcu_noslip]

    #  Define the variational problem
    (u, p) = TrialFunctions(W)
    (v, q) = TestFunctions(W)

    #  Material viscosity, Pa-sec.
    mu=0.1
    a = ( 
        mu * inner(grad(v), grad(u)) 
        - p*div(v) - q*div(u) ) * dx

    b1 = ( 
        dot(dot(grad(u), Constant((1.0,0.0))), v) 
    ) * dx

    b2 = ( 
        dot(dot(grad(u), Constant((0.0,1.0))), v) 
    ) * dx

    #  Matrix assembly.
    ## With B(coeff)=coeff[u]*B1+coeff[v]*B2,
    ## we will do (A+B(coeff)) @ coeff = load_vector
    ## Therefore, we can give the loss | (A+B(coeff)) @ coeff -load_vector |^2
    A = assemble(a)
    for bc in bcs:
        bc.apply(A)
    A = A.array()

    B1 = assemble(b1)
    for bc in bcs:
        bc.apply(B1)
    B1 = B1.array()

    B2 = assemble(b2)
    for bc in bcs:
        bc.apply(B2)
    B2 = B2.array()

    ne=mesh.cells().shape[0]

    pos_u1=W.sub(0).sub(0).collapse().tabulate_dof_coordinates()
    pos_u2=W.sub(0).sub(1).collapse().tabulate_dof_coordinates()
    pos_p=W.sub(1).collapse().tabulate_dof_coordinates()

    pos_all=W.tabulate_dof_coordinates()
    idx_u1=W.sub(0).sub(0).dofmap().dofs()
    idx_u2=W.sub(0).sub(1).dofmap().dofs()
    idx_p=W.sub(1).dofmap().dofs()

    row_to_index_u1 = {tuple(row): i for i, row in enumerate(pos_u1)}
    row_to_index_u2 = {tuple(row): i for i, row in enumerate(pos_u2)}
    row_to_index_p = {tuple(row): i for i, row in enumerate(pos_p)}

    perm_u1 = np.array([row_to_index_u1[tuple(row)] for row in pos_all[idx_u1]])
    perm_u2 = np.array([row_to_index_u2[tuple(row)] for row in pos_all[idx_u2]])
    perm_p = np.array([row_to_index_p[tuple(row)] for row in pos_all[idx_p]])
  
    p=pos_all
    ng=p.shape[0]
  
    idx_sol=[idx_u1,idx_u2,idx_p]
    idx_sol=np.array(idx_sol, dtype=object)

    print("Num of Elements : {}, Num of points : {}".format(ne, ng))

    # Generate training and validation data
    train_coeff_fs = []
    train_forcing_term = []
    train_load_vectors = []
    train_fenics_u1 = []
    train_fenics_u2 = []
    train_fenics_p = []

    validate_coeff_fs = []
    validate_forcing_term = []
    validate_load_vectors = []
    validate_fenics_u1 = []
    validate_fenics_u2 = []
    validate_fenics_p = []

    V = W.sub(0).collapse()


    # TRAINING SET
    np.random.seed(5)
    converged_train = 0
    not_converged_train = 0
    while(converged_train < num_input[0]):
        m0, m1 = np.random.rand(2)
        n0, n1, n2, n3 = np.pi * np.random.rand(4)

        if FORCE == "sincos":
            f = Expression(("m0*sin(n0*x[0]+n1*x[1])", "m1*cos(n2*x[0]+n3*x[1])"),
                   degree=deg_f, m0=m0, m1=m1, n0=n0, n1=n1, n2=n2, n3=n3)

        elif FORCE == "grf":
            u_x, u_y = generate_grf_sample(p[idx_sol[0]])
            # Create Function and assign values
            f = Function(V)
            f_values = np.empty(2 * len(u_x))
            f_values[0::2] = u_x
            f_values[1::2] = u_y
            f.vector().set_local(f_values)
            f.vector().apply('insert')

        #  Define the variational problem: a(u,v) = L(v).
        w = Function(W)
        (u, pr) = split(w)
        (v, q) = TestFunctions (W)

        a = ( 
            0.5 * mu * inner(grad(v), grad(u)) 
            - pr*div(v) - q*div(u) 
            + dot(dot(grad(u), u), v)
            ) * dx
        
        l = inner(f, v) * dx
        L = assemble(l)
        bc.apply(L)
        
        F = a - l

        #  Solution.
        J = derivative(F, w)
        #solve ( F == 0, w, bc, J = J )
        try:
            solve(F == 0, w, bcs, J=J,
                    solver_parameters={
                        "newton_solver": {
                        "absolute_tolerance": 1e-10,
                        "relative_tolerance": 1e-10,
                        "maximum_iterations": 50,
                        "linear_solver": "mumps",
                        "error_on_nonconvergence": True
                    }
                })
            converged_train += 1
            u, pressure = w.split(deepcopy=True)
            sol_u1 = u.sub(0, deepcopy=True).vector()[:]
            sol_u2 = u.sub(1, deepcopy=True).vector()[:]
            sol_p = pressure.vector()[:]

            if ordering:
                sol_u1 = sol_u1[perm_u1]
                sol_u2 = sol_u2[perm_u2]
                sol_p = sol_p[perm_p]

            # Store data
            train_coeff_fs.append(np.array([m0, m1, n0, n1, n2, n3]))
            if FORCE == "sincos":
                train_forcing_term.append(np.array([0.0,0.0]))
            elif FORCE == "grf":
                train_forcing_term.append(np.concatenate((u_x, u_y)))
            train_load_vectors.append(L.get_local())
            train_fenics_u1.append(sol_u1)
            train_fenics_u2.append(sol_u2)
            train_fenics_p.append(sol_p)

        except RuntimeError:
            not_converged_train += 1
            print("Solver did not fully converge, using best available solution.")
        
        

    # VALIDATION SET
    np.random.seed(10)
    converged_val = 0
    not_converged_val = 0
    while(converged_val < num_input[1]):
        m0, m1 = np.random.rand(2)
        n0, n1, n2, n3 = np.pi * np.random.rand(4)

        if FORCE == "sincos":
            f = Expression(("m0+sin(n0*x[0]+n1*x[1])", "m1+cos(n2*x[0]+n3*x[1])"),
                   degree=deg_f, m0=m0, m1=m1, n0=n0, n1=n1, n2=n2, n3=n3)
        elif FORCE == "grf":
            u_x, u_y = generate_grf_sample(p[idx_sol[0]])

            # Create Function and assign values
            f = Function(V)
            f_values = np.empty(2 * len(u_x))
            f_values[0::2] = u_x
            f_values[1::2] = u_y
            f.vector().set_local(f_values)
            f.vector().apply('insert')

        
        #  Define the variational problem: a(u,v) = L(v).
        w = Function(W)
        (u, pr) = split(w)
        (v, q) = TestFunctions (W)

        a = ( 
            0.5 * mu * inner(grad(v), grad(u)) 
            - pr*div(v) - q*div(u) 
            + dot(dot(grad(u), u), v)
            ) * dx
        
        l = inner(f, v) * dx
        L = assemble(l)
        bc.apply(L)
        
        F = a - l

        #  Solution.
        J = derivative(F, w)
        #solve ( F == 0, w, bc, J = J )
        try:
            solve(F == 0, w, bcs, J=J,
                    solver_parameters={
                        "newton_solver": {
                        "absolute_tolerance": 1e-10,
                        "relative_tolerance": 1e-10,
                        "maximum_iterations": 100,
                        "linear_solver": "mumps",
                        "error_on_nonconvergence": True
                    }
                })
            
            converged_val += 1
            u, pressure = w.split(deepcopy=True)
            sol_u1 = u.sub(0, deepcopy=True).vector()[:]
            sol_u2 = u.sub(1, deepcopy=True).vector()[:]
            sol_p = pressure.vector()[:]

            if ordering:
                sol_u1 = sol_u1[perm_u1]
                sol_u2 = sol_u2[perm_u2]
                sol_p = sol_p[perm_p]

            # Store data
            validate_coeff_fs.append(np.array([m0, m1, n0, n1, n2, n3]))
            if FORCE == "sincos":
                validate_forcing_term.append(np.array([0.0,0.0]))
            elif FORCE == "grf":
                validate_forcing_term.append(np.concatenate((u_x, u_y)))
            validate_load_vectors.append(L.get_local())
            validate_fenics_u1.append(sol_u1)
            validate_fenics_u2.append(sol_u2)
            validate_fenics_p.append(sol_p)

        except RuntimeError:
            not_converged_val += 1
            print("Solver did not fully converge, using best available solution.")

    print("Converged train :", converged_train)
    print("Converged val :", converged_val)
    print("Not converged train: ", not_converged_train)
    print("Not converged val: ", not_converged_val)

    return ne, ng, p, idx_sol, pos_u1, pos_p, A, B1, B2, np.array(train_coeff_fs), np.array(train_forcing_term), np.array(train_load_vectors), np.array(train_fenics_u1), np.array(train_fenics_u2), np.array(train_fenics_p), np.array(validate_coeff_fs), np.array(validate_forcing_term), np.array(validate_load_vectors), np.array(validate_fenics_u1), np.array(validate_fenics_u2), np.array(validate_fenics_p)

order='2x1'
list_num_xy=[15]
num_input=[1000, 1000]
typ='stokes'
deg_f = 5

for idx, num in enumerate(list_num_xy):
    ne, ng, p, idx_sol, pos_u, pos_p, A, B1, B2, train_coeff_fs, train_forcing_term, train_load_vectors, train_fenics_u1, train_fenics_u2, train_fenics_p, validate_coeff_fs, validate_forcing_term, validate_load_vectors, validate_fenics_u1, validate_fenics_u2, validate_fenics_p=create_data(mesh, num_input, num, deg_f, ordering=True)

    # build filename
    base = f"data_ordered/P{order}_ne{ne}_{typ}"
    if gparams["bc"] is not None:
        mesh_path = f"{base}_{gparams['bc']}_BC_{FORCE}.npz"
    else:
        mesh_path = f"{base}.npz"

    # save with mesh_path
    np.savez(
        mesh_path,
        ne=ne, ng=ng, p=p, idx_sol=idx_sol,
        pos_u=pos_u, pos_p=pos_p, 
        A=A, B1=B1, B2=B2,
        train_coeff_fs=train_coeff_fs,
        train_forcing_term=train_forcing_term,
        train_load_vectors = train_load_vectors,
        train_fenics_u1=train_fenics_u1, train_fenics_u2=train_fenics_u2,
        train_fenics_p=train_fenics_p,
        validate_coeff_fs=validate_coeff_fs,
        validate_forcing_term=validate_forcing_term,
        validate_load_vectors= validate_load_vectors,
        validate_fenics_u1=validate_fenics_u1,
        validate_fenics_u2=validate_fenics_u2,
        validate_fenics_p=validate_fenics_p
    )
    print(f"Saved data at {mesh_path} for num_xy = {num}")