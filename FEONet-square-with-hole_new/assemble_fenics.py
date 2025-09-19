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

# ARGS
parser = argparse.ArgumentParser("SEM")
parser.add_argument("--bc", type=str, choices=['lower', 'channel_flow'])
parser.add_argument("--forcing_term", type=str, choices=['sincos', 'grf'])
args = parser.parse_args()
gparams = args.__dict__

BC = gparams['bc']
FORCE = gparams['forcing_term']

num = 14
domain = Rectangle(Point(-1,-1), Point(1,1)) - Circle(Point(0,0), 0.5)
mesh = generate_mesh(domain, num)

def create_data(mesh, num_input ,deg_f, ordering=False):
    
    V = VectorElement('CG', triangle, 2)
    Q = FiniteElement('CG', triangle, 1)
    TH = V * Q
    W = FunctionSpace(mesh, TH)

    if BC == 'channel_flow':
        x_c, y_c = 0.0, 0.0  
        class Circle(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and (x[0]-x_c)**2 + (x[1]-y_c)**2 < 0.5**2
    
        facet_marker = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
        Circle().mark(facet_marker, 2)

        # Define boundary conditions
        u_inflow = Expression(("1 - x[1]*x[1]", "0"), degree=1)
        noslip = DirichletBC(W.sub(0), (0, 0), "on_boundary && (x[1] >= 0.9 || x[1] < 0.1)")

        inflow = DirichletBC(W.sub(0), u_inflow, "on_boundary && x[0] <= 0.1")
        circle = DirichletBC(W.sub(0), (0, 0), facet_marker, 2)
        outflow = DirichletBC(W.sub(1), 0, "on_boundary && x[0] >= 0.9")
        bcs = [noslip, inflow, outflow, circle]

    elif BC == "lower":
        SlipRate = Expression ( ( "-5.0", "0.0" ), degree=3)
        def LowerBoundary ( x, on_boundary ):
            return x[1] < DOLFIN_EPS and on_boundary
        bc = DirichletBC(W.sub ( 0 ), SlipRate, LowerBoundary)
        bcs = [bc]

    #  Define the variational problem
    ( u, p ) = TrialFunctions ( W )
    ( v, q ) = TestFunctions ( W )

    mu = 0.1
    a = ( 0.5 * mu * inner ( grad ( v ) + grad ( v ).T, grad ( u ) + grad ( u ).T ) \
        - div ( v )* p + q * div ( u ) ) * dx

    # Assemble matrix
    A = assemble(a)
    for bc in bcs:
        bc.apply(A)
    matrix = A.array()

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

    idx_bdry0_pts=list(bc.get_boundary_values().keys())

    gfl = np.zeros((ng,1))
    gfl[idx_bdry0_pts]=1
  
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
    for _ in tqdm(range(num_input[0])):
        m0, m1 = np.random.rand(2)
        n0, n1, n2, n3 = np.pi * np.random.rand(4)
        if FORCE == "sincos":
            f = Expression(("m0*sin(n0*x[0]+n1*x[1])", "m1*cos(n2*x[0]+n3*x[1])"),
                   degree=deg_f, m0=m0, m1=m1, n0=n0, n1=n1, n2=n2, n3=n3)
            train_coeff_fs.append(np.array([m0, m1, n0, n1, n2, n3]))
            train_forcing_term.append(np.array([0.0,0.0]))

        elif FORCE == "grf":
            u_x, u_y = generate_grf_sample(p[idx_sol[0]])
            # Create Function and assign values
            f = Function(V)
            f_values = np.empty(2 * len(u_x))
            f_values[0::2] = u_x
            f_values[1::2] = u_y
            f.vector().set_local(f_values)
            f.vector().apply('insert')

            train_coeff_fs.append(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
            train_forcing_term.append(np.concatenate((u_x, u_y)))
        
        l = inner(f, v) * dx
        L = assemble(l)
        for bc in bcs:
            bc.apply(A, L)

        # Solve system in FunctionSpace W
        w = Function(W)
        solve(A, w.vector(), L)

        # Split mixed solution into components
        u, pressure = w.split(deepcopy=True)
        sol_u1 = u.sub(0, deepcopy=True).vector()[:]
        sol_u2 = u.sub(1, deepcopy=True).vector()[:]
        sol_p = pressure.vector()[:]

        if ordering:
            sol_u1 = sol_u1[perm_u1]
            sol_u2 = sol_u2[perm_u2]
            sol_p = sol_p[perm_p]

        # Store data
        train_load_vectors.append(L.get_local())
        train_fenics_u1.append(sol_u1)
        train_fenics_u2.append(sol_u2)
        train_fenics_p.append(sol_p)

    # VALIDATION SET
    np.random.seed(10)
    for _ in tqdm(range(num_input[1])):
        m0, m1 = np.random.rand(2)
        n0, n1, n2, n3 = np.pi * np.random.rand(4)
        if FORCE == "sincos":
            f = Expression(("m0+sin(n0*x[0]+n1*x[1])", "m1+cos(n2*x[0]+n3*x[1])"),
                   degree=deg_f, m0=m0, m1=m1, n0=n0, n1=n1, n2=n2, n3=n3)
            validate_coeff_fs.append(np.array([m0, m1, n0, n1, n2, n3]))
            validate_forcing_term.append(np.array([0.0,0.0]))

        elif FORCE == "grf":
            u_x, u_y = generate_grf_sample(p[idx_sol[0]])

            # Create Function and assign values
            f = Function(V)
            f_values = np.empty(2 * len(u_x))
            f_values[0::2] = u_x
            f_values[1::2] = u_y
            f.vector().set_local(f_values)
            f.vector().apply('insert')

            validate_coeff_fs.append(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
            validate_forcing_term.append(np.concatenate((u_x, u_y)))
            
        l = inner(f, v) * dx
        L = assemble(l)
        for bc in bcs:
            bc.apply(A,L)

        w = Function(W)
        solve(A, w.vector(), L)

        # Split mixed solution into components
        u, pressure = w.split(deepcopy=True)
        sol_u1 = u.sub(0, deepcopy=True).vector()[:]
        sol_u2 = u.sub(1, deepcopy=True).vector()[:]
        sol_p = pressure.vector()[:]

        if ordering:
            sol_u1 = sol_u1[perm_u1]
            sol_u2 = sol_u2[perm_u2]
            sol_p = sol_p[perm_p]

        validate_load_vectors.append(L.get_local())
        validate_fenics_u1.append(sol_u1)
        validate_fenics_u2.append(sol_u2)
        validate_fenics_p.append(sol_p)

    print(type(p))

    return ne, ng, p, gfl, idx_sol, pos_u1, pos_p, matrix, np.array(train_coeff_fs), np.array(train_forcing_term), np.array(train_load_vectors), np.array(train_fenics_u1), np.array(train_fenics_u2), np.array(train_fenics_p), np.array(validate_coeff_fs), np.array(validate_forcing_term), np.array(validate_load_vectors), np.array(validate_fenics_u1), np.array(validate_fenics_u2), np.array(validate_fenics_p)

order='2x1'
list_num_xy=[14]
num_input=[1000, 1000]
typ='stokes'
deg_f = 5

for idx, num in enumerate(list_num_xy):
    ne, ng, p, gfl, idx_sol, pos_u, pos_p, matrix, train_coeff_fs, train_forcing_term, train_load_vectors, train_fenics_u1, train_fenics_u2, train_fenics_p, validate_coeff_fs, validate_forcing_term, validate_load_vectors, validate_fenics_u1, validate_fenics_u2, validate_fenics_p=create_data(mesh, num_input, deg_f, ordering=True)

    # build filename
    base = f"data_ordered/P{order}_ne{ne}_{typ}"
    if gparams["bc"] is not None:
        mesh_path = f"{base}_{gparams['bc']}_BC_{FORCE}.npz"
    else:
        mesh_path = f"{base}.npz"

    # save with mesh_path
    np.savez(
        mesh_path,
        ne=ne, ng=ng, p=p, gfl=gfl, idx_sol=idx_sol,
        pos_u=pos_u, pos_p=pos_p, matrix=matrix,
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