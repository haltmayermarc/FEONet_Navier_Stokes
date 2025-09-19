import numpy as np
import pandas as pd
import scipy
from scipy import io
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
parser.add_argument("--bc", type=str, choices=['lower', 'zero', 'channel_flow'])
args = parser.parse_args()
gparams = args.__dict__

BC = gparams['bc']

def create_data(num_input ,num_xy ,deg_f, ordering=False):
    mesh = RectangleMesh(Point(0, 0), Point(1,1), num_xy, num_xy)

    # Define the function spaces:
    V = VectorElement('CG', triangle, 2)
    Q = FiniteElement('CG', triangle, 1)
    TH = V * Q
    W = FunctionSpace(mesh, TH)


    # Define the BC
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
    ( u, p ) = TrialFunctions ( W )
    ( v, q ) = TestFunctions ( W )
    mu = 0.1
    a = (mu * inner(grad(v), grad(u)) - p*div(v) - q*div(u) ) * dx

    ### What's the difference here?
    #a = ( 0.5 * mu * inner ( grad ( v ) + grad ( v ).T, grad ( u ) + grad ( u ).T ) \
    #    - div ( v )* p + q * div ( u ) ) * dx

    # Assemble matrix
    A = assemble(a)
    for bc in bcs:
        bc.apply(A)
    matrix = A.array()

    ne=mesh.cells().shape[0]

    # FENiCS ordering
    pos_u1=W.sub(0).sub(0).collapse().tabulate_dof_coordinates()
    pos_u2=W.sub(0).sub(1).collapse().tabulate_dof_coordinates()
    pos_p=W.sub(1).collapse().tabulate_dof_coordinates()

    # numerical ordering
    pos_all=W.tabulate_dof_coordinates()
    idx_u1=W.sub(0).sub(0).dofmap().dofs()
    idx_u2=W.sub(0).sub(1).dofmap().dofs()
    idx_p=W.sub(1).dofmap().dofs()

    # Construct the permutations for re-indexing
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
    train_load_vectors = []
    train_fenics_u1 = []
    train_fenics_u2 = []
    train_fenics_p = []

    validate_coeff_fs = []
    validate_load_vectors = []
    validate_fenics_u1 = []
    validate_fenics_u2 = []
    validate_fenics_p = []


    # TRAINING SET
    np.random.seed(5)
    for _ in tqdm(range(num_input[0])):
        m0, m1 = np.random.rand(2)
        n0, n1, n2, n3 = np.pi * np.random.rand(4)

        f = Expression(("m0*sin(n0*x[0]+n1*x[1])", "m1*cos(n2*x[0]+n3*x[1])"),
                   degree=deg_f, m0=m0, m1=m1, n0=n0, n1=n1, n2=n2, n3=n3)
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
        train_coeff_fs.append(np.array([m0, m1, n0, n1, n2, n3]))
        train_load_vectors.append(L.get_local())
        train_fenics_u1.append(sol_u1)
        train_fenics_u2.append(sol_u2)
        train_fenics_p.append(sol_p)

    # VALIDATION SET
    np.random.seed(10)
    for _ in tqdm(range(num_input[1])):
        m0, m1 = np.random.rand(2)
        n0, n1, n2, n3 = np.pi * np.random.rand(4)

        f = Expression(("m0*sin(n0*x[0]+n1*x[1])", "m1*cos(n2*x[0]+n3*x[1])"),
                   degree=deg_f, m0=m0, m1=m1, n0=n0, n1=n1, n2=n2, n3=n3)
        l = inner(f, v) * dx
        L = assemble(l)
        for bc in bcs:
            bc.apply(A, L)

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

        validate_coeff_fs.append(np.array([m0, m1, n0, n1, n2, n3]))
        validate_load_vectors.append(L.get_local())
        validate_fenics_u1.append(sol_u1)
        validate_fenics_u2.append(sol_u2)
        validate_fenics_p.append(sol_p)

    print(type(p))

    return ne, ng, p, gfl, idx_sol, pos_u1, pos_p, matrix, np.array(train_coeff_fs), np.array(train_load_vectors), np.array(train_fenics_u1), np.array(train_fenics_u2), np.array(train_fenics_p), np.array(validate_coeff_fs), np.array(validate_load_vectors), np.array(validate_fenics_u1), np.array(validate_fenics_u2), np.array(validate_fenics_p)

order='2x1'
list_num_xy=[6,15]
num_input=[1000, 1000]
typ='stokes'
deg_f = 5

for idx, num in enumerate(list_num_xy):
    ne, ng, p, gfl, idx_sol, pos_u, pos_p, matrix, train_coeff_fs, train_load_vectors, train_fenics_u1, train_fenics_u2, train_fenics_p, validate_coeff_fs, validate_load_vectors, validate_fenics_u1, validate_fenics_u2, validate_fenics_p=create_data(num_input, num, deg_f, ordering=True)

    # build filename
    base = f"data_ordered/P{order}_ne{ne}_{typ}"
    if gparams["bc"] is not None:
        mesh_path = f"{base}_{gparams['bc']}_BC.npz"
    else:
        mesh_path = f"{base}.npz"

    # save with mesh_path
    np.savez(
        mesh_path,
        ne=ne, ng=ng, p=p, gfl=gfl, idx_sol=idx_sol,
        pos_u=pos_u, pos_p=pos_p, matrix=matrix,
        train_coeff_fs=train_coeff_fs,
        train_load_vectors = train_load_vectors,
        train_fenics_u1=train_fenics_u1, train_fenics_u2=train_fenics_u2,
        train_fenics_p=train_fenics_p,
        validate_coeff_fs=validate_coeff_fs,
        validate_load_vectors= validate_load_vectors,
        validate_fenics_u1=validate_fenics_u1,
        validate_fenics_u2=validate_fenics_u2,
        validate_fenics_p=validate_fenics_p
    )
    print(f"Saved data at {mesh_path} for num_xy = {num}")