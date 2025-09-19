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

import math
import numpy as np
from dolfin import Expression, UserExpression

parser = argparse.ArgumentParser("SEM")
parser.add_argument("--dt", type=float, default=0.01)
parser.add_argument("--bc", type=str, choices=['lower', 'zero', 'channel_flow'])
parser.add_argument("--forcing_term", type=str, choices=['five', 'zero'])

args = parser.parse_args()
gparams = args.__dict__

BC = gparams['bc']
DT = gparams['dt']
FORCE = gparams['forcing_term']

class RandomNoSlipIC(UserExpression):
    def __init__(self, ks, ls, amps, degree=4, **kwargs):
        super().__init__(degree=degree, **kwargs)
        self.ks = np.array(ks, dtype=int)
        self.ls = np.array(ls, dtype=int)
        self.amps = np.array(amps, dtype=float)
        self.pi = math.pi

    def eval(self, values, x):
        X, Y = float(x[0]), float(x[1])
        ux, uy = 0.0, 0.0
        for k, l, a in zip(self.ks, self.ls, self.amps):
            sx = math.sin(k * self.pi * X)
            sy = math.sin(l * self.pi * Y)
            cx = math.cos(k * self.pi * X)
            cy = math.cos(l * self.pi * Y)
            # u = (∂ψ/∂y, -∂ψ/∂x), ψ = sin²(kπx) sin²(lπy)
            ux += a * (2.0 * (sx * sx) * sy * cy * l * self.pi)
            uy += a * (-2.0 * sx * cx * (sy * sy) * k * self.pi)
        values[0] = ux
        values[1] = uy

    def value_shape(self):
        return (2,)

def random_ic_expression(n_modes=3, kmax=4, lmax=4, amp=1.0, seed=None, degree=4):
    rng = np.random.default_rng(seed)
    ks = rng.integers(1, kmax+1, size=n_modes)
    ls = rng.integers(1, lmax+1, size=n_modes)
    scales = (ks**2 + ls**2)**(-0.5)  # decay to keep smooth
    amps = amp * rng.normal(size=n_modes) * scales
    return RandomNoSlipIC(ks=ks, ls=ls, amps=amps, degree=degree)


def create_data(num_input, num_xy, deg_u_init, dt, ordering=False):
    T=[0,1]
    grid_t=np.arange(T[0],T[1]+dt,dt)

    mesh = RectangleMesh(Point(0, 0), Point(1,1), num_xy, num_xy)

    # Define the function spaces:
    V = VectorElement('CG', triangle, 2)
    Q = FiniteElement('CG', triangle, 1)
    TH = V * Q
    W = FunctionSpace(mesh, TH)

    # Define the Dirichlet BC
    if BC == "lower":
        SlipRate = Expression ( ( "-5.0", "0.0" ), degree=3)
        def LowerBoundary ( x, on_boundary ):
            return x[1] < DOLFIN_EPS and on_boundary
        bc = DirichletBC(W.sub ( 0 ), SlipRate, LowerBoundary)
        bcs = [bc]
    elif BC == "zero":
        zero = Constant((0.0, 0.0))   # if W.sub(0) is vector-valued
        bc = DirichletBC(W.sub(0), zero, 'on_boundary')
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
    #  Material viscosity, Pa-sec.
    mu=1
    a = ( mu * inner(grad(v), grad(u)) \
        - div ( v )* p + q * div ( u ) ) * dx
    if FORCE == "five":
        f = Constant ((5.0, -5.0))
    elif FORCE == "zero":
        f = Constant (( 0.0, 0.0))
    l = inner ( v, f ) * dx

    w = Function(W)

    #  Matrix assembly.
    ## We will do (S/dt+A) @ newcoeff = (S/dt) @ old_coeff + load_vector
    ## Therefore, newcoeff = (S/dt+A)^-1 @ [(S/dt) @ old_coeff + load_vector]
    s = inner(v,u) * dx
    S = assemble(s)
    for bc in bcs:
        bc.apply(S)
    S = S.array()

    A = assemble(a)
    for bc in bcs:
        bc.apply(A)
    A = A.array()

    L = assemble(l)
    for bc in bcs:
        bc.apply(L)
    load_vector = L.get_local()

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
  
    ng=pos_all.shape[0]

    #idx_bdry0_pts=list(bc.get_boundary_values().keys())

    gfl = np.zeros((ng,1))
    #gfl[idx_bdry0_pts]=1
  
    idx_sol=[idx_u1,idx_u2,idx_p]
    idx_sol=np.array(idx_sol, dtype=object)

    print("Num of Elements : {}, Num of points : {}".format(ne, ng))

    train_coeffs_init = []
    train_values_init = []
    train_fenics_u1 = []
    train_fenics_u2 = []
    train_fenics_p = []

    # TRAINING SET
    np.random.seed(5)
    for _ in tqdm(range(num_input[0])):
        if BC == "lower":
            m0, m1 = 2+np.random.rand(2)
            n0, n1= 2*np.pi*(np.random.rand(2))
            train_coeffs_init.append(np.array([m0,n0,m1,n1]))
            u_init = Expression ( ( "-5.0+m0*sin(n0*x[0])*sin(x[1])", "0.0+m1*cos(n1*x[0])*sin(x[1])" ), degree=deg_u_init, m0=m0, m1=m1, n0=n0, n1=n1 )
        elif BC == "zero":
            u_init = random_ic_expression(n_modes=5, kmax=4, lmax=4, amp=1.0, seed=5, degree=4)
        elif BC == "channel_flow":
            m0, m1 = np.random.rand(2)
            train_coeffs_init.append(np.array([m0,m1]))
            u_init = Expression(("0.1*m0*(1 - x[1])*x[1]", "0.01*m1*sin(pi*x[0])*(1-x[1])*x[1]"), degree=deg_u_init, m0=m0, m1=m1)
        u_b = project(u_init, W.sub ( 0 ).collapse())
        #u_b_vector = u_b.vector()[:]
        u_b_vector_x = u_b.sub(0, deepcopy=True).vector()[:]
        u_b_vector_y = u_b.sub(1, deepcopy=True).vector()[:]
        u_b_vector_x_ordered = u_b_vector_x[perm_u1]
        u_b_vector_y_ordered = u_b_vector_y[perm_u2]
        train_values_init.append(np.array([u_b_vector_x_ordered,u_b_vector_y_ordered]))


        DT = Constant(dt)
        u_t = (1/DT) * inner ( v, (u-u_b) ) * dx

        F = u_t + a - l

        LHS = lhs(F)
        RHS = rhs(F)

        traj_u1=[]
        traj_u2=[]
        traj_p=[]
        traj_u1.append(u_b.sub(0, deepcopy=True).vector()[:])
        traj_u2.append(u_b.sub(1, deepcopy=True).vector()[:])
        traj_p.append(np.zeros(len(idx_p)))

        for t in grid_t[1:]:
            LHS = lhs(F)
            RHS = rhs(F)
            solve(LHS==RHS, w, bcs)
            (u, p) = w.split(deepcopy=True)
            u_b.assign(u)
            sol_u1=u.sub(0, deepcopy=True).vector()[:]
            sol_u2=u.sub(1, deepcopy=True).vector()[:]
            sol_p=p.vector()[:]
            traj_u1.append(sol_u1)
            traj_u2.append(sol_u2)
            traj_p.append(sol_p)

        if ordering:
            traj_u1_ordered = []
            traj_u2_ordered = []
            traj_p_ordered = []

            for t in range(len(traj_u1)):
                traj_u1_ordered.append(traj_u1[t][perm_u1])
                traj_u2_ordered.append(traj_u2[t][perm_u2])
                traj_p_ordered.append(traj_p[t][perm_p])

            traj_u1_ordered = np.array(traj_u1_ordered)
            traj_u2_ordered = np.array(traj_u2_ordered)
            traj_p_ordered = np.array(traj_p_ordered)

            train_fenics_u1.append(traj_u1_ordered)
            train_fenics_u2.append(traj_u2_ordered)
            train_fenics_p.append(traj_p_ordered)
        else:
            traj_u1=np.array(traj_u1)
            traj_u2=np.array(traj_u2)
            traj_p=np.array(traj_p)

            train_fenics_u1.append(traj_u1)
            train_fenics_u2.append(traj_u2)
            train_fenics_p.append(traj_p)

    validate_coeffs_init = []
    validate_values_init = []
    validate_fenics_u1 = []
    validate_fenics_u2 = []
    validate_fenics_p = []

    # TRAINING SET
    np.random.seed(10)
    for _ in tqdm(range(num_input[1])):
        if BC == "lower":
            m0, m1 = 2+np.random.rand(2)
            n0, n1= 2*np.pi*(np.random.rand(2))
            validate_coeffs_init.append(np.array([m0,m1,n0,n1]))
            u_init = Expression ( ( "-5.0+m0*sin(n0*x[0])*sin(x[1])", "0.0+m1*cos(n1*x[0])*sin(x[1])" ), degree=deg_u_init, m0=m0, m1=m1, n0=n0, n1=n1 )
        elif BC == "zero":
            u_init = random_ic_expression(n_modes=5, kmax=4, lmax=4, amp=1.0, seed=10, degree=4)
        elif BC == "channel_flow":
            m0, m1 = np.random.rand(2)
            u_init = Expression(("0.1*m0*(1 - x[1])*x[1]", "0.1*m1*sin(pi*x[0])*(1-x[1])*x[1]"), degree=deg_u_init, m0=m0, m1=m1)
            validate_coeffs_init.append(np.array([m0,m1]))
        u_b = project(u_init, W.sub ( 0 ).collapse())
        u_b_vector_x = u_b.sub(0, deepcopy=True).vector()[:]
        u_b_vector_y = u_b.sub(1, deepcopy=True).vector()[:]
        u_b_vector_x_ordered = u_b_vector_x[perm_u1]
        u_b_vector_y_ordered = u_b_vector_y[perm_u2]
        validate_values_init.append(np.array([u_b_vector_x_ordered,u_b_vector_y_ordered]))

        DT = Constant(dt)
        u_t = (1/DT) * inner(v, (u-u_b) ) * dx

        F = u_t + a - l

        LHS = lhs(F)
        RHS = rhs(F)

        traj_u1=[]
        traj_u2=[]
        traj_p=[]
        traj_u1.append(u_b.sub(0, deepcopy=True).vector()[:])
        traj_u2.append(u_b.sub(1, deepcopy=True).vector()[:])
        traj_p.append(np.zeros(len(idx_p)))

        for t in grid_t[1:]:
            LHS = lhs(F)
            RHS = rhs(F)
            solve(LHS==RHS, w, bc)
            (u, p) = w.split(deepcopy=True)
            u_b.assign(u)
            sol_u1=u.sub(0, deepcopy=True).vector()[:]
            sol_u2=u.sub(1,deepcopy=True).vector()[:]
            sol_p=p.vector()[:]
            traj_u1.append(sol_u1)
            traj_u2.append(sol_u2)
            traj_p.append(sol_p)

        if ordering:
            traj_u1_ordered = []
            traj_u2_ordered = []
            traj_p_ordered = []

            for t in range(len(traj_u1)):
                traj_u1_ordered.append(traj_u1[t][perm_u1])
                traj_u2_ordered.append(traj_u2[t][perm_u2])
                traj_p_ordered.append(traj_p[t][perm_p])

            traj_u1_ordered = np.array(traj_u1_ordered)
            traj_u2_ordered = np.array(traj_u2_ordered)
            traj_p_ordered = np.array(traj_p_ordered)

            validate_fenics_u1.append(traj_u1_ordered)
            validate_fenics_u2.append(traj_u2_ordered)
            validate_fenics_p.append(traj_p_ordered)

        else:
            traj_u1=np.array(traj_u1)
            traj_u2=np.array(traj_u2)
            traj_p=np.array(traj_p)

            validate_fenics_u1.append(traj_u1)
            validate_fenics_u2.append(traj_u2)
            validate_fenics_p.append(traj_p)

    return ne, ng, pos_all, gfl, idx_sol, pos_u1, pos_p, S, A, load_vector, np.array(train_coeffs_init), np.array(train_values_init), np.array(train_fenics_u1), np.array(train_fenics_u2), np.array(train_fenics_p), np.array(validate_coeffs_init), np.array(validate_values_init), np.array(validate_fenics_u1), np.array(validate_fenics_u2), np.array(validate_fenics_p)

order='2x1'
list_num_xy=[10]
num_input=[1000, 1000]
typ='stokes'
deg_u_init = 5

for idx, num in enumerate(list_num_xy):
    ne, ng, p, gfl, idx_sol, pos_u, pos_p, S, A, load_vector, train_coeffs_init, train_values_init, train_fenics_u1, train_fenics_u2, train_fenics_p, validate_coeffs_init, validate_values_init, validate_fenics_u1, validate_fenics_u2, validate_fenics_p = create_data(
        num_input, num, deg_u_init, dt=DT, ordering=True
    )

    # build filename
    dt_str = str(DT).replace(".", "_")
    base = f"data_ordered/P{order}_ne{ne}_{typ}"
    if gparams["bc"] is not None:
        mesh_path = f"{base}_{gparams['bc']}_BC_{FORCE}_dt_{dt_str}.npz"
    else:
        mesh_path = f"{base}.npz"

    # save with mesh_path
    np.savez(
        mesh_path,
        ne=ne, ng=ng, p=p, gfl=gfl, idx_sol=idx_sol,
        pos_u=pos_u, pos_p=pos_p, S=S, A=A, load_vector=load_vector,
        train_coeffs_init=train_coeffs_init,
        train_values_init=train_values_init,
        train_fenics_u1=train_fenics_u1, train_fenics_u2=train_fenics_u2,
        train_fenics_p=train_fenics_p,
        validate_coeffs_init=validate_coeffs_init,
        validate_values_init=validate_values_init,
        validate_fenics_u1=validate_fenics_u1,
        validate_fenics_u2=validate_fenics_u2,
        validate_fenics_p=validate_fenics_p
    )
    print(f"Saved data at {mesh_path} for num_xy = {num}")
