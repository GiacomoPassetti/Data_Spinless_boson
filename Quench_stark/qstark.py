# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 11:10:56 2021

@author: giaco
"""
import sys
sys.path.append('C:/Users/giaco/Desktop/Cluster/Quench_stark')
import numpy as np
from N_cons import ansatz_wf, Suz_trot_im, H_Peier_bond, full_sweep, U_bond
import tenpy.linalg.np_conserved as npc
import pickle
import time

Nmax=20
L=int(sys.argv[1])
g_0=float(sys.argv[2])
g=g_0/np.sqrt(L)
Omega  = 10
J=1
h=1
V=0
dt=0.005
tmax=10
ID_gs='Psi_GS_Nmax_'+str(Nmax)+'L_'+str(L)+'Omega_'+str(Omega)+'J_'+str(J)+'h_'+str(0.1)+'V_'+str(V)+'g_0'+str(g_0)+'imTEBD'
ID='Quench_ws_Nmax'+str(Nmax)+'L_'+str(L)+'Omega_'+str(Omega)+'J_'+str(J)+'h_'+str(h)+'V_'+str(V)+'g_0'+str(g_0)

trunc_param={'chi_max':120,'svd_min': 1.e-13, 'verbose': False}




#Here i will have to load the psi from the ground_data

with open(ID_gs+'.pkl', 'rb') as f:
    psi = pickle.load(f)

Id=ons_r=npc.outer(psi.sites[0].Id.replace_labels(['p','p*'],['p0', 'p0*']),npc.outer(psi.sites[1].Id.replace_labels(['p', 'p*'], ['p1', 'p1*']),psi.sites[1].Id.replace_labels(['p', 'p*'], ['p2', 'p2*'])) ).itranspose([0,2,4,1,3,5])
U=[]
for i in range(L-1):
   U.append(U_bond(-1j*dt, H_Peier_bond(psi, g, J, Omega,V, (2*i+1)*h, (2*i+2)*h, L)))

n_av=[]
NN=[]
A=[]
eps=0
errors=[]
n_av.append(psi.expectation_value('N'))
NN.append(psi.expectation_value('NN', [0]))
A.append(psi.expectation_value(psi.sites[0].B+psi.sites[0].Bd, [0]))
errors.append(eps)
ts=time.time()
for i in range(int(tmax//(10*dt))):
    eps += full_sweep(psi, 10, U, Id, trunc_param, L).eps
    print('test', time.time()-ts)
    n_av.append(psi.expectation_value('N'))
    NN.append(psi.expectation_value('NN', [0]))
    A.append(psi.expectation_value(psi.sites[0].B+psi.sites[0].Bd, [0]))
    errors.append(eps)
    
np.save(ID+'n_av.npy', n_av)
np.save(ID+'NN.npy', NN)
np.save(ID+'A.npy', A)
np.save(ID+'eps.npy', errors)
    

