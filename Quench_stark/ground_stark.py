# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 11:10:56 2021

@author: giaco
"""
import sys

import numpy as np
from N_cons import ansatz_wf, Suz_trot_im, H_Peier_bond
import tenpy.linalg.np_conserved as npc
import pickle


Nmax=20
L=int(sys.argv[1])
g_0=float(sys.argv[2])
g=g_0/np.sqrt(L)
Omega  = 10
J=1
h=0.1
V=0
max_error_E=[1.e-8, 1.e-8, 1.e-8, 1.e-8, 1.e-8, 1.e-8]
ID='Psi_GS_Nmax_'+str(Nmax)+'L_'+str(L)+'Omega_'+str(Omega)+'J_'+str(J)+'h_'+str(h)+'V_'+str(V)+'g_0'+str(g_0)
N_steps=[10, 10, 10, 10, 10, 10]
delta_t_im=[0.1, 1.e-2, 1.e-3, 1.e-4, 1.e-5]
trunc_param={'chi_max':120,'svd_min': 1.e-13, 'verbose': False}
psi=ansatz_wf(Nmax, L)
Id=ons_r=npc.outer(psi.sites[0].Id.replace_labels(['p','p*'],['p0', 'p0*']),npc.outer(psi.sites[1].Id.replace_labels(['p', 'p*'], ['p1', 'p1*']),psi.sites[1].Id.replace_labels(['p', 'p*'], ['p2', 'p2*'])) ).itranspose([0,2,4,1,3,5])
H_bond=[]
for i in range(L-1):
   H_bond.append(H_Peier_bond(psi, g, J, Omega,V, (2*i+1)*h, (2*i+2)*h, L))
   
#Generate the GS from the initial Ansatz
Suz_trot_im(psi, delta_t_im, max_error_E, N_steps, H_bond, trunc_param, L, Id)

with open(ID+'imTEBD.pkl', 'wb') as f:
       pickle.dump(psi, f)

