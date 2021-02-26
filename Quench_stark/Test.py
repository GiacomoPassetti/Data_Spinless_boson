import sys

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

print(psi.expectation_value('N'))
    

