# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 09:34:50 2021

@author: giaco
"""

import sys
import numpy as np
from N_cons_ws import ansatz_wf, Suz_trot_im, H_Peier_bond, ansatz_left, Energy, U_bond, full_sweep
import time
import tenpy.linalg.np_conserved as npc
import pickle



Nmax=15
L=int(sys.argv[1])
g_0=float(sys.argv[2])
g=0
Omega  = 10
J=1
h=0.1
V=0
max_error_E=[1.e-8, 1.e-7, 1.e-6]
ID='Psi_GS_Nmax_'+str(Nmax)+'L_'+str(L)+'Omega_'+str(Omega)+'J_'+str(J)+'h_'+str(h)+'V_'+str(V)+'g_0'+str(g_0)
N_steps=[20, 20, 20]
delta_t_im=[0.1, 1.e-2, 1.e-3]
trunc_param={'chi_max':120,'svd_min': 1.e-13, 'verbose': False}
psi=ansatz_left(Nmax, L)
ts=time.time()
Id=ons_r=npc.outer(psi.sites[0].Id.replace_labels(['p','p*'],['p0', 'p0*']),npc.outer(psi.sites[1].Id.replace_labels(['p', 'p*'], ['p1', 'p1*']),psi.sites[1].Id.replace_labels(['p', 'p*'], ['p2', 'p2*'])) ).itranspose([0,2,4,1,3,5])


H_bond=[H_Peier_bond(psi, g, J, Omega,V, h,h/2, L)]
for i in range(L-3):
   H_bond.append(H_Peier_bond(psi, g, J, Omega,V, (i+2)*h/2,(i+3)*h/2, L))
H_bond.append(H_Peier_bond(psi, g, J, Omega,V, (L-1)*h/2,L*h, L))


#Generate the GS from the initial Ansatz

Suz_trot_im(psi, delta_t_im, max_error_E, N_steps, H_bond, trunc_param, L, Id)
with open(ID+'imTEBD.pkl', 'wb') as f:
       pickle.dump(psi, f)
psi=0

with open(ID+'imTEBD.pkl', 'rb') as f:
       psi=pickle.load(f)


gg=np.arange(0.2,2,0.1)
for g in gg:
    print('finding ground for:', g)
    ID='Psi_GS_Nmax_'+str(Nmax)+'L_'+str(L)+'Omega_'+str(Omega)+'J_'+str(J)+'h_'+str(h)+'V_'+str(V)+'g'+str(g)
    N_steps=[10]
    delta_t_im=[1.e-3]
    max_error_E=[1.e-6]
    H_bond=[H_Peier_bond(psi, g, J, Omega,V, h,h/2, L)]
    for i in range(L-3):
       H_bond.append(H_Peier_bond(psi, g, J, Omega,V, (i+2)*h/2,(i+3)*h/2, L))
    H_bond.append(H_Peier_bond(psi, g, J, Omega,V, (L-1)*h/2,L*h, L))
    Suz_trot_im(psi, delta_t_im, max_error_E, N_steps, H_bond, trunc_param, L, Id)
    with open(ID+'imTEBD.pkl', 'wb') as f:
        pickle.dump(psi, f)
  


"""
h=1
dt=0.005
tmax=10
ID_gs='Psi_GS_Nmax_'+str(Nmax)+'L_'+str(L)+'Omega_'+str(Omega)+'J_'+str(J)+'h_'+str(0.1)+'V_'+str(V)+'g_0'+str(g_0)+'imTEBD'
ID='Quench_ws_Nmax'+str(Nmax)+'L_'+str(L)+'Omega_'+str(Omega)+'J_'+str(J)+'h_'+str(h)+'V_'+str(V)+'g_0'+str(g_0)

trunc_param={'chi_max':120,'svd_min': 1.e-13, 'verbose': False}
psi=ansatz_wf(Nmax, L)
Id=ons_r=npc.outer(psi.sites[0].Id.replace_labels(['p','p*'],['p0', 'p0*']),npc.outer(psi.sites[1].Id.replace_labels(['p', 'p*'], ['p1', 'p1*']),psi.sites[1].Id.replace_labels(['p', 'p*'], ['p2', 'p2*'])) ).itranspose([0,2,4,1,3,5])
U=[U_bond(1j*dt, H_Peier_bond(psi, g, J, Omega,V, h,h/2, L))]
for i in range(L-3):
   U.append(U_bond(1j*dt, H_Peier_bond(psi, g, J, Omega,V, (i+2)*h/2,(i+3)*h/2, L)))
U.append(U_bond(1j*dt, H_Peier_bond(psi, g, J, Omega,V, (L-1)*h/2,L*h, L)))

#Here i will have to load the psi from the ground_data




n_av=[]
NN=[]
A=[]
eps=0
errors=[]
n_av.append(psi.expectation_value('N'))
NN.append(psi.expectation_value('NN', [0]))
A.append(psi.expectation_value(psi.sites[0].B+psi.sites[0].Bd, [0]))
errors.append(eps)
for i in range(int(tmax//(10*dt))):
    print('Time ev step:', 10*dt)
    eps += full_sweep(psi, 10, U, Id, trunc_param, L).eps
    
    n_av.append(psi.expectation_value('N'))
    NN.append(psi.expectation_value('NN', [0]))
    A.append(psi.expectation_value(psi.sites[0].B+psi.sites[0].Bd, [0]))
    errors.append(eps)
    if time.time()-ts > 100000:
        with open(ID+'imTEBD_interrupted_at_t_'+str(i*dt*10)+'.pkl', 'wb') as f:
          pickle.dump(psi, f)
        
    
np.save(ID+'n_av.npy', n_av)
np.save(ID+'NN.npy', NN)
np.save(ID+'A.npy', A)
np.save(ID+'eps.npy', errors)
"""
    
