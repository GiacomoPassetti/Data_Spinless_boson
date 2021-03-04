import sys
import numpy as np
from N_cons_ws import ansatz_wf, Suz_trot_im, H_Peier_bond, ansatz_left, Energy, U_bond, full_sweep
import time
import tenpy.linalg.np_conserved as npc
import pickle



Nmax=15
L=int(sys.argv[1])
g_0=float(sys.argv[2])
g=g_0/(np.sqrt(L))
Omega  = 10
J=1
h=1
V=0
dt=0.005
tmax=10
ID='Psi_GS_Nmax_'+str(Nmax)+'L_'+str(L)+'Omega_'+str(Omega)+'J_'+str(J)+'h_'+str(0.1)+'V_'+str(V)+'g_0'+str(g_0)

trunc_param={'chi_max':120,'svd_min': 1.e-13, 'verbose': False}
psi=ansatz_left(Nmax, L)
ts=time.time()
Id=ons_r=npc.outer(psi.sites[0].Id.replace_labels(['p','p*'],['p0', 'p0*']),npc.outer(psi.sites[1].Id.replace_labels(['p', 'p*'], ['p1', 'p1*']),psi.sites[1].Id.replace_labels(['p', 'p*'], ['p2', 'p2*'])) ).itranspose([0,2,4,1,3,5])


#Load the grouns state
with open(ID+'imTEBD.pkl', 'rb') as f:
       psi=pickle.load(f)

U=[U_bond(1j*dt, H_Peier_bond(psi, g, J, Omega,V, h,h/2, L))]
for i in range(L-3):
   U.append(U_bond(1j*dt, H_Peier_bond(psi, g, J, Omega,V, (i+2)*h/2,(i+3)*h/2, L)))
U.append(U_bond(1j*dt, H_Peier_bond(psi, g, J, Omega,V, (L-1)*h/2,L*h, L)))

#Here i will have to load the psi from the ground_data


ID='Quench_stark_'+str(Nmax)+'L_'+str(L)+'Omega_'+str(Omega)+'J_'+str(J)+'h_'+str(0.1)+'V_'+str(V)+'g_0'+str(g_0)+'dt_'+str(dt)+'tmax'+str(tmax)

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
