import tenpy
import copy
import sys
import numpy as np
import numpy.linalg as alg
from tenpy import models
from tenpy.networks.site import FermionSite
from tenpy.networks.site import BosonSite
from tenpy.networks.mps import MPS
from tenpy.tools.params import get_parameter
import tenpy.linalg.np_conserved as npc
from tenpy.networks.mpo import MPO, MPOEnvironment
import tenpy.linalg.charges as charges
from tenpy.models.lattice import Chain
from scipy.linalg import expm
import pickle




def sites(L,Nmax):
 FSite=FermionSite(None, filling=0.5)
 BSite=BosonSite(Nmax=Nmax,conserve=None, filling=0 )
 sites=[]
 sites.append(BSite)
 for i in range(L):
     sites.append(FSite)
 return sites

def product_state(L):
    ps=['vac']
    for i in range(int(L/2)):
        ps.append('empty')
        ps.append('full')
    return ps

def mixed_state(L):
    ps=['vac']
    ms = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
    for i in range(int(L/2)):
        ps.append(ms)
        ps.append(ms)
    return ps

def H_Peier(g, J, Omega, h, U):
    Peier=npc.outer(npc.expm(1j*g*(psi.sites[0].B+psi.sites[0].Bd)).replace_labels(['p', 'p*'], ['p0', 'p0*']),npc.outer(-J*psi.sites[1].Cd.replace_labels(['p', 'p*'], ['p1', 'p1*']),psi.sites[1].C.replace_labels(['p', 'p*'], ['p2', 'p2*']))).itranspose([0,2,4,1,3,5])
    Peier_hc=npc.outer(npc.expm(-1j*g*(psi.sites[0].B+psi.sites[0].Bd)).replace_labels(['p', 'p*'], ['p0', 'p0*']), npc.outer(-J*psi.sites[1].C.replace_labels(['p', 'p*'], ['p1', 'p1*']),psi.sites[1].Cd.replace_labels(['p', 'p*'], ['p2', 'p2*']))).itranspose([0,2,4,1,3,5])
    cav=npc.outer((Omega/((L-1)))*psi.sites[0].N.replace_labels(['p','p*'],['p0', 'p0*']),npc.outer(psi.sites[1].Id.replace_labels(['p', 'p*'], ['p1', 'p1*']),psi.sites[1].Id.replace_labels(['p', 'p*'], ['p2', 'p2*'])) ).itranspose([0,2,4,1,3,5])
    rep=npc.outer(psi.sites[0].Id.replace_labels(['p','p*'],['p0', 'p0*']),npc.outer(U*psi.sites[1].N.replace_labels(['p', 'p*'], ['p1', 'p1*']),psi.sites[1].N.replace_labels(['p', 'p*'], ['p2', 'p2*'])) ).itranspose([0,2,4,1,3,5])
    ons_l=npc.outer(psi.sites[0].Id.replace_labels(['p','p*'],['p0', 'p0*']),npc.outer(h*psi.sites[1].N.replace_labels(['p', 'p*'], ['p1', 'p1*']),psi.sites[1].Id.replace_labels(['p', 'p*'], ['p2', 'p2*'])) ).itranspose([0,2,4,1,3,5])
    ons_r=npc.outer(psi.sites[0].Id.replace_labels(['p','p*'],['p0', 'p0*']),npc.outer(psi.sites[1].Id.replace_labels(['p', 'p*'], ['p1', 'p1*']),h*psi.sites[1].N.replace_labels(['p', 'p*'], ['p2', 'p2*'])) ).itranspose([0,2,4,1,3,5])
    H_bond_tebd_ev=Peier+Peier_hc+cav+rep+ons_l+ons_r
    H_bond_tebd_odd=Peier+Peier_hc+cav+rep #This is the energetic term that will be used in the TEBD algorithm
    H_bond_ev=Peier+Peier_hc+rep+ons_l+ons_r
    H_bond_odd=Peier+Peier_hc+rep#This takes into account only the Peier term and is useful to evaluate the energy of the bonds
    return H_bond_tebd_ev, H_bond_tebd_odd, H_bond_ev, H_bond_odd
  
def onsite(psi, h):
    ons= h*psi.sites[1].N
    return ons
    


def U_bond(dt, H_bond):
    
    H2 = H_bond.combine_legs([('p0', 'p1', 'p2'), ('p0*', 'p1*', 'p2*')], qconj=[+1, -1])
    H2= (-dt)*H2
    U=npc.expm(H2).split_legs()
    return U



def Energy(psi, H_bond_ev, H_bond_odd):
        E=[psi.expectation_value(Omega*sites[0].N, [0])]
        
        for i in range(int(L/2)-1): # First Odd sweep
            
            psi.swap_sites(2*i, swap_op=None, trunc_par=trunc_param[1])
            E.append(psi.expectation_value(H_bond_odd, [2*i+1]))
            psi.swap_sites(2*i+1, swap_op=None, trunc_par=trunc_param[1])

        for i in range(int(L/2)-1):
            

            E.append(psi.expectation_value(H_bond_ev, [L-2-2*i]))
            psi.swap_sites(L-3-2*i, swap_op=None, trunc_par=trunc_param[1])
            psi.swap_sites(L-4-2*i, swap_op=None, trunc_par=trunc_param[1])

        E.append(psi.expectation_value(H_bond_ev, [0]))
        E_tot=np.sum(E)
        E_bond=np.average(E[1:L+1])
        
        return E_tot, E_bond

def Suz_trot_im(psi, delta_t, max_error_E, N_steps, H_bond_tebd_ev,H_bond_tebd_odd, H_bond_ev, H_bond_odd):
 DeltaE=2*max_error_E
 E_old=Energy(psi, H_bond_ev, H_bond_odd)[1]
 for dt in range(len(delta_t)):
    print("delta_tau =", delta_t[dt])
    U_ev=U_bond(delta_t[dt], H_bond_tebd_ev)
    U_odd= U_bond(delta_t[dt]/2, H_bond_tebd_odd)
    DeltaE= 2*max_error_E
    step=0
    while (DeltaE > max_error_E):
      for T in range(N_steps): 

        print("Step:", T)

            
        for i in range(int(L/2)-1): # First Odd sweep

      
            psi.swap_sites(2*i, swap_op=None, trunc_par=trunc_param[dt])


            psi.apply_local_op((2*i)+1 , U_odd, unitary=False, renormalize=True)


            psi.swap_sites(2*i+1, swap_op=None, trunc_par=trunc_param[dt])
        
        for i in range(int(L/2)-1):# Even sweep
            

            psi.apply_local_op(L-2-2*i, U_ev, unitary=False, renormalize=True)
            psi.swap_sites(L-3-2*i, swap_op=None, trunc_par=trunc_param[dt])
            psi.swap_sites(L-4-2*i, swap_op=None, trunc_par=trunc_param[dt])


        psi.apply_local_op(0, U_ev, unitary=True)



        for i in range(int(L/2)-1): # Second Odd sweep
            
            psi.swap_sites(2*i, swap_op=None, trunc_par=trunc_param[dt])
            psi.apply_local_op((2*i)+1 , U_odd, unitary=False, renormalize=True)
            psi.swap_sites(2*i+1, swap_op=None, trunc_par=trunc_param[dt])
        for i in range(int(L/2)-1):
            psi.swap_sites(L-3-2*i, swap_op=None, trunc_par=trunc_param[dt])
            psi.swap_sites(L-4-2*i, swap_op=None, trunc_par=trunc_param[dt])
        
        

      
        psi.compress_svd(trunc_param[dt])


      step += N_steps
      E=Energy(psi, H_bond_ev, H_bond_odd)[1]
      DeltaE=np.abs(E_old-E)
      E_old=E

      
      print("After", step, "steps, bond_E = ", E, "and DeltaE = ", DeltaE )

Nmax=4
L=18
g= 0
Omega  = 1/20
J=1/18
h=0
V=0
max_error_E=1.e-5
ID='Psi_GS_Nmax_'+str(Nmax)+'L_'+str(L)+'Omega_'+str(Omega)+'J_'+str(J)+'h_'+str(h)+'V_'+str(V)
N_steps=8
sites = sites(L,Nmax)
ps= product_state(L)
psi=MPS.from_product_state(sites, ps)
H_bond_tebd_ev=H_Peier(g, J, Omega, h, V)[0]
H_bond_tebd_odd=H_Peier(g, J, Omega, h, V)[1]
H_bond_ev=H_Peier(g, J, Omega, h, V)[2]
H_bond_odd=H_Peier(g, J, Omega, h, V)[3]

delta_t_im=[0.1, 0.01, 0.001, 1.e-4, 1.e-5]
chis=[50, 60, 60, 80, 100]
verbose=False
trunc_param=[]
for i in range(len(chis)):
    trunc_param.append({'chi_max':chis[i],'svd_min': 0.00000000000001, 'verbose': verbose})
    
N_b=[]
g_s=[]
for g in list(np.arange(0,0.05, 0.005)):
    ID='Psi_GS_Nmax_'+str(Nmax)+'L_'+str(L)+'Omega_'+str(Omega)+'J_'+str(J)+'h_'+str(h)+'V_'+str(V)
    g_s.append(g)
    psi=MPS.from_product_state(sites, ps)
    H_bond_tebd_ev=H_Peier(g, J, Omega, h, V)[0]
    H_bond_tebd_odd=H_Peier(g, J, Omega, h, V)[1]
    H_bond_ev=H_Peier(g, J, Omega, h, V)[2]
    H_bond_odd=H_Peier(g, J, Omega, h, V)[3]
    Suz_trot_im(psi, delta_t_im, max_error_E, N_steps, H_bond_tebd_ev, H_bond_tebd_odd, H_bond_ev, H_bond_odd)
    with open(ID+'.pkl', 'wb') as f:
       pickle.dump(psi, f)
    N_b.append(psi.expectation_value('N', [0]))
    print('For g=',g, 'N_b=', N_b)


np.save('N_bos'+ID, N_b)

