import tenpy
import copy
import sys
import numpy as np
import numpy.linalg as alg

from tenpy import models
from tenpy.networks.site import SpinSite
from tenpy.networks.site import FermionSite
from tenpy.networks.site import BosonSite
from tenpy.models.model import CouplingModel
from tenpy.models.model import CouplingMPOModel
from tenpy.models.spins import SpinModel
from tenpy.algorithms import dmrg
from tenpy.networks.mps import MPS
from tenpy.models.lattice import Lattice
from tenpy.tools.params import get_parameter
import tenpy.linalg.np_conserved as npc
from tenpy.networks.mpo import MPO, MPOEnvironment
import tenpy.linalg.charges as charges
from tenpy.models.lattice import Chain
from scipy.linalg import expm
from tenpy.models.fermions_spinless import FermionModel
from tenpy.algorithms.tebd import Engine
import pickle
import copy



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



def single_el(L):
    ps=['vac']
    for i in range(int(L/2)):
        ps.append('empty')
        ps.append('empty')
    ps[int(L/2)]='full'
    return ps

def psi(sites,ps):
    psi=MPS.from_product_state(sites, ps)
    return psi

def H_Peier(g, J, Omega, pert):
    Peier=npc.outer(npc.expm(1j*g*(psi.sites[0].B+psi.sites[0].Bd)).replace_labels(['p', 'p*'], ['p0', 'p0*']),npc.outer(-J*psi.sites[1].Cd.replace_labels(['p', 'p*'], ['p1', 'p1*']),psi.sites[1].C.replace_labels(['p', 'p*'], ['p2', 'p2*']))).itranspose([0,2,4,1,3,5])
    Peier_hc=npc.outer(npc.expm(-1j*g*(psi.sites[0].B+psi.sites[0].Bd)).replace_labels(['p', 'p*'], ['p0', 'p0*']), npc.outer(-J*psi.sites[1].C.replace_labels(['p', 'p*'], ['p1', 'p1*']),psi.sites[1].Cd.replace_labels(['p', 'p*'], ['p2', 'p2*']))).itranspose([0,2,4,1,3,5])
    cav=npc.outer((Omega/(2*(L-1)))*psi.sites[0].N.replace_labels(['p','p*'],['p0', 'p0*']),npc.outer(psi.sites[1].Id.replace_labels(['p', 'p*'], ['p1', 'p1*']),psi.sites[1].Id.replace_labels(['p', 'p*'], ['p2', 'p2*'])) ).itranspose([0,2,4,1,3,5])
    perturbation_ev=npc.outer(psi.sites[0].Id.replace_labels(['p', 'p*'], ['p0', 'p0*']),npc.outer(psi.sites[1].Id.replace_labels(['p', 'p*'], ['p1', 'p1*']),pert*psi.sites[1].N.replace_labels(['p', 'p*'], ['p2', 'p2*']))).itranspose([0,2,4,1,3,5])
    perturbation_odd=npc.outer(psi.sites[0].Id.replace_labels(['p', 'p*'], ['p0', 'p0*']),npc.outer(pert*psi.sites[1].N.replace_labels(['p', 'p*'], ['p1', 'p1*']),psi.sites[1].Id.replace_labels(['p', 'p*'], ['p2', 'p2*']))).itranspose([0,2,4,1,3,5])
    H_bond_tebd=Peier+Peier_hc+cav  #This is the energetic term that will be used in the TEBD algorithm
    H_bond=Peier+Peier_hc  
    H_perturbed_ev=H_bond_tebd+perturbation_ev
    H_perturbed_odd=H_bond_tebd+perturbation_odd#This takes into account only the Peier term and is useful to evaluate the energy of the bonds
    return H_bond_tebd, H_bond, H_perturbed_ev, H_perturbed_odd
    




def U_bond(dt, H_bond):
    
    H2 = H_bond.combine_legs([('p0', 'p1', 'p2'), ('p0*', 'p1*', 'p2*')], qconj=[+1, -1])
    H2= (-dt)*H2
    U=npc.expm(H2).split_legs()
    return U  



        
def Suz_trot_real_pert(psi, dt, N_steps, H_bond_tebd, H_perturbed_ev, H_perturbed_odd):

    U_ev=[U_bond(1j*dt, H_bond_tebd)]*int(L/2)
    U_odd= [U_bond((1j*dt)/2, H_bond)]*(int(L/2)-1)
    U_ev[int(L/4)]=U_bond(1j*dt, H_perturbed_ev)
    U_odd[int(L/4)]=U_bond((1j*dt)/2, H_perturbed_odd)
    
    for T in range(N_steps):
        print("Step number: ", T)

        for i in range(int(L/2)-1): # First Odd sweep
            
            psi.swap_sites(2*i, swap_op=None, trunc_par=trunc_param)
            psi.apply_local_op((2*i)+1 , U_odd[i], unitary=True)
            psi.swap_sites(2*i+1, swap_op=None, trunc_par=trunc_param)

        for i in range(int(L/2)-1):
           

            psi.apply_local_op(L-2-2*i, U_ev[i], unitary=True)
            psi.swap_sites(L-3-2*i, swap_op=None, trunc_par=trunc_param)
            psi.swap_sites(L-4-2*i, swap_op=None, trunc_par=trunc_param)

        psi.apply_local_op(0, U_ev[0], unitary=True)

        for i in range(int(L/2)-1): # First Odd sweep
            
            psi.swap_sites(2*i, swap_op=None, trunc_par=trunc_param)
            psi.apply_local_op((2*i)+1 , U_odd[i], unitary=True)
            psi.swap_sites(2*i+1, swap_op=None, trunc_par=trunc_param)
        for i in range(int(L/2)-1):
            psi.swap_sites(L-3-2*i, swap_op=None, trunc_par=trunc_param)
            psi.swap_sites(L-4-2*i, swap_op=None, trunc_par=trunc_param)
         
        psi.compress_svd(trunc_param)
        print(psi)
        


Nmax=8
L=16
g= float(sys.argv[1])
Omega  = float(sys.argv[2])
J=1   
pert=0
dt=0.01
tmax=1
NB=3
sites = sites(L,Nmax)
ps= mixed_state(L)
psi=MPS.from_product_state(sites, ps)
H_perturbed_ev=H_Peier(g, J, Omega, pert)[2]
H_perturbed_odd=H_Peier(g, J, Omega, pert)[3]


H_bond_tebd=H_Peier(g, J, Omega, pert)[0]
H_bond=H_Peier(g, J, Omega, pert)[1]

verbose=False
trunc_param={'chi_max':80,'svd_min': 0.00000000000001, 'verbose': verbose}

for i in range(NB):
  psi.apply_local_op(0, 'Bd')


psi.apply_local_op(int(L/2),sites[1].Cd-0.9*sites[1].C, unitary=True)



ID='g_'+str(g)+'Nb_'+str(NB)+'Omega_'+str(Omega)



n_i_t=[]
n_i=[]
NBt=[]
for i in range(L):
    n_i.append(psi.expectation_value('N', i+1))
n_i_t.append(n_i)
for i in range(int(tmax/dt)):
   print("time_step:", i)
   Suz_trot_real_pert(psi, dt, 1, H_bond_tebd, H_perturbed_ev, H_perturbed_odd)
   n_i=[]
   for j in range(L):
       n_i.append(psi.expectation_value('N', j+1))
   n_i_t.append(n_i)
   NBt.append(psi.expectation_value('N', 0))

np.save('LC_NB'+ID, NBt)
np.save('LC_nit'+ID, n_i_t)
