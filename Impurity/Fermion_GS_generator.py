# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 09:53:15 2021

@author: giaco
"""

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




def sites(L):
 FSite=FermionSite(None, filling=0.5)

 sites=[]
 
 for i in range(L):
     sites.append(FSite)
 return sites

def product_state(L):
    ps=[]
    for i in range(int(L/2)):
        ps.append('empty')
        ps.append('full')
    return ps
J=1
dt=0.05
L=40
V=0
mu=0
steps=40
sites=sites(L)
ps=product_state(L)
psi=MPS.from_product_state(sites, ps)
psi2=MPS.from_product_state(sites, ps)
model_params={'bc_MPS':'finite', 'bc_x':'open', 'explicit_plus_hc':True, 'lattice':'Chain', 'J':J, 'conserve':None, 'V':V, 'mu':mu, 'L':L}
FC=tenpy.models.fermions_spinless.FermionChain(model_params)
print(FC.calc_H_bond()[0])
verbose=True
trunc_param={'svd_min': 0.000000000000000001, 'verbose': verbose, 'keys':'sorted'}
options={
            'compression_method': 'SVD',
            'trunc_param': trunc_param,
            'keys':'sorted',
            'verbose': verbose 
            }
tebd_params = {
        'order': 2,
        'delta_tau_list': [0.1, 0.01, 0.001, 1.e-4, 1.e-5],
        'N_steps': 20,
        'max_error_E': 1.e-8,
        'trunc_params': {
            'chi_max': 120,
            'svd_min': 1.e-10
        },
        'verbose': verbose,
    }

ID='GS_J_'+str(J)+'V_'+str(V)+'L_'+str(L)

"""#Generate with IMTE
eng = Engine(psi, FC, tebd_params)
eng.run_GS()
"""
dmrg_params = {
        'mixer': True,  # setting this to True is essential for the 1-site algorithm to work.
        'max_E_err': 1.e-18,
        'trunc_params': {
            'chi_max': 200,
            'svd_min': 1.e-12
        },
        'verbose': verbose,
        'combine': False,
         # specifies single-site
    }
info = dmrg.run(psi, FC, dmrg_params)






with open(ID+'DMRG.pkl', 'wb') as f:
       pickle.dump(psi, f)
    


    
