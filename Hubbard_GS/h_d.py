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
from scipy.linalg import expm
import pickle

import time


Nmax=20
L=20
g= float(sys.argv[1])
Omega  = 10
J=float(sys.argv[2])
h=0
V=float(sys.argv[3])
max_error_E=[1.e-6, 1.e-6, 1.e-6, 1.e-6, 1.e-6]
ID='Psi_GS_Nmax_'+str(Nmax)+'L_'+str(L)+'g_'+str(g)+'Omega_'+str(Omega)+'J_'+str(J)+'h_'+str(h)+'V_'+str(V)

with open(ID+'.pkl', 'rb') as f:
    psi = pickle.load(f)

n_i=psi.expectation_value('N')
np.save('Occupancy'+ID+'.npy', n_i)
