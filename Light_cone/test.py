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


a, b = float(sys.argv[1]), float(sys.argv[2])
array=np.array([a,b])
ID=str(a)+'_'+str(b)
np.save(ID, array)

