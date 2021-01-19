import tenpy 
import pickle
import sys
import numpy as np

a=np.array([1,2,3,4,float(sys.argv[1])])
np.save(str(sys.argv[1])+str(sys.argv[2])+'aaa', a)
