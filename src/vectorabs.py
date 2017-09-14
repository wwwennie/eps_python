# vectorizing fc-direct code by minimizing indexing 
# Methods for calculating dielectric function and absorption
# based on RPA
# based on Eq. 5-13 in Electronic States and Optical Transitions by 
#   Bassan and Parravicini 

import smear as sm
import numpy as np 

def vecdirac(Fdelta,diffeigs,freq,smear,ismear):
  """ Procedure to vectorize absorption code

      Inputs:
         Fdelta: vectorized Dirac delta smearing function 
         diffeigs: [nkpt][nempty][nfilled], (e_jk - e_ik)
         freq: frequency
         smear: constant smearing factor

      Outputs:
         fdeig_vec : contained (f_ik - f_jk) 
               [nkpt][nempty][nfilled]
  """
  diff_vec = np.subtract(diffeigs,freq) # e_jk - e_ik - hbar*omega
  if (ismear == 1):
    delta_vec = Fdelta((diff_vec/smear))/smear
  elif (ismear == 2):
    delta_vec = Fdelta(diff_vec/smear,2)

  return delta_vec 



