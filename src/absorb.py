# Methods for calculating dielectric function and absorption
# based on RPA
# based on Eq. 5-13 in Electronic States and Optical Transitions by 
#   Bassan and Parravicini 

import smear as sm

# dirac-delta & FD factor vectorized...
def absvec(pmn,delta_vec,fd_vec,kptwt,nkpt,nbnd,nfilled):
  """ Procedure to calculate the absorption coefficient
      at particular frequency, embedded in vecdirac (vectorization
      of dirac delta smearing)
      Computed in atomic units      

      Inputs:
         pmn: complex momentum matrix elements; 
              [ikpt][ibnd][jbnd][coord]
         delta_vec: smearing dirac delta
              [nkpt][nempty][nfilled]
         fd_vec: f_ik; [nkpt][nbnd]
         kptwt: k-point weights
         nkpt: number of kpts
         nbnd: number electronic bands
         nfilled: num filled bands; for index tracking

      Outputs:
         alpha: absorption coefficient at freq
  """
  alpha = float(0)
  for ikpt in range(nkpt):  
    for ibnd in range(nbnd-nfilled): # empty bands
       fd_ik = fd_vec[ikpt][ibnd+nfilled] 
       pmntmp = pmn[ikpt][ibnd]
       deltatmp = delta_vec[ikpt][ibnd]
       for jbnd in range(nfilled): #filled bands
          # Fermi-dirac factor
          fd_jk = fd_vec[ikpt][jbnd]
          fd_fac = fd_jk-fd_ik
 
          alpha += kptwt[ikpt] * fd_fac * (pmntmp[jbnd]**2) * deltatmp[jbnd]
  return alpha

# non-vectorized version
def absorb(pmn,eigs,efermi,T,freq,kptwt,nkpt,nbnd,aa,smear,nfilled):
  # need choose which smearing method....
  """ Procedure to calculate the absorption coefficient
      at particular frequency 
      Computed in atomic units      

      Inputs:
         pmn: complex momentum matrix elements; 
              [ikpt][ibnd][jbnd][coord]
         eigs: eigenenergies [ikpt, bnd]
         T: temperature
         freq: frequency to calculate over
         kptwt: k-point weights
         nkpt: number of kpts
         nbnd: number electronic bands
         aa: adaptive smearing parameter
         smear: regular constant smearing
         nfilled: num filled bands; for index tracking

      Outputs:
         alpha: absorption coefficient at freq
  """
  alpha = float(0)
  for ikpt in range(nkpt):  
     for ibnd in range(nbnd-nfilled): # empty bands
        eig_ik = eigs[ikpt,ibnd+nfilled]
        fd_ik = sm.fermidirac(eig_ik,efermi,T)
        
        for jbnd in range(nfilled): #filled bands
           eig_jk = eigs[ikpt,jbnd]

           # Fermi-dirac factor
           fd_jk = sm.fermidirac(eig_jk,efermi,T)
           fd_fac = fd_jk-fd_ik
           
           # smearing
           arg = eig_ik - eig_jk - freq
           
           # Gaussian smearing
           argsm = arg/(smear**2)
           dirac = sm.w0gauss(argsm)
           
           # MP smearing
           #argsm = arg/smear
           #dirac = sm.mpdelta(argsm,2)
  
           #momentum matrix element
           ##pmn_tmp = pmn[ikpt][ibnd][jbnd]
           ##pmn_k = float(0)
           ##for coord in range(3):
           ##  pmn_k += pmn_tmp[coord]**2

           alpha += kptwt[ikpt] * fd_fac * (pmn[ikpt][ibnd][jbnd]**2) * dirac

#       # Debugging 
#       print "ik", eig_ik
#       print "jk", eig_jk
#       print "arg", arg
#       print "pmn", pmn_tmp
#       print pmn_k       
#       print "dirac", dirac
#       print "fd", fd_fac
#       print alpha

  # fermidirac(ebnd,efermi,T)
  # w0gauss(x)
  # sig_nk(nk1,nk2,nk3,vk,aa) 
  
  return alpha

