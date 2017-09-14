#!/usr/bin/env python
# This is a collection of functions that computes the absorption
# coefficient for direct transitions
# 
#
# Written by Wennie Wang (wwwennie@gmail.com)
# Created: 1 June 2016
# Last modified:
#
# Absorption coefficient is calculated by:
# [see Kioupakis, et al. PRB 81, 241201 (2010)]
# [see Eq. 5-13 in Electronic States and Optical Transitions, Bassani & Parravicini]
#

# THINGS TO TEST/DO:
#    - MP smearing
#    - rescaling of p_mn terms
#    - vectorization of the code

import sys
# specific to where special packages like f90nml are
sys.path.append("/u/wwwennie/.local/lib/python2.6/site-packages/")
import numpy as np
import f90nml
import smear as sm
import absorb as a
import vectorabs as vec
import time
from multiprocessing import Pool


#*************************************************************
#****************        Read Input          *****************
#*************************************************************

## Assumes VASP format files for input 

def read_input():
  """ Read input from Transmatrix, EIGENVAL, IBZKPT
      Assumes VASP files  
        Transmatrix: momentum matrix elements, in units of hbar/a_bohr
        EIGENVAL: eigenvalues at KPOINTS, unused currently
        IBZKPT: list of k-points and respective weights 

      Other inputs
        nbnd   = number bands
        n_r    = refractive index (assumed constant)
        efermi = Fermi energy (eV)
        temp   = temperature (K)
        smear  = constant smearing amount (eV)
        aa     = adaptive smearing factor (eV)
        kgrid  = k-point grid size 
         
  """

  # Read input card
  nml  = f90nml.read('input')

  fil_trans = nml['input']['fil_trans']
  fil_eig = nml['input']['fil_eig']
  fil_ibzkpt = nml['input']['fil_ibzkpt']
  
  # Assign values 
  nbnd = nml['input']['nbnd']
  n_r = nml['input']['n_r']
  efermi = nml['input']['efermi']
  temp = nml['input']['T']
  smear = nml['input']['smear']
  ismear = nml['input']['ismear']
  aa =  nml['input']['aa']
  volume = nml['input']['volume']
  kgrid = nml['input']['kgrid'] 
  rfreq = nml['input']['rfreq']
  stepfreq = nml['input']['stepfreq']

  # read in files
  f_trans = np.loadtxt(fil_trans)   # Transmatrix
  f_ibzkpt = np.loadtxt(fil_ibzkpt,skiprows=3) # ibzkpt

  # Make things into floats as appropriate
  aa   = float(aa)  
  temp = float(temp)
  kgrid  = [int(x) for x in kgrid]
  minfreq = float(rfreq[0])
  maxfreq = float(rfreq[1])
  stepfreq = float(stepfreq) 

  # Assign values from IBZKPT 
  nkpt = f_ibzkpt.shape[0]
  kpts = f_ibzkpt[:,0:3]
  kptwt = f_ibzkpt[:,3]/sum(f_ibzkpt[:,3]) 
  
  # Assign values of Transmatrix* file
  #kptwt = f_trans[:,0]	#kpt weight
  #kptind = f_trans[:,1]	#kpt index num
  ibnd = f_trans[:,1]	#empty band index, BEWARE of new Transmatrix format
  jbnd = f_trans[:,2]   #filled band index

  nfilled = int(max(jbnd))        # num filled bands
  nempty = int(max(ibnd)-nfilled) # num empty bands
  
  pmn_k = np.zeros((nkpt,nempty,nfilled)) 
  eigs = np.zeros((nkpt,nbnd))	# eigenvals (in eV)
  diffeigs = np.zeros((nkpt,nempty,nfilled))
  counter = 0
  ikpt = -1  

  for line in f_trans:
     #ikpt = int(line[1])-1   # new Transmatrix format omits
     ibnd = int(line[1])-1 - nfilled
     jbnd = int(line[2])-1
     
     # tracking index of kpt, new Transmatrix format
     if np.mod(counter,nfilled*nempty) == 0:
        counter += 1
        ikpt += 1
     else:
        counter += 1
     
     # eigenvalues- lots of overwriting values here; more efficient way?
     eigs[ikpt,ibnd+nfilled] = float(line[3])
     eigs[ikpt,jbnd] = float(line[4])     

     diffeigs[ikpt,ibnd,jbnd] = float(line[3])-float(line[4])

     # modulus over coordinates 
     pmntmp = line[5:]
     pmn_k[ikpt][ibnd][jbnd] = np.linalg.norm(pmntmp) 
  

  ##Debug
  ##print pmn[10][1][1], pmn[0][0][0]  
  ##print eigs[10,30], eigs[10,3]
  
  return nkpt, nbnd, nfilled, kgrid, n_r, pmn_k, diffeigs, eigs, kpts, kptwt, efermi, temp, smear, ismear, aa, volume, minfreq, maxfreq, stepfreq


#*************************************************************
#********************   Main program   ***********************
#*************************************************************

def main():
  """ Usage: Place relevant input parameters in file 'input'
      In conjunction with: smear.py, abs.py
      Execute: python fc_direct.py  
     
      Computed in atomic units """  
  # Track program run time
  start_time = time.time()
   
  # Get pmn, eigenvalues, kpts, kpt weights
  nkpt, nbnd, nfilled, kgrid, n_r, pmn_unit, diffeigs_unit, eigs_unit, kpts, kptwt, efermi_unit, temp, smear,ismear, aa, volume, minfreq, maxfreq, stepfreq  = read_input()
 
  # Frequencies to calculate over, eV
  freqs = np.arange(minfreq,maxfreq,stepfreq)
 
  # Constants for unit conversion
  RytoeV = 13.60569253   # Ryd to eV conversion
  RytoHar =  0.5         # Ryd to Hartree conversion        
  Hartocm1 = 219474.63      # Hartree to cm^-1 conversion
  tpi = 2.0*np.pi           # 2pi
  au = 0.52917720859        # Bohr radius 
  me = 9.10938215e-31       # electron mass
  eV = 1.602176487e-19      # electron charge and eV to J
  hbar = 1.054571628e-34    # J
  KtoJ = 1.38064852e-23     # Boltzmann constant 
  KtoeV = 8.6173324e-5      # Boltzmann constant
  KtoHar = KtoeV / RytoeV * RytoHar
  eps0 = 8.854187817e-12    # Vac permittivity
  c_light = 137.035999070   # Hartree
  
  # Convert input to atomic units
  pmn    = pmn_unit #* 2*au*RytoeV  # VASP p_mn in units hbar/a_bohr
                                    # looks like it is converted to non-a.u.
                                    # in Transmatrix file
  T      = temp*KtoHar             
  eigs   = eigs_unit / RytoeV * RytoHar    
  efermi = efermi_unit/ RytoeV * RytoHar
  vol    = volume / (au**3)
  smear  = smear / RytoeV * RytoHar
  harfreqs  = freqs / RytoeV * RytoHar

  fdeigs = np.zeros((nkpt,nbnd))
  fdeigs = (eigs-efermi)/T
  diffeigs   = diffeigs_unit / RytoeV * RytoHar    
  

  # # DEBUG: checking Fermi-Dirac distribtuion
  # eigtest = np.arange(-0.5,0.5,0.05)
  # fd = [sm.fermidirac(energy,0,T) for energy in eigtest]
  # np.savetxt("fd-check",np.c_[eigtest,fd]) 
    
  ##### Calculate absorption coefficient  #####
  nfreqs = len(freqs)
  alpha = np.zeros(nfreqs)
  runtime = np.zeros(nfreqs)
  
  prog_freq = np.ceil(nfreqs/10)
  
  #### Vectorization #####
  #  relevant smearing functions
  Ffd = np.vectorize(sm.fd)
  if (ismear == 1):
     Fdelta = np.vectorize(sm.w0gauss)
  elif (ismear == 2):
     Fdelta = np.vectorize(sm.mpdelta)

  # array of FD-terms [nkpt][nbnd]
  fd_vec = Ffd(fdeigs)
  ########################

  fileout = open('progress','w')
  fileout.write("================ Progress ============\n")
  ##### Calculate absorption coefficient ####
  for f in range(nfreqs):
     delta_vec = vec.vecdirac(Fdelta,diffeigs,harfreqs[f],smear,ismear)
     alpha[f] = a.absvec(pmn,delta_vec,fd_vec,kptwt,nkpt,nbnd,nfilled)
     #alpha[f] = a.absorb(pmn,eigs,efermi,T,harfreqs[f],kptwt,nkpt,nbnd,aa,smear,nfilled)
     # progress bar
     if (np.mod(int(f),int(prog_freq)) == 0): 
        runtime[f] = time.time() - start_time
        fileout.write("frequency {0} of {1}: {2} s\n".format(int(f),int(nfreqs),runtime[f]))
        fileout.flush() 
  
  # pre-factor to absorption coefficient
  pre = 2 * 4 * np.pi**2 / (n_r * c_light)
  pre = pre / vol
  invfreq = [1.0/freq for freq in harfreqs]
  pre = np.multiply(pre,invfreq) 

  alpha = np.multiply(pre,alpha)
 
  # imag dielectric for double-checking
  # from numbers in a.u.
  imeps = (n_r * c_light) * np.multiply(invfreq,alpha) 
  
  # Convert to conventional units
  alpha = alpha * Hartocm1 # au to cm^-1
  
  # Output to file
  omalpha = np.c_[freqs,alpha]
  np.savetxt("runtime", runtime)
  np.savetxt("alpha",omalpha)
  np.savetxt("imepsilon",np.c_[freqs,imeps])

if __name__ == "__main__":
  main()
