# Smearing schemes for delta function
# Intended for use with fc_direct.py for calculation of 
# free-carrier direct absorption transitions
#
# Methods:
# w0gauss : standard Gaussian smearing
# w1gauss : FD smearing 
# sig_nk# : variable smearing for Dirac-delta involvinv e_{nk} - e_{m,k+q}
#    TODO: implement band velocity for variable smearing....
#            with/without inclusion of symmetry?

# mpdelta : Methfessel-Paxton; order n = 0 is equivalent to Gaussian smearing
#            supposedly can extract more physical meaning from MP smearing
#            is what is used by default in VASP
#            in practice, n = 1, 2 is sufficient;

import numpy as np 

# standard gaussian
def w0gauss(x):
   sqrtpm1 = 1.0/1.7724538509055160
   arg = np.min([500,x**2.0])
   w0gauss = np.float64(0.0)
   w0gauss = sqrtpm1 * np.exp( -arg )
  
   return w0gauss

def mpdelta(x,n):
   """ x : argument 
       n : order of MP; n = 0 equivalent to Gaussian smearing 
       adapted from DELSTP subroutine in dos.F of VASP 5.4.1"""

   A = 1.0/np.sqrt(np.pi)
   K = 0
   H1 = 1.0
   H2 = 2.0*x
   D = A
   
   # coeffecient of Hermite polynomials
   for i in range(n):
      A = A/(-4.0*(i+1))
      K = K + 1
      H3 = H1
      H1 = H2
      H2 = 2.0*x*H2 - 2*K*H3
    
      K = K + 1
      H3 = H1
      H1 = H2
      H2 = 2.0*x*H2 - 2*K*H3
      D = D + A*H1
   
   D = D*np.exp(-(x*x))

   return D
  
   

# FD smearing
def fd(x):                    
  """ Fermi-Dirac occupation occupation factor:
       x : (ebnd - efermi)/T
  """ 
  fd = np.float64(0.0)
  fd = 1/(np.exp(x)+1)
  return fd

def fermidirac(ebnd,efermi,T):                    
  """ Fermi-Dirac occupation occupation factor:
       ebnd: energy (in eV) of band of interest
       efermi: Fermi energy (in eV)
       T: temperature (in eV) """
 
  fd = np.float64(0.0)
  energy = ebnd-efermi
  fd = 1/(np.exp(energy/T)+1)
  return fd

# adaptive smearing
def sig_nk(nk1,nk2,nk3,vk,aa):
  
   dF1 = np.abs(vk[0] * 1.0/nk1) 
   dF2 = np.abs(vk[1] * 1.0/nk2) 
   dF3 = np.abs(vk[2] * 1.0/nk3) 
   
   sig_nk = aa * np.sqrt(np.abs(dF1**2 + dF2**2 + dF3**2)) 
   return sig_nk

##
def sig_nk1(nk1,nk2,nk3,vk,aa):
  
   dF1 = np.abs(vk[0] * 1.0) 
   dF2 = np.abs(vk[1] * 1.0) 
   dF3 = np.abs(vk[2] * 1.0) 
   
   sig_nk1 = aa * np.sqrt(np.abs(dF1**2 + dF2**2 + dF3**2)) 
   return sig_nk1

##
def sig_nk2(nk1,nk2,nk3,vk,aa):
  
   dF1 = np.abs(vk[0] * 1.0/nk1) 
   dF2 = np.abs(vk[1] * 1.0/nk2) 
   dF3 = np.abs(vk[2] * 1.0/nk3) 
   
   sig_nk2 = aa * np.max([dF1,dF2,dF3]) 
   return sig_nk2

##
def sig_nk3(nk1,nk2,nk3,vk,aa):
  
   dF1 = np.abs(vk[0] * 1.0/nk1) 
   dF2 = np.abs(vk[1] * 1.0/nk2) 
   dF3 = np.abs(vk[2] * 1.0/nk3) 
   
   sig_nk3 = aa * np.min([dF1,dF2,dF3]) 
   return sig_nk3
