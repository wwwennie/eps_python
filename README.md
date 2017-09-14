# eps_python

(Naive) Implementation of imaginary part of dielectric function.
For calculations of absorption properties in semiconductors.

Requires generating file containing matrix elements in units hbar/a_Bohr containing the following information:

| kpt wt | ibnd  | jbnd  | eig_i | eig_j  | Re_i   | Im_i |

where i = x,y,z 

Also install f90nml to handle input files structured in namelists as found in Fortran.
