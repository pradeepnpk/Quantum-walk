# Quantum-walk: Open quantum walk simulations in python

Install Qutip and all other dependencies http://qutip.org/docs/3.1.0/installation.html

Packages nedded: 

Python	2.7+	

Numpy	1.7+	                  Not tested on lower versions.

Scipy	0.14+	                  Lower versions have missing features.

Matplotlib	                  1.2.0+	Some plotting does not work on lower versions.

Cython	0.15+	                Needed for compiling some time-dependent Hamiltonians.

GCC Compiler	4.2+	          Needed for compiling Cython files.

Fortran Compiler	Fortran 90	Needed for compiling the optional Fortran-based Monte Carlo solver.

BLAS library	1.2+	          Optional, Linux & Mac only. Needed for installing Fortran Monte Carlo solver.

Mayavi	4.1+	                Optional. Needed for using the Bloch3d class.

Python Headers	2.7+	        Linux only. Needed for compiling Cython files.

LaTeX	TexLive 2009+	Optional. Needed if using LaTeX in figures.

nose	1.1.2+	                Optional. For running tests.

scikits.umfpack	5.2.0+	      Optional. Faster (~2-5x) steady state calculations.

