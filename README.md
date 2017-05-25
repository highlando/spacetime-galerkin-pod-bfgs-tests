space-time Galerkin POD 
---

Code and hardware documentation of the numerical tests reported in our first submission of *Space-time Galerkin POD with application in optimal control of semi-linear parabolic partial differential equations*.

[![DOI](https://zenodo.org/badge/73575460.svg)](https://zenodo.org/badge/latestdoi/73575460)

## Space-time Galerkin POD (Python)

Implementation of space-time generalized POD and space-time Galerkin for solving Burgers' equation. This is a snapshot of the repo [spacetime-genpod-burgers](https://gitlab.mpi-magdeburg.mpg.de/heiland/spacetime-genpod-burgers) that is under continuous development.

### Setup
Change directory to `spacetime-pod-python/`.
To rerun the numerical examples run 
```
python2.7 run_numtests_genpod.py
python2.7 run_numtests_bfgspod_step.py
python2.7 run_numtests_bfgspod_heartshape.py
```
The script `run_numtests_genpod.py` has a switch for the two testcases `inival` and `heartshape`.

To get the timings, set `timingsonly = True` in the scripts.

### Dependencies:
 * dolfin (FEniCS) (v 1.3.0; 1.5.0; 2016.2.0)
 * scipy (v 0.13.3; 0.17.0)
 * numpy (v 1.8.2; 1.11.0)
 * scikit-sparse (v 0.3)
    * used for factorization of the mass matrices
	* critical for critical parameters (like small `nu`)
    * scipy's, numpy's built-in Cholesky routines can be used instead but the algorithm may fail at critical parameters
 * and my home-brew python modules
   * [dolfin_navier_scipy](https://github.com/highlando/dolfin_navier_scipy) -- interface between `scipy` and `fenics`, a timer function, and caching of intermediate results
   * [mat-lib-plots](https://github.com/highlando/mat-lib-plots) -- routines for plotting
   * [sadptprj_riclyap_adi](https://github.com/highlando/sadptprj_riclyap_adi) -- linear algebra helper routines
 * this repo already contains the home-brew modules and -- for the default parameters -- the factorized mass matrices

## Hardware documentation
See `logs_pc764` for the *logfiles* of the numerical tests and the hardware specification of the PC used for the tests reported in the manuscript.
