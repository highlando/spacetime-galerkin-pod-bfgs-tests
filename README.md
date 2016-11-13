space-time Galerkin POD 
---

Code and hardware documentation of the numerical tests reported in our preprint *Space-time Galerkin POD with application in optimal control of semi-linear parabolic partial differential equations*.

[![DOI](https://zenodo.org/badge/73575460.svg)](https://zenodo.org/badge/latestdoi/73575460)

## Space-time Galerkin POD (Python)

Implementation of space-time generalized POD and space-time Galerkin for solving Burgers' equation. This is a snapshot of the repo [spacetime-genpod-burgers](https://gitlab.mpi-magdeburg.mpg.de/heiland/spacetime-genpod-burgers) that is under continuous development.

### Setup
Change directory to `spacetime-pod-python/`.
To rerun the numerical examples run `python run_numtests.py`. Set `timingsonly = True` there, to only get the timings.

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

## Classical POD and BFGS (Matlab)
Implementation of classical POD for the space reduction and BFGS for the optimization.

### Setup 
Change directory to `pod-bfgs-matlab/`. To rerun the numerical tests, run `driver_tab9.m`, `driver_tab10.m`, or `driver_tab11.m`.

### Dependencies
 * Matlab (v 8.0.0.783 (R2012b); 9.1.0.441655 (R2016b))

## Hardware documentation
See `logs_adelheid` for the *logfiles* of the numerical tests and the hardware specifications.
