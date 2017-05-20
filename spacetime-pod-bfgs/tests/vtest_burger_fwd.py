import dolfin_burgers_scipy as dbs
import gen_pod_utils as gpu
import numpy as np
from plot_utils import plotmat

Nq = 100  # dimension of the spatial discretization
Nts = 100  # number of time sampling points
t0, tE = 0., 1.
tmesh = np.linspace(t0, tE, Nts)
nu = 1e-3
(My, A, rhs, nfunc, femp) = dbs.\
    burgers_spacedisc(N=Nq, nu=nu, retfemdict=True)

iniv = np.r_[np.ones(((Nq-1)/2, 1)), np.zeros(((Nq)/2, 1))]
simudict = dict(iniv=iniv, A=A, M=My,
                nfunc=nfunc, rhs=rhs, tmesh=tmesh)
vv = gpu.time_int_semil(**simudict)
plotmat(vv, fignum=1234)
