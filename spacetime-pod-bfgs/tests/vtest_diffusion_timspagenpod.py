import numpy as np
import scipy.sparse as sps

import dolfin_navier_scipy.data_output_utils as dou
import sadptprj_riclyap_adi.lin_alg_utils as lau

import dolfin_burgers_scipy as dbs
import gen_pod_utils as gpu
from plot_utils import plotmat
from spacetime_pod_utils import get_spacetimepodres

# import burgers_genpod_utils as bgu


Nq = 100  # dimension of the spatial discretization
Ns = 65  # dimension of the temporal discretization
hs = 7  # reduced S dim
hq = 33  # reduced Y dim
Nts = 300  # number of time sampling points
nu = 1e-2
t0, tE = 0., 1.
plotsvs = True
mockUky = np.eye(Nq)
mockUks = np.eye(Ns)
(M, A, rhs, nfunc, femp) = dbs.\
    burgers_spacedisc(N=Nq, nu=nu, retfemdict=True)
# +1 bc of boundary conditions that are eliminated
femp, ay, my = femp, A, M
tmesh = np.linspace(t0, tE, Nts)
iniv = np.r_[np.zeros((np.floor(Nq*.5), 1)), np.ones((np.ceil(Nq*.5)-1, 1))]
# iniv = np.ones((Nq-1, 1))
datastr = 'burgv_contim_nu{1}_N{0}_Nts{2}'.format(Nq, nu, Nts)

simudict = dict(iniv=iniv, A=A, M=M, nfunc=None, rhs=rhs, tmesh=tmesh)
vv = dou.load_or_comp(filestr='data/'+datastr, comprtn=gpu.time_int_semil,
                      comprtnargs=simudict, arraytype='dense', debug=True)
plotmat(vv)
ared, mred, nonlred, rhsred, inired, Uky = gpu.\
    get_podred_model(M=M, A=A, nonl=nfunc, rhs=rhs,
                     sol=vv.T, tmesh=tmesh, verbose=True,
                     poddim=hq, sdim=Ns, plotsvs=plotsvs,
                     genpod=True, basfuntype='pl')

(msx, ms) = gpu.get_genmeasuremat(sol=vv.T, tmesh=tmesh, sdim=Ns)
x = lau.apply_massinv(sps.csc_matrix(ms), msx.T).T
Uks = gpu.get_podmats(sol=x.T, poddim=hs, plotsvs=plotsvs, M=M)
hshysol = np.dot(Uks.T, np.dot(x.T, Uky)).\
    reshape((hs*hq, 1), order='C')
dms = gpu.get_dms(sdim=Ns, tmesh=tmesh, basfuntype='pl')
hdms = np.dot(Uks.T, np.dot(dms, Uks))
hms = np.dot(Uks.T, np.dot(ms, Uks))
nres, timespaceres = get_spacetimepodres(tvvec=hshysol, dms=hdms, ms=hms,
                                         my=mred, ared=None,
                                         nfunc=None, rhs=None)
trsrs = timespaceres.reshape((hq, hs))
plotmat((np.dot(Uky, np.dot(trsrs, Uks.T))).T, fignum=121)
