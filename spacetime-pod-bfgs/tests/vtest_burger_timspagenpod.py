import numpy as np

import dolfin_navier_scipy.data_output_utils as dou
import sadptprj_riclyap_adi.lin_alg_utils as lau

import dolfin_burgers_scipy as dbs
import burgers_genpod_utils as bgu
import gen_pod_utils as gpu
from plot_utils import plotmat

from spacetime_pod_utils import get_spacetimepodres


Nq = 60  # dimension of the spatial discretization
Ns = 33  # dimension of the temporal discretization
hs = 8  # reduced S dim
hq = 16  # reduced Y dim
Nts = 200  # number of time sampling points
nu = 1e-2
t0, tE = 0., 1.
noconv = False
plotsvs = False
debug = True
mockUky = np.eye(Nq)
mockUks = np.eye(Ns)

(M, A, rhs, nfunc, femp) = dbs.\
    burgers_spacedisc(N=Nq, nu=nu, retfemdict=True)
# +1 bc of boundary conditions that are eliminated
femp, ay, my = femp, A, M
tmesh = np.linspace(t0, tE, Nts)
iniv = np.r_[np.zeros((np.floor(Nq*.5), 1)), np.ones((np.ceil(Nq*.5)-1, 1))]
# iniv = np.ones((Nq-1, 1))
datastr = 'burgv_vtest_nu{1}_N{0}_Nts{2}_noconv{3}'.format(Nq, nu, Nts, noconv)

if noconv:
    nfunc = None
simudict = dict(iniv=iniv, A=A, M=M, nfunc=nfunc, rhs=rhs, tmesh=tmesh)
vv = dou.load_or_comp(filestr='data/'+datastr, comprtn=gpu.time_int_semil,
                      comprtnargs=simudict, arraytype='dense', debug=True)
plotmat(vv)
raise Warning('TODO: debug')
ared, mred, nonlred, rhsred, inired, Uky = gpu.\
    get_podred_model(M=M, A=A, nonl=nfunc, rhs=rhs,
                     sol=vv.T, tmesh=tmesh, verbose=True,
                     poddim=hq, sdim=Ns, plotsvs=plotsvs,
                     genpod=True, basfuntype='pl')

(msx, ms) = gpu.get_genmeasuremat(sol=vv.T, tmesh=tmesh, sdim=Ns)
import scipy.sparse as sps
x = lau.apply_massinv(sps.csc_matrix(ms), msx.T).T
Uks = gpu.get_podmats(sol=x.T, poddim=hs, plotsvs=plotsvs, M=M)
uvvdxl = dbs.get_burgertensor_spacecomp(podmat=Uky, **femp)
htittl = bgu.get_burgertensor_timecomp(podmat=Uks, sdim=Ns,
                                       tmesh=tmesh, basfuntype='pl')
hshysol = np.dot(Uks.T, np.dot(x.T, Uky)).\
    reshape((hs*hq, 1), order='C')
dms = gpu.get_dms(sdim=Ns, tmesh=tmesh, basfuntype='pl')
hdms = np.dot(Uks.T, np.dot(dms, Uks))
hms = np.dot(Uks.T, np.dot(ms, Uks))


def evabrgquadterm(tvvec):
    return bgu.\
        eva_burger_quadratic(tvvec=tvvec, htittl=htittl, uvvdxl=uvvdxl)
if noconv:
    eva_burger_quadratic = None
nres, timespaceres = get_spacetimepodres(tvvec=hshysol, dms=hdms, ms=hms,
                                         my=mred, ared=ared,
                                         nfunc=evabrgquadterm, rhs=None)
trsrs = timespaceres.reshape((hq, hs))
plotmat((np.dot(Uky, np.dot(trsrs, Uks.T))).T, fignum=121)


def optifun(tvvec):
    allo = get_spacetimepodres(tvvec=tvvec.reshape((tvvec.size, 1)),
                               dms=hdms, ms=hms, my=mred, ared=ared,
                               nfunc=evabrgquadterm, rhs=None, retnorm=False)
    return allo.flatten()
from scipy.optimize import fsolve
optix = fsolve(optifun, x0=np.zeros((hs*hq, )))
optix = optix.reshape((hs, hq))
plotmat((np.dot(Uky, np.dot(optix.T, Uks.T))).T, fignum=123)
