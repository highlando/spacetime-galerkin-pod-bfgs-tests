import numpy as np
import scipy.sparse as sps

import dolfin_navier_scipy.data_output_utils as dou
import sadptprj_riclyap_adi.lin_alg_utils as lau

import dolfin_burgers_scipy as dbs
# import burgers_genpod_utils as bgu
import gen_pod_utils as gpu
from plot_utils import plotmat  # , get_spacetimepodres


Nq = 70  # dimension of the spatial discretization
Ns = 33  # dimension of the temporal discretization
hs = 8  # reduced S dim
hq = 16  # reduced Y dim
Nts = 100  # number of time sampling points
nu = 1e-3
t0, tE = 0., 1.
plotsvs = True
noconv = True
mockUky = np.eye(Nq)
mockUks = np.eye(Ns)
(M, A, rhs, nfunc, femp) = dbs.\
    burgers_spacedisc(N=Nq, nu=nu, retfemdict=True)
# +1 bc of boundary conditions that are eliminated
femp, ay, my = femp, A, M
tmesh = np.linspace(t0, tE, Nts)
iniv = np.r_[np.ones(((Nq-1)/2, 1)), np.zeros(((Nq)/2, 1))]
# iniv = np.ones((Nq-1, 1))
datastr = 'burgv_contim_nu{1}_N{0}_Nts{2}'.format(Nq, nu, Nts)
if noconv:
    nfunc = None

simudict = dict(iniv=iniv, A=A, M=M, nfunc=nfunc, rhs=rhs, tmesh=tmesh)
vv = dou.load_or_comp(filestr='data/'+datastr, comprtn=gpu.time_int_semil,
                      comprtnargs=simudict, arraytype='dense', debug=False)
# plotmat(vv)
ared, mred, nonlred, rhsred, inired, Uky = gpu.\
    get_podred_model(M=M, A=A, nonl=nfunc, rhs=rhs,
                     sol=vv.T, tmesh=tmesh, verbose=True,
                     poddim=hq, sdim=Ns, plotsvs=plotsvs,
                     genpod=True, basfuntype='pl')
inivred = np.dot(Uky.T, iniv)
(msx, ms) = gpu.get_genmeasuremat(sol=vv.T, tmesh=tmesh, sdim=Ns)
x = lau.apply_massinv(sps.csc_matrix(ms), msx.T).T
sini = np.r_[1, np.zeros((Ns-1, ))].reshape((Ns, 1))
xz = np.copy(x)
xz[:, 0] = 0  # zero out nu0 - the ini condition needs extra treatment
Uks = gpu.get_podmats(sol=xz.T, poddim=hs-1, plotsvs=False, M=M)
Uks = np.hstack([sini, Uks])

dms = gpu.get_dms(sdim=Ns, tmesh=tmesh, basfuntype='pl')
hdms = np.dot(Uks.T, np.dot(dms, Uks))
hms = np.dot(Uks.T, np.dot(ms, Uks))
# nres, timespaceres = get_spacetimepodres(tvvec=hshysol, dms=hdms, ms=hms,
#                                          my=mred, ared=None,
#                                          nfunc=None, rhs=None)
# trsrs = timespaceres.reshape((hq, hs))
# plotmat((np.dot(Uky, np.dot(trsrs, Uks.T))).T, fignum=121)


def sollintimspasys(dms=None, ms=None, ay=None, my=None, iniv=None,
                    opti=False):

    lns = ms.shape[0]
    lnq = ay.shape[0]
    dmsz = dms[1:, :1]
    msz = ms[1:, :1]

    dmsI = dms[1:, 1:]
    msI = ms[1:, 1:]

    solmat = np.kron(dmsI, my) + np.kron(msI, ay)

    rhs = -np.kron(dmsz, lau.mm_dnssps(my, iniv)) \
        - np.kron(msz, lau.mm_dnssps(ay, iniv))

    if opti:
        import scipy.optimize as sco

        def _linres(tsvec):
            tsvecs = tsvec.reshape((tsvec.size, 1))
            lres = (np.dot(solmat, tsvecs) - rhs)
            return lres.flatten()

        def _lrprime(tsvec):
            return solmat
        sol = sco.fsolve(_linres, np.zeros(((lns-1)*lnq, )), fprime=_lrprime)
    else:
        sol = np.linalg.solve(solmat, rhs)

    fsol = np.hstack([iniv, sol.reshape((lns-1, lnq)).T])
    return fsol


fullsyslincheck = True
fopti = False
if fullsyslincheck:
    fsol = sollintimspasys(dms=dms, ms=ms, ay=np.array(ay.todense()),
                           my=np.array(my.todense()), iniv=iniv, opti=fopti)
    plotmat(fsol.T, fignum=122)
smasyslincheck = True
sopti = True
if smasyslincheck:
    ssol = sollintimspasys(dms=hdms, ms=hms, ay=ared, my=mred, iniv=inivred,
                           opti=sopti)
    fssolt = np.dot(np.dot(Uks, ssol.T), Uky.T)
    plotmat(fssolt, fignum=124)

plotmat(fssolt - fsol.T, fignum=133)
# plotmat(fssolt - x.T, fignum=144)
