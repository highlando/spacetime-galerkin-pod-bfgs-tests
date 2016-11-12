import dolfin_burgers_scipy as dbs
import burgers_genpod_utils as bgu
import gen_pod_utils as gpu
import numpy as np
import scipy.sparse as sps

from plot_utils import plotmat

Nq = 120  # dimension of the spatial discretization
Nts = 120  # number of time sampling points
t0, tE = 0., 1.

Ns = 30  # Number of measurement functions = Num of snapshots
hq = 20  # number of space modes

tmesh = np.linspace(t0, tE, Nts)
nu = 2e-3

plotplease = True
# plotplease = False

(My, A, rhs, nfunc, femp) = dbs.\
    burgers_spacedisc(N=Nq, nu=nu, retfemdict=True)

iniv = np.r_[np.ones(((Nq-1)/2, 1)), np.zeros(((Nq)/2, 1))]


def fwdrhs(t):
    return rhs
simudict = dict(iniv=iniv, A=A, M=My,
                nfunc=nfunc, rhs=fwdrhs, tmesh=tmesh)
vv = gpu.time_int_semil(**simudict)
if plotplease:
    plotmat(vv, fignum=1234)

(xms, Ms) = gpu.get_genmeasuremat(sol=vv.T, tmesh=tmesh, sdim=Ns)

from ext_cholmod import SparseFactorMassmat
myfac = SparseFactorMassmat(sps.csc_matrix(My))
msfac = SparseFactorMassmat(sps.csc_matrix(Ms))
# My = myfac.F*myfac.Ft
# Ms = msfac.F*msfac.Ft

lytxms = myfac.Ft*xms
lytxlst = (msfac.solve_Ft(lytxms.T)).T
xlst = (msfac.solve_Ft(xms.T)).T

Uky, _ = gpu.get_podbases(measmat=lytxlst, nlsvecs=hq, nrsvecs=0)

rlrdlft = myfac.solve_Ft(np.dot(Uky, np.dot(Uky.T, myfac.Ft*iniv)))
rldiffprj = iniv - rlrdlft
rlprojerr = np.dot(rldiffprj.T, My*rldiffprj)

rlrdlftsol = myfac.solve_Ft(np.dot(Uky, np.dot(Uky.T, myfac.Ft*xlst)))
rldiffprjsol = xlst - rlrdlftsol
rlprojersol = np.linalg.norm(myfac.Ft*rldiffprjsol, ord='fro')

print 'Factorization works: {0}'.\
    format(np.allclose(xlst, myfac.Ft*myfac.solve_Ft(xlst)))

print 'nu={0}, Nq={1}, hq={2}'.format(nu, Nq, hq)
print 'real direct proj error in inival: {0}'.format(rlprojerr)
print 'real direct proj error in sol: {0}'.format(rlprojersol)
# print 'diff in proj: {0}'.format(difprjer)

lyitUky = myfac.solve_Ft(Uky)  # for the Galerkin projection of the system
lytUky = myfac.F*Uky  # to project down, e.g., the initial value
# note that tx = uy.-T*Uky beta*hx  = Ly.-T*Uky*Uky.T*Ly.T*x
Ak, Mk, nonl_red, rhs_red, liftcoef, projcoef =\
    gpu.get_spaprjredmod(M=My, A=A, nonl=nfunc, rhs=fwdrhs,
                         Uk=lyitUky, prjUk=lytUky)
redsimudict = dict(iniv=projcoef(iniv).flatten(), A=Ak, M=Mk, nfunc=nonl_red,
                   rhs=rhs_red, tmesh=tmesh)
hvv = gpu.time_int_semil(**redsimudict)
lifthvv = liftcoef(hvv.T).T
if plotplease:
    plotmat(lifthvv, fignum=1235)
    plotmat(lifthvv-vv, fignum=1236)

uvvdxl = dbs.get_burgertensor_spacecomp(V=femp['V'], podmat=lyitUky,
                                        ininds=femp['ininds'],
                                        diribc=femp['diribc'])

# testvk = projcoef(vv[-2, :].T)
# bnonlvalk = nonl_red(testvk, None).reshape((len(testvk), 1))


# the reduced nonlinearity through the tensor evaluations
def tens_red_nonl(svec, t):
    return bgu.eva_burger_spacecomp(uvvdxl=uvvdxl, svec=svec).flatten()

tensredsimudict = dict(iniv=projcoef(iniv).flatten(), A=Ak, M=Mk,
                       nfunc=tens_red_nonl, rhs=rhs_red, tmesh=tmesh)

thvv = gpu.time_int_semil(**tensredsimudict)
liftthvv = liftcoef(thvv.T).T
if plotplease:
    plotmat(liftthvv, fignum=1237)
    plotmat(liftthvv-vv, fignum=1238)
