import dolfin_burgers_scipy as dbs
import gen_pod_utils as gpu
import numpy as np
from plot_utils import plotmat

Nq = 40  # dimension of the spatial discretization
Nts = 50  # number of time sampling points
t0, tE = 0., 1.

plotplease = True
# plotplease = False

tmesh = np.linspace(t0, tE, Nts)
nu = 2e-2
(My, A, rhs, nfunc, femp) = dbs.\
    burgers_spacedisc(N=Nq, nu=nu, retfemdict=True)

iniv = np.r_[np.ones(((Nq-1)/2, 1)), np.zeros(((Nq)/2, 1))]

xmesh = np.linspace(0, 1, Nq+1)
frqx = 3
frqt = 1
ukx = np.sin(frqx*2*np.pi*xmesh)


def fwdrhs(t):
    return np.sin(frqt*2*np.pi*t)*ukx[femp['ininds']]

simudict = dict(iniv=iniv, A=A, M=My,
                nfunc=nfunc, rhs=fwdrhs, tmesh=tmesh)
vv = gpu.time_int_semil(**simudict)
if plotplease:
    plotmat(vv, fignum=1234)


def getthedisctimes(t, tmesh):
    itk = np.floor(t/(tmesh[-1]-tmesh[0])*(tmesh.size-1)).astype(int)
    return tmesh[itk], tmesh[itk+1]

vdict = dict(zip(tmesh, vv.tolist()))


def vfun(t):
    """ interpolating the data """
    try:
        itk = np.floor(t/(tmesh[-1]-tmesh[0])*(tmesh.size-1)).astype(int)
        frct = (tmesh[itk+1]-t)/(tmesh[itk+1]-tmesh[itk])
        return (frct*np.array(vdict[tmesh[itk]]) +
                (1-frct)*np.array(vdict[tmesh[itk+1]]))
    except IndexError:
        return np.array(vdict[t])

vdxop, fnctnl = dbs.burgers_bwd_spacedisc(V=femp['V'], ininds=femp['ininds'],
                                          diribc=femp['diribc'])
te = tmesh[-1]


def vstar(t):
    return iniv.flatten()


def burger_bwd_rhs(t):
    return -fnctnl(vfun(te-t)).flatten() + iniv.flatten()


def burger_bwd_nonl(lvec, t):
    vdx = vdxop(vfun(te-t))
    return -(vdx*lvec).flatten()


bbwdsimudict = dict(iniv=0*iniv, A=A, M=My, nfunc=burger_bwd_nonl,
                    rhs=burger_bwd_rhs, tmesh=tmesh)

bwdl = gpu.time_int_semil(**bbwdsimudict)
fwdl = np.flipud(bwdl)
if plotplease:
    plotmat(fwdl, fignum=1235)
