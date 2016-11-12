import numpy as np
import scipy.sparse as sps

import dolfin_burgers_scipy as dbs
import gen_pod_utils as gpu

from plot_utils import plotmat
from run_burger_optcont import eva_costfun, Xtoxvec
# import dolfin_navier_scipy.data_output_utils as dou

Nq = 200
x0, xE = 0.0, 1.0
t0 = 0.
tE = 1.
Nts = 150
Ns = 100
nu = 10e-3
alpha = 0*1e-4
qalpha = 1.  # weight of the state in the cost fun
qvstar = 1.  # weight of the state in the cost fun
plotplease = True
dmndct = dict(tE=1., t0=0., x0=0., xE=1.)

iniv = np.r_[np.zeros(((Nq)/2, 1)), np.ones(((Nq-1)/2, 1))]
tmesh = np.linspace(t0, tE, Nts)
snapshottmesh = np.linspace(t0, tE, Ns)
Ms = sps.csc_matrix(gpu.get_ms(sdim=Ns, tmesh=tmesh))

(My, A, rhs, nfunc, femp) = dbs.\
    burgers_spacedisc(N=Nq, nu=nu, x0=x0, xE=xE, retfemdict=True)


def fwdrhs(t):
    return rhs
simudict = dict(iniv=iniv, A=A, M=My, nfunc=nfunc, rhs=fwdrhs, tmesh=tmesh)

# with dou.Timer('fwd'):
#     vv = dou.load_or_comp(filestr=None, comprtn=gpu.time_int_semil,
#                           arraytype='dense', comprtnargs=simudict,
#                           debug=True)
#     # vv = gpu.time_int_semil(**simudict)
# if plotplease:
#     plotmat(vv, fignum=1234, **dmndct)


# def burger_contrl_rhs(t):
#     return np.zeros((Nq-1, 1))

uopt = np.ones((Nq-1, Ns)).T

simudict = dict(iniv=iniv, A=A, M=My, nfunc=nfunc,
                rhs=fwdrhs, tmesh=snapshottmesh)

print 'back check...'

vv, infodict = gpu.time_int_semil(full_output=True, **simudict)
print infodict['nst'][-1]
raise Warning('TODO: debug')

vopt = np.tile(iniv.T, (Ns, 1))
if plotplease:
    plotmat(vv, fignum=12341, **dmndct)
    vvd = vv - np.tile(iniv.T, (Ns, 1))

    plotmat(vvd, fignum=12342, **dmndct)

    plotmat(vopt, fignum=12343, **dmndct)


def vstar(t):
    return qvstar*iniv.T

valdict = eva_costfun(vopt=Xtoxvec(qalpha*vv.T), uopt=Xtoxvec(uopt),
                      qmat=qalpha*My, rmat=alpha*My, ms=Ms,
                      vstar=vstar, tmesh=snapshottmesh)
print('nu = {0}, alpha = {1}, Nq={2}'.format(nu, alpha, Nq))
print('Value of the cost functional: {0}'.format(valdict['value']))
normofvpart = np.sqrt(2*valdict['vterm'])
print('|| v - {0}*vstar || = {1}'.format(qvstar, normofvpart))
