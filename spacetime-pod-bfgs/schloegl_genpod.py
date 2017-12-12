import numpy as np
import matplotlib.pyplot as plt

import dolfin_navier_scipy.data_output_utils as dou

from schloegl_spacetime_pod import get_schloegl_modls
import genpod_opti_utils as gou

import spacetime_function_utils as sfu

t0, tE, Nts = 0., 3., 100
Nq, Ns = 30, 90
hq, hs = 10, 15

nonlpara, mu = 5., 1e-1
alpha = 1e-6

utrial = 'sine'
inivtype = 'allzero'

spacebasscheme = 'VandL'


def testu(t):
    return np.sin(2*t/tE*np.pi)


modeldict = dict(t0=t0, tE=tE, Nts=Nts,
                 nonlpara=nonlpara, mu=mu,
                 Nq=Nq, Ns=Ns,
                 hq=hq, hs=hs,
                 spacebasscheme=spacebasscheme,
                 # spacepod=True,
                 spacetimepod=True,
                 utrial=utrial,
                 inivtype=inivtype,
                 alpha=alpha)
probstr = 'genpodopti_beta{0}mu{1}alpha{4}hqhs{2}{3}'.\
    format(nonlpara, mu, hq, hs, alpha)

(clres, eva_fulfwd, eva_fulfwd_intp, eva_fulcostfun,
    cmoddct) = get_schloegl_modls(**modeldict)

redtmesh = cmoddct['redtmesh']
tmesh = cmoddct['tmesh']
msrmtmesh = cmoddct['msrmtmesh']

ystarfun = cmoddct['ystarfun']
inflatehxv = cmoddct['inflatehxv']
inflatehxl = cmoddct['inflatehxl']

cB = cmoddct['B']
cC = cmoddct['C']
rmo = cmoddct['rmo']
ystarfun = cmoddct['ystarfun']
NV, nu = cB.shape

cuk = np.ones((nu, redtmesh.size))
ziniv = np.zeros((NV, 1))
hziniv = np.zeros((hq, 1))

print('DOIN: ' + probstr)
print('optimizing...')
with dou.Timer('... the spacetimepod-reduced problem'):
    genpodsol = gou.spacetimesolve(func=clres,
                                   inival=np.zeros((2*(hs-1)*hq, )),
                                   message='cl problem - no jacobian')

gpoptiv = inflatehxv(genpodsol[:(hs-1)*hq])
gpoptil = inflatehxl(genpodsol[(hs-1)*hq:])
optiu = rmo*(cB.T).dot(gpoptil)

print('checking backing...')
ucheck = testu(msrmtmesh)
vnopt = eva_fulfwd_intp(ucheck, inival=ziniv, utmesh=msrmtmesh)
ynopt = cC.dot(sfu.xvectoX(vnopt, ns=tmesh.size, nq=NV)).flatten()
noptivalv, noptivalu = eva_fulcostfun(ucheck, vvec=vnopt, inival=ziniv,
                                      utmesh=msrmtmesh, retparts=True)
print('J(u_test): {0}+{1}'.format(noptivalv, noptivalu))

vopt = eva_fulfwd_intp(optiu, inival=ziniv, utmesh=msrmtmesh)
yopt = cC.dot(sfu.xvectoX(vopt, ns=tmesh.size, nq=NV)).flatten()
optivalv, optivalu = eva_fulcostfun(optiu, vvec=vopt, inival=ziniv,
                                    utmesh=msrmtmesh, retparts=True)
print('J(u_opt): {0}+{1}'.format(optivalv, optivalu))

ystar = ystarfun(msrmtmesh)
redcyopt = cC.dot(gpoptiv)

plotplease = True
if plotplease:
    plt.figure(1)
    plt.plot(tmesh, yopt, label='y_opt')
    plt.plot(tmesh, ynopt, label='y_test')
    # plt.plot(tmesh, redcynopt, label='redy_u0')
    plt.plot(msrmtmesh, redcyopt.flatten(), label='redy_opt')
    plt.plot(msrmtmesh, ystar.flatten(), label='ystar')
    plt.legend(loc='best')
    tiksplz = True
    if tiksplz:
        from matplotlib2tikz import save
        save(probstr + '.tikz')
    plt.show()
print('DONE: ' + probstr)
