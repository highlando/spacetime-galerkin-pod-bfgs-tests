import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import fmin_bfgs

import dolfin_navier_scipy.data_output_utils as dou

from schloegl_spacetime_pod import get_schloegl_modls

import spacetime_function_utils as sfu

t0, tE, Nts = 0., 3., 100
Nq, Ns = 30, 90

hq, hs = 20, 20

nonlpara, mu = 5., 1e-1
alpha = 1e-6

bfgsiters = 120
gtol = 1e-5

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
                 spacepod=True,
                 utrial=utrial,
                 inivtype=inivtype,
                 alpha=alpha)

probstr = 'sqpopti_beta{0}mu{1}alpha{4}hqhs{2}{3}'.\
    format(nonlpara, mu, hq, hs, alpha)

(eva_redcostfun, eva_redcostgrad, eva_fulfwd, eva_fulfwd_intp, eva_fulcostfun,
 eva_redfwd, cmoddct) = get_schloegl_modls(**modeldict)

redtmesh = cmoddct['redtmesh']
tmesh = cmoddct['tmesh']
msrmtmesh = cmoddct['msrmtmesh']
cB = cmoddct['B']
cC = cmoddct['C']
ystarfun = cmoddct['ystarfun']
NV, nu = cB.shape

cuk = np.ones((nu, redtmesh.size))
ziniv = np.zeros((NV, 1))
hziniv = np.zeros((hq, 1))


def redcostfun(uvec):
    return eva_redcostfun(uvec, inival=hziniv)


def redcostgrad(uvec):
    return eva_redcostgrad(uvec, inival=hziniv)


print('optimizing...')
with dou.Timer('BFGS - reduced problem'):
    uopt, fopt, gopt, Bopt, nfc, ngc, wflag = \
        fmin_bfgs(redcostfun, 0*sfu.Xtoxvec(cuk), full_output=True,
                  fprime=redcostgrad,
                  maxiter=bfgsiters, gtol=gtol)

print('checking backing...')

ucheck = testu(msrmtmesh)
vnopt = eva_fulfwd_intp(ucheck, inival=ziniv, utmesh=msrmtmesh)
ynopt = cC.dot(sfu.xvectoX(vnopt, ns=tmesh.size, nq=NV)).flatten()
noptivlv, noptivlu = eva_fulcostfun(ucheck, vvec=vnopt,
                                    utmesh=msrmtmesh, retparts=True)
print('J(u_test): {0}+{1}'.format(noptivlv, noptivlu))

vopt = eva_fulfwd_intp(uopt, inival=ziniv, utmesh=redtmesh)
yopt = cC.dot(sfu.xvectoX(vopt, ns=tmesh.size, nq=NV)).flatten()
optivalv, optivalu = eva_fulcostfun(uopt, vvec=vopt, retparts=True,
                                    inival=ziniv, utmesh=redtmesh)
yopt = cC.dot(sfu.xvectoX(vopt, ns=tmesh.size, nq=NV)).flatten()

optivalv, optivalu = eva_fulcostfun(uopt, vvec=vopt, retparts=True,
                                    inival=ziniv, utmesh=redtmesh)
print('J(u_opt): {0}+{1}'.format(optivalv, optivalu))

ystar = ystarfun(tmesh)
redcyopt = eva_redfwd(uopt, inival=hziniv, tmesh=tmesh, rety=True)
redcynopt = eva_redfwd(ucheck, inival=hziniv, utmesh=msrmtmesh, rety=True)

plt.figure(1)
plt.plot(tmesh, yopt, label='y_opt')
plt.plot(tmesh, ynopt, label='y_test')
# plt.plot(tmesh, redcynopt, label='redy_u0')
plt.plot(tmesh, redcyopt.flatten(), label='redy_opt')
plt.plot(tmesh, ystar.flatten(), label='ystar')
plt.legend(loc='best')
tiksplz = True
if tiksplz:
    from matplotlib2tikz import save as tikz_save
    tikz_save(probstr + '.tikz')
plt.show()
print('DONE: ' + probstr)
