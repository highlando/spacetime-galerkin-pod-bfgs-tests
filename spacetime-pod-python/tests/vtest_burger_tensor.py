import numpy as np
import scipy.sparse as sps

import dolfin_navier_scipy.data_output_utils as dou
import sadptprj_riclyap_adi.lin_alg_utils as lau

import dolfin_burgers_scipy as dbs
import gen_pod_utils as gpu

from plot_utils import plotmat
from numpy.linalg import norm
from burgers_genpod_utils import get_podredmod_burger


# coeffs of the tests
Nq = 50  # dimension of the spatial discretization
Ns = 40  # dimension of the temporal discretization
hq, hs = 10, 10  # reduced dimension
Nts = 90  # number of time sampling points

redmodel = True
linearcase = False
plotplease = False

t0, tE = 0., 1.
timebasfuntype = 'pl'
tmesh = np.linspace(t0, tE, Nts)
iniv = np.r_[np.zeros((np.floor(Nq*.5), 1)), np.ones((np.ceil(Nq*.5)-1, 1))]

nu = 3.e-3
nul = [nu]
datastr = 'burgv_contim_nu{1}N{0}Nts{2}t0{3}tE{4}'.format(Nq, nu, Nts, t0, tE)


nustr = 'nu{0}'.format(nul[0])
for curnu in nul[1:]:
    nustr = nustr + '_{0}'.format(curnu)


def getburgdisc():
    '''the burger discretization and full solution
    '''
    (My, A, rhs, nfunc, femp) = dbs.\
        burgers_spacedisc(N=Nq, nu=nu, retfemdict=True)

    simudict = dict(iniv=iniv, A=A, M=My,
                    nfunc=nfunc, rhs=rhs, tmesh=tmesh)
    vv = dou.\
        load_or_comp(filestr='data/'+datastr, comprtn=gpu.time_int_semil,
                     comprtnargs=simudict, arraytype='dense', debug=True)

    V, ininds, diribc = femp['V'], femp['ininds'], femp['diribc']
    return burgdata, femp

if not linearcase:
    uvvdxlfull = dbs.\
        get_burgertensor_spacecomp(V=V, podmat=np.eye(len(ininds)),
                                   ininds=ininds, diribc=diribc)

    def eva_burger_spacecomp(uvvdxl=None, svec=None):
        evall = []
        for uvvdx in uvvdxl:
            evall.append(np.dot(svec.T, uvvdx.dot(svec)))
        return np.array(evall).reshape((len(uvvdxl), 1))

    def nonl_red(v, t):
        return np.dot(Uky.T, nfunc(np.dot(Uky, v), t)).flatten()

    testv = vv[-1, :].T
    bnonlval = nfunc(testv, None).reshape((len(testv), 1))
    btensval = eva_burger_spacecomp(uvvdxl=uvvdxlfull, svec=testv)

    # compute the reduced nonlinearities and tensor
    (msx, Ms) = gpu.get_genmeasuremat(sol=vv.T, tmesh=tmesh, sdim=Ns)
    x = lau.apply_massinv(sps.csc_matrix(Ms), msx.T).T

    Uky, Uks = gpu.get_timspapar_podbas(hs=hs, hq=hq, plotsvs=plotplease,
                                        My=My, Ms=Ms, snapsl=[x])

    rbd = get_podredmod_burger(Uky=Uky, Uks=Uks, Nq=Nq,
                               nu=nu, iniv=iniv, nustr=None)

    uvvdxl = rbd['uvvdxl']
    testvk = np.dot(Uky.T, vv[-1, :].T)

    bnonlvalk = nonl_red(testvk, None).reshape((len(testvk), 1))
    btensvalk = eva_burger_spacecomp(uvvdxl=uvvdxl, svec=testvk)

    print 'rel norm: ass. nonl - tensor evaluation: {0}'.\
        format(norm(bnonlval-btensval)/norm(testv))
    print 'rel norm: reduced: ass. nonl - tensor evaluation: {0}'.\
        format(norm(bnonlvalk-btensvalk)/norm(testvk))
    print 'rel projection error in the nonl. evaluation: {0}'.\
        format(norm(bnonlval-np.dot(Uky, btensvalk))/norm(testvk))
    print 'rel projection error in the state: {0}'.\
        format(norm(testv-np.dot(Uky, testvk))/norm(testv))

# check the linear space time residual
plotplease = False

linsimudict = dict(iniv=iniv, A=A, M=My, nfunc=None,
                   rhs=rhs, tmesh=tmesh)
vvl = dou.\
    load_or_comp(filestr='data/lincase_'+datastr, comprtn=gpu.time_int_semil,
                 comprtnargs=linsimudict, arraytype='dense', debug=True)
if plotplease:
    plotmat(vvl, fignum=125)

(msxlin, Ms) = gpu.get_genmeasuremat(sol=vvl.T, tmesh=tmesh, sdim=Ns)
xlin = lau.apply_massinv(sps.csc_matrix(Ms), msxlin.T).T

if not redmodel:
    Ukylin, Ukslin = np.eye(Nq-1), np.eye(Ns)
    hs, hq = Ns, Nq-1
    Ms = gpu.get_ms(sdim=Ns, tmesh=tmesh, basfuntype='pl')
    dms = gpu.get_dms(sdim=Ns, tmesh=tmesh, basfuntype=timebasfuntype)
    rbd = dict(hmy=My.todense(), hay=A.todense(), hms=Ms, hdms=dms,
               inivred=iniv, hrhs=rhs, hnonl=nfunc)

else:
    Ukylin, Ukslin = gpu.get_timspapar_podbas(hs=hs, hq=hq, plotsvs=plotplease,
                                              My=My, Ms=Ms, snapsl=[xlin])

    rbd = get_podredmod_burger(Uky=Uky, Uks=Uks, Nq=Nq,
                               nu=nu, iniv=iniv, nustr=nustr)

dmsz = rbd['hdms'][1:, :1]  # part of the mass matrix related to the ini value
msz = rbd['hms'][1:, :1]

dmsI = rbd['hdms'][1:, 1:]  # mass matrices w/o ini values (test and trial)
msI = rbd['hms'][1:, 1:]

solmat = np.kron(dmsI, rbd['hmy']) + np.kron(msI, rbd['hay'])

hshysol = np.dot(Ukslin.T, np.dot(xlin.T, Ukylin))
hshysolvec = hshysol.reshape((hs*hq, 1), order='C')

inivlin = hshysolvec[:hq, :]
insollin = hshysolvec[hq:, :]

linrhs = - np.kron(dmsz, lau.mm_dnssps(rbd['hmy'], inivlin)) \
    - np.kron(msz, lau.mm_dnssps(rbd['hay'], inivlin))

print 'residual projsol: ', norm(np.dot(solmat, insollin) - linrhs)
print norm(xlin.T - np.dot(Ukslin, np.dot(hshysol, Ukylin.T)))

linsyssolvec = np.linalg.solve(solmat, linrhs)
diff = linsyssolvec - insollin
print 'diff ', norm(diff)
print 'residual linsyssol: ', norm(np.dot(solmat, linsyssolvec) - linrhs)
print 'cond A ', np.linalg.cond(solmat)
linsyssol = linsyssolvec.reshape((hs-1, hq))

plotmat(vvl, fignum=113)
plotmat(np.dot(Ukslin, np.dot(hshysol, Ukylin.T)), fignum=114)
plotmat(linsyssol, fignum=115)
plotmat(linsyssol-np.dot(Ukslin, np.dot(hshysol, Ukylin.T)), fignum=116)

if plotplease:
    plotmat(vv, fignum=155, tikzfile='full sol')
