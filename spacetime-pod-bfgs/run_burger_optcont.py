import numpy as np
import scipy.sparse as sps
from scipy.optimize import fsolve
from scipy.interpolate import interp1d
try:
    import numdifftools as nd
except ImportError:
    print 'Cannot import numdifftools -- hope we don`t need it'

import dolfin_navier_scipy.data_output_utils as dou

import dolfin_burgers_scipy as dbs
import burgers_genpod_utils as bgu
import burgers_optcont_utils as bou
import gen_pod_utils as gpu
import spacetime_pod_utils as spu

from plot_utils import plotmat

import logging
import logging.config
import yaml
logging.config.dictConfig(yaml.load(open('logging.conf', 'r')))
logger = logging.getLogger('basic')
# logger.setLevel('INFO')  # default is DEBUG can be ERROR, INFO
logger.setLevel('INFO')  # default is DEBUG can be ERROR, INFO

debug = True
debug = False


# helper functions
def stateadjspatibas(xms=None, lms=None, Ms=None, My=None,
                     nspacevecs=None, ntimevecs=None, spacebasscheme='VandL'):

    # if spacebasscheme == 'VandL' or (spacebasscheme == 'onlyV'
    #                                  or spacebasscheme == 'onlyL'):
    lyitUVy, lyUVy, lsitUVs, lsUVs = gpu.\
        get_podbases_wrtmassmats(xms=xms, Ms=Ms, My=My, nspacevecs=nspacevecs,
                                 xtratreatini=True, ntimevecs=ntimevecs)

    if lms is None:
        lyitULy, lyULy, lsitULs, lsULs = None, None, None, None

    else:
        lyitULy, lyULy, lsitULs, lsULs = gpu.\
            get_podbases_wrtmassmats(xms=lms, My=My, nspacevecs=nspacevecs,
                                     Ms=Ms, ntimevecs=ntimevecs,
                                     xtratreattermi=True)
    if spacebasscheme == 'onlyV':
        lyULy, lyitULy = lyUVy, lyitUVy

    if spacebasscheme == 'onlyL':
        lyUVy, lyitUVy = lyULy, lyitULy

    if spacebasscheme == 'combined':
        lyitUVLy, lyUVLy, _, _ = gpu.\
            get_podbases_wrtmassmats(xms=[xms, lms], My=My, Ms=Ms,
                                     nspacevecs=nspacevecs, ntimevecs=0)
        lyUVy, lyitUVy = lyUVLy, lyitUVLy
        lyULy, lyitULy = lyUVLy, lyitUVLy

    return lyitUVy, lyUVy, lsitUVs, lsUVs, lyitULy, lyULy, lsitULs, lsULs


def spacetimesolve(func=None, funcjaco=None, inival=None, timerecord=False,
                   usendjaco=False, message=None, printstats=True):
    if logger.level == 10:
        myjaco = funcjaco(inival.flatten())
        ndjaco = nd.Jacobian(func)(inival.flatten())
        logger.debug('diffnorm of the analytical and the numerical jacobian' +
                     ' `dJ={0}`'.format(np.linalg.norm(myjaco-ndjaco)))

    if usendjaco:
        funcjaco = numerical_jacobian(func)

    tfd = {}  # if timerecord else None
    with dou.Timer(message, timerinfo=tfd):
        if printstats:
            sol, infdct, _, _ = fsolve(func, x0=inival.flatten(),
                                       fprime=funcjaco, full_output=True)
            logger.info(message + ': nfev={0}, normf={1}'.
                        format(infdct['nfev'], np.linalg.norm(infdct['fvec'])))
        else:
            sol = fsolve(func, x0=inival.flatten(), fprime=funcjaco)

    if timerecord:
        return sol, tfd['elt']
    else:
        return sol


def Xtoxvec(X):
    " X - (nq, ns) array to (ns*nq, 1) array by stacking the columns "

    return X.T.reshape((X.size, 1))


def numerical_jacobian(func):
    def ndjaco(tvvec):
        ndjc = nd.Jacobian(func)(tvvec)
        return ndjc
    return ndjaco


def companandjacos(tvvec, func=None, jacofunc=None):
    ndjaco = nd.Jacobian(func)(tvvec)
    anajaco = jacofunc(tvvec)
    vld = tvvec.size/2
    jlln, jlla = ndjaco[vld:, vld:], anajaco[vld:, vld:]
    jlvn, jlva = ndjaco[vld:, :vld], anajaco[vld:, :vld]
    jvvn, jvva = ndjaco[:vld, :vld], anajaco[:vld, :vld]
    jvln, jvla = ndjaco[:vld, vld:], anajaco[:vld, vld:]

    print 'diff in dJ: ', np.linalg.norm(ndjaco-anajaco)
    print 'diff in dJ_11: ', np.linalg.norm(jvvn - jvva)
    print 'diff in dJ_12: ', np.linalg.norm(jvln - jvla)
    print 'diff in dJ_21: ', np.linalg.norm(jlvn - jlva)
    print 'diff in dJ_22: ', np.linalg.norm(jlln - jlla)


def eva_costfun(vopt=None, uopt=None, qmat=None, rmat=None, ms=None,
                vstar=None, tmesh=None,
                iniv=None, termiu=None):
    if iniv is not None:
        vopt = np.vstack([iniv, vopt])
    if termiu is not None:
        uopt = np.vstack([uopt, termiu])
    if tmesh is not None:
        vstarvec = np.atleast_2d(vstar(tmesh[0])).T
        for t in tmesh[1:]:
            vstarvec = np.vstack([vstarvec, np.atleast_2d(vstar(t)).T])
        diffv = vopt - vstarvec
    else:
        diffv = vopt - vstar
    vterm = 0.5*spu.krontimspaproduct(diffv, ms=ms, my=qmat)
    uterm = 0.5*spu.krontimspaproduct(uopt, ms=ms, my=rmat)
    return dict(vterm=vterm, uterm=uterm, value=vterm+uterm)


def testit(Nq=None, Nts=None,
           inivtype=None, dmndct=None, Ns=None, hq=None, hs=None,
           spacebasscheme=None, nu=None, alpha=None,
           genpodstate=False, genpodadj=False, genpodcl=False,
           plotplease=False, tikzplease=False, tikzprefikz='',
           adjplotdict=None,
           onlytimings=False, **kwargs):

    t0, tE = dmndct['t0'], dmndct['tE']

    # to tikz or not to tikz
    fullmodelv_tkzf, fullmodelL_tkzf, redoptiv_tkzf = None, None, None
    redmodelL_tkzf = None
    redmodelV_tkzf = None
    redoptlambda_tkzf, backcheck_tkzf = None, None
    if tikzplease:
        fullmodelv_tkzf = tikzprefikz + 'fullmodelV'
        fullmodelL_tkzf = tikzprefikz + 'fullmodelL'
        redmodelL_tkzf = tikzprefikz + 'redmodelL'
        redmodelV_tkzf = tikzprefikz + 'redmodelV'
        # redoptiv_tkzf = tikzprefikz + 'redoptiv'
        # redoptlambda_tkzf = tikzprefikz + 'redoptlambda'
        backcheck_tkzf = tikzprefikz + 'backcheck'
    tmesh = np.linspace(t0, tE, Nts)
    snapshottmesh = np.linspace(t0, tE, Ns)
    # ### define the model
    x0, xE = dmndct['x0'], dmndct['xE']
    (My, A, rhs, nfunc, femp) = dbs.\
        burgers_spacedisc(N=Nq, nu=nu, x0=x0, xE=xE, retfemdict=True)
    # define the initial value
    if inivtype == 'smooth':
        xrng = np.linspace(0, 2*np.pi, Nq-1)
        iniv = 0.5 - 0.5*np.sin(xrng + 0.5*np.pi)
        iniv = 0.5*iniv.reshape((Nq-1, 1))
    elif inivtype == 'step':
        # iniv = np.r_[np.ones(((Nq-1)/2, 1)), np.zeros(((Nq)/2, 1))]
        iniv = np.r_[np.zeros(((Nq)/2, 1)), np.ones(((Nq-1)/2, 1))]
    elif inivtype == 'ramp':
        iniv = np.r_[np.linspace(0, 1, ((Nq-1)/2)).reshape(((Nq-1)/2, 1)),
                     np.zeros(((Nq)/2, 1))]
    # ### compute the forward snapshots

    def fwdrhs(t):
        return rhs
    simudict = dict(iniv=iniv, A=A, M=My, nfunc=nfunc, rhs=fwdrhs, tmesh=tmesh)
    with dou.Timer('fwd'):
        datastr = 'data/fwdsol_iniv' + inivtype + \
            'Nq{0}Nts{1}nu{2}'.format(Nq, Nts, nu)
        vv = dou.load_or_comp(filestr=datastr, comprtn=gpu.time_int_semil,
                              arraytype='dense', comprtnargs=simudict,
                              debug=debug)
        # vv = gpu.time_int_semil(**simudict)
    if plotplease:
        plotmat(vv, fignum=1234, tikzfile=fullmodelv_tkzf, **dmndct)

    (xms, Ms) = gpu.get_genmeasuremat(sol=vv.T, tmesh=tmesh, sdim=Ns)
    # ### compute the backward snapshots
    if genpodadj or genpodcl:
        vfun = interp1d(tmesh, vv, axis=0)
        vdxoperator, fnctnl = dbs.\
            burgers_bwd_spacedisc(V=femp['V'], ininds=femp['ininds'],
                                  diribc=femp['diribc'])
        te = tmesh[-1]

        def vfun_ext(t):
            if t < t0:  # the integrator may require values outside [t0, tE]
                # print 'omg I am out of range'
                return vfun(t0)
            else:
                return vfun(t)

        def vstar(t):
            return iniv.flatten()

        def burger_bwd_rhs(t):
            # TODO: -----------------------------> here we need vstar
            return (-fnctnl(vfun_ext(te-t)).flatten()
                    + fnctnl(vstar(t)).flatten())

        def burger_bwd_nonl(lvec, t):
            vdx = vdxoperator(vfun_ext(te-t))
            return -(vdx*lvec).flatten()

        termiL = np.zeros((Nq-1, 1))
        bbwdsimudict = dict(iniv=termiL, A=A, M=My, nfunc=burger_bwd_nonl,
                            rhs=burger_bwd_rhs, tmesh=tmesh)

        with dou.Timer('bwd'):
            datastr = 'data/bwdsol_iniv' + inivtype + \
                'Nq{0}Nts{1}nu{2}'.format(Nq, Nts, nu)
            bwdll = dou.\
                load_or_comp(filestr=datastr, comprtn=gpu.time_int_semil,
                             arraytype='dense', comprtnargs=bbwdsimudict,
                             debug=debug)
            # bwdll = gpu.time_int_semil(**bbwdsimudict)
        ll = np.flipud(bwdll)  # flip the to make it forward time
        if plotplease:
            if adjplotdict is None:
                adjplotdict = dmndct
            plotmat(ll, fignum=1235, tikzfile=fullmodelL_tkzf, **adjplotdict)

        (Lms, _) = gpu.get_genmeasuremat(sol=ll.T, tmesh=tmesh, sdim=Ns)
    else:
        Lms = None
    # ### compute the projection matrices, i.e. optimal bases
    (lyitUVy, lyUVy, lsitUVs, lsUVs, lyitULy, lyULy, lsitULs,
     lsULs) = stateadjspatibas(xms=xms, lms=Lms, Ms=Ms, My=My, nspacevecs=hq,
                               ntimevecs=hs, spacebasscheme=spacebasscheme)
    # ### the fwd projection scheme
    AVk, MVk, nonl_red, rhs_red, liftcoef, projcoef =\
        gpu.get_spaprjredmod(M=My, A=A, nonl=nfunc, rhs=fwdrhs,
                             Uk=lyitUVy, prjUk=lyUVy)

    hiniv = projcoef(iniv)

    Ms = sps.csc_matrix(gpu.get_ms(sdim=Ns, tmesh=tmesh))
    dms = sps.csc_matrix(gpu.get_dms(sdim=Ns, tmesh=tmesh))
    Vhms = np.dot(lsitUVs.T, Ms*lsitUVs)
    Vhdms = np.dot(lsitUVs.T, dms*lsitUVs)

    print 'assembling the reduced tensor...'
    datastr = 'data/fwd_iniv' + inivtype + '_tnsr_' + spacebasscheme + \
        '_Nts{5}Nq{0}Ns{1}hq{2}hs{3}nu{4}'.format(Nq, Ns, hq, hs, nu, Nts)
    uvvdxl, htittl = bgu.\
        get_burger_tensor(Uky=lyitUVy, Uks=lsitUVs, sdim=Ns, bwd=True,
                          Vhdms=Vhdms, Vhms=Vhms, Vhmy=MVk, Vhay=AVk,
                          tmesh=tmesh, datastr=datastr, debug=debug, **femp)

    vres, vresprime = \
        bou.setup_burger_fwdres(Vhdms=Vhdms, Vhms=Vhms, Vhmy=MVk, Vhay=AVk,
                                htittl=htittl, uvvdxl=uvvdxl, hiniv=hiniv)

    def ndvresprime(tvvec):
        ndjaco = nd.Jacobian(vres)(tvvec)
        return ndjaco
    # ### the bwd projection scheme
    if genpodadj or genpodcl:
        Lhms = np.dot(lsitULs.T, Ms*lsitULs)
        Lhdms = np.dot(lsitULs.T, dms*lsitULs)

        LVhms = np.dot(lsitULs.T, Ms*lsitUVs)
        LVhmy = np.dot(lyitULy.T, My*lyitUVy)

        Lhay, Lhmy, _, _, Lliftcoef, Lprojcoef =\
            gpu.get_spaprjredmod(M=My, A=A, Uk=lyitULy, prjUk=lyULy)

        print 'assembling the bwd reduced tensor...'
        datastr = 'data/bwdtnsr_iniv' + inivtype + '_' + spacebasscheme +\
            '_Nts{5}Nq{0}Ns{1}hq{2}hs{3}nu{4}'.format(Nq, Ns, hq, hs, nu, Nts)
        Luvvdxl, Lhtittl = bgu.\
            get_burger_tensor(Uky=lyitULy, bwd=True, Uks=lsitULs,
                              Ukyconv=lyitUVy, Uksconv=lsitUVs, sdim=Ns,
                              tmesh=tmesh, datastr=datastr, debug=debug,
                              **femp)

        tgtst = np.tile(iniv, Ns)
        htgst = np.dot(lyUVy.T, np.dot(tgtst, lsUVs))
        htgstvec = Xtoxvec(htgst)

        hcurst = np.dot(lyUVy.T, np.dot(xms, lsitUVs))
        hcurstvec = Xtoxvec(hcurst)

        htermiL = np.zeros((hq, 1))

        lres, lresprime = bou.\
            setup_burger_bwdres(Lhdms=Lhdms, Lhms=Lhms, Lhmy=Lhmy, Lhay=Lhay,
                                LVhms=LVhms, LVhmy=LVhmy,
                                Lhtittl=Lhtittl, Luvvdxl=Luvvdxl,
                                hiniv=hiniv, htermiL=htermiL,
                                tsVvec=hcurstvec[hq:, :], tsVtrgtvec=htgstvec)
        # ### the optimal cont problem
        VLhms = np.dot(lsitUVs.T, Ms*lsitULs)
        VLhmy = np.dot(lyitUVy.T, My*lyitULy)
        clres, clresprime = bou.\
            setup_burger_clres(Vhdms=Vhdms, Vhms=Vhms, Vhmy=MVk, Vhay=AVk,
                               VLhms=VLhms, VLhmy=VLhmy, alpha=alpha,
                               htittl=htittl, uvvdxl=uvvdxl,
                               tsVtrgtvec=htgstvec,
                               Lhdms=Lhdms, Lhms=Lhms, Lhmy=Lhmy, Lhay=Lhay,
                               LVhms=LVhms, LVhmy=LVhmy,
                               Lhtittl=Lhtittl, Luvvdxl=Luvvdxl,
                               hiniv=hiniv, htermiL=htermiL)
    # ### the tests
    # # forward problem
    if genpodstate:
        optiniV = np.tile(hiniv.T, hs-1).T
        print 'solving the optimization problem (state)...'
        sol = spacetimesolve(func=vres, funcjaco=vresprime, inival=optiniV,
                             message='fwd problem - analytical jacobian')
        optiV = np.r_[hiniv.flatten(), sol]
        optiV = optiV.reshape((hs, hq))
        if plotplease:
            plotmat((np.dot(lyitUVy, np.dot(optiV.T, lsitUVs.T))).T,
                    tikzfile=redmodelV_tkzf, fignum=1233, **dmndct)

    if genpodadj:
        optiniL = np.zeros((hq*(hs-1), 1))
        print 'solving the optimization problem (adjoint)...'
        sol = spacetimesolve(func=lres, funcjaco=lresprime, inival=optiniL,
                             message='bwd problem - analytical jacobian')
        optiL = np.r_[sol, htermiL.flatten()]
        optiL = optiL.reshape((hs, hq))
        if plotplease:
            plotmat((np.dot(lyitULy, np.dot(optiL.T, lsitULs.T))).T,
                    fignum=124, tikzfile=redmodelL_tkzf, **adjplotdict)

    if genpodcl:
        print 'solving the optimization problem (fwdbwd)...'
        optiniV = np.tile(hiniv.T, hs-1).T
        optiniL = np.zeros((hq*(hs-1), 1))
        optiniVL = np.vstack([optiniV, optiniL])
        sol, optconttime = \
            spacetimesolve(func=clres, funcjaco=clresprime, inival=optiniVL,
                           message='optcont problem - analytical jacobian',
                           timerecord=True)
        if onlytimings:
            return optconttime

        optiV = np.r_[hiniv.flatten(), sol[:hq*(hs-1)]]
        optiV = optiV.reshape((hs, hq))

        optiL = np.r_[sol[-hq*(hs-1):], htermiL.flatten()]
        optiL = optiL.reshape((hs, hq))

        fulloptiL = (np.dot(lyitULy, np.dot(optiL.T, lsitULs.T))).T
        if plotplease:
            plotmat((np.dot(lyitUVy, np.dot(optiV.T, lsitUVs.T))).T,
                    tikzfile=redoptiv_tkzf, fignum=123, **dmndct)
            plotmat(1./alpha*fulloptiL, tikzfile=redoptlambda_tkzf,
                    fignum=1241, **dmndct)

        # ### SECTION: fwd problem with reduced costates
        redmodu = 1./alpha*fulloptiL
        redmodufun = interp1d(snapshottmesh, redmodu, axis=0)

        def burger_contrl_rhs(t):
            if t > tE:  # the integrator may require values outside [t0, tE]
                return fnctnl(redmodufun(tE)).flatten()
            else:
                return fnctnl(redmodufun(t)).flatten()

        simudict = dict(iniv=iniv, A=A, M=My, nfunc=nfunc,
                        rhs=burger_contrl_rhs, tmesh=snapshottmesh)

        print 'back check...'
        with dou.Timer('check back'):
            vv = gpu.time_int_semil(**simudict)
            if plotplease:
                plotmat(vv, fignum=12341, tikzfile=backcheck_tkzf, **dmndct)
                vvd = vv - np.tile(iniv.T, (Ns, 1))

                plotmat(vvd, fignum=12342, **dmndct)

                plotmat(np.tile(iniv.T, (Ns, 1)), fignum=12343,
                        tikzfile=tikzprefikz+'zstar', **dmndct)

        valdict = eva_costfun(vopt=Xtoxvec(vv.T), uopt=Xtoxvec(redmodu),
                              qmat=My, rmat=alpha*My, ms=Ms,
                              vstar=vstar, tmesh=snapshottmesh)

        logger.info('Value of the cost functional: {0}'.
                    format(valdict['value']))

        return valdict, optconttime
    else:
        return None, None


if __name__ == '__main__':
    # ### define the problem
    testitdict = \
        dict(Nq=150,  # dimension of the spatial discretization
             Nts=150,  # number of time sampling points
             # t0=0., tE=1.,
             # x0=0., xE=1.,
             inivtype='step',  # 'ramp', 'smooth'
             dmndct=dict(tE=1., t0=0., x0=0., xE=1.),  # for the plots
             Ns=70,  # Number of measurement functions=Num of snapshots
             hq=12,  # number of space modes
             hs=12,  # number of time modes
             spacebasscheme='combined',  # 'onlyL' 'VandL' 'combined'
             plotplease=True, tikzplease=False,
             nu=5e-3,
             alpha=1e-3,
             genpodcl=True)

    value, timerinfo = testit(**testitdict)
