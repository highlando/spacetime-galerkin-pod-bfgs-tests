import numpy as np
import scipy.sparse as sps
from scipy.interpolate import interp1d
try:
    import numdifftools as nd
except ImportError:
    print('Cannot import numdifftools -- hope we don`t need it')

import dolfin_navier_scipy.data_output_utils as dou
import spacetime_galerkin_pod.gen_pod_utils as gpu

import dolfin_burgers_scipy as dbs
import burgers_genpod_utils as bgu
import burgers_optcont_utils as bou
import spacetime_pod_utils as spu
import genpod_opti_utils as gou

from plot_utils import plotmat

from heartshape import get_spcheartfun

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
def stateadjspatibas(xms=None, lms=None, Ms=None, My=None, onlyfwd=False,
                     nspacevecs=None, ntimevecs=None, spacebasscheme='VandL'):

    lyULy, lyitULy, lsitULs, lsULs = None, None, None, None

    if spacebasscheme == 'combined' and lms is not None:
        lyitUVLy, lyUVLy, _, _ = gpu.\
            get_podbases_wrtmassmats(xms=[xms, lms], My=My, Ms=Ms,
                                     nspacevecs=nspacevecs, ntimevecs=0)
        lyUVy, lyitUVy = lyUVLy, lyitUVLy

        _, _, lsitUVs, lsUVs = gpu.\
            get_podbases_wrtmassmats(xms=xms, Ms=Ms, My=My,
                                     xtratreatini=True, ntimevecs=ntimevecs)

        if not onlyfwd:
            lyULy, lyitULy = lyUVLy, lyitUVLy
            _, _, lsitULs, lsULs = gpu.\
                get_podbases_wrtmassmats(xms=lms, Ms=Ms, My=My,
                                         xtratreattermi=True,
                                         ntimevecs=ntimevecs)

    else:
        if spacebasscheme == 'combined' and lms is None:
            raise UserWarning('you want `combined` pod bases but there is' +
                              ' no lms -- gonna use `onlyV`')
            cms = xms
        if spacebasscheme == 'onlyV':
            cms = xms
        if spacebasscheme == 'onlyL':
            cms = lms
        lyUVLy, lyitUVLy, lsitUVs, lsUVs = gpu.\
            get_podbases_wrtmassmats(xms=cms, Ms=Ms, My=My,
                                     nspacevecs=nspacevecs,
                                     xtratreatini=True, ntimevecs=ntimevecs)
        lyUVy, lyitUVy = lyUVLy, lyitUVLy
        if not onlyfwd:
            lyULy, lyitULy = lyUVLy, lyitUVLy
            _, _, lsitULs, lsULs = gpu.\
                get_podbases_wrtmassmats(xms=cms, Ms=Ms, My=My,
                                         xtratreattermi=True,
                                         ntimevecs=ntimevecs)

    return lyitUVy, lyUVy, lsitUVs, lsUVs, lyitULy, lyULy, lsitULs, lsULs


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

    print('diff in dJ: ', np.linalg.norm(ndjaco-anajaco))
    print('diff in dJ_11: ', np.linalg.norm(jvvn - jvva))
    print('diff in dJ_12: ', np.linalg.norm(jvln - jvla))
    print('diff in dJ_21: ', np.linalg.norm(jlvn - jlva))
    print('diff in dJ_22: ', np.linalg.norm(jlln - jlla))


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
           adjplotdict=None, target='inival',
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

    if target == 'inival':
        def vstar(t):
            return iniv.flatten()

    elif target == 'heart' or 'invheart':
        invertt = True if target == 'invheart' else False
        myheartfun = get_spcheartfun(NY=Nq-1, invertt=invertt)

        def vstar(t):
            return myheartfun(t).flatten()

    # ### define the model
    x0, xE = dmndct['x0'], dmndct['xE']
    (My, A, rhs, nfunc, femp) = dbs.\
        burgers_spacedisc(N=Nq, nu=nu, x0=x0, xE=xE, retfemdict=True)
    # define the initial value
    iniv = dbs.burger_onedim_inival(inivtype=inivtype, V=femp['V'],
                                    ininds=femp['ininds'])
    # nqbytwo = np.int(np.floor(Nq/2))
    # nqmobytwo = np.int(np.floor((Nq-1)/2))
    # if inivtype == 'smooth':
    #     xrng = np.linspace(0, 2*np.pi, Nq-1)
    #     iniv = 0.5 - 0.5*np.sin(xrng + 0.5*np.pi)
    #     iniv = 0.5*iniv.reshape((Nq-1, 1))
    # elif inivtype == 'step':
    #     # iniv = np.r_[np.ones(((Nq-1)/2, 1)), np.zeros(((Nq)/2, 1))]
    #     # iniv = np.r_[np.zeros((nqbytwo, 1)), np.ones((nqmobytwo, 1))]
    #     iniv = np.r_[np.ones((nqmobytwo, 1)), np.zeros((nqbytwo, 1))]
    # elif inivtype == 'ramp':
    #     iniv = np.r_[np.linspace(0, 1, (nqmobytwo)).reshape((nqmobytwo, 1)),
    #                  np.zeros((nqbytwo, 1))]
    # elif inivtype == 'zero':
    #     iniv = np.zeros((Nq-1, 1))
    # ### compute the forward snapshots

    def fwdrhs(t):
        return rhs
    simudict = dict(iniv=iniv, A=A, M=My, nfunc=nfunc, rhs=fwdrhs, tmesh=tmesh)
    with dou.Timer('fwd'):
        datastr = 'data/fwdsol_iniv' + inivtype + '_target_' + target +\
            'Nq{0}Nts{1}nu{2}'.format(Nq, Nts, nu)
        vv = dou.load_or_comp(filestr=datastr, comprtn=gpu.time_int_semil,
                              arraytype='dense', comprtnargs=simudict,
                              debug=debug)
        # vv = gpu.time_int_semil(**simudict)
    if plotplease:
        plotmat(vv, fignum=1234, tikzfile=fullmodelv_tkzf, **dmndct)

    (xms, Ms) = gpu.get_genmeasuremat(sol=vv.T, tmesh=tmesh, sdim=Ns)

    # XXX: a little crime here
    # Ms[0, 1] = 0
    # Ms[1, 0] = 0
    Ms = sps.csc_matrix(Ms)

    # ### compute the backward snapshots
    if genpodadj or genpodcl or spacebasscheme == 'combined':
        _vfun = interp1d(tmesh, vv, axis=0)  # , fill_value='extrapolate')

        def vfun(t):
            if t < tmesh[0]:
                return _vfun(tmesh[0])
            elif t > tmesh[-1]:
                return _vfun(tmesh[-1])
            else:
                return _vfun(t)

        vdxoperator, fnctnl = dbs.\
            burgers_bwd_spacedisc(V=femp['V'], ininds=femp['ininds'],
                                  diribc=femp['diribc'])
        te = tmesh[-1]

        def burger_bwd_rhs(t):
            # TODO: -----------------------------> here we need vstar
            return -fnctnl(vfun(te-t)).flatten()+fnctnl(vstar(te-t)).flatten()

        def burger_bwd_nonl(lvec, t):
            vdx = vdxoperator(vfun(te-t))
            return -(vdx*lvec).flatten()

        termiL = np.zeros((Nq-1, 1))
        bbwdsimudict = dict(iniv=termiL, A=A, M=My, nfunc=burger_bwd_nonl,
                            rhs=burger_bwd_rhs, tmesh=tmesh)

        with dou.Timer('bwd'):
            datastr = 'data/bwdsol_iniv' + inivtype + '_target' + target +\
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
     lsULs) = gou.stateadjspatibas(xms=xms, lms=Lms, Ms=Ms, My=My,
                                   nspacevecs=hq, ntimevecs=hs,
                                   spacebasscheme=spacebasscheme)
    # (alyitUVy, alyUVy, alsitUVs, alsUVs, alyitULy, alyULy, alsitULs,
    #  alsULs) = stateadjspatibas(xms=xms, lms=Lms, Ms=Ms, My=My,
    #                             nspacevecs=hq, ntimevecs=hs,
    #                             spacebasscheme=spacebasscheme)
    # print('TODO: debug hack')
    # lsitUVs = alsitUVs
    # lsUVs = alsUVs
    # lyitUVy = alyitUVy
    # lyUVy = alyUVy
    # import ipdb; ipdb.set_trace()
    # ### the fwd projection scheme
    AVk, MVk, nonl_red, rhs_red, liftcoef, projcoef =\
        gpu.get_spaprjredmod(M=My, A=A, nonl=nfunc, rhs=fwdrhs,
                             Uk=lyitUVy, prjUk=lyUVy)

    hiniv = projcoef(iniv)
    hconstone = lsUVs.T.sum(axis=1, keepdims=True)  # the constant 1 in hat S
    spatimIniv = np.kron(hiniv, hconstone.T)  # the space-time ini value
    if plotplease:
        plotmat((np.dot(lyitUVy, np.dot(spatimIniv, lsitUVs.T))).T,
                fignum=323, **dmndct)
        hconstonel = lsULs.T.sum(axis=1, keepdims=True)  # the 1 in hat R
        lspatimIniv = np.kron(hiniv, hconstonel.T)  # the space-time ini value
        plotmat((np.dot(lyitULy, np.dot(lspatimIniv, lsitULs.T))).T,
                fignum=322, **dmndct)

    spatimInivvec = gou.Xtoxvec(spatimIniv)
    timeIniv = spatimInivvec[:hq, :]
    # hiniv = spatimInivvec[:hq, :]

    # XXX see above
    # Ms = sps.csc_matrix(gpu.get_ms(sdim=Ns, tmesh=tmesh))
    dms = sps.csc_matrix(gpu.get_dms(sdim=Ns, tmesh=tmesh))
    Vhms = np.dot(lsitUVs.T, Ms*lsitUVs)
    Vhdms = np.dot(lsitUVs.T, dms*lsitUVs)

    locdebug = debug
    # locdebug = True
    # print('assembling the reduced tensor... -- TODO: locdebug is True')
    print('assembling the reduced tensor...')
    datastr = 'data/fwd_iniv' + inivtype + '_tnsr_' + spacebasscheme + \
        '_target_' + target +\
        '_Nts{5}Nq{0}Ns{1}hq{2}hs{3}nu{4}'.format(Nq, Ns, hq, hs, nu, Nts)
    uvvdxl, htittl = bgu.\
        get_burger_tensor(Uky=lyitUVy, Uks=lsitUVs, sdim=Ns, bwd=True,
                          Vhdms=Vhdms, Vhms=Vhms, Vhmy=MVk, Vhay=AVk,
                          tmesh=tmesh, datastr=datastr, debug=locdebug,
                          **femp)

    vres, vresprime = \
        bou.setup_burger_fwdres(Vhdms=Vhdms, Vhms=Vhms, Vhmy=MVk, Vhay=AVk,
                                htittl=htittl, uvvdxl=uvvdxl, hiniv=timeIniv)

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

        print('assembling the bwd reduced tensor...')
        datastr = 'data/bwdtnsr_iniv' + inivtype + '_' + spacebasscheme +\
            '_target_' + target +\
            '_Nts{5}Nq{0}Ns{1}hq{2}hs{3}nu{4}'.format(Nq, Ns, hq, hs, nu, Nts)
        Luvvdxl, Lhtittl = bgu.\
            get_burger_tensor(Uky=lyitULy, bwd=True, Uks=lsitULs,
                              Ukyconv=lyitUVy, Uksconv=lsitUVs, sdim=Ns,
                              tmesh=tmesh, datastr=datastr, debug=locdebug,
                              **femp)

        tgtst = gou.xvectoX(gou.functovec(vstar, snapshottmesh),
                            ns=Ns, nq=Nq-1)

        htgst = np.dot(lyUVy.T, np.dot(tgtst, lsUVs))
        htgstvec = Xtoxvec(htgst)

        hcurst = np.dot(lyUVy.T, np.dot(xms, lsitUVs))
        hcurstvec = Xtoxvec(hcurst)

        htermiL = np.zeros((hq, 1))

        lres, lresprime = bou.\
            setup_burger_bwdres(Lhdms=Lhdms, Lhms=Lhms, Lhmy=Lhmy, Lhay=Lhay,
                                LVhms=LVhms, LVhmy=LVhmy,
                                Lhtittl=Lhtittl, Luvvdxl=Luvvdxl,
                                hiniv=timeIniv, htermiL=htermiL,
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
                               hiniv=timeIniv, htermiL=htermiL)
    # ### the tests
    # # forward problem
    if genpodstate:
        optiniV = np.tile(timeIniv.T, hs-1).T
        # optiniV = spatimInivvec[hq:, :]
        print('solving the optimization problem (state)...')
        sol = gou.spacetimesolve(func=vres, funcjaco=vresprime,
                                 # inival=optiniV,
                                 inival=0*spatimInivvec[hq:],
                                 message='fwd problem - analytical jacobian')
        optiV = np.r_[timeIniv.flatten(), sol]
        optiV = optiV.reshape((hs, hq))
        if plotplease:
            plotmat((np.dot(lyitUVy, np.dot(optiV.T, lsitUVs.T))).T,
                    tikzfile=redmodelV_tkzf, fignum=1233, **dmndct)

    if genpodadj:
        optiniL = np.zeros((hq*(hs-1), 1))
        print('solving the optimization problem (adjoint)...')
        sol = gou.spacetimesolve(func=lres, funcjaco=lresprime, inival=optiniL,
                                 message='bwd problem - analytical jacobian')
        optiL = np.r_[sol, htermiL.flatten()]
        optiL = optiL.reshape((hs, hq))
        if plotplease:
            plotmat((np.dot(lyitULy, np.dot(optiL.T, lsitULs.T))).T,
                    fignum=124, tikzfile=redmodelL_tkzf, **adjplotdict)

    if genpodcl:
        print('solving the optimization problem (fwdbwd)...')
        fwdbwdini = False
        if fwdbwdini:
            hcurst = np.dot(lyUVy.T, np.dot(xms, lsitUVs))
            hcurstvec = gou.Xtoxvec(hcurst)
            # plotmat((np.dot(lyitUVy,
            #                 np.dot(gou.xvectoX(hcurstvec, nq=hq, ns=hs),
            #                        lsitUVs.T))).T, fignum=9999)
            optiniL = np.zeros((hq*(hs-1), 1))
            hcuradj = np.dot(lyULy.T, np.dot(Lms, lsitULs))
            hcuradjvec = gou.Xtoxvec(hcuradj)
            # plotmat((np.dot(lyitULy,
            #                 np.dot(gou.xvectoX(hcuradjvec, nq=hq, ns=hs),
            #                        lsitULs.T))).T, fignum=9998)
            optiniV = hcurstvec[hq:, :]
            optiniL = hcuradjvec[:-hq, :]

        else:
            # optiniV = np.zeros((hq*(hs-1), 1))
            optiniV = spatimInivvec[hq:].reshape((hq*(hs-1), 1))
            optiniL = np.zeros((hq*(hs-1), 1))
        optiniVL = np.vstack([optiniV, optiniL])
        sol, timingsdict = \
            gou.spacetimesolve(func=clres, funcjaco=clresprime,
                               inival=optiniVL,
                               message='optcont problem - analytical jacobian',
                               timerecord=True)
        if onlytimings:
            return timingsdict

        optiV = np.r_[timeIniv.flatten(), sol[:hq*(hs-1)]]
        optiV = optiV.reshape((hs, hq))

        optiL = np.r_[sol[-hq*(hs-1):], htermiL.flatten()]
        optiL = optiL.reshape((hs, hq))

        fulloptiL = (np.dot(lyitULy, np.dot(optiL.T, lsitULs.T))).T
        if plotplease:
            plotmat((np.dot(lyitUVy, np.dot(optiV.T, lsitUVs.T))).T,
                    tikzfile=redoptiv_tkzf, fignum=123, **dmndct)
            plotmat(1./alpha*fulloptiL, tikzfile=redoptlambda_tkzf,
                    fignum=1241, **dmndct)
            plotmat(fulloptiL, tikzfile=redoptlambda_tkzf,
                    fignum=12411, **dmndct)

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

        print('back check...')
        with dou.Timer('check back'):
            vv = gpu.time_int_semil(**simudict)
            if plotplease:
                plotmat(vv, fignum=12341, tikzfile=backcheck_tkzf, **dmndct)
                vvd = vv - tgtst.T

                plotmat(vvd, fignum=12342, **dmndct)

                plotmat(tgtst.T, fignum=12343,
                        tikzfile=tikzprefikz+'zstar', **dmndct)

        valdict = eva_costfun(vopt=Xtoxvec(vv.T), uopt=Xtoxvec(redmodu),
                              qmat=My, rmat=alpha*My, ms=Ms,
                              vstar=vstar, tmesh=snapshottmesh)
        valdict.update(dict(unormsqrd=2*1./alpha*valdict['uterm']))

        logger.info('Value of the cost functional: {0}'.
                    format(valdict['value']))

        return valdict, timingsdict
    else:
        return None, None


if __name__ == '__main__':
    # ### define the problem
    testitdict = \
        dict(Nq=500,  # dimension of the spatial discretization
             Nts=120,  # number of time sampling points
             # t0=0., tE=1.,
             # x0=0., xE=1.,
             # inivtype='step',  # 'step',  # 'ramp', 'smooth'
             inivtype='step',  # 'zero', 'step',  # 'ramp', 'smooth'
             dmndct=dict(tE=1., t0=0., x0=0., xE=1.),  # for the plots
             Ns=120,  # Number of measurement functions=Num of snapshots
             hq=12,  # number of space modes
             hs=12,  # number of time modes
             target='inival',  # 'inival', 'heart'
             # target='heart',  # 'inival', 'heart'
             genpodstate=True,
             genpodcl=True,
             # genpodadj=True,
             # spacebasscheme='VandL',
             spacebasscheme='combined',
             # spacebasscheme='onlyV',
             # spacebasschemes: 'onlyV', 'onlyL', 'VandL', 'combined'
             plotplease=True, tikzplease=False,
             nu=5e-3, alpha=1e-3)

    value, timerinfo = testit(**testitdict)
    print('Back check: value of costfunction: {0}'.format(value['value']))
    print('Back check: value of vterm: {0}'.format(value['vterm']))
