import numpy as np
from scipy.interpolate import interp1d

from scipy.optimize import fmin_bfgs

import dolfin_navier_scipy.data_output_utils as dou

import dolfin_burgers_scipy as dbs
import gen_pod_utils as gpu
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


def bfgs_opti(Nq=None, Nts=None,
              inivtype=None, dmndct=None, Ns=None, hq=None, hs=None,
              redrtol=None, redatol=None,
              spacebasscheme=None, nu=None, alpha=None,
              podstate=False, podadj=False, genpodcl=False,
              plotplease=None, tikzplease=False, tikzprefikz='',
              adjplotdict=None, bfgsiters=None, gtol=1e-5,
              target='inival',
              checkbwdtimings=False, onlytimings=False, **kwargs):

    dmndct.update(dict(plotplease=plotplease))

    t0, tE = dmndct['t0'], dmndct['tE']

    tmesh = np.linspace(t0, tE, Nts)
    # snapshottmesh = np.linspace(t0, tE, Ns)
    redtmesh = np.linspace(t0, tE, hs)
    # ### define the model

    if target == 'inival':
        def vstar(t):
            return iniv.flatten()

    elif target == 'heart' or 'invheart':
        invertt = True if target == 'invheart' else False
        myheartfun = get_spcheartfun(NY=Nq-1, invertt=invertt)

        def vstar(t):
            return myheartfun(t).flatten()

    x0, xE = dmndct['x0'], dmndct['xE']

    (My, A, rhs, nfunc, femp) = dbs.\
        burgers_spacedisc(N=Nq, nu=nu, x0=x0, xE=xE, retfemdict=True)
    iniv = dbs.burger_onedim_inival(Nq=Nq, inivtype=inivtype)

    # ### compute the forward snapshots
    def fwdrhs(t):
        return rhs
    simudict = dict(iniv=iniv, A=A, M=My, nfunc=nfunc, rhs=fwdrhs, tmesh=tmesh)
    with dou.Timer('fwd'):
        datastr = 'data/fwdsol_iniv' + inivtype + '_target' + target +\
            'Nq{0}Nts{1}nu{2}'.format(Nq, Nts, nu)
        # vv = gpu.time_int_semil(**simudict)
        vv = dou.load_or_comp(filestr=datastr, comprtn=gpu.time_int_semil,
                              arraytype='dense', comprtnargs=simudict,
                              debug=debug)
    plotmat(vv, fignum=1234, **dmndct)

    (xms, Ms) = gpu.get_genmeasuremat(sol=vv.T, tmesh=tmesh, sdim=Ns)
    # ### compute the backward snapshots
    vfun = interp1d(tmesh, vv, axis=0, fill_value='extrapolate')
    vdxoperator, fnctnl = dbs.\
        burgers_bwd_spacedisc(V=femp['V'], ininds=femp['ininds'],
                              diribc=femp['diribc'])
    te = tmesh[-1]

    def burger_bwd_rhs(t):
        # TODO: -----------------------------> here we need vstar
        return -fnctnl(vfun(te-t)).flatten() + fnctnl(vstar(te-t)).flatten()

    def burger_bwd_nonl(lvec, t):
        vdx = vdxoperator(vfun(te-t))
        return -(vdx*lvec).flatten()

    termiL = np.zeros((Nq-1, 1))
    bbwdsimudict = dict(iniv=termiL, A=A, M=My, nfunc=burger_bwd_nonl,
                        rhs=burger_bwd_rhs, tmesh=tmesh)

    with dou.Timer('bwd'):
        datastr = 'data/bwdsol_iniv' + inivtype + '_target' + target + \
            'Nq{0}Nts{1}nu{2}'.format(Nq, Nts, nu)
        bwdll = dou.\
            load_or_comp(filestr=datastr, comprtn=gpu.time_int_semil,
                         arraytype='dense', comprtnargs=bbwdsimudict,
                         debug=debug)
        # bwdll = gpu.time_int_semil(**bbwdsimudict)
    ll = np.flipud(bwdll)  # flip the to make it forward time
    if adjplotdict is None:
        adjplotdict = dmndct
    plotmat(ll, fignum=1235, **adjplotdict)

    (Lms, _) = gpu.get_genmeasuremat(sol=ll.T, tmesh=tmesh, sdim=Ns)

    # ### compute the projection matrices, i.e. optimal bases
    (lyitUVy, lyUVy, lsitUVs, lsUVs, lyitULy, lyULy, lsitULs, lsULs) = gou.\
        stateadjspatibas(xms=xms, lms=Lms, Ms=Ms, My=My, nspacevecs=hq,
                         ntimevecs=hs, spacebasscheme=spacebasscheme)
    # ## the fwd projection scheme
    AVk, MVk, nonl_red, rhs_red, liftVcoef, projVcoef =\
        gpu.get_spaprjredmod(M=My, A=A, nonl=nfunc, rhs=fwdrhs,
                             Uk=lyitUVy, prjUk=lyUVy)
    hiniv = projVcoef(iniv)
    # ## preassemble reduced nonlinearity as tensor
    print 'assembling the reduced tensor...'
    import burgers_genpod_utils as bgu
    datastr = 'data/fwd_iniv' + inivtype + '_tnsr_' + spacebasscheme + \
        '_target_' + target +\
        '_Nts{4}Nq{0}Ns{1}hq{2}nu{3}'.format(Nq, Ns, hq, nu, Nts)
    uvvdxl, _ = bgu.\
        get_burger_tensor(Uky=lyitUVy, Uks=lsitUVs, bwd=True, spaceonly=True,
                          # Vhdms=None, Vhms=None, Vhmy=MVk, Vhay=AVk,
                          # sdim=Ns, tmesh=tmesh,
                          datastr=datastr, debug=debug, **femp)

    def redfwdnln_tns(vvec, t):
        return bgu.eva_burger_spacecomp(uvvdxl=uvvdxl, svec=vvec).flatten()

    # ## the bwd projection scheme
    ALk, MLk, _, _, liftLcoef, projLcoef =\
        gpu.get_spaprjredmod(M=My, A=A, nonl=burger_bwd_nonl,
                             rhs=burger_bwd_rhs, Uk=lyitULy, prjUk=lyULy)
    htermiL = projLcoef(termiL)

    red_vdxop = gpu.get_redmatfunc(matfunc=vdxoperator, ULk=lyitULy,
                                   UVk=lyitUVy)

    print 'assembling the bwd reduced tensor...'
    datastr = 'data/bwdtnsr_iniv' + inivtype + '_' + spacebasscheme +\
        '_target_' + target +\
        '_Nts{4}Nq{0}Ns{1}hq{2}nu{3}'.format(Nq, Ns, hq, nu, Nts)
    Luvvdxl, _ = bgu.\
        get_burger_tensor(Uky=lyitULy, bwd=True, spaceonly=True,
                          Ukyconv=lyitUVy,
                          datastr=datastr, debug=debug,
                          **femp)

    # ## the mixed matrices
    VLhmy = np.dot(lyitUVy.T, My*lyitULy)
    LVhmy = np.dot(lyitULy.T, My*lyitUVy)

    redvstarvec = gou.functovec(vstar, redtmesh, projcoef=projVcoef)
    vstarvec = gou.functovec(vstar, tmesh)

    eva_redfwd = gou.get_eva_fwd(iniv=hiniv, MV=MVk, AV=AVk, MVU=VLhmy,
                                 rhs=rhs_red, nonlfunc=redfwdnln_tns,
                                 # nonlfunc=nonl_red,
                                 solvrtol=redrtol, solvatol=redatol,
                                 tmesh=redtmesh)

    eva_redbwd = gou.get_eva_bwd(vstarvec=redvstarvec, MLV=LVhmy, ML=MLk,
                                 AL=ALk, termiL=htermiL, tmesh=redtmesh,
                                 bwdvltens=Luvvdxl,
                                 solvrtol=redrtol, solvatol=redatol,
                                 vdxoperator=red_vdxop)
    if checkbwdtimings:
        rdvvec = eva_redfwd(np.zeros((hq, hs)))
        return eva_redbwd, rdvvec

    eva_liftufwd = gou.get_eva_fwd(iniv=iniv, MV=My, AV=A, MVU=My,
                                   rhs=fwdrhs, nonlfunc=nfunc,
                                   tmesh=tmesh, redtmesh=redtmesh,
                                   liftUcoef=liftLcoef)

    eva_redcostfun, eva_redcostgrad = \
        gou.get_eva_costfun(tmesh=redtmesh, MVs=MVk, MUs=alpha*MLk,
                            vstarvec=redvstarvec, getcompgrad=True,
                            eva_fwd=eva_redfwd, eva_bwd=eva_redbwd)

    eva_liftucostfun = \
        gou.get_eva_costfun(tmesh=tmesh, MVs=My,
                            utmesh=redtmesh, MUs=alpha*MLk,
                            vstarvec=vstarvec, eva_fwd=eva_liftufwd)

    checktest = True
    checktest = False
    if checktest:
        dmndct.update(dict(plotplease=True))
        rdvvec = eva_redfwd(np.zeros((hq, hs)))
        rdvv = gou.xvectoX(rdvvec, nq=hq, ns=hs)
        plotmat(np.dot(lyitUVy, rdvv).T, fignum=2135, **dmndct)

        rdlvec = eva_redbwd(rdvvec)
        rdll = gou.xvectoX(rdlvec, nq=hq, ns=hs)
        plotmat(np.dot(lyitULy, rdll).T, fignum=2136, **dmndct)

        plotmat(np.dot(lyitUVy, gou.xvectoX(redvstarvec, ns=hs, nq=hq)).T,
                fignum=324, **dmndct)
        checktestjaco = True
        checktestjaco = False
        if checktestjaco:
            import numdifftools as nd
            uk = gou.Xtoxvec(np.zeros((hq, hs)))
            myjaco = eva_redcostgrad(gou.Xtoxvec(uk).flatten())
            ndjaco = nd.Jacobian(eva_redcostfun)(gou.Xtoxvec(uk).flatten())
            print 'diffnorm of the analytical and the numerical jacobian' +\
                  ' `dJ={0}`'.format(np.linalg.norm(myjaco-ndjaco))

    uk = gou.Xtoxvec(np.zeros((hq, hs)))

    tfd = {}  # if timerecord else None
    print 'solving the reduced optimization problem'

    profileit = False
    if profileit:
        from pycallgraph import PyCallGraph
        from pycallgraph.output import GraphvizOutput
        with PyCallGraph(output=GraphvizOutput()):
            uopt, fopt, gopt, Bopt, nfc, ngc, wflag = \
                fmin_bfgs(eva_redcostfun, gou.Xtoxvec(uk), full_output=True,
                          fprime=eva_redcostgrad, maxiter=bfgsiters, gtol=gtol)
    with dou.Timer('bfgs optimization', timerinfo=tfd):
        uopt, fopt, gopt, Bopt, nfc, ngc, wflag = \
            fmin_bfgs(eva_redcostfun, gou.Xtoxvec(uk), full_output=True,
                      fprime=eva_redcostgrad, maxiter=bfgsiters, gtol=gtol)
        tfd.update(dict(nfc=nfc, ngc=ngc))

    if onlytimings:
        minfcallgcall = min(nfc, ngc)  # minimum of func/grad calls
        tfwdclt = []
        tfwdclo = []
        dumfwd = 0*uk
        for k in range(3):
            ltfd = {}
            with dou.Timer('', verbose=False, timerinfo=ltfd):
                dumfwd = eva_redfwd(uopt)
            tfwdclo.append(ltfd['elt'])
            ltfd = {}
            with dou.Timer('', verbose=False, timerinfo=ltfd):
                dumfwd = eva_redfwd(gou.Xtoxvec(uk))
            tfwdclt.append(ltfd['elt'])
        dumfwd = 0*dumfwd
        medfwdcall = 0.5*(np.median(np.array(tfwdclo)) +
                          np.median(np.array(tfwdclt)))
        funccalloverhead = minfcallgcall*medfwdcall
        tfd.update(dict(overhead=funccalloverhead))

        return tfd

    else:
        print 'evaluating the full cost function'
        fulcostvpart, fulcostupart = eva_liftucostfun(uopt, retparts=True)

        if plotplease:
            vvecopt = eva_liftufwd(uopt)
            vopt = gou.xvectoX(vvecopt, nq=Nq-1, ns=Nts)
            plotmat(vopt.T, fignum=3234, **dmndct)

        return dict(vterm=fulcostvpart,
                    uterm=fulcostupart,
                    value=fulcostupart+fulcostvpart,
                    unormsqrd=2*1./alpha*fulcostupart), tfd


if __name__ == '__main__':
    # ### define the problem
    plotplease = True
    testitdict = \
        dict(Nq=220,  # dimension of the spatial discretization
             Nts=250,  # number of time sampling points
             # t0=0., tE=1.,
             # x0=0., xE=1.,
             inivtype='step',  # 'ramp', 'smooth'
             dmndct=dict(tE=1., t0=0., x0=0., xE=1.),
             # for the plots
             Ns=120,  # Number of measurement functions=Num of snapshots
             hq=15,  # number of space modes
             hs=15,  # number of time points
             redrtol=1e-5, redatol=1e-5,
             bfgsiters=100,
             # spacebasschemes: 'onlyV', 'onlyL', 'VandL', 'combined'
             podstate=False, spacebasscheme='combined',
             podadj=False,
             # target='heart',  # 'invheart',  # 'inival', 'heart'
             target='inival',
             tikzplease=False, plotplease=plotplease,
             nu=5e-3, alpha=6.25e-5)

    value, timerinfo = bfgs_opti(**testitdict)

    print 'Back check: value of costfunction: {0}'.format(value['value'])
    print 'Back check: value of vterm: {0}'.format(value['vterm'])
