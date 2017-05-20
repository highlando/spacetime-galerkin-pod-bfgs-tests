import numpy as np
from scipy.interpolate import interp1d

import dolfin_navier_scipy.data_output_utils as dou
import sadptprj_riclyap_adi.lin_alg_utils as lau

import dolfin_burgers_scipy as dbs
import gen_pod_utils as gpu
import genpod_opti_utils as gou

from plot_utils import plotmat

import logging
import logging.config
import yaml
logging.config.dictConfig(yaml.load(open('logging.conf', 'r')))
logger = logging.getLogger('basic')
# logger.setLevel('INFO')  # default is DEBUG can be ERROR, INFO
logger.setLevel('INFO')  # default is DEBUG can be ERROR, INFO

debug = False
debug = True


def space_genpod_burger(Nq=None, Nts=None,
                        inivtype=None, dmndct=None, Ns=None, hq=None, hs=None,
                        spacebasscheme=None, nu=None, alpha=None,
                        plotplease=False, tikzprefikz='',
                        adjplotdict=None,
                        onlytimings=False, **kwargs):

    t0, tE = dmndct['t0'], dmndct['tE']

    tmesh = np.linspace(t0, tE, Nts)
    # snapshottmesh = np.linspace(t0, tE, Ns)
    redtmesh = np.linspace(t0, tE, hs)
    # ### define the model
    x0, xE = dmndct['x0'], dmndct['xE']

    (My, A, rhs, nfunc, femp) = dbs.\
        burgers_spacedisc(N=Nq, nu=nu, x0=x0, xE=xE, retfemdict=True)
    iniv = dbs.burger_onedim_inival(Nq=Nq, inivtype=inivtype)

    # ### compute the forward snapshots
    def fwdrhs(t):
        return rhs
    simudict = dict(iniv=iniv, A=A, M=My, nfunc=nfunc, rhs=fwdrhs, tmesh=tmesh)
    with dou.Timer('fwd'):
        datastr = 'data/fwdsol_iniv' + inivtype + \
            'Nq{0}Nts{1}nu{2}'.format(Nq, Nts, nu)
        # vv = gpu.time_int_semil(**simudict)
        vv = dou.load_or_comp(filestr=datastr, comprtn=gpu.time_int_semil,
                              arraytype='dense', comprtnargs=simudict,
                              debug=debug)
    plotmat(vv, fignum=1236, **dmndct)

    (xms, Ms) = gpu.get_genmeasuremat(sol=vv.T, tmesh=tmesh, sdim=Ns)

    # ### compute the backward snapshots
    vfun = interp1d(tmesh, vv, axis=0, fill_value='extrapolate')
    vdxoperator, fnctnl = dbs.\
        burgers_bwd_spacedisc(V=femp['V'], ininds=femp['ininds'],
                              diribc=femp['diribc'])
    te = tmesh[-1]

    def vstar(t):
        return iniv.flatten()

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
        datastr = 'data/bwdsol_iniv' + inivtype + \
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

    # ## the bwd projection scheme
    ALk, MLk, bwdl_red, bwdrhs_red, liftLcoef, projLcoef =\
        gpu.get_spaprjredmod(M=My, A=A, nonl=burger_bwd_nonl,
                             rhs=burger_bwd_rhs, Uk=lyitULy, prjUk=lyULy)
    htermiL = projLcoef(termiL)
    red_vdxop = gpu.get_redmatfunc(matfunc=vdxoperator, ULk=lyitULy,
                                   UVk=lyitUVy)

    # ## the mixed mats
    VLhmy = np.dot(lyitUVy.T, My*lyitULy)
    LVhmy = np.dot(lyitULy.T, My*lyitUVy)

    # ### the tests
    # ## reduced forward problem
    redsimudict = dict(iniv=hiniv, A=AVk, M=MVk, nfunc=nonl_red,
                       rhs=rhs_red, tmesh=redtmesh)
    with dou.Timer('redfwd'):
        print 'solving the reduced  problem (state)...'
        redvv = gpu.time_int_semil(**redsimudict)
    redvv = redvv.reshape((hs, hq))
    plotmat(np.dot(redvv, lyitUVy.T), fignum=1233, **dmndct)
    # ## the reduced fwd as a function
    eva_redfwd = gou.get_eva_fwd(iniv=hiniv, MV=MVk, AV=AVk, MVU=VLhmy,
                                 rhs=rhs_red, nonlfunc=nonl_red,
                                 tmesh=redtmesh)

    rdvvec = eva_redfwd(np.zeros((hq, hs)))
    rdvv = gou.xvectoX(rdvvec, nq=hq, ns=hs)
    plotmat(np.dot(rdvv.T, lyitUVy.T), fignum=1237, **dmndct)

    # ## the reduced bwd problem
    redbwdsimudict = dict(iniv=htermiL, A=ALk, M=MLk, nfunc=bwdl_red,
                          rhs=bwdrhs_red, tmesh=redtmesh)
    with dou.Timer('redbwd'):
        print 'solving the reduced  problem (adjoint)...'
        redll = gpu.time_int_semil(**redbwdsimudict)
    redll = np.flipud(redll)  # flip the to make it forward time
    redll = redll.reshape((hs, hq))
    plotmat(np.dot(redll, lyitULy.T), fignum=2232, **adjplotdict)

    # ## the reduced bwd problem with reduced fwd solution
    redvfun = interp1d(redtmesh, rdvv, axis=1, fill_value='extrapolate')

    def _liftredvfun(t):
        return np.dot(lyitUVy, redvfun(t))

    def _burger_bwd_rhs(t):
        # TODO: -----------------------------> here we need vstar
        return (-fnctnl(_liftredvfun(te-t)).flatten() +
                fnctnl(vstar(te-t)).flatten())

    def _burger_bwd_nonl(lvec, t):
        vdx = vdxoperator(_liftredvfun(te-t))
        return -(vdx*lvec).flatten()

    ALk, MLk, _bwdl_red, _bwdrhs_red, liftLcoef, projLcoef =\
        gpu.get_spaprjredmod(M=My, A=A, nonl=_burger_bwd_nonl,
                             rhs=_burger_bwd_rhs, Uk=lyitULy, prjUk=lyULy)

    _redbwdsimudict = dict(iniv=htermiL, A=ALk, M=MLk, nfunc=_bwdl_red,
                           rhs=_bwdrhs_red, tmesh=redtmesh)
    redll = gpu.time_int_semil(**_redbwdsimudict)
    redll = np.flipud(redll)  # flip the to make it forward time
    redll = redll.reshape((hs, hq))
    plotmat(np.dot(redll, lyitULy.T), fignum=2233, **adjplotdict)

    redvstarvec = gou.functovec(vstar, redtmesh, projcoef=projVcoef)
    redvstar = gou.xvectoX(redvstarvec, nq=hq, ns=hs)
    _redvstarfun = interp1d(redtmesh, redvstar, axis=1,
                            fill_value='extrapolate')

    # ## the reduced bwd problem with reduced fwd solution
    # ## and manual setup of the coeffs

    def redbvdxl(rlvec, t):
        rvdx = red_vdxop(redvfun(te-t))
        return -np.dot(rvdx, rlvec).flatten()

    def _bbwdrhs(t):
        # TODO: -----------------------------> here we need vstar
        return (-lau.mm_dnssps(LVhmy, redvfun(te-t)).flatten() +
                lau.mm_dnssps(LVhmy, _redvstarfun(t)).flatten())

    redbwdsimudict = dict(iniv=htermiL, A=ALk, M=MLk, nfunc=redbvdxl,
                          rhs=_bbwdrhs, tmesh=redtmesh)
    redll = gpu.time_int_semil(**redbwdsimudict)
    redll = np.flipud(redll)  # flip it to make it forward time
    redll = redll.reshape((hs, hq))
    plotmat(np.dot(redll, lyitULy.T), fignum=2234, **adjplotdict)

    eva_redbwd = gou.get_eva_bwd(vstarvec=redvstarvec, MLV=LVhmy, ML=MLk,
                                 AL=ALk, termiL=htermiL, tmesh=redtmesh,
                                 vdxoperator=red_vdxop)

    # ## the reduced bwd problem as function
    rdlvec = eva_redbwd(rdvvec)
    rdll = gou.xvectoX(rdlvec, nq=hq, ns=hs)
    plotmat(np.dot(rdll.T, lyitULy.T), fignum=2235, **adjplotdict)

    return None, None


if __name__ == '__main__':
    # ### define the problem
    plotplease = True
    testitdict = \
        dict(Nq=150,  # dimension of the spatial discretization
             Nts=150,  # number of time sampling points
             # t0=0., tE=1.,
             # x0=0., xE=1.,
             inivtype='step',  # 'ramp', 'smooth'
             dmndct=dict(tE=1., t0=0., x0=0., xE=1., plotplease=plotplease),
             # for the plots
             Ns=70,  # Number of measurement functions=Num of snapshots
             hq=7,  # number of space modes
             hs=8,  # number of time points
             # spacebasschemes: 'onlyV', 'onlyL', 'VandL', 'combined'
             podstate=True, spacebasscheme='combined',
             plotplease=plotplease,
             nu=1e-2, alpha=1e-3)

    value, timerinfo = space_genpod_burger(**testitdict)
