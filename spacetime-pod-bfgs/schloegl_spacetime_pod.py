# ## TODO: normalize `xms`, `lms` for use in `SVD[xms, lms]`
import dolfin
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spsla
from scipy.interpolate import interp1d

import dolfin_navier_scipy.data_output_utils as dou
import spacetime_galerkin_pod.gen_pod_utils as gpu

import spacetime_function_utils as sfu
import spacetime_pod_utils as spu
import genpod_opti_utils as gou

import schloegl_model as sml

import logging
import logging.config
import yaml
logging.config.dictConfig(yaml.load(open('logging.conf', 'r')))
logger = logging.getLogger('basic')
# logger.setLevel('INFO')  # default is DEBUG can be ERROR, INFO
logger.setLevel('INFO')  # default is DEBUG can be ERROR, INFO

debug = True
debug = False
plotplease = True
plotplease = False


def get_schloegl_modls(
    t0=0., tE=2., Nts=100,
    Nq=40, Ns=90,
    hq=10, hs=10,
    nonlpara=.02, mu=1e-2,
    alpha=1e-3,
    # spacebasscheme='VandL',
    # spacebasscheme='combined',
    spacebasscheme='onlyV',
    spacepod=False, spacetimepod=False,
    inivtype='blob',  # 'allzero'
    utrial='allzero',  # 'allone'
    test_jacos=False, test_redfwd=False, test_redbwd=False,
    test_redcl=False,
    plotplease=False,
    test_intp=False, test_misc=False
):

    tmesh = np.linspace(t0, tE, Nts+1)
    msrmtmesh = np.linspace(t0, tE, Ns)
    redtmesh = np.linspace(t0, tE, hs)

    # ## Chap: Forward simulation
    M, A, B, C, rhsa, schlgl_nonl, schlgl_nnl_ptw, schlgl_plotit, femp =\
        sml.schloegl_spacedisc(N=Nq, nonlpara=nonlpara, mu=mu, retfemdict=True)
    V, ininds = femp['V'], femp['ininds']

    if inivtype == 'blob':
        inivalexp = dolfin.\
            Expression('.5*(1+cos(pi*(2*x[0]-1)))*.5*(1+cos(pi*(2*x[1]-1)))',
                       element=femp['V'].ufl_element())
        inivalfunc = dolfin.interpolate(inivalexp, V)
        inivalvec = inivalfunc.vector().array()[ininds].\
            reshape((ininds.size, 1))
    elif inivtype == 'allzero':
        inivalvec = np.zeros((ininds.size, 1))
    else:
        raise NotImplementedError('no such inival type')

    if utrial == 'allzero':

        def _utv(t):
            return 0.*t
    elif utrial == 'allone':

        def _utv(t):
            return 0.*t + 1.
    elif utrial == 'sine':

        def _utv(t):
            return np.sin(t/(tE-t0)*2*np.pi)
    else:
        raise NotImplementedError('no such utrial type')

    def fwdrhs(t):
        return rhsa

    def fwdrhsbutrial(t):
        return rhsa + B.dot(_utv(t))

    # TODO: get Butv right in the red mods

    simudict = dict(iniv=inivalvec, A=A, M=M,
                    nfunc=schlgl_nonl,
                    rhs=fwdrhsbutrial, tmesh=tmesh)

    probstr = 'schlglb{0}n{1}_iniv{2}'.format(nonlpara, mu, inivtype) +\
        '_utrial{0}'.format(utrial) +\
        'Nq{0}Nts{1}tE{2}'.format(Nq, Nts, tE)
    redpstr = 'red_' + probstr + 'hq{0}Ns{1}hs{2}'.format(hq, Ns, hs)

    # ## Chap: the forward data
    with dou.Timer('fwd'):
        datastr = 'data/' + probstr + '_fwd'

        vv = dou.load_or_comp(filestr=datastr, comprtn=gpu.time_int_semil,
                              arraytype='dense', comprtnargs=simudict,
                              debug=debug)

    sfilestr = 'plots/' + probstr + '_fwd.pvd'

    if plotplease:
        schlgl_plotit(sol=vv, sfilestr=sfilestr, tmesh=tmesh)

    # Generalized measurements
    (xms, Ms) = gpu.get_genmeasuremat(sol=vv.T, tmesh=tmesh, sdim=Ns)
    Ms = sps.csc_matrix(Ms)

    # ## Chap: the backward problem
    _vfun = interp1d(tmesh, vv, axis=0)
    _rstrtt = gou.get_restrictt(t0=t0, tE=tE)

    def vfun(t):
        return _vfun(_rstrtt(t))

    theonevec = 0*inivalvec + 1.
    # TODO: debug hack
    # theonevec = 0*inivalvec
    cone = C.dot(theonevec)

    def _adj(t):
        if t < tE/6:
            return 0.
        if t > 5.*tE/6:
            return 1.
        else:
            return (t-tE/6.)/(2./3*tE)

    _vadj = np.vectorize(_adj)

    def vstar(t):
        return .5*(1 - np.cos(_vadj(t)*np.pi))*theonevec

    def cvstar(t):
        return .5*(1 - np.cos(_vadj(t)*np.pi))*cone

    nlvdvop, fnctnl, nlvdvop_ptw = sml.\
        schloegl_bwd_spacedisc(V=femp['V'], ininds=femp['ininds'],
                               nonlpara=nonlpara, diribc=femp['diribc'])

    def schlgl_bwd_rhs(t):
        return -np.dot(C.T, C.dot(vfun(tE-t).flatten()-vstar(tE-t).flatten()))

    termiL = 0*inivalvec
    schlgl_bwd_nonl = gou.get_bwd_nonl(vfun, nlvdvop, reversetime=True, tE=tE)

    bbwdsimudict = dict(iniv=termiL, A=A, M=M,
                        nfunc=schlgl_bwd_nonl,
                        rhs=schlgl_bwd_rhs, tmesh=tmesh)
    with dou.Timer('bwd'):
        datastr = 'data/' + probstr + '_bwd'
        bwdll = dou.\
            load_or_comp(filestr=datastr, comprtn=gpu.time_int_semil,
                         arraytype='dense', comprtnargs=bbwdsimudict,
                         debug=debug)
        # bwdll = gpu.time_int_semil(**bbwdsimudict)
    ll = np.flipud(bwdll)  # flip the to make it forward time
    (lms, _) = gpu.get_genmeasuremat(sol=ll.T, tmesh=tmesh, sdim=Ns)

    # ## Chap: The adjoint problem -- with interpolated nonlinearity
    if test_intp:
        termiL = 0*inivalvec

        schlgl_bwd_nonl_intp = gou.\
            get_bwd_nonl(vfun, nlvdvop_ptw, ptwise=True, M=M,
                         reversetime=True, tE=tE)
        bbwdsimudict = dict(iniv=termiL, A=A, M=M,
                            # TODO: nfunc=None,
                            nfunc=schlgl_bwd_nonl_intp,
                            rhs=schlgl_bwd_rhs, tmesh=tmesh)
        with dou.Timer('bwd - with interpolation'):
            bwdll = gpu.time_int_semil(**bbwdsimudict)
        llintp = np.flipud(bwdll)  # flip the to make it forward time
        sfilestr = 'plots/' + probstr + '_bwd_intpnonl.pvd'
        if plotplease:
            schlgl_plotit(sol=llintp, sfilestr=sfilestr, tmesh=tmesh)
        nrmllllintp = gpu.space_time_norm(spatimvals=ll-llintp, tmesh=tmesh,
                                          spacemmat=M)
        print('|ll - ll_intp|: {0}'.format(nrmllllintp))

    # full interpolating
    def intp_nonl(vvec, t=None):
        return M*schlgl_nnl_ptw(vvec)

    eva_fulfwd = gou.get_eva_fwd(iniv=inivalvec, MV=M, AV=A, B=B, C=C,
                                 rhs=fwdrhs, nonlfunc=schlgl_nonl,
                                 tmesh=tmesh)

    eva_fulfwd_intp = gou.\
        get_eva_fwd(iniv=inivalvec, MV=M, AV=A, B=B, C=C,
                    rhs=fwdrhs, nonlfunc=intp_nonl, tmesh=tmesh)

    fulystarvec = (cvstar(tmesh)).reshape((tmesh.size, 1))
    eva_fulcostfun = \
        gou.get_eva_costfun(tmesh=tmesh, vmat=np.eye(1),
                            rmat=alpha*np.eye(1),
                            ystarvec=fulystarvec, cmat=C,
                            eva_fwd=eva_fulfwd)

    if spacepod:
        # POD Bases
        lyitUVy, lyUVy, _, _, lyitULy, lyULy, _, _ = \
            gou.stateadjspatibas(xms=xms, lms=lms, Ms=Ms, My=M,
                                 nspacevecs=hq, spaceonly=True,
                                 spacebasscheme=spacebasscheme)

        # reduced fwd problem
        AVk, MVk, Bk, nonl_red, rhs_red, liftcoef, projcoef =\
            gpu.get_spaprjredmod(M=M, A=A, B=B, nonl=schlgl_nonl, rhs=fwdrhs,
                                 Uk=lyitUVy, prjUk=lyUVy)

        hiniv = projcoef(inivalvec)

        redsfilestr = 'plots/' + redpstr + '_fwd.pvd'

        # Solve with inflating for the nonlinearity
        if test_redfwd:

            def redfwdrhsbutrial(t):
                return rhs_red(t) + Bk.dot(_utv(t))

            redsimudict = dict(iniv=hiniv, A=AVk, M=MVk, nfunc=nonl_red,
                               rhs=redfwdrhsbutrial, tmesh=tmesh)
            with dou.Timer('reduced fwd'):
                rdvv = gpu.time_int_semil(**redsimudict)
            liftrdvv = np.dot(rdvv, lyitUVy.T)
            diffrdvv = liftrdvv - vv
            print('|vv - redvv|: {0}'.
                  format(gpu.space_time_norm(spatimvals=diffrdvv, tmesh=tmesh,
                                             spacemmat=M)))

        # Solve with inflating and interpolating
        def infl_intp_nonl(vvec, t=None):
            inflvvc = liftcoef(vvec)
            return np.dot(lyitUVy.T, M*schlgl_nnl_ptw(inflvvc))

        # ## DEIM? -- no need -- inflate/interpolate speeds up by factor 60
        if test_intp and test_redfwd:
            redsimudict = dict(iniv=hiniv, A=AVk, M=MVk, nfunc=infl_intp_nonl,
                               rhs=redfwdrhsbutrial, tmesh=tmesh)
            with dou.Timer('reduced fwd -- inflated and interpolated'):
                iirdvv = gpu.time_int_semil(**redsimudict)

            liftiirdvv = np.dot(iirdvv, lyitUVy.T)
            diffrdvv = liftiirdvv - vv
            print('|vv - iiredvv|: {0}'.format
                  (gpu.space_time_norm(spatimvals=diffrdvv, tmesh=tmesh,
                                       spacemmat=M)))
        # reduced bwd problem
        get_red_nldv = gou.get_infl_intp_nldv(lyitULy=lyitULy, lyULy=lyULy,
                                              lyitUVy=lyitUVy,
                                              nonl_ptw=nlvdvop_ptw)
        lredct = (C.dot(lyitULy)).T
        vredc = C.dot(lyitUVy)

        ALk, MLk, BLk, _, _, liftLcoef, projLcoef =\
            gpu.get_spaprjredmod(M=M, A=A, B=B, Uk=lyitULy, prjUk=lyULy,
                                 nonl=None, rhs=None)

        htermiL = projLcoef(termiL)

        # ## chap: costfun and gradient
        ystarvec = (cvstar(redtmesh)).reshape((redtmesh.size, 1))

        eva_redbwd = gou.get_eva_bwd(ML=MLk, AL=ALk,
                                     ystarfun=cvstar, cmattrp=lredct,
                                     cmat=vredc,
                                     termiL=htermiL, tmesh=redtmesh,
                                     vdxoperator=get_red_nldv)

        eva_redfwd = gou.get_eva_fwd(iniv=hiniv, MV=MVk, AV=AVk, B=Bk,
                                     C=vredc,
                                     rhs=rhs_red, nonlfunc=infl_intp_nonl,
                                     tmesh=redtmesh)

        eva_redcostfun, eva_redcostgrad = \
            gou.get_eva_costfun(tmesh=redtmesh, vmat=np.eye(1),
                                rmat=alpha*np.eye(1),
                                bmat=BLk, getcompgrad=True,
                                ystarvec=ystarvec, cmat=vredc,
                                eva_fwd=eva_redfwd, eva_bwd=eva_redbwd)

        if test_redfwd or test_intp:
            Nv, Nu = B.shape
            testuvec = np.ones((redtmesh.size, Nu))
            fulvv = eva_fulfwd(testuvec, utmesh=redtmesh, tmesh=tmesh)
            fulvvV = gou.xvectoX(fulvv, ns=tmesh.size, nq=Nv)

        if test_intp:
            with dou.Timer('fwd as function handle with input: ' +
                           '-- interpolated'):
                intpvv = eva_fulfwd_intp(testuvec, utmesh=redtmesh)
            intpV = gou.xvectoX(intpvv, ns=tmesh.size, nq=Nv)
            nrmvvitpvv = gpu.space_time_norm(spatimvals=(fulvvV-intpV).T,
                                             tmesh=tmesh, spacemmat=M)
            print('|vv - intpvv|: {0}'.format(nrmvvitpvv))

        if test_redfwd:
            with dou.Timer('redfwd as function handle with input: ' +
                           '-- inflated and interpolated'):
                rdvv = eva_redfwd(testuvec, utmesh=redtmesh, tmesh=tmesh)
            rdvvV = gou.xvectoX(rdvv, ns=tmesh.size, nq=hq)
            inflrdvvV = lyitUVy.dot(rdvvV)
            nrmvvrdvv = gpu.space_time_norm(spatimvals=(fulvvV-inflrdvvV).T,
                                            tmesh=tmesh, spacemmat=M)
            print('|vv - rdvv|: {0}'.format(nrmvvrdvv))

        if test_jacos:
            import numdifftools as nd
            uk = np.ones((1, hs))
            myjaco = eva_redcostgrad(gou.Xtoxvec(uk).flatten())
            ndjaco = nd.Jacobian(eva_redcostfun)(gou.Xtoxvec(uk).flatten())
            print('diffnorm of the analytical and the numerical jacobian' +
                  ' `dJ={0}`'.format(np.linalg.norm(myjaco-ndjaco)))
            print(myjaco/ndjaco)

        return (eva_redcostfun, eva_redcostgrad, eva_fulfwd, eva_fulfwd_intp,
                eva_fulcostfun, eva_redfwd, dict(B=B, C=C, redtmesh=redtmesh,
                                                 tmesh=tmesh,
                                                 msrmtmesh=msrmtmesh,
                                                 ystarfun=cvstar))

    # ## Chap: Space-time Galerkin POD
    if spacetimepod:
        (lyitUVy, lyUVy, lsitUVs, lsUVs,
         lyitULy, lyULy, lsitULs, lsULs) = \
            gou.stateadjspatibas(xms=xms, lms=lms, Ms=Ms, My=M,
                                 nspacevecs=hq, ntimevecs=hs,
                                 spacebasscheme='combined')
        dms = sps.csc_matrix(gpu.get_dms(sdim=Ns, tmesh=tmesh))

        # reduced time mats
        Vhms = np.dot(lsitUVs.T, Ms*lsitUVs)
        VLhms = np.dot(lsitUVs.T, Ms*lsitULs)
        Vhdms = np.dot(lsitUVs.T, dms*lsitUVs)

        # reduced space mats fwd prob
        AVk, MVk, BVk, CVk, _, rhs_red, liftcoef, projcoef =\
            gpu.get_spaprjredmod(M=M, A=A, B=B, C=C, rhs=fwdrhs,
                                 Uk=lyitUVy, prjUk=lyUVy)
        hiniv = projcoef(inivalvec)

        # space time inival
        hconstone = lsUVs.T.sum(axis=1, keepdims=True)
        spatimIniv = np.kron(hiniv, hconstone.T)

        if plotplease:
            inflaspatiminiv = np.dot(lyitUVy, np.dot(spatimIniv, lsitUVs.T))
            redsfilestr = 'plots/' + redpstr + '_spatiminival.pvd'
            schlgl_plotit(sol=inflaspatiminiv.T, sfilestr=redsfilestr,
                          tmesh=msrmtmesh)

        spatimInivvec = gou.Xtoxvec(spatimIniv)
        htimeIniv = spatimInivvec[:hq, :]
        # fwdIniv = spatimInivvec[hq:, :]

        schloeglfwdnonl = sfu.\
            get_spacetime_inflt_intrp(nonlfunc=schlgl_nnl_ptw, inival=None,
                                      lyitUVy=lyitUVy, lyUVy=lyUVy,
                                      lsitUVs=lsitUVs, lsUVs=lsUVs)

        if plotplease:
            stvschlgld = np.vstack([schloeglfwdnonl(spatimInivvec)])
            stvschlgldX = gou.xvectoX(stvschlgld, nq=hq, ns=hs)
            inflstvschlgld = np.dot(lyitUVy, np.dot(stvschlgldX, lsitUVs.T))
            cursfilestr = 'plots/' + redpstr + '_inival_schloegled.pvd'
            schlgl_plotit(sol=inflstvschlgld.T, sfilestr=cursfilestr,
                          tmesh=msrmtmesh)

        inflatehxv = sfu.get_appnd_inflt_reshp_func(timeinfl=lsitUVs,
                                                    spaceinfl=lyitUVy,
                                                    timeinival=htimeIniv)

        if test_misc or test_redfwd or test_redbwd:
            hatX = np.dot(np.dot(lyUVy.T, xms), lsitUVs)
            xnoms = spsla.spsolve(Ms, xms.T).T
        if test_misc:
            # ## Chap: some checks
            nrmxspip = np.sqrt(spu.krontimspaproduct(gou.Xtoxvec(xnoms),
                                                     my=M, ms=Ms))
            nrmprjxspi = np.sqrt(spu.krontimspaproduct(gou.Xtoxvec(hatX),
                                                       my=MVk, ms=Vhms))
            nrmxstn = gpu.space_time_norm(spatimvals=vv, tmesh=tmesh,
                                          spacemmat=M)
            print('|x| through space-time inner product: {0}'.format(nrmxspip))
            print('|x| through prjctd space-time inner product: {0}'.
                  format(nrmprjxspi))
            print('|x| through space-time norm: {0}'.format(nrmxstn))

        # ## Chap: genpod for the forward prob
        get_fwdres = sfu.\
            getget_spacetime_fwdres(dms=Vhdms, ms=Vhms, my=MVk, ared=AVk,
                                    msr=VLhms, bmat=BVk,
                                    nfunc=schloeglfwdnonl, rhs=None)

        msrmtutrialvec = (_utv(msrmtmesh).reshape((1, Ns)))
        hutrvec = (msrmtutrialvec.dot(lsULs)).T

        schlgl_fwdres = get_fwdres(uvec=hutrvec, iniv=htimeIniv)
        if test_misc:
            vvvec = gou.Xtoxvec(hatX)
            vvvecinner = vvvec[hq:, :]
            solres = schlgl_fwdres(vvvecinner)
            print('fwd genpod: residual of projected solution: {0}'.
                  format(np.linalg.norm(solres)))

        if test_redfwd:
            print('solving the optimization problem (state)...')
            fwdsol = gou.spacetimesolve(func=schlgl_fwdres,
                                        inival=np.zeros(((hs-1)*hq, )),
                                        # inival=spatimInivvec[hq:],
                                        message='fwd problem - no jacobian')

            if plotplease:
                inflstschlglfwd = inflatehxv(fwdsol)
                cursfilestr = 'plots/' + redpstr + '_fwd_genpod.pvd'
                schlgl_plotit(sol=inflstschlglfwd.T, sfilestr=cursfilestr,
                              tmesh=msrmtmesh)
            cslres = schlgl_fwdres(fwdsol)
            print('residual of computed solution: {0}'.
                  format(np.linalg.norm(cslres)))
            difvgp = gou.Xtoxvec(xnoms - inflatehxv(fwdsol))
            nrmdfvgp = np.sqrt(spu.krontimspaproduct(difvgp, my=M, ms=Ms))
            print('|x - xhat| through space-time inner product: {0}'.
                  format(nrmdfvgp))

        # ## Chap: genpod adjoint

        Lhms = np.dot(lsitULs.T, Ms*lsitULs)
        Lhdms = np.dot(lsitULs.T, dms*lsitULs)
        LVhms = np.dot(lsitULs.T, Ms*lsitUVs)

        Lhay, Lhmy, BLk, CLk, _, _, Lliftcoef, Lprojcoef =\
            gpu.get_spaprjredmod(M=M, A=A, B=B, C=C, Uk=lyitULy, prjUk=lyULy)

        # hvcmat = np.kron(lsitUVs, CVk)
        # hlcmat = np.kron(lsitULs, CLk)
        spaceonlyhtermiL = Lprojcoef(termiL)

        hconstone = lsULs.T.sum(axis=1, keepdims=True)
        spatimIniv = np.kron(spaceonlyhtermiL, hconstone.T)
        spatimInivvec = gou.Xtoxvec(spatimIniv)
        htermiL = spatimInivvec[-hq:, :]

        inflthlv = sfu.\
            get_appnd_inflt_reshp_func(timeinfl=lsitULs, spaceinfl=lyitULy,
                                       timeinival=htermiL, endappend=True)

        msrmtystarvec = (cvstar(msrmtmesh).reshape((1, Ns)))
        hystarvec = msrmtystarvec.dot(lsUVs)
        if test_misc:
            prjinflysvdf = np.linalg.norm(msrmtystarvec
                                          - hystarvec.dot(lsitUVs.T))
            print('prjinfltd ystarvec: {0}'.format(prjinflysvdf))

        get_bwdres = sfu.\
            getget_spacetime_bwdres(lsitUVs=lsitUVs, lyitUVy=lyitUVy,
                                    lyitULy=lyitULy, lyULy=lyULy,
                                    lsitULs=lsitULs, lsULs=lsULs,
                                    # termiv=htermiL,
                                    # iniv=htimeIniv,
                                    ndv_ptw=nlvdvop_ptw,
                                    ctrp=CLk.T, cmat=CVk,
                                    ystarvec=hystarvec,
                                    dms=Lhdms, ms=Lhms, my=Lhmy, ared=Lhay,
                                    mrs=LVhms)

        if plotplease:
            inflaspatiminiv = np.dot(lyitULy, np.dot(spatimIniv, lsitULs.T))
            redsfilestr = 'plots/' + redpstr + '_spatiminival_bwd.pvd'
            schlgl_plotit(sol=inflaspatiminiv.T, sfilestr=redsfilestr,
                          tmesh=msrmtmesh)

        if plotplease:
            hatL = np.dot(np.dot(lyULy.T, lms), lsitULs)
            prjinflasol = np.dot(lyitULy, np.dot(hatL, lsitULs.T))
            redsfilestr = 'plots/' + redpstr + '_prjinfl_bwd.pvd'
            schlgl_plotit(sol=prjinflasol.T, sfilestr=redsfilestr,
                          tmesh=msrmtmesh)

        if test_redbwd:
            schlgl_bwdres = get_bwdres(sfu.Xtoxvec(hatX))
            lnoms = spsla.spsolve(Ms, lms.T).T
            llvec = gou.Xtoxvec(hatL)
            llvecinner = llvec[:-hq, ]
            solres = schlgl_bwdres(llvec)
            print('bwd genpod: residual of projected solution: {0}'.
                  format(np.linalg.norm(solres)))
            print('solving the optimization problem (adjoint)...')
            bwdsol = gou.spacetimesolve(func=schlgl_bwdres,
                                        # inival=spatimInivvec[:-hq],
                                        inival=np.zeros(((hs-1)*hq, )),
                                        message='bwd problem - no jacobian')

            if plotplease:
                inflstschlgl = inflthlv(bwdsol)
                cursfilestr = 'plots/' + redpstr + '_bwd_genpod.pvd'
                schlgl_plotit(sol=inflstschlgl.T, sfilestr=cursfilestr,
                              tmesh=msrmtmesh)

            lnoms = spsla.spsolve(Ms, lms.T).T
            prjinflasol = np.dot(lyitULy, np.dot(hatL, lsitULs.T))

            hatL = np.dot(np.dot(lyULy.T, lms), lsitULs)
            llvec = gou.Xtoxvec(hatL)

            llvecinner = llvec[:-hq, ]
            solres = schlgl_bwdres(llvecinner)
            cslres = schlgl_bwdres(bwdsol)
            print('residual of projected solution: {0}'.
                  format(np.linalg.norm(solres)))
            print('residual of computed solution: {0}'.
                  format(np.linalg.norm(cslres)))

            hatl = gou.Xtoxvec(inflstschlgl)
            diflgp = gou.Xtoxvec(lnoms - inflstschlgl)
            difprjinfl = gou.Xtoxvec(lnoms - prjinflasol)

            nrmdfprjinfl = np.sqrt(spu.krontimspaproduct(difprjinfl,
                                                         my=M, ms=Ms))
            print('|bwdsol - infllhat| through space-time inner product: {0}'.
                  format(nrmdfprjinfl))

            nrmdflgp = np.sqrt(spu.krontimspaproduct(diflgp, my=M, ms=Ms))
            print('|l - lhat| through space-time inner product: {0}'.
                  format(nrmdflgp))
            nrmhtl = np.sqrt(spu.krontimspaproduct(hatl, my=M, ms=Ms))
            print('|lhat| through space-time inner product: {0}'.
                  format(nrmhtl))

        clres = sfu.get_clres(get_fwdres=get_fwdres, get_bwdres=get_bwdres,
                              bmattrp=BLk.T, rmo=1./alpha,
                              htermiL=htermiL, hiniv=htimeIniv,
                              hsv=hs, hqv=hq, hsl=hs, hql=hq)
        if test_redcl:
            print('`(v,l) = 0` --> ini value res norm {0}:'.
                  format(np.linalg.norm(clres(np.zeros((2*(hs-1)*hq, ))))))
        clres = sfu.get_clres(get_fwdres=get_fwdres, get_bwdres=get_bwdres,
                              bmattrp=BLk.T, rmo=1./alpha,
                              htermiL=htermiL, hiniv=htimeIniv,
                              hsv=hs, hqv=hq, hsl=hs, hql=hq)
        if test_redcl:
            print('`(v,l) = 0` --> ini value res norm {0}:'.
                  format(np.linalg.norm(clres(np.zeros((2*(hs-1)*hq, ))))))
            gou.spacetimesolve(func=clres,
                               inival=np.zeros((2*(hs-1)*hq, )),
                               message='cl problem - no jacobian')
        return (clres, eva_fulfwd, eva_fulfwd_intp, eva_fulcostfun,
                dict(rmo=1./alpha, bmattrp=BLk.T, C=C, CVk=CVk, B=B,
                     inflatehxv=inflatehxv, inflatehxl=inflthlv,
                     htermil=htermiL, hiniv=hiniv,
                     lsitULs=lsitULs, lsULs=lsULs,
                     msrmtmesh=msrmtmesh, redtmesh=redtmesh, tmesh=tmesh,
                     ystarfun=cvstar, get_fwdres=get_fwdres
                     ))

if __name__ == '__main__':
    spacepod = False
    spacetimepod = False
    test_redfwd = False
    test_redbwd = False
    test_redcl = False
    test_intp = False
    test_jacos = False
    test_misc = False

    # ## make it come true
    # debug = True
    # spacepod = True
    spacetimepod = True
    # test_redfwd = True
    # test_redbwd = True
    test_redcl = True
    test_intp = True
    plotplease = True
    test_jacos = True
    test_misc = True

    t0, tE, Nts = 0., 2., 100
    Nq, Ns = 25, 20

    hq, hs = 20, 10
    alpha = 1e-3

    utrial = 'allzero'
    utrial = 'allone'
    utrial = 'sine'

    inivtype = 'blob'
    inivtype = 'allzero'

    spacebasscheme = 'VandL'
    spacebasscheme = 'combined'

    get_schloegl_modls(Nq=Nq, Ns=Ns,
                       test_redfwd=test_redfwd, test_redbwd=test_redbwd,
                       test_redcl=test_redcl,
                       plotplease=plotplease,
                       test_jacos=test_jacos,
                       test_misc=test_misc,
                       utrial=utrial, inivtype=inivtype,
                       spacepod=spacepod, spacetimepod=spacetimepod,
                       spacebasscheme=spacebasscheme,
                       test_intp=test_intp)
