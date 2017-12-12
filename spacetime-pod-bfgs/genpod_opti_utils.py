import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spsla

from scipy.optimize import fsolve
import scipy.optimize as sco

import spacetime_pod_utils as spu

from scipy.interpolate import interp1d
import dolfin_navier_scipy.data_output_utils as dou

import spacetime_galerkin_pod.gen_pod_utils as gpu

import logging
logger = logging.getLogger('basic')

try:
    import numdifftools as nd
except ImportError:
    print('can not import numdifftool -- can do without')


__all__ = ['stateadjspatibas',
           'get_eva_fwd',
           'get_eva_bwd',
           'eva_costfun',
           'get_eva_costfun',
           'Xtoxvec',
           'xvectoX',
           'numerical_jacobian',
           'costfun',
           'functovec',
           'get_bwd_nonl']


# helper functions
def get_restrictt(t0=None, tE=None):
    def rstt(t):
        return np.max([t0, np.min([t, tE])])
    return rstt


def stateadjspatibas(xms=None, lms=None, Ms=None, My=None, onlyfwd=False,
                     spaceonly=False,
                     nspacevecs=None, ntimevecs=None, spacebasscheme='VandL'):

    gpw = gpu.get_podbases_wrtmassmats
    lyULy, lyitULy, lsitULs, lsULs = None, None, None, None
    if spacebasscheme == 'VandL':
        if spaceonly:
            lyitUVy, lyUVy, _, _ = gpw(xms=xms, My=My, Ms=Ms,
                                       nspacevecs=nspacevecs, ntimevecs=0)
            lyitULy, lyULy, _, _ = gpw(xms=lms, My=My, Ms=Ms,
                                       nspacevecs=nspacevecs, ntimevecs=0)
            return lyitUVy, lyUVy, None, None, lyitULy, lyULy, None, None
        else:
            lyitUVy, lyUVy, lsitUVs, lsUVs = \
                gpw(xms=xms, My=My, Ms=Ms,
                    nspacevecs=nspacevecs, ntimevecs=ntimevecs,
                    xtratreatini=True)
            lyitULy, lyULy, lsitULs, lsULs = \
                gpw(xms=lms, My=My, Ms=Ms,
                    nspacevecs=nspacevecs, ntimevecs=ntimevecs,
                    xtratreattermi=True)
            return (lyitUVy, lyUVy, lsitUVs, lsUVs,
                    lyitULy, lyULy, lsitULs, lsULs)

    elif spacebasscheme == 'onlyV':
        cms = xms
    elif spacebasscheme == 'onlyL':
        cms = lms
    elif spacebasscheme == 'combined':
        cms = [xms, lms]
    if spaceonly:
        lyitUVLy, lyUVLy, _, _ = gpw(xms=cms, My=My, Ms=Ms,
                                     nspacevecs=nspacevecs, ntimevecs=0)
        lyUVy, lyitUVy, lyULy, lyitULy = lyUVLy, lyitUVLy, lyUVLy, lyitUVLy
        return lyitUVy, lyUVy, None, None, lyitULy, lyULy, None, None
    else:
        lyitUVy, lyUVy, lsitUVs, lsUVs = \
            gpw(xms=cms, My=My, Ms=Ms,
                nspacevecs=nspacevecs, ntimevecs=ntimevecs,
                xtratreatini=True)
        lyitULy, lyULy, lsitULs, lsULs = \
            gpw(xms=cms, My=My, Ms=Ms,
                nspacevecs=nspacevecs, ntimevecs=ntimevecs,
                xtratreattermi=True)
        # print('TODO: debug hack')
        # return lyitUVy, lyUVy, lsitUVs, lsUVs, None, None, None, None
        return lyitUVy, lyUVy, lsitUVs, lsUVs, lyitULy, lyULy, lsitULs, lsULs


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


def Xtoxvec(X):
    ''' X - (nq, ns) array to (ns*nq, 1) array by stacking the columns '''
    return X.T.reshape((X.size, 1))


def xvectoX(xvec, nq=None, ns=None):
    ''' xvec - (ns*nq, 1) vector to (nq, ns) by writing columnwise '''
    return xvec.reshape((ns, nq)).T


def numerical_jacobian(func):
    def ndjaco(tvvec):
        ndjc = nd.Jacobian(func)(tvvec)
        return ndjc
    return ndjaco


def trapz_int_mat(tmesh):
    hvec = np.zeros((tmesh.size+1, 1))
    hvec[1:-1, 0] = tmesh[1:] - tmesh[:-1]
    trpdiag = .5*(hvec[:-1]+hvec[1:])
    try:
        Mtrpz = sps.diags(trpdiag.flatten())
    except TypeError:
        Mtrpz = sps.diags(trpdiag.flatten(), 0)
    return Mtrpz


def costfun(vmat=None, MYt=None, rmat=None, MUt=None,
            bmat=None,
            tmesh=None, utmesh=None,
            vstarvec=None, ystarvec=None, cmat=None):
    ''' Define the costfunctionals

    Parameters:
    ---
    MYt, MUt: {(hs, hs) sparse array, 'trapez'}
        mass matrix of the time discretization,
        if `'trapez'`, then the piecewise trapezoidal rule is used
    '''
    if utmesh is None:
        utmesh = tmesh

    if MUt == 'trapez':
        MUt = trapz_int_mat(utmesh)
    if MYt == 'trapez':
        MYt = trapz_int_mat(tmesh)

    locns = MUt.shape[0]  # time dimension
    try:
        locnq = cmat.shape[1]  # space dimension
    except AttributeError:
        locnq = vmat.shape[0]

    def evacostfun(vvec, uvec, utmesh=None, retparts=False):
        if utmesh is not None:
            lMUt = trapz_int_mat(utmesh)
        else:
            lMUt = MUt

        if ystarvec is None:
            diffv = vvec - vstarvec
        else:
            diffv = Xtoxvec(cmat.dot(xvectoX(vvec, ns=locns,
                                             nq=locnq))) - ystarvec
        vterm = 0.5*spu.krontimspaproduct(diffv, ms=MYt, my=vmat)
        uterm = 0.5*spu.krontimspaproduct(uvec, ms=lMUt, my=rmat)
        if retparts:
            return vterm.flatten()[0], uterm.flatten()[0]
        else:
            return uterm.flatten()[0] + vterm.flatten()[0]

    def compcfgrad(lvec=None, uvec=None):
        # TODO: why is there a minus <-- bc we need the negative gradient
        # TODO: what is bmat
        return -(spu.krontimspamatvec(lvec, ms=MYt, my=bmat.T) +
                 spu.krontimspamatvec(uvec, ms=MUt, my=rmat))

    return evacostfun, compcfgrad


def functovec(func, tmesh, projcoef=None):
    if projcoef is None:
        prfunc = func
    else:
        def prfunc(t):
            try:
                return projcoef(func(t))
            except TypeError:
                return np.dot(projcoef, func(t))
    finiv = prfunc(tmesh[0])
    dimv = finiv.size
    funvlst = [finiv.reshape((dimv, 1))]
    for tk in tmesh[1:]:
        funvlst.append(prfunc(tk).reshape((dimv, 1)))
    return np.vstack(funvlst)


def get_eva_costfun(tmesh=None, vmat=None, rmat=None, bmat=None,
                    MVt='trapez', MUt='trapez',
                    ystarvec=None, cmat=None,
                    vstarvec=None, utmesh=None,
                    eva_bwd=None, eva_fwd=None, getcompgrad=False):
    ''' get a function that evaluates J(u)=J(v(u),u)'''

    def _sparsifymat(mmat):
        if sps.issparse(mmat):
            return mmat
        nlfmmat = np.copy(mmat)
        nlfmmat[np.abs(nlfmmat) < 1e-15] = 0
        return sps.csr_matrix(nlfmmat)

    # ## remove zeros and make it sparse for fast evaluation
    vmat = _sparsifymat(vmat)
    rmat = _sparsifymat(rmat)

    _costfun, _costgrad = costfun(vmat=vmat, MYt=MVt, rmat=rmat, MUt=MUt,
                                  bmat=bmat,
                                  tmesh=tmesh, utmesh=utmesh,
                                  cmat=cmat, ystarvec=ystarvec,
                                  vstarvec=vstarvec)

    def eva_costfun(uvec, inival=None, vvec=None, retparts=False, utmesh=None):
        if vvec is None:
            vvec = eva_fwd(uvec, utmesh=utmesh)
        return _costfun(vvec, uvec.flatten(), utmesh=utmesh, retparts=retparts)

    if getcompgrad:
        def eva_costgrad(uvec, inival=None, vvec=None, lvec=None):
            if lvec is None:
                if vvec is None:
                    vvec = eva_fwd(uvec, inival=None)
                lvec = eva_bwd(vvec)
            return _costgrad(lvec=lvec.reshape((lvec.size, 1)),
                             uvec=uvec.reshape((uvec.size, 1))).flatten()

        return eva_costfun, eva_costgrad

    else:
        return eva_costfun


def get_eva_fwd(iniv=None, MV=None, AV=None, B=None, rhs=None,
                C=None,
                solvrtol=None, solvatol=None,
                nonlfunc=None, tmesh=None, utmesh=None, liftUcoef=None):
    ''' set up the function that evaluates the fwd problem
    '''
    tE = tmesh[-1]
    # NQ = MV.shape[0]
    # NS = tmesh.size
    if utmesh is None:
        gutmesh = tmesh
    else:
        gutmesh = utmesh
    gtmesh = tmesh
    giniv = iniv

    def eva_fwd(uvec, tmesh=None, rety=False,
                ufun=None, utmesh=None,
                debug=False, inival=None):

        if utmesh is not None:
            cutmesh = utmesh
        else:
            cutmesh = gutmesh
        if tmesh is not None:
            ltmesh = tmesh
        else:
            ltmesh = gtmesh
        ciniv = inival if inival is not None else giniv

        ns = cutmesh.size

        if ufun is None:
            nu = B.shape[1]
            U = xvectoX(uvec, nq=nu, ns=ns)
            _ufun = interp1d(cutmesh, U, axis=1)
            if liftUcoef is not None:
                def ufun(t):
                    return liftUcoef(_ufun(t))
            else:
                ufun = _ufun

        def _contrl_rhs(t):
            if t > tE:  # the integrator may require values outside [t0, tE]
                return rhs(tE).flatten()+(B.dot(ufun(tE))).flatten()
            else:
                return rhs(t).flatten()+(B.dot(ufun(t))).flatten()

        simudict = dict(iniv=ciniv, A=AV, M=MV, nfunc=nonlfunc,
                        rhs=_contrl_rhs, tmesh=ltmesh,
                        rtol=solvrtol, atol=solvatol)

        vv = gpu.time_int_semil(**simudict)

        # vv = vv.reshape((NS, NQ))
        if rety:
            vv = (C.dot(vv.T)).T

        if debug:
            return Xtoxvec(vv.T), simudict
        else:
            return Xtoxvec(vv.T)

    return eva_fwd


def get_eva_bwd(ML=None, AL=None, vmat=None,
                vstarvec=None, vstarfun=None,
                ystarfun=None, cmattrp=None, cmat=None,
                solvrtol=None, solvatol=None, bwdvltens=None,
                termiL=None, tmesh=None,
                vdxoperator=None, vdxlinop=None):
    ''' setup the backward problem

    Parameters:
    ---

    vdxoperator: callable
        a function that returns `vdx = N(v(t))_v` ---
        the derivative of the forward problem nonlinearity --
        to compute `vdx.dot(l)`
    '''

    nq, ns, tE = ML.shape[1], tmesh.size, tmesh[-1]
    _rstt = get_restrictt(t0=tmesh[0], tE=tE)

    if vstarfun is None and ystarfun is None:
        vstar = xvectoX(vstarvec, nq=nq, ns=ns)
        __vstarfun = interp1d(tmesh, vstar, axis=1)

        def _vstarfun(t):
            return __vstarfun(_rstt(t))
    else:
        _vstarfun = vstarfun

    def eva_bwd(vvec, debug=False):
        vv = xvectoX(vvec, nq=nq, ns=ns)
        _vfun = interp1d(tmesh, vv, axis=1)

        def vfun(t):
            return _vfun(_rstt(t))

        if ystarfun is None:
            def _bbwdrhs(t):
                return (-vmat.dot(vfun(tE-t)).flatten() +
                        vmat.dot(_vstarfun(tE-t)).flatten())
        else:
            def _bbwdrhs(t):
                return -cmattrp.dot(cmat.dot(vfun(tE-t)).flatten()
                                    - ystarfun(tE-t).flatten())
                # turn -lredct.dot(vredc.dot(rdvfun(tE-t).flatten())
                #                  - cvstar(tE-t).flatten())

        if vdxoperator is not None:
            def _bbwdnl(lvec, t):
                vdx = vdxoperator(vfun(tE-t))
                return (vdx.dot(lvec)).flatten()

        # elif vdxlinop is not None:
        #     def _bbwdnl(lvec, t):
        #         vdx = vdxlinop(vfun(tE-t))
        #         return -(vdx*lvec).flatten()

        elif bwdvltens is not None:
            import burgers_genpod_utils as bgu

            def _bbwdnl(lvec, t):
                return -bgu.eva_burger_spacecomp(uvvdxl=bwdvltens, svec=lvec,
                                                 convvec=vfun(tE-t)).flatten()

        simudict = dict(iniv=termiL, A=AL, M=ML, nfunc=_bbwdnl,
                        rhs=_bbwdrhs, tmesh=tmesh,
                        rtol=solvrtol, atol=solvatol)

        if debug:
            simudict.update(dict(full_output=True))
            with dou.Timer('eva red bwd'):
                ll, infodict = gpu.time_int_semil(**simudict)
            return (np.flipud(ll), infodict, _bbwdnl,
                    AL, _vfun, _bbwdrhs, simudict)
        ll = gpu.time_int_semil(**simudict)
        ll = np.flipud(ll)  # flip it to make it forward time
        # ll = ll.reshape((ns, nq))
        return Xtoxvec(ll.T)
    return eva_bwd


def spacetimesolve(func=None, funcjaco=None, inival=None, timerecord=False,
                   usendjaco=False, message=None, printstats=True):

    useoptimizer = False
    if logger.level == 10:
        myjaco = funcjaco(inival.flatten())
        ndjaco = nd.Jacobian(func)(inival.flatten())
        logger.debug('diffnorm of the analytic and the numerical jacobian' +
                     ' `dJ={0}`'.format(np.linalg.norm(myjaco-ndjaco)))

    if usendjaco:
        funcjaco = numerical_jacobian(func)

    tfd = {}  # if timerecord else None
    with dou.Timer(message, timerinfo=tfd):
        if useoptimizer:
            def funcsqrd(vec):
                return 0.5*np.linalg.norm(func(vec.flatten()))**2

            def jac(vec):
                return np.dot(func(vec), funcjaco(vec))

            def funcnjac(vec):
                fvec = func(vec)
                jvec = funcjaco(vec)
                return 0.5*np.linalg.norm(fvec), np.dot(fvec, jvec)

            # ndfuncnjac = numerical_jacobian(funcsqrd)
            # anajaco = jac(inival)
            # ndjaco = ndfuncnjac(inival)
            # print(np.linalg.norm(ndjaco - anajaco))

            # optisol = sco.minimize(funcnjac, inival,  jac=True,
            # optisol = sco.minimize(funcsqrd, inival, jac=jac,
            #                        method='BFGS')
            # sol = optisol.x
            optisol, fopt, gopt, Bopt, nfc, ngc, wflag = \
                sco.fmin_bfgs(funcsqrd, inival, full_output=True,
                              fprime=jac, maxiter=500, gtol=5e-1)
            print('norm of Gradient: {0}'.format(np.linalg.norm(gopt)))
            sol = optisol

        else:
            if printstats:
                sol, infdct, _, _ = fsolve(func, x0=inival.flatten(),
                                           fprime=funcjaco, full_output=True)
                logger.info(message + ': nfev={0}, normf={1}'.
                            format(infdct['nfev'],
                                   np.linalg.norm(infdct['fvec'])))
            else:
                sol = fsolve(func, x0=inival.flatten(), fprime=funcjaco)

    if timerecord:
        return sol, tfd
    else:
        return sol


def get_bwd_nonl(vfun, nldvop, ptwise=False, M=None,
                 reversetime=False, tE=None):
    ''' returns a function that returns `N(v(t))_v*l`

    for a the adjoint state `l`
    '''
    if reversetime:
        def _vf(t):
            return vfun(tE-t)
    else:
        _vf = vfun
    if ptwise:
        def bwd_nonl(lvec, t):
            return (M*(nldvop(_vf(t)).flatten()*lvec.flatten())).flatten()

    else:
        def bwd_nonl(lvec, t):
            nldv = nldvop(_vf(t))
            return (nldv*lvec).flatten()
    return bwd_nonl


def get_infl_intp_nldv(lyitULy=None, lyULy=None, lyitUVy=None,
                       nonl_ptw=None):

    def infl_intp_nldv(vvec):
        nldvintp = (nonl_ptw(lyitUVy.dot(vvec))).flatten()

        def nldvintpxl(lvec):
            return (lyULy.T).dot(nldvintp*(lyitULy.dot(lvec)).flatten())

        nldvlinop = spsla.LinearOperator((lyitULy.shape[1],
                                          lyitULy.shape[1]),
                                         matvec=nldvintpxl)
        return nldvlinop
    return infl_intp_nldv
