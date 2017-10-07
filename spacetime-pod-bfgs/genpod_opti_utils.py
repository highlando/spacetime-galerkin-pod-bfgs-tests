import numpy as np
import scipy.sparse as sps

import gen_pod_utils as gpu
import spacetime_pod_utils as spu

from scipy.interpolate import interp1d
import sadptprj_riclyap_adi.lin_alg_utils as lau
import dolfin_navier_scipy.data_output_utils as dou

try:
    import numdifftools as nd
except ImportError:
    print('can not import numdifftool -- can do without')


__all__ = ['stateadjspatibas',
           'eva_costfun',
           'Xtoxvec',
           'xvectoX',
           'numerical_jacobian',
           'costfun',
           'functovec']


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


def costfun(MYs=None, MYt=None, MUs=None, MUt=None,
            tmesh=None, utmesh=None, vstarvec=None):
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
        uzhvec = np.zeros((utmesh.size+1, 1))
        uzhvec[1:-1, 0] = utmesh[1:] - utmesh[:-1]
        utrpdiag = .5*(uzhvec[:-1]+uzhvec[1:])
        try:
            MUt = sps.diags(utrpdiag.flatten())
        except TypeError:
            MUt = sps.diags(utrpdiag.flatten(), 0)
    if MYt == 'trapez':
        zhvec = np.zeros((tmesh.size+1, 1))
        zhvec[1:-1, 0] = tmesh[1:] - tmesh[:-1]
        trpdiag = .5*(zhvec[:-1]+zhvec[1:])
        try:
            MYt = sps.diags(trpdiag.flatten())
        except TypeError:
            MYt = sps.diags(trpdiag.flatten(), 0)

    def evacostfun(vvec, uvec, retparts=False):
        diffv = vvec - vstarvec
        vterm = 0.5*spu.krontimspaproduct(diffv, ms=MYt, my=MYs)
        uterm = 0.5*spu.krontimspaproduct(uvec, ms=MUt, my=MUs)
        if retparts:
            return vterm.flatten()[0], uterm.flatten()[0]
        else:
            return uterm.flatten()[0] + vterm.flatten()[0]

    def compcfgrad(lvec=None, uvec=None):
        # TODO: why is there a minus
        return -(spu.krontimspamatvec(lvec, ms=MYt, my=MYs) +
                 spu.krontimspamatvec(uvec, ms=MUt, my=MUs))

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


def get_eva_costfun(tmesh=None, MVs=None, MUs=None, MVt='trapez', MUt='trapez',
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
    MVs = _sparsifymat(MVs)
    MUs = _sparsifymat(MUs)

    _costfun, _costgrad = costfun(MYs=MVs, MYt=MVt, MUs=MUs, MUt=MUt,
                                  tmesh=tmesh, utmesh=utmesh,
                                  vstarvec=vstarvec)

    def eva_costfun(uvec, vvec=None, retparts=False):
        if vvec is None:
            vvec = eva_fwd(uvec)
        return _costfun(vvec, uvec.flatten(), retparts=retparts)

    if getcompgrad:
        def eva_costgrad(uvec, vvec=None, lvec=None):
            if lvec is None:
                if vvec is None:
                    vvec = eva_fwd(uvec)
                lvec = eva_bwd(vvec)
            return _costgrad(lvec=lvec.reshape((lvec.size, 1)),
                             uvec=uvec.reshape((lvec.size, 1))).flatten()

        return eva_costfun, eva_costgrad

    else:
        return eva_costfun


def get_eva_fwd(iniv=None, MV=None, AV=None, MVU=None, rhs=None,
                solvrtol=None, solvatol=None,
                nonlfunc=None, tmesh=None, redtmesh=None, liftUcoef=None):
    ''' set up the function that evaluates the reduced fwd problem
    '''
    tE = tmesh[-1]
    NQ = MV.shape[0]
    NS = tmesh.size
    if redtmesh is None:
        redtmesh = tmesh

    def eva_fwd(uvec):

        ns = redtmesh.size
        nq = np.int(uvec.size / ns)

        U = xvectoX(uvec, nq=nq, ns=ns)
        _ufun = interp1d(redtmesh, U, axis=1)
        if liftUcoef is not None:
            def ufun(t):
                return liftUcoef(_ufun(t))
        else:
            ufun = _ufun

        def _contrl_rhs(t):
            if t > tE:  # the integrator may require values outside [t0, tE]
                return rhs(t).flatten()+lau.mm_dnssps(MVU, ufun(tE)).flatten()
            else:
                return rhs(t).flatten()+lau.mm_dnssps(MVU, ufun(t)).flatten()

        simudict = dict(iniv=iniv, A=AV, M=MV, nfunc=nonlfunc,
                        rhs=_contrl_rhs, tmesh=tmesh,
                        rtol=solvrtol, atol=solvatol)

        vv = gpu.time_int_semil(**simudict)

        vv = vv.reshape((NS, NQ))

        return Xtoxvec(vv.T)

    return eva_fwd


def get_eva_bwd(vstarvec=None, MLV=None, ML=None, AL=None,
                solvrtol=None, solvatol=None, bwdvltens=None,
                termiL=None, tmesh=None, vdxoperator=None):

    nq, ns, te = ML.shape[1], tmesh.size, tmesh[-1]
    vstar = xvectoX(vstarvec, nq=nq, ns=ns)
    __vstarfun = interp1d(tmesh, vstar, axis=1)  # , fill_value='extrapolate')

    def _vstarfun(t):
        if t < tmesh[0]:
            return __vstarfun(tmesh[0])
        elif t > tmesh[-1]:
            return __vstarfun(tmesh[-1])
        else:
            return __vstarfun(t)

    def eva_bwd(vvec, debug=False):
        vv = xvectoX(vvec, nq=nq, ns=ns)
        _vfun = interp1d(tmesh, vv, axis=1)

        def vfun(t):
            if t < tmesh[0]:
                return _vfun(tmesh[0])
            elif t > tmesh[-1]:
                return _vfun(tmesh[-1])
            else:
                return _vfun(t)

        def _bbwdrhs(t):
            # TODO: -----------------------------> here we need vstar
            return (-lau.mm_dnssps(MLV, vfun(te-t)).flatten() +
                    lau.mm_dnssps(MLV, _vstarfun(te-t)).flatten())

        if vdxoperator is not None:
            def _bbwdnl_vdx(lvec, t):
                vdx = vdxoperator(vfun(te-t))
                return -(lau.mm_dnssps(vdx, lvec)).flatten()

        if bwdvltens is not None:
            import burgers_genpod_utils as bgu

            def _bbwdnl_tens(lvec, t):
                return -bgu.eva_burger_spacecomp(uvvdxl=bwdvltens, svec=lvec,
                                                 convvec=vfun(te-t)).flatten()

        # print _bbwdnl_tens(vfun(0), 0.)-_bbwdnl_vdx(vfun(0), 0.)
        # print _bbwdnl_tens(vfun(0.1), 0.5)-_bbwdnl_vdx(vfun(0.1), 0.5)
        # import ipdb; ipdb.set_trace()
        _bbwdnl = _bbwdnl_tens

        simudict = dict(iniv=termiL, A=AL, M=ML, nfunc=_bbwdnl,
                        rhs=_bbwdrhs, tmesh=tmesh,
                        rtol=solvrtol, atol=solvatol)

        if debug:
            simudict.update(dict(full_output=True))
            with dou.Timer('eva red bwd'):
                ll, infodict = gpu.time_int_semil(**simudict)
            return ll, infodict, _bbwdnl_tens, _bbwdnl_vdx, AL, _vfun
        ll = gpu.time_int_semil(**simudict)
        ll = np.flipud(ll)  # flip it to make it forward time
        ll = ll.reshape((ns, nq))
        return Xtoxvec(ll.T)
    return eva_bwd
