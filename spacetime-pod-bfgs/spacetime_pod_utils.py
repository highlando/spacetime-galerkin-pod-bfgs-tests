import numpy as np
import sadptprj_riclyap_adi.lin_alg_utils as lau
import scipy.sparse as sps

__all__ = ['eva_quadform',
           'expand_stpodsol',
           'apply_time_space_kronprod',
           'get_spacetimepodres',
           'get_spatimpodres_jacobian',
           'timspanorm',
           'krontimspaproduct']


def eva_quadform(tsvecone=None, tsvectwo=None, htittl=None, uvvdxl=None,
                 retjacobian=False, retjacobian_wrt_tsvecone=False,
                 retjacobian_wrt_tsvectwo=False):
    """evaluate a time-space quadratic form

    Parameters
    ---
    tsvecone: (`hsdim*hqdim`, 1) array
        a space-time state
    tsvectwo: (`hsdim*hqdim`, 1) array
        another space-time state
    htittl: list of `hsdim` (`hsdim, hsdim`) arrays
        the time component
    uvvdxl: list of `hqdim` (`hqdim`, `hqdim`) arrays
        the space component
    retjacobian: boolean, optional
        whether to return the jacobian `dH(v,v)/dv`
    retjacobian_wrt_tsvecone: boolean, optional
        whether to return the jacobian `dH(v1,v2)/dv1 = f(v2)`
    retjacobian_wrt_tsvectwo: boolean, optional
        whether to return the jacobian `dH(v1,v2)/dv2 = f(v1)`

    Notes
    ---
    The quadratic gives a vector with components `[ltsv.T*H_i*rtsv]`.
    Thus the Jacobian `J(v)` is defined
    through `[J(v)*w]_i = w.T*H_i*v + v.T*H_i*w = v.T*(H+H.T)*w`

    """
    if tsvectwo is None:
        # print 'no leftvec given - I assume you want to compute H(rvec, rvec)'
        tsvectwo = tsvecone

    if retjacobian:
        if tsvecone is not tsvectwo:
            raise UserWarning('I can not compute the quadratic Jacobian if' +
                              ' lvec =/ rvec')
        vthhtl = []
        for htitt in htittl:
            for uvvdx in uvvdxl:
                hvij = np.dot(np.kron(htitt, uvvdx), tsvectwo)
                vthij = np.dot(tsvecone.T, np.kron(htitt, uvvdx))
                vthhtl.append(vthij+hvij.T)
        return np.vstack(vthhtl)

    if retjacobian_wrt_tsvecone:
        vthhtl = []
        for htitt in htittl:
            for uvvdx in uvvdxl:
                hvij = np.dot(np.kron(htitt, uvvdx), tsvectwo)
                vthhtl.append(hvij.T)
        return np.vstack(vthhtl)

    elif retjacobian_wrt_tsvectwo:
        vthhtl = []
        for htitt in htittl:
            for uvvdx in uvvdxl:
                vthij = np.dot(tsvecone.T, np.kron(htitt, uvvdx))
                vthhtl.append(vthij)
        return np.vstack(vthhtl)

    else:
        tvtvxl = []
        for htitt in htittl:
            for uvvdx in uvvdxl:
                tvxlij = np.dot(np.kron(htitt, uvvdx), tsvectwo)
                tvtvxl.append(np.dot(tsvecone.T, tvxlij))
        return np.array(tvtvxl).reshape(len(htittl)*len(uvvdxl), 1)


def expand_stpodsol(stpsol=None, Uks=None, Uky=None):
    solmat = stpsol.reshape((Uks.shape[1], Uky.shape[1]))
    solmatexps = np.dot(Uks, solmat)
    return np.dot(solmatexps, Uky.T)


def apply_time_space_kronprod(tvvec=None, smat=None, qmat=None, iniv=None):
    """evaluate a time-space linear term

    Parameters
    ---
    tvvec: (`hsdim*hqdim`, 1) array
        the current space-time state
    smat: (`sdim, sdim`) array
        the time component
    qmat: (`qdim`, `qdim`) array
        the space component
    iniv: (`hqdim`, 1) array, optional
        the initial value, if not `None` it eliminates the dof associated with
        the first time coefficient
    """

    if iniv is None:
        return np.dot(np.kron(smat, qmat), tvvec)
    else:
        inivqvec = np.vstack([iniv, tvvec])
        smatr = smat[1:, :]
        return np.dot(np.kron(smatr, qmat), inivqvec)


def get_spacetimepodres(tvvec=None, dms=None, ms=None, my=None, ared=None,
                        nfunc=None, rhs=None, retnorm=False,
                        termiv=None, iniv=None):
    """ the residual of a space semi-explicit semi-linear space-time galerkin

    `R(tv) = [kron(dms, my) + kron(ms, ared)]*tv + nonl(tv) - rhs`

    Parameters:
    ---
    ms : (ns, ns) array
        the time mass matrix
    dms : (ns, ns) array
        the time "stiffness" matrix
    my : (nq, nq) array
        the space mass matrix
    ared : (nq, nq) array
        the space stiffness matrix
    nfunc : callable f(v)
        the nonlinearity
    rhs : (ns*nq, 1) array, optional
        the right hand side in space time tensor coefficients
    iniv: (`hqdim`, 1) array, optional
        the initial value, if not `None`, the dof associated with
        the first time coefficient will be eliminated
    termiv: (`hqdim`, 1) array, optional
        the terminal value, if not `None`, the dof associated with
        the last time coefficient will be eliminated
    """

    if iniv is not None:
        try:
            tvvec = np.vstack([iniv, tvvec])
        except ValueError:
            tvvec = np.vstack([iniv, tvvec.reshape((tvvec.size, 1))])
    if termiv is not None:
        try:
            tvvec = np.vstack([tvvec, termiv])
        except ValueError:
            tvvec = np.vstack([tvvec.reshape((tvvec.size, 1)), termiv])
    dtpart = apply_time_space_kronprod(tvvec=tvvec, smat=dms, qmat=my)
    if ared is None:
        apart = 0
    else:
        apart = apply_time_space_kronprod(tvvec=tvvec, smat=ms, qmat=ared)
    if nfunc is None:
        nonlpart = 0
    else:
        nonlpart = nfunc(tvvec=tvvec)
    rhspart = 0 if rhs is None else rhs
    stpr = dtpart + apart + nonlpart - rhspart
    if iniv is not None:
        stpr = stpr[iniv.size:, :]
    if termiv is not None:
        stpr = stpr[:-termiv.size, :]
    if retnorm:
        # compute the norm of the residual in the M.-1 scalar product
        tsm = np.kron(ms, my)
        resn = np.dot(stpr.T, np.linalg.solve(tsm, stpr))
        return resn, stpr
    else:
        return stpr


def get_spatimpodres_jacobian(tvvec=None, dms=None, ms=None, my=None,
                              ared=None, nfuncjac=None, addmat=None,
                              termiv=None, iniv=None):
    """ generate the jacobian for the `spatimpod residual`
    """

    if ared is None:
        ared = 0*my

    if iniv is not None:
        try:
            tvvec = np.vstack([iniv, tvvec])
        except ValueError:
            tvvec = np.vstack([iniv, tvvec.reshape((tvvec.size, 1))])
        apartmat = np.kron(ms[1:, 1:], ared)
        dtpartmat = np.kron(dms[1:, 1:], my)
    elif termiv is not None:
        try:
            tvvec = np.vstack([tvvec, termiv])
        except ValueError:
            tvvec = np.vstack([tvvec.reshape((tvvec.size, 1)), termiv])
        apartmat = np.kron(ms[:-1, :-1], ared)
        dtpartmat = np.kron(dms[:-1, :-1], my)
    else:
        apartmat = np.kron(ms, ared)
        dtpartmat = np.kron(dms, my)

    if nfuncjac is None:
        nonlpartmat = 0*apartmat
    else:
        nonlpartmat = nfuncjac(tvvec=tvvec)
        if iniv is not None:
            nonlpartmat = nonlpartmat[iniv.size:, iniv.size:]
        elif termiv is not None:
            nonlpartmat = nonlpartmat[:-termiv.size, :-termiv.size]
    jacmat = dtpartmat + apartmat + nonlpartmat
    if addmat is not None:
        jacmat = jacmat + addmat
    return jacmat


def timspanorm(x, mspace=None, mtime=None):
    ynorml = []
    for col in range(x.shape[1]):
        ccol = x[:, col]
        ynorml.append(np.dot(ccol.T, lau.mm_dnssps(mspace, ccol)))
    ynormvec = np.array(ynorml).reshape((x.shape[1], 1))

    return np.sqrt(np.dot(ynormvec.T, lau.mm_dnssps(mtime, ynormvec)))


def krontimspaproduct(vone, vtwo=None, my=None, ms=None):
    if vtwo is None:
        vtwo = vone
    return np.dot(vone.T, sps.kron(ms, my)*vtwo)
