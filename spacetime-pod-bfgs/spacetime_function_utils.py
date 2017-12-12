import numpy as np

import spacetime_pod_utils as spu


def Xtoxvec(X):
    ''' X - (nq, ns) array to (ns*nq, 1) array by stacking the columns '''
    return X.T.reshape((X.size, 1))


def xvectoX(xvec, nq=None, ns=None):
    ''' xvec - (ns*nq, 1) vector to (nq, ns) by writing columnwise '''
    return xvec.reshape((ns, nq)).T


def get_spacetime_inflt_intrp(nonlfunc=None, inival=None,
                              lyitUVy=None, lyUVy=None,
                              lsitUVs=None, lsUVs=None):
    hs, hq = lsUVs.shape[1], lyUVy.shape[1]

    def eva_spati_rhs(tvvec):
        if inival is not None:
            try:
                tvvec = np.vstack([inival, tvvec])
            except ValueError:
                tvvec = np.vstack([inival, tvvec.reshape((tvvec.size, 1))])
        tvX = xvectoX(tvvec, nq=hq, ns=hs)
        NinflX = nonlfunc(np.dot(lyitUVy, np.dot(tvX, lsitUVs.T)))
        prjNinflX = np.dot(lyUVy.T, np.dot(NinflX, lsUVs))
        ninflvec = Xtoxvec(prjNinflX)
        if inival is not None:
            ninflvec = ninflvec[hq:]
        return ninflvec

    return eva_spati_rhs


def get_appnd_inflt_reshp_func(spaceinfl=None, timeinfl=None,
                               timeinival=None, endappend=False):
    nq, ns = spaceinfl.shape[1], timeinfl.shape[1]

    def appinfres(tsvx, tiniv=None):
        if tsvx.size == (ns - 1)*nq:
            liniv = tiniv if tiniv is not None else timeinival
        if tsvx.shape[0] == nq and tsvx.size == (ns - 1)*nq:
            if endappend:
                X = np.hstack([tsvx, liniv])
            else:
                X = np.hstack([liniv, tsvx])
        elif tsvx.shape[0] == tsvx.size:  # it's a vector
            Xvec = tsvx.reshape((tsvx.size, 1))  # now it is a column
            if Xvec.size == (ns - 1)*nq:
                if endappend:
                    Xvec = np.vstack([Xvec, liniv])
                else:
                    Xvec = np.vstack([liniv, Xvec])
            X = xvectoX(Xvec, ns=ns, nq=nq)
        else:
            X = tsvx
        return np.dot(spaceinfl, np.dot(X, timeinfl.T))
    return appinfres


def getget_spacetime_bwdres(ndv_ptw=None,
                            lsitUVs=None, lyitUVy=None,
                            lyitULy=None, lyULy=None,
                            lsitULs=None, lsULs=None,
                            iniv=None, termiv=None,
                            ctrp=None, cmat=None, ystarvec=None,
                            dms=None, ms=None, my=None, ared=None,
                            mrs=None):
    globiniv = iniv

    def get_bwdres(vvec, iniv=None):
        if iniv is not None:
            vvec = np.vstack([iniv, vvec])
        elif globiniv is not None:
            vvec = np.vstack([globiniv, vvec])

        nq, hq = lyitUVy.shape
        ns, hs = lsitUVs.shape

        hvV = xvectoX(vvec, ns=hs, nq=hq)
        vV = np.dot(lyitUVy, np.dot(hvV, lsitUVs.T))
        inflvvec = Xtoxvec(vV).flatten()

        if ndv_ptw is None:
            bwdnl = None
        else:
            def bwdnl(tvvec=None):
                infllvec = np.kron(lsitULs, lyitULy).dot(tvvec)
                nllvec = (ndv_ptw(inflvvec)).flatten()
                nllvX = xvectoX(nllvec*infllvec.flatten(), nq=nq, ns=ns)
                clvx = np.dot(lyULy.T, np.dot(nllvX, lsULs))
                return Xtoxvec(clvx).reshape(tvvec.shape)

        cvvc = cmat.dot(xvectoX(vvec, ns=hs, nq=hq))
        bwdrhs = -(np.kron(mrs, ctrp)).dot(cvvc.T - ystarvec.T)

        def bwdres(lvec):
            return spu.spacetimepodres(tvvec=lvec, dms=-dms, ms=ms,
                                       my=my, ared=ared,
                                       nfunc=bwdnl, rhs=bwdrhs, retnorm=False,
                                       termiv=termiv).flatten()
        return bwdres
    return get_bwdres


def getget_spacetime_fwdres(dms=None, ms=None, my=None, ared=None,
                            nfunc=None, rhs=None, iniv=None,
                            msr=None, bmat=None):
        globiniv = iniv

        def get_fwdres(uvec=None, iniv=None):
            if iniv is None:
                lociniv = globiniv
            else:
                lociniv = iniv
            if uvec is None:
                crhs = rhs
            else:
                bu = (np.kron(msr, bmat)).dot(uvec)
                if rhs is None:
                    crhs = bu
                else:
                    crhs = rhs + bu

            def fwdres(tsVvec):
                vres = spu.spacetimepodres(tvvec=tsVvec, dms=dms, ms=ms,
                                           my=my, ared=ared,
                                           nfunc=nfunc,
                                           rhs=crhs, iniv=lociniv,
                                           retnorm=False)
                return vres.flatten()
            return fwdres
        return get_fwdres


def get_clres(get_fwdres=None, get_bwdres=None, bmattrp=None, rmo=None,
              hsv=None, hqv=None, hsl=None, hql=None,
              htermiL=None, hiniv=None):
    ghtl, ghnv = htermiL, hiniv

    def clres(tsvlvec, htermiL=None, hiniV=None):
        if not tsvlvec.size == (hsv-1)*hqv + (hsl-1)*hql:
            raise UserWarning('wrong dimensions')
        lhnv = hiniV if hiniV is not None else ghnv
        lhtl = htermiL if htermiL is not None else ghtl
        vvec = np.r_[lhnv.flatten(), tsvlvec[:(hsv-1)*hqv]]
        lvec = np.r_[tsvlvec[-(hsl-1)*hql:], lhtl.flatten()]

        uvec = (rmo*bmattrp.dot(xvectoX(lvec, ns=hsl, nq=hql))).T
        fwdres = get_fwdres(uvec=uvec)
        vres = fwdres(vvec)[hqv:]

        bwdres = get_bwdres(vvec=vvec)
        lres = bwdres(lvec)[:-hql]

        return np.r_[vres, lres]
    return clres
