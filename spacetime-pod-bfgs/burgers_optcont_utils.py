import numpy as np

import spacetime_pod_utils as spu

import logging
logger = logging.getLogger('basic')

__all__ = ['burger_vres',
           'burger_lres',
           'burger_clres_jaco',
           'setup_burger_clres',
           'setup_burger_bwdres',
           'setup_burger_fwdres']


def burger_vres(tsVvec=None, tsLvec=None,
                Vhdms=None, Vhms=None, Vhmy=None, Vhay=None,
                alpha=None, VLhms=None, VLhmy=None,
                htittl=None, uvvdxl=None,
                hiniv=None, htermiL=None,
                retjacobian=False, retjacobian_wrtl=False):

    if retjacobian:  # wrt. tvvec
        def brgquadtrmjaco(tvvec):
            return spu.\
                eva_quadform(tsvecone=tvvec, tsvectwo=tvvec, htittl=htittl,
                             uvvdxl=uvvdxl, retjacobian=True)

        return spu.get_spatimpodres_jacobian(tvvec=tsVvec, dms=Vhdms, ms=Vhms,
                                             my=Vhmy, ared=Vhay, iniv=hiniv,
                                             nfuncjac=brgquadtrmjaco)
    elif retjacobian_wrtl:
        if htermiL is None and hiniv is None:
            return -1./alpha*np.dot(np.kron(VLhms, VLhmy))
        elif htermiL is not None and hiniv is not None:
            return -1./alpha*np.kron(VLhms[1:, :-1], VLhmy)
        else:
            raise NotImplementedError

    else:
        if tsLvec is not None:
            fwdrhs = 1./alpha*np.dot(np.kron(VLhms, VLhmy),
                                     np.vstack([tsLvec, htermiL]))
        else:
            fwdrhs = None

        def fwdnonl(tvvec):
            return spu.eva_quadform(tsvecone=tvvec, tsvectwo=tvvec,
                                    htittl=htittl, uvvdxl=uvvdxl)

        res = spu.spacetimepodres(tvvec=tsVvec, dms=Vhdms, ms=Vhms,
                                  my=Vhmy, ared=Vhay, nfunc=fwdnonl,
                                  rhs=fwdrhs, iniv=hiniv, retnorm=False)
        return res


def burger_lres(tsVvec=None, tsLvec=None, tsVtrgtvec=None,
                Lhdms=None, Lhms=None, Lhmy=None, Lhay=None,
                LVhms=None, LVhmy=None,
                Lhtittl=None, Luvvdxl=None,
                hiniv=None, htermiL=None,
                nononlinearity=False,
                retjacobian=False, retjacobian_wrtv=False):

    inivtsVvec = np.vstack([hiniv, tsVvec])
    tsLvectrml = np.vstack([tsLvec, htermiL])

    if retjacobian:  # wrt. tslvec
        if nononlinearity:
            brgbwdconvjaco = None
        else:
            def brgbwdconvjaco(tvvec):
                return spu.eva_quadform(tsvecone=-inivtsVvec, htittl=Lhtittl,
                                        uvvdxl=Luvvdxl,
                                        retjacobian_wrt_tsvectwo=True)

        return spu.get_spatimpodres_jacobian(tvvec=None, dms=-Lhdms, ms=Lhms,
                                             my=Lhmy, ared=Lhay,
                                             nfuncjac=brgbwdconvjaco,
                                             termiv=htermiL)
    if retjacobian_wrtv:
        convprt = -spu.eva_quadform(tsvectwo=tsLvectrml, htittl=Lhtittl,
                                    uvvdxl=Luvvdxl,
                                    retjacobian_wrt_tsvecone=True)
        if htermiL is None and hiniv is None:
            return np.dot(np.kron(LVhms, LVhmy))
        elif htermiL is not None and hiniv is not None:
            convprt = convprt[:-len(htermiL), len(hiniv):]
            return np.kron(LVhms[:-1, 1:], LVhmy) + convprt

        else:
            raise NotImplementedError

    else:
        adjrhs = np.dot(np.kron(LVhms, LVhmy),
                        tsVtrgtvec-inivtsVvec)
        if nononlinearity:
            bwdnonl = None

        else:
            def bwdnonl(tvvec=None):
                return spu.eva_quadform(tsvectwo=tvvec, tsvecone=-inivtsVvec,
                                        htittl=Lhtittl, uvvdxl=Luvvdxl)
        res = spu.spacetimepodres(tvvec=tsLvec, dms=-Lhdms, ms=Lhms,
                                  my=Lhmy, ared=Lhay, nfunc=bwdnonl,
                                  rhs=adjrhs, termiv=htermiL)
        return res


def burger_clres_jaco(tsVvec=None, tsLvec=None,
                      Lhdms=None, Lhms=None, Lhmy=None, Lhay=None,
                      Vhdms=None, Vhms=None, Vhmy=None, Vhay=None,
                      LVhms=None, LVhmy=None,
                      VLhms=None, VLhmy=None,
                      alpha=None,
                      Vhtittl=None, Vuvvdxl=None,
                      Lhtittl=None, Luvvdxl=None,
                      hiniv=None, htermiL=None):

    jvv = burger_vres(tsVvec=tsVvec, tsLvec=tsLvec,
                      Vhdms=Vhdms, Vhms=Vhms, Vhmy=Vhmy, Vhay=Vhay,
                      htittl=Vhtittl, uvvdxl=Vuvvdxl,
                      hiniv=hiniv, htermiL=htermiL,
                      retjacobian=True)

    jvl = burger_vres(alpha=alpha, VLhms=VLhms, VLhmy=VLhmy,
                      hiniv=hiniv, htermiL=htermiL,
                      retjacobian_wrtl=True)

    jlv = burger_lres(tsLvec=tsLvec,
                      LVhms=LVhms, LVhmy=LVhmy,
                      Lhtittl=Lhtittl, Luvvdxl=Luvvdxl,
                      hiniv=hiniv, htermiL=htermiL,
                      retjacobian_wrtv=True)

    jll = burger_lres(tsVvec=tsVvec,
                      Lhdms=Lhdms, Lhms=Lhms, Lhmy=Lhmy, Lhay=Lhay,
                      Lhtittl=Lhtittl, Luvvdxl=Luvvdxl,
                      hiniv=hiniv, htermiL=htermiL,
                      retjacobian=True)

    return np.vstack([np.hstack([jvv, jvl]), np.hstack([jlv, jll])])


def setup_burger_fwdres(Vhdms=None, Vhms=None, Vhmy=None, Vhay=None,
                        htittl=None, uvvdxl=None, hiniv=None):

    def vres(tvvec):
        allo = burger_vres(tsVvec=tvvec.reshape((tvvec.size, 1)),
                           Vhdms=Vhdms, Vhms=Vhms, Vhmy=Vhmy, Vhay=Vhay,
                           htittl=htittl, uvvdxl=uvvdxl, hiniv=hiniv)
        return allo.flatten()

    def vresprime(tvvec):
        return burger_vres(tsVvec=tvvec.reshape((tvvec.size, 1)),
                           Vhdms=Vhdms, Vhms=Vhms, Vhmy=Vhmy, Vhay=Vhay,
                           htittl=htittl, uvvdxl=uvvdxl, hiniv=hiniv,
                           retjacobian=True)

    return vres, vresprime


def setup_burger_bwdres(Lhdms=None, Lhms=None, Lhmy=None, Lhay=None,
                        LVhms=None, LVhmy=None,
                        Lhtittl=None, Luvvdxl=None,
                        hiniv=None, htermiL=None,
                        tsVvec=None, tsVtrgtvec=None):

    def lres(tsLvec):
        allo = burger_lres(tsLvec=tsLvec.reshape((tsLvec.size, 1)),
                           tsVvec=tsVvec, tsVtrgtvec=tsVtrgtvec,
                           Lhdms=Lhdms, Lhms=Lhms, Lhmy=Lhmy, Lhay=Lhay,
                           LVhms=LVhms, LVhmy=LVhmy,
                           Lhtittl=Lhtittl, Luvvdxl=Luvvdxl,
                           hiniv=hiniv, htermiL=htermiL)
        return allo.flatten()

    def lresprime(tsLvec):
        return burger_lres(tsVvec=tsVvec, Lhdms=Lhdms, Lhms=Lhms,
                           Lhmy=Lhmy, Lhay=Lhay,
                           Lhtittl=Lhtittl, Luvvdxl=Luvvdxl,
                           hiniv=hiniv, htermiL=htermiL, retjacobian=True)

    return lres, lresprime


def setup_burger_clres(Vhdms=None, Vhms=None, Vhmy=None, Vhay=None,
                       VLhms=None, VLhmy=None, alpha=None,
                       htittl=None, uvvdxl=None,
                       tsVtrgtvec=None,
                       Lhdms=None, Lhms=None, Lhmy=None, Lhay=None,
                       LVhms=None, LVhmy=None,
                       Lhtittl=None, Luvvdxl=None,
                       hiniv=None, htermiL=None):
    hq = hiniv.size
    hs = Vhms.shape[0]

    def clres(tsvec):
        tsVvec = tsvec[:hq*(hs-1)]
        tsLvec = tsvec[-hq*(hs-1):]
        vres = burger_vres(tsVvec=tsVvec.reshape((tsVvec.size, 1)),
                           tsLvec=tsLvec.reshape((tsLvec.size, 1)),
                           Vhdms=Vhdms, Vhms=Vhms, Vhmy=Vhmy, Vhay=Vhay,
                           VLhms=VLhms, VLhmy=VLhmy, alpha=alpha,
                           htittl=htittl, uvvdxl=uvvdxl,
                           hiniv=hiniv, htermiL=htermiL)
        lres = burger_lres(tsVvec=tsVvec.reshape((tsVvec.size, 1)),
                           tsLvec=tsLvec.reshape((tsLvec.size, 1)),
                           tsVtrgtvec=tsVtrgtvec,
                           Lhdms=Lhdms, Lhms=Lhms, Lhmy=Lhmy, Lhay=Lhay,
                           LVhms=LVhms, LVhmy=LVhmy,
                           Lhtittl=Lhtittl, Luvvdxl=Luvvdxl,
                           hiniv=hiniv, htermiL=htermiL)
        return np.r_[vres.flatten(), lres.flatten()]

    def clresprime(tsvec):
        tsVvec = tsvec[:hq*(hs-1)].reshape((hq*(hs-1), 1))
        tsLvec = tsvec[-hq*(hs-1):].reshape((hq*(hs-1), 1))
        return burger_clres_jaco(tsVvec=tsVvec, tsLvec=tsLvec,
                                 Lhdms=Lhdms, Lhms=Lhms, Lhmy=Lhmy, Lhay=Lhay,
                                 Vhdms=Vhdms, Vhms=Vhms, Vhmy=Vhmy, Vhay=Vhay,
                                 LVhms=LVhms, LVhmy=LVhmy,
                                 VLhms=VLhms, VLhmy=VLhmy,
                                 alpha=alpha,
                                 Vhtittl=htittl, Vuvvdxl=uvvdxl,
                                 Lhtittl=Lhtittl, Luvvdxl=Luvvdxl,
                                 hiniv=hiniv, htermiL=htermiL)
    return clres, clresprime
