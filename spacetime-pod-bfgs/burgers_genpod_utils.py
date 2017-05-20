import numpy as np
import scipy.integrate as sii
import scipy.sparse as sps
import json
import warnings

import dolfin_navier_scipy.data_output_utils as dou
import sadptprj_riclyap_adi.lin_alg_utils as lau

import gen_pod_utils as gpu
import dolfin_burgers_scipy as dbs

__all__ = ['get_podredmod_burger',
           'get_snaps_burger',
           'get_snaps_heateq',
           'get_burger_tensor',
           'get_burgertensor_timecomp',
           'eva_burger_quadratic',
           'eva_burger_spacecomp']


def burger_fwd_gpresidual(ms=None, ay=None, dms=None, rhs=None,
                          my=None, iniv=None, htittl=None, uvvdxl=None
                          ):
    """ function to return the residuals of the burger genpod apprxm

    DEPRECATED
    """
    import spacetime_pod_utils as spu

    if rhs is not None and np.linalg.norm(rhs) > 0:
        raise NotImplementedError('rhs not considered in the alg')
    # lns = ms.shape[0]  # dimension of time disc
    lnq = ay.shape[0]  # dimension of space disc

    dmsz = dms[1:, :1]  # the part of the mass matrix related to the ini value
    msz = ms[1:, :1]

    dmsI = dms[1:, 1:]  # mass matrices w/o ini values (test and trial)
    msI = ms[1:, 1:]

    def _nonlinres(tsvec):
        def evabrgquadterm(tvvec):
            return eva_burger_quadratic(tvvec=tvvec, htittl=htittl,
                                        uvvdxl=uvvdxl)
        bres = spu.get_spacetimepodres(tvvec=tsvec,
                                       dms=dms, ms=ms,
                                       my=my, ared=ay,
                                       nfunc=evabrgquadterm,
                                       rhs=None, retnorm=False, iniv=iniv)
        return bres[0*lnq:, ].flatten()

    solmat = np.kron(dmsI, my) + np.kron(msI, ay)
    linrhs = - np.kron(dmsz, lau.mm_dnssps(my, iniv)) \
        - np.kron(msz, lau.mm_dnssps(ay, iniv))

    def _linres(tsvec):
        tsvecs = tsvec.reshape((tsvec.size, 1))
        lres = (np.dot(solmat, tsvecs) - linrhs)
        return lres.flatten()

    def _lrprime(tsvec):
        return solmat


def get_burger_tensor(Uky=None, Uks=None, datastr=None, V=None, diribc=None,
                      bwd=None, Ukyconv=None, Uksconv=None,
                      ininds=None, sdim=None, tmesh=None,
                      debug=False, **kwargs):
    # loading/assembling the components of the tensor
    try:
        if datastr is None or debug:
            print "no datastr specified or `debug` -- won't load/save any data"
            raise IOError
        else:
            fjs = open(datastr)
            burgernonldict = json.load(fjs)
            fjs.close()
            uvvdxl = nonlty_listtoarray(burgernonldict['uvvdxl'])
            htittl = nonlty_listtoarray(burgernonldict['htittl'])
            print 'Loaded the coefficients for the nonlinearity from {0}'.\
                format(datastr)
    except IOError:
        with dou.Timer('assemble the nonlinear coefficients'):
            print '... in space'
            uvvdxl = dbs.get_burgertensor_spacecomp(podmat=Uky, V=V,
                                                    Ukyleft=Ukyconv,
                                                    bwd=bwd,
                                                    ininds=ininds,
                                                    diribc=diribc)
            print '... in time'
            htittl = get_burgertensor_timecomp(podmat=Uks, sdim=sdim,
                                               Uksconv=Uksconv,
                                               tmesh=tmesh, basfuntype='pl')
        print '...done'

        # saveit
        if datastr is not None:
            burgernonldict = dict(uvvdxl=nonlty_arraytolist(uvvdxl),
                                  htittl=nonlty_arraytolist(htittl))
            f = open(datastr, 'w')
            f.write(json.dumps(burgernonldict))
            f.close()
            print 'Saved the coefficients for the nonlinearity to {0}'.\
                format(datastr)
    return uvvdxl, htittl


def get_podredmod_burger(Uky=None, Uks=None, nu=None, iniv=None,
                         Nq=None, plotsvs=False,
                         Ns=None, tmesh=None, timebasfuntype=None,
                         datastr=None):
    ''' get the coeffs of the space/time reduced Burgers model

    Parameters
    ---
    Uky : (Ny, hy) np.array
        containing the POD vectors in space dimension
    Uks : (Ns, hs) np.array
        containing the POD vectors in time dimension
    nu : float
        the considered viscosity
    iniv : (Ny, 1) np.array
        the initial value

    Returns
    ---
    redburgdict : dictionary
        a dictionary with the following keys coeffs of the reduced system:
            * `hmy`: space mass matrix
            * `hay`: space stiffness matrix
            * `hms`: time mass matrix
            * `hdms`: time derivative mass matrix
            * `inivred`: initial value
            * `uvvdxl`: the space component of the burger nonlinearity
            * `htittl`: the time component of the burger nonlinearity
            * `hnonl`: f(t, v) that returns the value of the nonlinearity
            * `hrhs`: right hand side
            * `nonlresjacobian`: f(v) that returns the Jacobian
                of the residual of the genpod burger approximation

    Examples
    ---
    redburgdict = get_podredmod_burger(Uky=x, Uks=x, Ms=x, curnu=1e-2, iniv=x)
    '''

    retdict = {}
    # get the space full model for the considered `nu`
    (M, A, rhs, nfunc, femp) = dbs.\
        burgers_spacedisc(N=Nq, nu=nu, retfemdict=True)

    # reducing the linear parts and the rhs
    ared, mred, _, rhsred, _, _ = gpu.\
        get_podred_model(M=M, A=A, rhs=rhs, Uk=Uky)

    retdict.update(dict(hay=ared, hmy=mred, inivred=np.dot(Uky.T, iniv),
                        hrhs=rhsred))

    # the time reduced model
    # TODO: this should be synchronized with comp of Ms
    Ms = gpu.get_ms(sdim=Ns, tmesh=tmesh, basfuntype=timebasfuntype)
    dms = gpu.get_dms(sdim=Ns, tmesh=tmesh, basfuntype=timebasfuntype)
    hdms = np.dot(Uks.T, np.dot(dms, Uks))
    hms = np.dot(Uks.T, np.dot(Ms, Uks))
    retdict.update(dict(hdms=hdms, hms=hms))

    uvvdxl, htittl = get_burger_tensor(Uky=Uky, Uks=Uks, datastr=datastr,
                                       sdim=Ns, tmesh=tmesh, **femp)
    retdict.update(dict(uvvdxl=uvvdxl, htittl=htittl))

    def burger_rednonl(tvvec):
        return eva_burger_quadratic(tvvec=tvvec, htittl=htittl, uvvdxl=uvvdxl,
                                    iniv=None, retjacobian=False)

    hdmsI = hdms[1:, 1:]  # mass matrices w/o ini values (test and trial)
    hmsI = hms[1:, 1:]
    solmat = np.kron(hdmsI, mred) + np.kron(hmsI, ared)

    def nlrprime(tsvec, curiniv=None):
        if curiniv is not None:
            try:
                tvvecini = np.vstack([curiniv, tsvec])
            except ValueError:
                tvvecini = np.vstack([curiniv, tsvec.reshape((tsvec.size, 1))])
        ebmat = eva_burger_quadratic(tvvec=tvvecini, htittl=htittl,
                                     uvvdxl=uvvdxl, retjacobian=True)
        ebmat = ebmat[curiniv.size:, curiniv.size:, ]
        return solmat+ebmat

    retdict.update(dict(hnonl=burger_rednonl, nonlresjacobian=nlrprime))

    return retdict


def get_snaps_burger(tmesh=None, Nq=None, Ns=None, nul=[],
                     iniv=None, datstrdict=None):
    ''' compute the gen. measurement matrix

    Parameters
    ---
    nul : list
        of `nu` values for which the measurements are taken
    datstrdict : dict, optional
        dict `{nu:filestr}` where to load/save the full solutions

    Returns
    ---
    snapsl : list
        of the gen. measurement matrices
    My : (Ny, Ny) sparse array
        the mass matrix of the space discretization
    Ms : (Ns, Ns) sparse array
        the mass matrix of the time discretization

    Examples
    ---
    snapsl, My, Ms = get_snaps_burger(Nts=300, Nq=300, Ns=65, nul=[1e-2])
    '''

    snapsl = []
    for curnu in nul:
        datastr = datstrdict[curnu] if datstrdict is not None else None
        (My, A, rhs, nfunc, femp) = dbs.\
            burgers_spacedisc(N=Nq, nu=curnu, retfemdict=True)
        simudict = dict(iniv=iniv, A=A, M=My,
                        nfunc=nfunc, rhs=rhs, tmesh=tmesh)
        vv = dou.\
            load_or_comp(filestr=datastr, comprtn=gpu.time_int_semil,
                         comprtnargs=simudict, arraytype='dense', debug=True)
        (msx, Ms) = gpu.get_genmeasuremat(sol=vv.T, tmesh=tmesh, sdim=Ns)
        x = lau.apply_massinv(sps.csc_matrix(Ms), msx.T).T
        snapsl.append(x)

    return snapsl, My, Ms


def get_snaps_heateq(tmesh=None, Nq=None, Ns=None, nul=[],
                     iniv=None, datstrdict=None):
    ''' compute the gen. measurement matrix as for burgers but

    w/o nonlinearity --> the heat eq

    Parameters
    ---
    nul : list
        of `nu` values for which the measurements are taken
    datstrdict : dict, optional
        dict `{nu:filestr}` where to load/save the full solutions

    Returns
    ---
    snapsl : list
        of the gen. measurement matrices
    My : (Ny, Ny) sparse array
        the mass matrix of the space discretization
    Ms : (Ns, Ns) sparse array
        the mass matrix of the time discretization

    Examples
    ---
    snapsl, My, Ms = get_snaps_heateq(Nts=300, Nq=300, Ns=65, nul=[1e-2])
    '''

    snapsl = []
    for curnu in nul:
        datastr = datstrdict[curnu] if datstrdict is not None else None
        (My, A, rhs, nfunc, femp) = dbs.\
            burgers_spacedisc(N=Nq, nu=curnu, retfemdict=True)
        simudict = dict(iniv=iniv, A=A, M=My,
                        nfunc=None, rhs=rhs, tmesh=tmesh)
        vv = dou.\
            load_or_comp(filestr=datastr, comprtn=gpu.time_int_semil,
                         comprtnargs=simudict, arraytype='dense', debug=True)
        (msx, Ms) = gpu.get_genmeasuremat(sol=vv.T, tmesh=tmesh, sdim=Ns)
        x = lau.apply_massinv(sps.csc_matrix(Ms), msx.T).T
        snapsl.append(x)

    return snapsl, My, Ms


def get_burgertensor_timecomp(podmat=None, sdim=None, tmesh=None,
                              Uksconv=None, basfuntype='pl'):
    """the temporal part of time space Galerkin applied to `uu.dx`

    Parameters
    ---

    """
    if not podmat.shape[0] == sdim:
        raise Warning("Looks like this is not the right POD basis")
    if Uksconv is None:
        Uksconv = podmat
    htittl = []
    bfuncl = []
    x0, xe = tmesh[0], tmesh[-1]
    for n in range(sdim):
        bfuncl.append(gpu.hatfuncs(n=n, N=sdim, x0=x0, xe=xe))

    for pm in podmat.T:
        def hatbfi(x):
            hfx = 0
            for k, fn in enumerate(bfuncl):
                hfx += pm[k]*fn(x)
            return hfx
        titt = np.zeros((sdim, sdim))
        for s in range(0, sdim):
            jhf = gpu.hatfuncs(n=s, x0=x0, xe=xe, N=sdim)
            for k in range(s, sdim):
                dkhf, pts = gpu.\
                    hatfuncs(n=k, x0=x0, xe=xe, N=sdim, retpts=True)

                def hbfiujuk(x):
                    return hatbfi(x)*dkhf(x)*jhf(x)
                for ts in range(len(pts)-1):
                    titt[k, s] += sii.\
                        fixed_quad(hbfiujuk, pts[ts], pts[ts+1], n=3)[0]
                titt[s, k] = titt[k, s]
        htittl.append(np.dot(Uksconv.T, np.dot(titt, podmat)))

    return htittl


def eva_convect_quadratic(tvvec=None, convtvvec=None, htittl=None, uvvdxl=None,
                          iniv=None, retjacobian=False):
    """evaluate the time-space-pod convection term

    Parameters
    ---
    tvvec: (`hsdim*hqdim`, 1) array
        the current space-time state
    convtvvec: (`hsdim*hqdim`, 1) array
        the current space-time convection-velocity state
    htittl: list of `hsdim` (`hsdim, hsdim`) arrays
        the time component
    uvvdxl: list of `hqdim` (`hqdim`, `hqdim`) arrays
        the space component
    iniv: (`hqdim`, 1) array, optional
        the initial value, if not `None` it replaces the dof associated with
        the first time coefficient
    retjacobian: boolean, optional
        whether to return the jacobian `df/dv (v)`

    Notes
    ---
    The function `convection` is of type `u.T*H*v`. Thus the Jacobian
    is `J(u) = u.T*H` TODO: add the burger option

    """

    warnings.warn('This function will be deprecated soon.' +
                  ' Rather use `spu.eva_quadform`')

    if iniv is not None:
        tvvec = np.vstack([iniv, tvvec])
    starttime = 0 if iniv is None else 1
    if retjacobian:
        vthhtl = []
        for htitt in htittl[starttime:]:
            for uvvdx in uvvdxl:
                # hvij = np.dot(np.kron(htitt, uvvdx), tvvec)  # cp from Burger
                vthij = np.dot(tvvec.T, np.kron(htitt, uvvdx))
                vthhtl.append(vthij)  # +hvij.T)
        return np.vstack(vthhtl)
    else:
        tvtvxl = []
        for htitt in htittl[starttime:]:
            for uvvdx in uvvdxl:
                tvxlij = np.dot(np.kron(htitt, uvvdx), tvvec)
                tvtvxl.append(np.dot(convtvvec.T, tvxlij))
        return np.array(tvtvxl).reshape((len(htittl)-starttime)*len(uvvdxl), 1)


def eva_burger_quadratic(tvvec=None, htittl=None, uvvdxl=None, iniv=None,
                         retjacobian=False):
    """evaluate the time-space-pod quadratic Burgers term

    Parameters
    ---
    tvvec: (`hsdim*hqdim`, 1) array
        the current space-time state
    htittl: list of `hsdim` (`hsdim, hsdim`) arrays
        the time component
    uvvdxl: list of `hqdim` (`hqdim`, `hqdim`) arrays
        the space component
    iniv: (`hqdim`, 1) array, optional
        the initial value, if not `None` it replaces the dof associated with
        the first time coefficient
    retjacobian: boolean, optional
        whether to return the jacobian `df/dv (v)`

    Notes
    ---
    The function `burger_quadratic` is of type `v.T*H*v`. Thus the Jacobian
    is defined through `J(v)*w = w.T*H*v + v.T*H*w = v.T*(H+H.T)*w`

    """

    warnings.warn('This function will be deprecated soon.' +
                  ' Rather use `spu.eva_quadform`')

    if iniv is not None:
        tvvec = np.vstack([iniv, tvvec])
    starttime = 0 if iniv is None else 1
    if retjacobian:
        vthhtl = []
        for htitt in htittl[starttime:]:
            for uvvdx in uvvdxl:
                hvij = np.dot(np.kron(htitt, uvvdx), tvvec)
                vthij = np.dot(tvvec.T, np.kron(htitt, uvvdx))
                vthhtl.append(vthij+hvij.T)
        return np.vstack(vthhtl)
    else:
        tvtvxl = []
        for htitt in htittl[starttime:]:
            for uvvdx in uvvdxl:
                tvxlij = np.dot(np.kron(htitt, uvvdx), tvvec)
                tvtvxl.append(np.dot(tvvec.T, tvxlij))
        return np.array(tvtvxl).reshape((len(htittl)-starttime)*len(uvvdxl), 1)


def eva_burger_spacecomp(uvvdxl=None, svec=None):
    evall = []
    for uvvdx in uvvdxl:
        evall.append(np.dot(svec.T, uvvdx.dot(svec)))
    return np.array(evall).reshape((len(uvvdxl), 1))


# functions that convert the burger arrays to lists for saving them as jsons
def nonlty_arraytolist(arraylist):
    listlist = []
    for aritem in arraylist:
        listlist.append(aritem.tolist())
    return listlist


def nonlty_listtoarray(listlist):
    arraylist = []
    for litem in listlist:
        arraylist.append(np.array(litem))
    return arraylist
