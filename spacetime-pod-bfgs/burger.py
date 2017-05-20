import numpy as np
import matplotlib.pyplot as plt

import dolfin_navier_scipy.data_output_utils as dou

import sadptprj_riclyap_adi.lin_alg_utils as lau

import gen_pod_utils as gpu
import spacetime_pod_utils as spu
import dolfin_burgers_scipy as dbs
import burgers_genpod_utils as bgu


if __name__ == '__main__':
    N = 200
    nu = 0.1
    t0, tE, Nts = 0.0, 1.0, 500
    sdim, hsdim, hydim = 33, 10, 20
    genpod = True
    basfuntype = 'hpl'
    iniv = np.r_[np.zeros((np.floor(N*.5), 1)), np.ones((np.ceil(N*.5)-1, 1))]
    # Nts is just the mesh where to store the values
    # the actual time grid is chosen by the solver through step size control

    tmesh = np.linspace(t0, tE, Nts)
    datastr = 'burgv_nu{1}_N{0}_Nts{2}'.format(N, nu, Nts)
    gpodstr = 'genpod{2}_hydim{0}_sdim{1}_basefuntyp{3}'.\
        format(hydim, sdim, genpod, basfuntype)
    print 'reduced model through ' + gpodstr
    # rfile = dolfin.File('results/' + datastr + '.pvd')

    M, A, rhs, nfunc, femp = dbs.burgers_spacedisc(N=N, nu=nu, retfemdict=True)

    # simudict = dict(iniv=iniv, A=A, M=M, nfunc=nfunc, rhs=rhs, tmesh=tmesh)
    simudict = dict(iniv=iniv, A=A, M=M, nfunc=None, rhs=rhs, tmesh=tmesh)
    vv = dou.load_or_comp(filestr='data/'+datastr, comprtn=time_int_semil,
                          comprtnargs=simudict, arraytype='dense', debug=True)

    # plotmat(vv)
    # (myx, my) = gpu.get_genmeasuremat(sol=vv.T, tmesh=tmesh, sdim=17)
    # print myx.shape
    # Uk = gpu.get_genpodmats(sol=vv.T, poddim=poddim, sdim=sdim, tmesh=tmesh,
    #                         plotsvs=True, basfuntype='pl')

    ared, mred, nonlred, rhsred, inired, Uky = gpu.\
        get_podred_model(M=M, A=A, nonl=nfunc, rhs=rhs,
                         sol=vv.T, tmesh=tmesh, verbose=True,
                         poddim=hydim, sdim=sdim,
                         genpod=genpod, basfuntype=basfuntype)
    # sdd = dict(iniv=inired, A=ared, M=mred, nfunc=nonlred, rhs=rhsred,
    #            tmesh=tmesh)
    # vvd = dou.load_or_comp(filestr='data/'+datastr+gpodstr, comprtn=ti_burg,
    #                        comprtnargs=sdd,
    #                        arraytype='dense', debug=True)

    # plotmat((np.dot(Uk, vvd.T)).T-vv)
    # # print np.linalg.norm(vvd - vv)
    # # plotmat((np.dot(rmd['Uk'], vvd.T)).T)
    # # plotmat(vv)
    # print np.linalg.norm((np.dot(Uk, vvd.T)).T-vv)/np.linalg.norm(vv)

    (msx, ms) = gpu.get_genmeasuremat(sol=vv.T, tmesh=tmesh, sdim=sdim)
    import scipy.sparse as sps
    x = lau.apply_massinv(sps.csc_matrix(ms), msx.T).T
    # sini = np.r_[1, np.zeros((sdim-1, ))].reshape((sdim, 1))
    # xz = np.copy(x)
    # xz[:, 0] = 0  # zero out nu0 - the ini condition needs extra treatment
    # Uks = gpu.get_podmats(sol=xz.T, poddim=hsdim-1, plotsvs=False, M=M)
    # Uks = np.hstack([sini, Uks])
    Uks = gpu.get_podmats(sol=x.T, poddim=hsdim, plotsvs=False, M=M)

    uvvdxl = dbs.get_burgertensor_spacecomp(podmat=Uky, **femp)
    htittl = bgu.get_burgertensor_timecomp(podmat=Uks, sdim=sdim,
                                           tmesh=tmesh, basfuntype='pl')

    # hsini = np.dot(Uks.T, sini)
    # hshyini = np.kron(hsini, inired.reshape((hydim, 1)))
    # iniproj = spu.expand_stpodsol(stpsol=hshyini, Uks=Uks, Uky=Uky)
    # plotmat(iniproj)
    hshysol = np.dot(Uks.T, np.dot(x.T, Uky)).\
        reshape((hsdim*hydim, 1), order='C')

    def evabrgquadterm(tvvec):
        return bgu.\
            eva_burger_quadratic(tvvec=tvvec, htittl=htittl, uvvdxl=uvvdxl)
    dms = gpu.get_dms(sdim=sdim, tmesh=tmesh, basfuntype='pl')
    hdms = np.dot(Uks.T, np.dot(dms, Uks))
    hms = np.dot(Uks.T, np.dot(ms, Uks))
    timespaceres = spu.get_spacetimepodres(tvvec=hshysol, dms=hdms, ms=hms,
                                           my=mred, ared=ared,
                                           nfunc=evabrgquadterm, rhs=None)
    print timespaceres
    # solproj = spu.expand_stpodsol(stpsol=hshysol, Uks=Uks, Uky=Uky)
    # plotmat(solproj)
