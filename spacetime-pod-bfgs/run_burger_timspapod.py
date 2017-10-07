import numpy as np
import json

import dolfin_navier_scipy.data_output_utils as dou
# import sadptprj_riclyap_adi.lin_alg_utils as lau
import burgers_genpod_utils as bgu
import gen_pod_utils as gpu
import spacetime_pod_utils as spu

from plot_utils import plotmat

__all__ = ['solnonlintimspasys']

# steps of the algorithm
# 1. get the snapshots
#   needs: Nts, Nq, Ns, [nu_1, ..., nu_k]
#   returns: My, Ms, [Y_1, ..., Y_k]
# 2. get the pod basis
#   needs: hs, hq, My, Ms, [Y_1, ..., Y_k]
#   returns, Uky, Uks
# 3. get the POD model
#   needs: Uky, Uks, nu
#   return: hms, hdms, hmy, ha, hrhs, burger_tensor
# 4. comp the approximation
#

# coeffs of the test
Nq = 130  # dimension of the spatial discretization
Ns = 65  # dimension of the temporal discretization
Nts = 500  # number of time sampling points

# # coeffs of the tries
# Nq = 100
# Ns = 65
# Nts = 100

t0, tE = 0., 1.
plotsvs = False
timebasfuntype = 'pl'
tmesh = np.linspace(t0, tE, Nts)
iniv = np.r_[np.ones(((Nq-1)/2, 1)), np.zeros(((Nq)/2, 1))]

'''
def solnonlintimspasys(dms=None, ms=None, ay=None, my=None, iniv=None,
                       htittl=None, uvvdxl=None, rhs=None):
    import scipy.optimize as sco
    nlslvini = np.tile(iniv.flatten(), lns-1)
    # nlslvini = np.zeros(((lns-1)*lnq, ))
    with dou.Timer('solve the nonlin system'):
        sol = sco.fsolve(nonlinres, nlslvini,
                         fprime=nlrprime)
    fsol = np.hstack([iniv, sol.reshape((lns-1, lnq)).T])
    return fsol
'''


def solnonlintimspasys(iniv=None, nonlinres=None, nlrprime=None,
                       lns=None, lnq=None):

    import scipy.optimize as sco
    nlslvini = np.tile(iniv.flatten(), lns-1)
    # nlslvini = np.zeros(((lns-1)*lnq, ))
    with dou.Timer('solve the nonlin system'):
        sol = sco.fsolve(nonlinres, nlslvini,
                         fprime=nlrprime)
    fsol = np.hstack([iniv, sol.reshape((lns-1, lnq)).T])
    return fsol


def comp_spatimepodgal(nul=[1e-2], hq=12, hs=12, nu=None,
                       plotplease=False, comperr=False, verbose=False):

    snapsl, My, Ms = bgu.get_snaps_burger(tmesh=tmesh, iniv=iniv,
                                          Nq=Nq, Ns=Ns, nul=nul)
    Uky, Uks = gpu.get_timspapar_podbas(hs=hs, hq=hq, plotsvs=plotsvs,
                                        My=My, Ms=Ms, snapsl=snapsl)

    nustr = 'nu{0}'.format(nul[0])
    for curnu in nul[1:]:
        nustr = nustr + '_{0}'.format(curnu)

    # get the dict with the coeffs of the reduced burger system
    hs, hq = Uks.shape[1], Uky.shape[1]
    gpnonldatastr = 'data/Nq{0}Ns{1}hs{2}hq{3}Nts{4}nu{5}t0{6}tE{7}'.\
        format(Nq, Ns, hs, hq, Nts, nustr, t0, tE) + '_genpodburgernonlrty'
    rbd = bgu.get_podredmod_burger(Uky=Uky, Uks=Uks, Nq=Nq, Ns=Ns, tmesh=tmesh,
                                   timebasfuntype=timebasfuntype,
                                   nu=nu, iniv=iniv, datastr=gpnonldatastr,
                                   plotsvs=False)

    # ssol = solnonlintimspasys(dms=rbd['hdms'], ms=rbd['hms'], ay=rbd['hay'],
    #                           my=rbd['hmy'], iniv=rbd['inivred'],
    #                           uvvdxl=rbd['uvvdxl'], htittl=rbd['htittl'],
    #                           rhs=rbd['hrhs'])

    def nonlinres(tvvec):
        return spu.\
            get_spacetimepodres(tvvec=tvvec, dms=rbd['hdms'], ms=rbd['hms'],
                                my=rbd['hmy'], ared=rbd['hay'],
                                nfunc=rbd['hnonl'],
                                retnorm=False, iniv=rbd['inivred']).flatten()

    def nlrprime(tvvec):
        return rbd['nonlresjacobian'](tvvec, curiniv=rbd['inivred'])

    ssol = solnonlintimspasys(iniv=rbd['inivred'], lns=hs, lnq=hq,
                              nonlinres=nonlinres,
                              nlrprime=nlrprime)

    fssolt = np.dot(np.dot(Uks, ssol.T), Uky.T)
    if plotplease:
        plotmat(fssolt, fignum=124, tikzfile='results/reduced sol')

    if comperr:
        with dou.Timer('solve the full system'):
            fullsol, _, _ = bgu.get_snaps_burger(tmesh=tmesh, Nq=Nq, Ns=Ns,
                                                 iniv=iniv, nul=[nu])
        fullsol = fullsol[0]
        if plotplease:
            plotmat(fssolt - fullsol.T, fignum=144,
                    tikzfile='results/difference')
            plotmat(fullsol.T, fignum=155, tikzfile='results/full sol')
        solnorm = spu.timspanorm(fullsol, mspace=My, mtime=Ms)
        apprxerr = spu.\
            timspanorm(fullsol-fssolt.T, mspace=My, mtime=Ms)/solnorm
        if verbose:
            print('Relative time/space approximation error: ', apprxerr)
        return apprxerr[0][0]
    else:
        return None

if __name__ == '__main__':
    factest = False
    nutest = True
    nul = [1e-2, 3.e-3, 1e-3]
    # nul = [1e-2]  # [1e-2, 3.e-3, 1e-3]
    nutestlist = np.logspace(-2, -3, 7)
    nutestlist = np.r_[10**(-1.9), nutestlist, 10**(-3.1)]

    smallnutestlist = np.logspace(-2, -3, 5)
    spatimpoddict = dict(nul=nul,  # [1e-3],  # , 1e-3],
                         hq=15,
                         hs=15,
                         nu=2e-3,
                         plotplease=True,
                         verbose=True,
                         comperr=True)
    aprxerr = comp_spatimepodgal(**spatimpoddict)
    raise Warning('TODO: debug')

    import matlibplots.conv_plot_utils as cpu
    if nutest:
        Klist = [20, 30, 40]
        cfac = 0.5
        datastr = 'results/nutest{0}{1}{2}K203040'.\
            format(nutestlist[0], nutestlist[-1], len(nutestlist))
        Kapprxl, leglist = [], []
        for K in Klist:
            hq = np.floor(cfac*K)
            hs = np.ceil((1-cfac)*K)
            spatimpoddict.update(dict(hq=hq, hs=hs))
            apprxl = []
            for cnu in nutestlist:
                spatimpoddict.update(dict(nu=cnu))
                apprxl.append(comp_spatimepodgal(**spatimpoddict))

            Kapprxl.append(apprxl)
            leglist.append('$K={0}$'.format(K))

        fctdct = dict(abscissa=nutestlist.tolist(),
                      datalist=Kapprxl,
                      leglist=leglist)

        f = open(datastr, 'w')
        f.write(json.dumps(fctdct))
        print('results dumped into json:', datastr)
        f.close()

        cpu.para_plot(**fctdct)

    if factest:
        f0, fe, nfs = .2, .8, 7
        datastr = 'results/factest{0}{1}{2}K30'.format(f0, fe, nfs)
        faclist = np.linspace(f0, fe, nfs)
        if nfs == 7:  # TODO: softcode the adaptivity towards fac=.5
            faclist = np.array([.2, .35, .425, .5, .575, .65, .8])

        K = 30
        nuapprxl, leglist = [], []
        for cnu in smallnutestlist:
            spatimpoddict.update(dict(nu=cnu))
            apprxl = []
            for cfac in faclist:
                hq = np.floor(cfac*K)
                hs = np.ceil((1-cfac)*K)
                spatimpoddict.update(dict(hq=hq, hs=hs))
                apprxl.append(comp_spatimepodgal(**spatimpoddict))
            nuapprxl.append(apprxl)
            leglist.append('$\\mu={0}$'.format(cnu))

        fctdct = dict(abscissa=faclist.tolist(),
                      datalist=nuapprxl,
                      leglist=leglist)

        f = open(datastr, 'w')
        f.write(json.dumps(fctdct))
        print('results dumped into json:', datastr)
        f.close()

        cpu.para_plot(**fctdct)
