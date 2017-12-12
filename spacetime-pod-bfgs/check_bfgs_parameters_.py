from bfgs_run_burger import bfgs_opti

from numtest_setup_utils import checkit

timingonly = False
numbertimings = 5
checkbwdtimings = False
plotplease = False

checkK = False
checkhqhs = False
checknu = False
checkalpha = False
checkgtol = False

# ## make it come true
checkK = True
checkhqhs = True
checknu = True
checkalpha = True
checkgtol = True

# ## other checks
# checkits = True
checkits = False
# checktols = True
checktols = False

baseits = 200  # 0  # it will terminate on its own -- due to precision loss
basegtol = 2.5e-4
basetol = 1e-4
basenu = 5e-3
basealpha = 1./2*6.25e-5
basehq, basehs = 18, 18
altbasehq, altbasehs = 18, 13
Nq, Nts, Ns = 220, 250, 120

checkbase = True
checkalt = True

testitdict = \
    dict(Nq=Nq,  # dimension of the spatial discretization
         Nts=Nts,  # number of time sampling points
         inivtype='step',  # 'ramp', 'smooth'
         dmndct=dict(tE=1., t0=0., x0=0., xE=1.),  # for the plots
         Ns=Ns,  # Number of measurement functions=Num of snapshots
         hq=basehq,  # number of space modes
         hs=basehs,  # number of time points
         redrtol=basetol, redatol=basetol,
         # tol in the solution of the redcd probs
         bfgsiters=baseits,  # number of BFGS iterations
         gtol=basegtol,  # tolerance for the gradient in BFGS
         # spacebasschemes: 'onlyV', 'onlyL', 'VandL', 'combined'
         podstate=True, spacebasscheme='combined',
         podadj=False, podopti=False,
         tikzplease=False, plotplease=plotplease,
         onlytimings=False,
         nu=basenu, alpha=basealpha)

if not checkbwdtimings:  # we need ipython functions later
    logstr = 'logs/' +\
        'sqppodlog_basenu{0}_basealpha{1}_basegtol{4}_basehqhs{2}{3}'.\
        format(basenu, basealpha, basehq, basehs, basegtol) + '_uterm_' +\
        'althqhs{0}{1}'.format(altbasehq, altbasehs)
    import dolfin_navier_scipy.data_output_utils as dou
    dou.logtofile(logstr=logstr)


def mycheckit(parupdate, parlist, infostr=None, vthfstr='3.0f'):
    valxtradict = {'unormsqrd': ['|u|**2', '.4f']}
    if timingonly:
        xtradict = {'overhead': ['fwd call overhead', '.4f'],
                    'nfc': ['n function calls', '3d'],
                    'ngc': ['n gradient calls', '3d']}
    else:
        xtradict = {'nfc': ['n function calls', '3d'],
                    'ngc': ['n gradient calls', '3d']}
    checkit(testitdict, testroutine=bfgs_opti, numbertimings=numbertimings,
            parlist=parlist, vthfstr=vthfstr, infostr=infostr,
            extradict=xtradict, valextradict=valxtradict,
            onlytimings=timingonly, parupdate=parupdate)

if checktols:
    tollist = [basetol*2**(x) for x in range(-1, 4)]

    def updatetestdict(testdict, ctol):
        testdict.update(dict(hq=basehq, hs=basehs, redatol=ctol,
                             redrtol=ctol, bfgsiters=baseits,
                             gtol=basegtol,
                             alpha=basealpha, nu=basenu))

    infostr = '### check the tols -- nu={0}, alpha={1}, N={2}, hqhs={3}-{4}'.\
        format(basenu, basealpha, baseits, basehq, basehs)
    mycheckit(updatetestdict, tollist, infostr=infostr, vthfstr='.2e')

if checkits:
    iterlist = [20, 30, 40, 50, 60, 70]

    def updatetestdict(testdict, citer):
        testdict.update(dict(hq=basehq, hs=basehs, redatol=basetol,
                             gtol=basegtol,
                             redrtol=basetol, bfgsiters=citer,
                             alpha=basealpha, nu=basenu))

    infostr = '### we check iters -- nu={0}, alpha={1}, hqhs={2}-{4}, tol={3}'.\
        format(basenu, basealpha, basehq, basetol, basehs)
    mycheckit(updatetestdict, iterlist, infostr=infostr)

if checkK:
    hqhslist = [(10, 10), (12, 12), (15, 15), (18, 18), (21, 21), (25, 25)]

    def updatetestdict(testdict, hqhs):
        testdict.update(dict(hq=hqhs[0], hs=hqhs[1], redatol=basetol,
                             gtol=basegtol,
                             redrtol=basetol, bfgsiters=baseits,
                             alpha=basealpha, nu=basenu))

    infostr = '### we check hqhs -- nu={0}, alpha={1}, N={2}, tol={3}'.\
        format(basenu, basealpha, baseits, basetol)
    mycheckit(updatetestdict, hqhslist, infostr=infostr)

if checkhqhs:
    hqupds = [-5, -3, -2,  1, +2,  0]
    hsupds = [0,   1,  2, -3, -2, -5]
    hqhslist = []
    for idx in range(len(hqupds)):
        hqhslist.append((basehq+hqupds[idx], basehs+hsupds[idx]))

    def updatetestdict(testdict, hqhs):
        testdict.update(dict(hq=hqhs[0], hs=hqhs[1], redatol=basetol,
                             gtol=basegtol,
                             redrtol=basetol, bfgsiters=baseits,
                             alpha=basealpha, nu=basenu))

    infostr = '### we check hqhs -- nu={0}, alpha={1}, N={2}, tol={3}'.\
        format(basenu, basealpha, baseits, basetol)
    mycheckit(updatetestdict, hqhslist, infostr=infostr)


if checknu:
    viscolist = [10**(-3)*2**(x) for x in range(-1, 6)]

    if checkbase:
        def updatetestdict(tstdct, cnu):
            tstdct.update(dict(hq=basehq, hs=basehs, alpha=basealpha,
                               redatol=basetol, redrtol=basetol, gtol=basegtol,
                               bfgsiters=baseits, nu=cnu))

        infostr = '### we check nu -- alpha={0}, N={1}, tol={2}, (hq,hs)=({3},{4})'.\
            format(basealpha, baseits, basetol, basehq, basehs)
        mycheckit(updatetestdict, viscolist, infostr=infostr, vthfstr='.2e')

    if checkalt:
        def updatetestdict(tstdct, cnu):
            tstdct.update(dict(hq=altbasehq, hs=altbasehs, alpha=basealpha,
                               redatol=basetol, redrtol=basetol, gtol=basegtol,
                               bfgsiters=baseits, nu=cnu))

        infostr = '### we check nu -- alpha={0}, N={1}, tol={2}, (hq,hs)=({3},{4})'.\
            format(basealpha, baseits, basetol, altbasehq, altbasehs)
        mycheckit(updatetestdict, viscolist, infostr=infostr, vthfstr='.2e')

if checkalpha:
    alphalist = [basealpha*2**(x) for x in range(-3, 3)]

    if checkbase:
        def updatetestdict(tstdct, alpha):
            tstdct.update(dict(hq=basehq, hs=basehs, alpha=alpha,
                               redatol=basetol, gtol=basegtol,
                               redrtol=basetol, bfgsiters=baseits, nu=basenu))

        infostr = '### we check alpha -- nu={0}, N={1}, tol={2}, (hq,hs)=({3},{4})'.\
            format(basenu, baseits, basetol, basehq, basehs)
        mycheckit(updatetestdict, alphalist, infostr=infostr, vthfstr='.2e')

    if checkalt:
        def updatetestdict(tstdct, alpha):
            tstdct.update(dict(hq=altbasehq, hs=altbasehs, alpha=alpha,
                               gtol=basegtol,
                               redatol=basetol, redrtol=basetol,
                               bfgsiters=baseits, nu=basenu))

        infostr = '### we check alpha -- nu={0}, N={1}, tol={2}, (hq,hs)=({3},{4})'.\
            format(basenu, baseits, basetol, altbasehq, altbasehs)
        mycheckit(updatetestdict, alphalist, infostr=infostr, vthfstr='.2e')

if checkgtol:
    gtollist = [basegtol*2**(.5*x) for x in range(-1, 6)]
    # gtollist = [basegtol*2**(.5*x) for x in range(-3, -1)]

    if checkbase:
        def updatetestdict(testdict, cgtol):
            testdict.update(dict(hq=basehq, hs=basehs, redatol=basetol,
                                 redrtol=basetol, bfgsiters=baseits,
                                 alpha=basealpha, nu=basenu, gtol=cgtol))

        infostr = '### we check gtol -- nu={0}, alpha={1}, hqhs={2}-{4}, tol={3}'.\
            format(basenu, basealpha, basehq, basetol, basehs)
        mycheckit(updatetestdict, gtollist, infostr=infostr, vthfstr='.2e')

    if checkalt:
        def updatetestdict(testdict, cgtol):
            testdict.update(dict(hq=altbasehq, hs=altbasehs, redatol=basetol,
                                 redrtol=basetol, bfgsiters=baseits,
                                 alpha=basealpha, nu=basenu, gtol=cgtol))

        infostr = '### we check gtol -- nu={0}, alpha={1}, hqhs={2}-{4}, tol={3}'.\
            format(basenu, basealpha, altbasehq, basetol, altbasehs)
        mycheckit(updatetestdict, gtollist, infostr=infostr, vthfstr='.2e')


if checkbwdtimings:
    testitdict.update(dict(checkbwdtimings=True))
    testitdict.update(hq=basehq, hs=basehs, alpha=basealpha, gtol=basegtol,
                      redatol=basetol, redrtol=basetol, nu=basenu)
    eva_redbwd, vv = bfgs_opti(**testitdict)
    (sol, odeintoutput, redbwdrhs_tns,
     redbwdrhs_vdx, reduceda, vfun) = eva_redbwd(vv, debug=True)
    midvindx = basehs/2
    print('number of function evaluations: {0}'.format
          (odeintoutput['nfe'][-1]))
    iniv = vv[0*basehq:basehq].flatten()
    midv = vv[midvindx*basehq:(midvindx+1)*basehq].flatten()
    endv = vv[-1*basehq:].flatten()
    print('do some \n\n%timeit redbwdrhs_vdx(iniv, 0.0) \n')
    print('do some \n\n%timeit redbwdrhs_tns(iniv, 0.0) \n')
    print('do some \n\n%timeit redbwdrhs(midv, 0.5) \n')
    print('do some \n\n%timeit redbwdrhs(endv, 1.) \n')
    print('and multiply it with {0}'.format(odeintoutput['nfe'][-1]) +
          ' to estimate the time spent on the nonlinearity')
    print('however, substract the time for the interpolation of v: \n')
    print('%timeit vfun(0.55555) \n')
    print('\ncompare it to \n\n%timeit np.dot(reduceda, midv) \n')
    print('to see what could be saved...')
    # from pycallgraph import PyCallGraph
    # from pycallgraph.output import GraphvizOutput
    # with PyCallGraph(output=GraphvizOutput()):
    #     sol = eva_redbwd(vv)
