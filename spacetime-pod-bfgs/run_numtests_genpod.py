# import numpy as np

from run_burger_optcont import testit
# import matlibplots.conv_plot_utils as cpu

from numtest_setup_utils import checkit

target = 'inival'
target = 'heart'

basnumbercheck = False
spatimbascheck = False
viscocheck = False
alphacheck = False

makeapic = True
makeapic = False

timingonly = False  # only timings for the optimization
timingonly = True

# ### make it come true
basnumbercheck = True
spatimbascheck = True
viscocheck = True
alphacheck = True

# ### define the problem
basenu = 5e-3
basealpha = 1e-3
basehq, basehs = 12, 12
altbasehq, altbasehs = 16, 8
if target == 'heart':
    altbasehq, altbasehs = 10, 15
    # basealpha = 1e-3

basecheck = True
altcheck = True

testitdict = \
    dict(Nq=220,  # dimension of the spatial discretization
         Nts=250,  # number of time sampling points
         dmndct=dict(tE=1., t0=0., x0=0., xE=1.),  # space time domain
         inivtype='step',  # 'ramp', 'smooth'
         target=target,
         Ns=120,  # Number of measurement functions=Num of snapshots
         hq=basehq,  # number of space modes
         hs=basehs,  # number of time modes
         spacebasscheme='combined',  # 'onlyL' 'VandL' 'combined'
         plotplease=False, tikzplease=False,
         nu=basenu,
         alpha=basealpha,
         genpodcl=True)

logstr = 'logs/genpodlog_' + target + '_basenu{0}_basealpha{1}_basehqhs{2}{3}'.\
    format(basenu, basealpha, basehq, basehs) + \
    'althqhs{0}{1}'.format(altbasehq, altbasehs) + '_uterm'
if timingonly:
    logstr = logstr + '__timings'
import dolfin_navier_scipy.data_output_utils as dou
dou.logtofile(logstr=logstr)


def mycheckit(parupdate, parlist, infostr=None, vthfstr='3.0f'):
    valxtradict = {'unormsqrd': ['|u|**2', '.4f']}
    # if timingonly:
    #     xtradict = {'overhead': ['fwd call overhead', '.4f'],
    #                 'nfc': ['n function calls', '3d'],
    #                 'ngc': ['n gradient calls', '3d']}
    # else:
    #     xtradict = {'nfc': ['n function calls', '3d'],
    #                 'ngc': ['n gradient calls', '3d']}
    checkit(testitdict, testroutine=testit, numbertimings=5,
            valextradict=valxtradict,
            parlist=parlist, vthfstr=vthfstr, infostr=infostr,
            onlytimings=timingonly, parupdate=parupdate)


# def printit(resultslist=None, fstrslist=None,
#             valv=None, valfun=None, times=None,
#             thing=None, thfstr=None):
#     if resultslist is not None:
#         fstl = ['.4f']*len(resultslist) if fstrslist is None else fstrslist
#         for k, item in enumerate(resultslist):
#             cpu.print_nparray_tex(np.array(item).flatten(), fstr=fstl[k])
#     else:
#         cpu.print_nparray_tex(np.array(valv).flatten())
#         cpu.print_nparray_tex(np.array(valfun).flatten())
#         cpu.print_nparray_tex(np.array(times).flatten())
#     if thing is not None:
#         cpu.print_nparray_tex(np.array(thing), fstr=thfstr)


# def checkit(testdict, testroutine=None, parlist=None, parupdate=None,
#             vthfstr=None, infostr='### ',
#             printplease=True, onlytimings=False, numbertimings=5):
#     if testroutine is None:
#         testroutine = testit
#     if onlytimings:
#         testdict.update(onlytimings=True)
#         timingslistlist = []
#         for run in range(numbertimings):
#             timingslist = []
#             for parval in parlist:
#                 parupdate(testdict, parval)
#                 timerinfo = testroutine(**testdict)
#                 timingslist.append(timerinfo)
#             timingslistlist.append(timingslist)
#         tarray = np.array(timingslistlist)
#         tmin = tarray.min(axis=0)
#         timingslistlist.append(tmin.tolist())
#         print infostr
#         printit(resultslist=timingslistlist, thing=parlist, thfstr=vthfstr)
#         return
#     vallistv, vallistfull, timingslist = [], [], []
#     for parval in parlist:
#         parupdate(testdict, parval)
#         value, timerinfo = testroutine(**testdict)
#         vallistv.append(value['vterm'])
#         vallistfull.append(value['value'])
#         timingslist.append(timerinfo)
#     print infostr
#     printit(valv=vallistv, valfun=vallistfull, times=timingslist,
#             thing=parlist, thfstr=vthfstr)


# ### checking the number of bas funcs
if basnumbercheck:
    hqhs = [(6, 6), (9, 9), (12, 12), (18, 18), (24, 24)]

    def updatetestdict(testdict, hqhs):
        testdict.update(dict(hq=hqhs[0], hs=hqhs[1],
                             alpha=basealpha, nu=basenu))

    infostr = '### we check K: nu={0}, alpha={1}'.\
        format(testitdict['nu'], testitdict['alpha'])
    mycheckit(updatetestdict, hqhs, vthfstr='3.0f', infostr=infostr)

if spatimbascheck:
    # hqhs = [(18, 6), (17, 7), (16, 8), (14, 10), (12, 12), (10, 14), (8, 16)]
    hqupds = [+4,  3,  0,  0, -2, -2, -4]
    hsupds = [-4, -2, -2,  0,  0,  3,  4]
    # if target == 'heart':
    #     hqupds = [+3, -1]
    #     hsupds = [-1,  2]
    hqhs = []
    for idx in range(len(hqupds)):
        hqhs.append((basehq+hqupds[idx], basehs+hsupds[idx]))

    def updatetestdict(testdict, hqhs):
        testdict.update(dict(hq=hqhs[0], hs=hqhs[1],
                             nu=basenu, alpha=basealpha))
    infostr = '### we check hqhs: nu={0}, alpha={1}'.\
        format(testitdict['nu'], testitdict['alpha'])
    mycheckit(updatetestdict, hqhs, vthfstr='4.0f', infostr=infostr)

if viscocheck:
    viscolist = [10**(-3)*2**(x) for x in range(-1, 6)]

    if makeapic:
        testitdict.update(hq=altbasehq, hs=altbasehs, nu=viscolist[3],
                          plotplease=True)
        print '### hq={0}, hs={1}, nu={2}, alpha={3}'.\
            format(altbasehq, altbasehs, testitdict['nu'], testitdict['alpha'])
        _, _ = testit(**testitdict)
        raise Warning('We only wanna plot')

    if altcheck:
        def updatetestdict(testdict, cnu):
            testitdict.update(dict(hq=altbasehq, hs=altbasehs, nu=cnu,
                                   alpha=basealpha))

        infostr = '### we check nu: hq={0}, hs={1}, alpha={2}'.\
            format(altbasehq, altbasehs, testitdict['alpha'])
        mycheckit(updatetestdict, viscolist, vthfstr='.1e', infostr=infostr)

    if basecheck:
        print 'we check nu...'

        def updatetestdict(testdict, cnu):
            testitdict.update(dict(hq=basehq, hs=basehs, nu=cnu,
                                   alpha=basealpha))

        infostr = '### we check nu: hq={0}, hs={1}, alpha={2}'.\
            format(basehq, basehs, testitdict['alpha'])
        mycheckit(updatetestdict, viscolist, vthfstr='.1e', infostr=infostr)

if alphacheck:
    # alphalist = [2**(-x) for x in range(12, 15)]
    alphalist = [basealpha*2**(x) for x in range(-2, 5)]
    if makeapic:
        testitdict.update(hq=altbasehq, hs=altbasehs, alpha=alphalist[-1],
                          plotplease=True)
        print '### hq={0}, hs={1}, nu={2}, alpha={3}'.\
            format(altbasehq, altbasehs, testitdict['nu'], alphalist[-1])
        _, _ = testit(**testitdict)
        raise Warning('We only wanna plot')

    if altcheck:

        def updatetestdict(testdict, calpha):
            testitdict.update(dict(hq=altbasehq, hs=altbasehs, nu=basenu,
                                   alpha=calpha))

        infostr = '### we check alpha: hq={0}, hs={1}, nu={2}'.\
            format(altbasehq, altbasehs, basenu)
        mycheckit(updatetestdict, alphalist, vthfstr='.2e', infostr=infostr)

    if basecheck:
        print 'we check alpha ...'

        def updatetestdict(testdict, calpha):
            testitdict.update(dict(hq=basehq, hs=basehs, nu=basenu,
                                   alpha=calpha))

        infostr = '### we check alpha: hq={0}, hs={1}, nu={2}'.\
            format(basehq, basehs, basenu)
        mycheckit(updatetestdict, alphalist, vthfstr='.2e', infostr=infostr)

if makeapic:
    testitdict.update(dict(hq=basehq, hs=basehs, nu=basenu, alpha=basealpha,
                      plotplease=True))
    _, _ = testit(**testitdict)
    # testitdict.update(dict(hq=16, hs=8, nu=2e-3, alpha=basealpha,
    #                   plotplease=True))
    # _, _ = testit(**testitdict)
