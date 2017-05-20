import numpy as np

from run_burger_optcont import testit
import matlibplots.conv_plot_utils as cpu

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

testitdict = \
    dict(Nq=220,  # dimension of the spatial discretization
         Nts=250,  # number of time sampling points
         dmndct=dict(tE=1., t0=0., x0=0., xE=1.),  # space time domain
         inivtype='step',  # 'ramp', 'smooth'
         Ns=120,  # Number of measurement functions=Num of snapshots
         hq=12,  # number of space modes
         hs=12,  # number of time modes
         spacebasscheme='combined',  # 'onlyL' 'VandL' 'combined'
         plotplease=False, tikzplease=False,
         nu=basenu,
         alpha=basealpha,
         genpodcl=True)

# value, timerinfo = testit(**testitdict)


def printit(resultslist=None, fstrslist=None,
            valv=None, valfun=None, times=None,
            thing=None, thfstr=None):
    if resultslist is not None:
        fstl = ['.4f']*len(resultslist) if fstrslist is None else fstrslist
        for k, item in enumerate(resultslist):
            cpu.print_nparray_tex(np.array(item).flatten(), fstr=fstl[k])
    else:
        cpu.print_nparray_tex(np.array(valv).flatten())
        cpu.print_nparray_tex(np.array(valfun).flatten())
        cpu.print_nparray_tex(np.array(times).flatten())
    if thing is not None:
        cpu.print_nparray_tex(np.array(thing), fstr=thfstr)


def checkit(testdict, parlist=None, parupdate=None, vthfstr=None,
            infostr='### ',
            printplease=True, onlytimings=False, numbertimings=5):
    if onlytimings:
        testdict.update(onlytimings=True)
        timingslistlist = []
        for run in range(numbertimings):
            timingslist = []
            for parval in parlist:
                parupdate(testdict, parval)
                timerinfo = testit(**testdict)
                timingslist.append(timerinfo)
            timingslistlist.append(timingslist)

        tarray = np.array(timingslistlist)
        tmin = tarray.min(axis=0)
        timingslistlist.append(tmin.tolist())
        print infostr
        printit(resultslist=timingslistlist, thing=parlist, thfstr=vthfstr)
        return

    vallistv, vallistfull, timingslist = [], [], []
    for parval in parlist:
        parupdate(testdict, parval)
        value, timerinfo = testit(**testdict)
        vallistv.append(value['vterm'])
        vallistfull.append(value['value'])
        timingslist.append(timerinfo)
    print infostr
    printit(valv=vallistv, valfun=vallistfull, times=timingslist,
            thing=parlist, thfstr=vthfstr)


# ### checking the number of bas funcs
if basnumbercheck:
    hqhs = [(6, 6), (9, 9), (12, 12), (18, 18), (24, 24)]

    def updatetestdict(testdict, hqhs):
        testdict.update(dict(hq=hqhs[0], hs=hqhs[1],
                             alpha=basealpha, nu=basenu))

    infostr = '### nu={0}, alpha={1}'.\
        format(testitdict['nu'], testitdict['alpha'])
    checkit(testitdict, parlist=hqhs, vthfstr='3.0f', infostr=infostr,
            onlytimings=timingonly, parupdate=updatetestdict)

if spatimbascheck:
    hqhs = [(18, 6), (17, 7), (16, 8), (14, 10), (12, 12), (10, 14), (8, 16)]

    def updatetestdict(testdict, hqhs):
        testdict.update(dict(hq=hqhs[0], hs=hqhs[1],
                             nu=basenu, alpha=basealpha))
    infostr = '### nu={0}, alpha={1}'.\
        format(testitdict['nu'], testitdict['alpha'])
    checkit(testitdict, parlist=hqhs, vthfstr='4.0f', infostr=infostr,
            onlytimings=timingonly, parupdate=updatetestdict)

if viscocheck:
    viscolist = [10**(-3)*2**(x) for x in range(-1, 6)]

    chq, chs = 16, 8
    if makeapic:
        chq, chs = 12, 12
        testitdict.update(hq=chq, hs=chs, nu=viscolist[3], plotplease=True)
        print '### hq={0}, hs={1}, nu={2}, alpha={3}'.\
            format(chq, chs, testitdict['nu'], testitdict['alpha'])
        _, _ = testit(**testitdict)
        raise Warning('We only wanna plot')

    def updatetestdict(testdict, cnu):
        testitdict.update(dict(hq=chq, hs=chs, nu=cnu, alpha=basealpha))

    infostr = '### hq={0}, hs={1}, alpha={2}'.\
        format(chq, chs, testitdict['alpha'])
    checkit(testitdict, parlist=viscolist, vthfstr='.1e', infostr=infostr,
            onlytimings=timingonly, parupdate=updatetestdict)

    chq, chs = 12, 12

    def updatetestdict(testdict, cnu):
        testitdict.update(dict(hq=chq, hs=chs, nu=cnu, alpha=basealpha))

    infostr = '### hq={0}, hs={1}, alpha={2}'.\
        format(chq, chs, testitdict['alpha'])
    checkit(testitdict, parlist=viscolist, vthfstr='.1e', infostr=infostr,
            onlytimings=timingonly, parupdate=updatetestdict)

if alphacheck:
    # alphalist = [2**(-x) for x in range(12, 15)]
    alphalist = [10**(-3)*2**(x) for x in range(-2, 5)]
    chq, chs = 16, 8
    # chq, chs = 18, 14
    if makeapic:
        testitdict.update(hq=chq, hs=chs, alpha=alphalist[-1], plotplease=True)
        print '### hq={0}, hs={1}, nu={2}, alpha={3}'.\
            format(chq, chs, testitdict['nu'], testitdict[-1])
        _, _ = testit(**testitdict)
        raise Warning('We only wanna plot')

    def updatetestdict(testdict, calpha):
        testitdict.update(dict(hq=chq, hs=chs, nu=basenu, alpha=calpha))

    infostr = '### hq={0}, hs={1}, nu={2}'.format(chq, chs, testitdict['nu'])
    checkit(testitdict, parlist=alphalist, vthfstr='.2e', infostr=infostr,
            onlytimings=timingonly, parupdate=updatetestdict)

    chq, chs = 12, 12

    def updatetestdict(testdict, calpha):
        testitdict.update(dict(hq=chq, hs=chs, nu=basenu, alpha=calpha))

    infostr = '### hq={0}, hs={1}, nu={2}'.format(chq, chs, testitdict['nu'])
    checkit(testitdict, parlist=alphalist, vthfstr='.2e', infostr=infostr,
            onlytimings=timingonly, parupdate=updatetestdict)

if makeapic:
    testitdict.update(dict(hq=12, hs=12, nu=basenu, alpha=basealpha,
                      plotplease=True))
    _, _ = testit(**testitdict)
    # testitdict.update(dict(hq=16, hs=8, nu=2e-3, alpha=basealpha,
    #                   plotplease=True))
    # _, _ = testit(**testitdict)
