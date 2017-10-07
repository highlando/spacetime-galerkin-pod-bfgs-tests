import numpy as np

import matlibplots.conv_plot_utils as cpu


def printit(resultslist=None, fstrslist=None,
            valv=None, valu=None, valfun=None, times=None,
            thing=None, thfstr=None):
    if resultslist is not None:
        fstl = ['.4f']*len(resultslist) if fstrslist is None else fstrslist
        for k, item in enumerate(resultslist):
            cpu.print_nparray_tex(np.array(item).flatten(), fstr=fstl[k])
    if valv is not None:
        cpu.print_nparray_tex(np.array(valv).flatten())
    if valu is not None:
        cpu.print_nparray_tex(np.array(valu).flatten())
    if valfun is not None:
        cpu.print_nparray_tex(np.array(valfun).flatten())
    if times is not None:
        cpu.print_nparray_tex(np.array(times).flatten())
    if thing is not None:
        cpu.print_nparray_tex(np.array(thing), fstr=thfstr)


def checkit(testdict, testroutine=None, parlist=None, parupdate=None,
            vthfstr=None, infostr='### ', extradict={},
            valextradict={},
            printplease=True, onlytimings=False, numbertimings=5):

    '''
    Parameters
    ---
    extradict: dict, optional
        of type {key: ['Name', fstr]}, to extract print extra things
        from the timerinfo dict
    '''

    if onlytimings:
        testdict.update(onlytimings=True)
        timingslistlist = []
        overheadlistlist = []
        xthingslistlistdct = {}
        for xkey in list(extradict.keys()):
            xthingslistlistdct.update({xkey: []})
        for run in range(numbertimings):
            timingslist = []
            overheadlist = []
            xthingslistdct = {}
            for xkey in list(extradict.keys()):
                xthingslistdct.update({xkey: []})
            for parval in parlist:
                parupdate(testdict, parval)
                timing = testroutine(**testdict)
                timingslist.append(timing['elt'])
                for xkey in list(extradict.keys()):
                    xthingslistdct[xkey].append(timing[xkey])

            timingslistlist.append(timingslist)
            overheadlistlist.append(overheadlist)
            for xkey in list(extradict.keys()):
                xthingslistlistdct[xkey].append(xthingslistdct[xkey])

        tarray = np.array(timingslistlist)
        tmin = tarray.min(axis=0)
        timingslistlist.append(tmin.tolist())
        print(infostr)
        printit(resultslist=timingslistlist, thing=parlist, thfstr=vthfstr)
        for xkey in list(extradict.keys()):
            print(extradict[xkey][0])  # the name
            printit(resultslist=xthingslistlistdct[xkey])
        return

    else:
        vallistv, vallistu, vallistfull, timingslist = [], [], [], []

        xthingslistdct = {}
        valxthingslistdct = {}
        for xkey in list(extradict.keys()):
            xthingslistdct.update({xkey: []})
        for xkey in list(valextradict.keys()):
            valxthingslistdct.update({xkey: []})

        for parval in parlist:
            parupdate(testdict, parval)
            value, timerinfo = testroutine(**testdict)
            vallistv.append(value['vterm'])
            try:
                vallistu.append(value['uterm'])
            except KeyError:
                pass
            vallistfull.append(value['value'])
            timingslist.append(timerinfo['elt'])
            for xkey in list(extradict.keys()):
                xthingslistdct[xkey].append(timerinfo[xkey])
            for xkey in list(valextradict.keys()):
                valxthingslistdct[xkey].append(value[xkey].flatten())

        print(infostr)
        printit(valv=vallistv, valu=vallistu,
                valfun=vallistfull, times=timingslist,
                thing=parlist, thfstr=vthfstr)

        for xkey in list(extradict.keys()):
            print(extradict[xkey][0])  # the name
            printit(thing=xthingslistdct[xkey], thfstr=extradict[xkey][1])
        for xkey in list(valextradict.keys()):
            print(valextradict[xkey][0])  # the name
            printit(thing=valxthingslistdct[xkey],
                    thfstr=valextradict[xkey][1])
        return
