import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

import genpod_opti_utils as gou

from plot_utils import plotmat

N = 2000  # discretization in polar coordinates
no, nt = 30, 30  # slices of the quadrants
NS, NY = 100, 100  # space time discretization

plotplease = False


def heartfun(t):
    return (np.sin(t)*np.sqrt(np.abs(np.cos(t)))/(np.sin(t)+1.4) -
            2*np.sin(t) + 2)

thetavec = np.linspace(0, 2*np.pi, N)
rvec = heartfun(thetavec)

if plotplease:
    plt.figure(1)
    plt.polar(thetavec, rvec)
    plt.show(block=False)

colist = []
for k in np.arange(N):
    yco = 0.2*(rvec[k]*np.sin(thetavec[k]+np.pi/2) + 2.5)
    xco = 0.2*(rvec[k]*np.cos(thetavec[k]+np.pi/2) + .8)
    colist.append(np.array([[yco, xco]]))

coarray = np.vstack(colist)
if plotplease:
    plt.figure(2)
    plt.plot(coarray[:, 1], coarray[:, 0])
    plt.show(block=False)

# get the upper half
inds = np.lexsort(np.fliplr(coarray).T)
Nhalf = np.int(np.floor(N/2))
scoarray = coarray[inds, :]
scoarray = scoarray[Nhalf:, :]

if plotplease:
    plt.figure(3)
    plt.plot(scoarray[:, 1], scoarray[:, 0])
    plt.show(block=False)

tcors = scoarray[:, 1]

xsortinds = np.lexsort(scoarray.T)
xscoarray = scoarray[xsortinds, :]

salientts = [0.032, 0.16, 0.96]  # start, bifurcation, end

sectoneids = salientts[0] < xscoarray[:, 1]
sectxs = xscoarray[sectoneids, :]
sectoneids = sectxs[:, 1] < salientts[1]
xscoaone = sectxs[sectoneids, :]

secttwoids = salientts[1] <= xscoarray[:, 1]
sectxs = xscoarray[secttwoids, :]
secttwoids = sectxs[:, 1] < salientts[2]
xscoatwo = sectxs[secttwoids, :]

tdsco = np.linspace(salientts[0], salientts[1], no)
firstpair = (xscoaone[0, 0], xscoaone[1, 0])
vallist = [[np.min(firstpair), np.max(firstpair)]]

for k, tk in enumerate(tdsco[:-1]):
    rghtofit = xscoaone[:, 1] >= tk
    xscoaone = xscoaone[rghtofit, :]
    itvinds = xscoaone[:, 1] < tdsco[k+1]
    ycors = xscoaone[itvinds, 0]
    vallist.append([np.min(ycors), np.max(ycors)])

sectoneborders = np.array(vallist)
sectoneborderfun = interp1d(tdsco, sectoneborders, axis=0)

tdsct = np.linspace(salientts[1], salientts[2], nt)
firstvals = (xscoatwo[0, 0], xscoatwo[1, 0])
vallist = [[np.mean(firstvals)]]

for k, tk in enumerate(tdsct[:-1]):
    rghtofit = xscoatwo[:, 1] >= tk
    xscoatwo = xscoatwo[rghtofit, :]
    itvinds = xscoatwo[:, 1] < tdsct[k+1]
    ycors = xscoatwo[itvinds, 0]
    vallist.append([np.mean(ycors)])

secttwoborders = np.array(vallist)
secttwoborderfun = interp1d(tdsct, secttwoborders, axis=0)

# funcs for the sects


def get_spcheartfun(NY=100, invertt=False):
    def spcheartshape(t):
        if invertt:
            t = 1. - t

        def _vti(xval):
            return np.int(np.round(xval*NY))

        if t < salientts[0] or t > salientts[2]:
            return np.zeros((1, NY))
        if t > salientts[0] and t <= salientts[1]:
            funcval = np.zeros((NY, 1))
            bvalsup = sectoneborderfun(t)
            bvalsdwn = 1. - bvalsup
            funcval[_vti(bvalsdwn[1]):_vti(bvalsdwn[0]), 0] = 1.
            funcval[_vti(bvalsup[0]):_vti(bvalsup[1]), 0] = 1.
            return funcval
        if t > salientts[1] and t <= salientts[2]:
            funcval = np.zeros((NY, 1))
            bvalup = secttwoborderfun(t)
            bvaldwn = 1. - bvalup
            funcval[_vti(bvaldwn[0]):_vti(bvalup[0]), 0] = 1.
            return funcval

    return spcheartshape

if __name__ == '__main__':
    spcheartshape = get_spcheartfun(NY=NY)
    heartfunvec = gou.functovec(spcheartshape, np.linspace(0, 1, 100))
    XX = gou.xvectoX(heartfunvec, nq=NY, ns=NS)
    plotmat(XX.T, fignum=2135)
