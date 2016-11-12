import numpy as np

import burgers_genpod_utils as bgu
import gen_pod_utils as gpu

Nq = 60  # dimension of the spatial discretization
Ns = 11  # dimension of the temporal discretization
hs = 8  # reduced S dim
hq = 16  # reduced Y dim
Nts = 200  # number of time sampling points
nu = 1e-2
t0, tE = 0., 1.
noconv = False
plotsvs = False
debug = True

tmesh = np.linspace(t0, tE, Nts)

mockUks = np.ones((Ns, 1))
mocksol = np.zeros((hq, Nts))

htittl = bgu.get_burgertensor_timecomp(podmat=mockUks, sdim=Ns,
                                       tmesh=tmesh, basfuntype='pl')

(msx, Ms) = gpu.get_genmeasuremat(sol=mocksol,
                                  tmesh=tmesh, sdim=Ns)
