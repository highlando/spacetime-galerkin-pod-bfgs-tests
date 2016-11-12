from gen_pod_utils import hatfuncs, get_dms
import matplotlib.pyplot as plt
import numpy as np

x0, xe, N = 0, 1, 100
n = 50

dhf, pts = hatfuncs(n=n, x0=x0, xe=xe, N=N, df=True)

x = np.linspace(x0, xe, 4*N+1)

dhfx = dhf(x)

dmy = get_dms(sdim=10, tmesh=x)

plt.plot(x, dhfx)
plt.show()
