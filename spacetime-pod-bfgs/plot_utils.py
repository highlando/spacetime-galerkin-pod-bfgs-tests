import matplotlib.pyplot as plt
import numpy as np


def plotmat(sol, t0=0.0, tE=1.0, x0=0.0, xE=1.0, vmin=None, vmax=None,
            fignum=None, tikzfile=None, vertflip=False, horiflip=False,
            cmapname='viridis', plotplease=True, **kwargs):
    if not plotplease:
        return
    if fignum is None:
        fignum = 111
    if vertflip:
        sol = np.flipud(sol)  # flip the to make it forward time
    if horiflip:
        sol = np.fliplr(sol)  # flip the space

    plt.figure(fignum)
    try:
        plt.imshow(sol, extent=[x0, xE, tE, t0], vmin=vmin, vmax=vmax,
                   cmap=plt.get_cmap(cmapname))
    except (NameError, ValueError) as e:
        plt.imshow(sol, extent=[x0, xE, tE, t0], vmin=vmin, vmax=vmax,
                   cmap=plt.get_cmap('gist_earth'))
    plt.xlabel('x')
    plt.ylabel('t')
    if tikzfile is not None:
        try:
            from matplotlib2tikz import save
            save(tikzfile + '.tikz')
        except ImportError:
            print 'matplotlib2tikz need to export tikz filez'

    plt.colorbar(orientation='horizontal')  # , shrink=.75)
    plt.show(block=False)
