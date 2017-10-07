import numpy as np
import scipy.sparse as sps
import scipy.io
import matplotlib.pyplot as plt
import gen_pod_utils as gpu


def get_podmats(Y, poddim, plotsvs=False):

    U, S, V = np.linalg.svd(Y)

    Uk = U[:, 0:poddim]

    if plotsvs:
        plt.plot(S, label='POD')
        plt.semilogy()
        plt.title('Singular Values of snapshot matrix')
        plt.legend()
        plt.show(block=False)
        print('POD-ratio: {0}'.format(np.sum(S[0:poddim]) / np.sum(S)))

    return Uk


def get_podred_model(M, A, Y, y0, f, poddim, k, tmesh=None,
                     genpod=False, plotsvs=False, verbose=False):

    if genpod:
        Uk = gpu.get_genpodmats(Y, poddim, k, tmesh, plotsvs=plotsvs)
    else:
        Uk = get_podmats(Y, poddim, plotsvs=plotsvs)

    Mk = np.dot(Uk.T * M, Uk)
    mki = np.linalg.inv(Mk)

    if sps.isspmatrix(A):
        Ak = A * Uk
    else:
        Ak = np.dot(A, Uk)

    A_red = np.dot(mki, np.dot(Uk.T, Ak))
    rhs_red = np.dot(mki, np.dot(Uk.T, f))

    y_red = np.dot(Uk.T, Y[:, 0])
    if verbose:
        print('projection error in initial value: {0}'.\
            format(np.linalg.norm(Y[:, 0] - np.dot(Uk, y_red).flatten())))

    return A_red, y_red, rhs_red, Uk


def get_mayf(N=10, Re=1e2, t0=0.0, tE=1.0, Nts=11, matprfx='dolfindata/dcmats',
             krylov=None, krpslvprms={}, krplsprms={}):

    def defdata(N=None, Re=None, t0=None, tE=None, Nts=None, prfx=''):
        return prfx + 'drivcavmats_Re{0}N{1}t0{2}tE{3}Nts{4}'.\
            format(Re, N, t0, tE, Nts)

    datstry = defdata(N=N, Re=Re, t0=t0, tE=tE, Nts=Nts, prfx=matprfx)
    if krylov:
        datstry = datstry + '_{0}'.format(krylov)
        try:
            datstry = datstry + 'tol{0}'.format(krpslvprms['tol'])
        except KeyError:
            datstry = datstry + 'tolnotspec'

    datstrtm = defdata(N=None, Re=None, t0=t0, tE=tE, Nts=Nts, prfx=matprfx)
    datstraf = defdata(N=N, Re=Re, t0=None, tE=None, Nts=None, prfx=matprfx)
    datstrmbj = defdata(N=N, Re=None, t0=None, tE=None, Nts=None, prfx=matprfx)

    print('Read data file: ' + datstry)

    try:
        # sysmats for system Mv' + Av = rhs + Bu
        # snapshots Y = v(tmesh)
        # J -- discrete div mat
        A = load_spa(datstraf + 'A')
        M = load_spa(datstrmbj + 'M')
        Y = load_npa(datstry + 'Y')
        f = load_npa(datstraf + 'rhs')
        B = load_spa(datstrmbj + 'B')
        tmesh = load_npa(datstrtm + 'tmesh')
        J = load_spa(datstrmbj + 'J')

    except IOError:
        from genpod_dolfin_interface import gopod
        M, A, Y, f, B, tmesh, J = gopod(problemname='drivencavity',
                                        N=N, Re=Re, t0=t0, tE=tE,
                                        Nts=Nts, NU=3, NY=3, paraoutput=False,
                                        krylov=krylov, krpslvprms=krpslvprms,
                                        krplsprms=krplsprms)

        save_spa(A, datstraf + 'A')
        save_spa(M, datstrmbj + 'M')
        save_npa(Y, datstry + 'Y')
        save_npa(f, datstraf + 'rhs')
        save_spa(B, datstrmbj + 'B')
        save_npa(tmesh, datstrtm + 'tmesh')
        save_spa(J, datstrmbj + 'J')

    return M, A, Y, f, B, tmesh, J


def save_npa(v, fstring='notspecified'):
    np.save(fstring, v)
    return


def load_npa(fstring):
    if not fstring[-4:] == '.npy':
        return np.load(fstring + '.npy')
    else:
        return np.load(fstring)


def save_spa(sparray, fstring='notspecified'):
    scipy.io.mmwrite(fstring, sparray)


def load_spa(fstring):
    return scipy.io.mmread(fstring).tocsc()


def timespace_diff_norm(tmesh, Y, Yred, M=None):

    dy = Y[:, 0] - Yred[:, 0]
    dtc = tmesh[1] - tmesh[0]
    err_old = 0.5 * dtc * np.dot(dy.T * M, dy)
    err = err_old

    for k in range(1, len(tmesh)):
        dy = Y[:, k] - Yred[:, k]
        dtc = tmesh[k] - tmesh[k - 1]
        err_new = 0.5 * dtc * np.dot(dy.T * M, dy)
        err += err_old + err_new
        err_old = err_new

    # dy = Y - Yred
    # dtvec = tmesh[1:] - tmesh[:-1]
    # trapvec = 0.5*np.dot(dy[:,:-1].T * M, dy[:,1:])
    # err = (dtvec*trapvec).sum()

    return np.sqrt(err)
