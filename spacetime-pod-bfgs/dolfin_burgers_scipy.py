import numpy as np

import dolfin
from dolfin import assemble, nabla_grad, dx, inner

import dolfin_navier_scipy.dolfin_to_sparrays as dts

__all__ = ['burgers_spacedisc',
           'burgers_bwd_spacedisc',
           'expandvfunc',
           'get_burgertensor_spacecomp',
           'plot_sol_paraview']


def burgers_spacedisc(N=10, nu=None, x0=0.0, xE=1.0, retfemdict=False,
                      condensemats=True):

    mesh = dolfin.IntervalMesh(N, x0, xE)
    V = dolfin.FunctionSpace(mesh, 'CG', 1)

    u = dolfin.TrialFunction(V)
    v = dolfin.TestFunction(V)

    # boundaries and conditions
    ugamma = dolfin.Expression('0', degree=1)

    def _spaceboundary(x, on_boundary):
        return on_boundary
    diribc = dolfin.DirichletBC(V, ugamma, _spaceboundary)

    mass = assemble(v*u*dx)
    stif = assemble(nu*inner(nabla_grad(v), nabla_grad(u))*dx)

    M = dts.mat_dolfin2sparse(mass)
    A = dts.mat_dolfin2sparse(stif)

    M, _, bcdict = dts.condense_velmatsbybcs(M, [diribc], return_bcinfo=True)
    ininds = bcdict['ininds']
    A, rhsa = dts.condense_velmatsbybcs(A, [diribc])

    def burger_nonl(vvec, t):
        v0 = expandvfunc(vvec, V=V, ininds=ininds)
        bnl = assemble(0.5*v*((v0*v0).dx(0))*dx)
        return bnl.array()[ininds]

    if retfemdict:
        return M, A, rhsa, burger_nonl, dict(V=V, diribc=diribc, ininds=ininds)
    else:
        return M, A, rhsa, burger_nonl


def burgers_bwd_spacedisc(V=None, diribc=None, ininds=None):
    """
    wrapper for dolfin functions that assemble

    `v*l.dx`, `v`, and `v*`

    `M` and `A` are the same as in the forward problem

    Notes
    ---
    Only implemented for homogeneous Dirichlet boundary conditions
    """
    u = dolfin.TrialFunction(V)
    v = dolfin.TestFunction(V)

    def vdxop(vvec):
        v0 = expandvfunc(vvec, V=V, ininds=ininds)
        bnl = assemble((v*v0*u.dx(0))*dx)
        Bnl = dts.mat_dolfin2sparse(bnl)
        Bnl, rhsa = dts.condense_velmatsbybcs(Bnl, [diribc])
        return Bnl

    def fnctnl(vvec):
        v0 = expandvfunc(vvec, V=V, ininds=ininds)
        fncnl = assemble((v*v0)*dx)
        return fncnl.array()[ininds]

    return vdxop, fnctnl


def burger_onedim_inival(Nq=None, inivtype='step'):
    # define the initial value
    if inivtype == 'smooth':
        xrng = np.linspace(0, 2*np.pi, Nq-1)
        iniv = 0.5 - 0.5*np.sin(xrng + 0.5*np.pi)
        iniv = 0.5*iniv.reshape((Nq-1, 1))
    elif inivtype == 'step':
        # iniv = np.r_[np.ones(((Nq-1)/2, 1)), np.zeros(((Nq)/2, 1))]
        iniv = np.r_[np.zeros(((Nq)/2, 1)), np.ones(((Nq-1)/2, 1))]
    elif inivtype == 'ramp':
        iniv = np.r_[np.linspace(0, 1, ((Nq-1)/2)).reshape(((Nq-1)/2, 1)),
                     np.zeros(((Nq)/2, 1))]
    return iniv


def expandvfunc(vvec, V=None, ininds=None):
    """
    Notes
    ---
    by now only zero boundary values
    """
    v0 = dolfin.Function(V)
    auxvec = np.zeros((V.dim(), 1))
    auxvec[ininds] = vvec.reshape((ininds.size, 1))
    v0.vector().set_local(auxvec)
    v0.rename('v', 'velocity')
    return v0


def get_burgertensor_spacecomp(V=None, podmat=None, ininds=None, diribc=None,
                               Ukyleft=None, bwd=False,
                               **kwargs):

    if not podmat.shape[0] == len(ininds):
        raise Warning("Looks like this is not the right POD basis")
    if Ukyleft is None:
        Ukyleft = podmat

    v = dolfin.TestFunction(V)
    u = dolfin.TrialFunction(V)

    if bwd:
        uvvdxl = []
        for ui in podmat.T:
            hatui = expandvfunc(ui, V=V, ininds=ininds)
            uivvdx = assemble(hatui*v*((u).dx(0))*dx)
            Uivvdx = dts.mat_dolfin2sparse(uivvdx)
            Uivvdx, _ = dts.condense_velmatsbybcs(Uivvdx, [diribc])
            uvvdxl.append(np.dot(Ukyleft.T, Uivvdx*podmat))
    else:
        uvvdxl = []
        for ui in podmat.T:
            hatui = expandvfunc(ui, V=V, ininds=ininds)
            uivvdx = assemble(0.5*hatui*((v*u).dx(0))*dx)
            Uivvdx = dts.mat_dolfin2sparse(uivvdx)
            Uivvdx, _ = dts.condense_velmatsbybcs(Uivvdx, [diribc])
            uvvdxl.append(np.dot(Ukyleft.T, Uivvdx*podmat))
    return uvvdxl


def plot_sol_paraview(sol, rfile=None, tmesh=None):
    for k, t in enumerate(tmesh.tolist()):
        cv = np.atleast_2d(sol[k, :]).T
        vf = expandvfunc(cv)
        rfile << vf, t
