import numpy as np

import matplotlib.pyplot as plt

import dolfin
from dolfin import assemble, nabla_grad, dx, inner

import dolfin_navier_scipy.dolfin_to_sparrays as dts
import dolfin_navier_scipy.data_output_utils as dou
import spacetime_galerkin_pod.gen_pod_utils as gpu


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


def schloegl_spacedisc(N=10, nonlpara=10., mu=1e-1,
                       retgetsdc=False, retfemdict=False):

    mesh = dolfin.UnitSquareMesh(N, N)
    V = dolfin.FunctionSpace(mesh, 'CG', 1)
    # domain of observation
    odcoo = dict(xmin=0.4,
                 xmax=0.6,
                 ymin=0.4,
                 ymax=0.6,
                 area=.2*.2)

    u = dolfin.TrialFunction(V)
    v = dolfin.TestFunction(V)

    # assembling the Neumann control operator
    class RightNeumannBoundary(dolfin.SubDomain):
        def inside(self, x, on_boundary):
            return (on_boundary and dolfin.near(x[0], 1))

    rnb = RightNeumannBoundary()
    bparts = dolfin.FacetFunction('size_t', mesh)
    bparts.set_all(0)
    rnb.mark(bparts, 1)
    ds = dolfin.Measure('ds', domain=mesh, subdomain_data=bparts)

    class ContShapeOne(dolfin.Expression):

        def eval(self, value, x):
            value[0] = .5*(1+np.cos(np.pi*(2*x[1]-1)))

        def value_shape(self):
            return (1,)

    contshfun = ContShapeOne(element=V.ufl_element())
    bneumann = assemble(v*contshfun*ds(1))

    # assembling the output operator
    class ChiDomObs(dolfin.Expression):
        def eval(self, value, x):
            mps = 1e-14
            if (x[0] < odcoo['xmax'] + mps
                    and x[0] > odcoo['xmin'] - mps
                    and x[1] < odcoo['ymax'] + mps
                    and x[1] > odcoo['ymin'] - mps):
                value[0] = 1.
            else:
                value[0] = 0.

        def value_shape(self):
            return (1,)

    chidomobs = ChiDomObs(element=V.ufl_element())
    obsop = 1./odcoo['area']*assemble(chidomobs*u*dx)

    # boundaries and conditions
    ugamma = dolfin.Expression('0', degree=1)

    def _diriboundary(x, on_boundary):
        return dolfin.near(x[0], 0) or dolfin.near(x[1], 1)

    diribc = dolfin.DirichletBC(V, ugamma, _diriboundary)

    mass = assemble(v*u*dx)
    stif = assemble(mu*inner(nabla_grad(v), nabla_grad(u))*dx)

    M = dts.mat_dolfin2sparse(mass)
    A = dts.mat_dolfin2sparse(stif)

    M, _, bcdict = dts.condense_velmatsbybcs(M, [diribc], return_bcinfo=True)
    ininds = bcdict['ininds']
    A, rhsa = dts.condense_velmatsbybcs(A, [diribc])

    bneumann = bneumann.array()[ininds]
    B = bneumann.reshape((bneumann.size, 1))

    cobsop = obsop.array()[ininds]
    C = cobsop.reshape((1, cobsop.size))

    def schloegl_nonl(vvec, t=None):
        v0 = expandvfunc(vvec, V=V, ininds=ininds)
        bnl = nonlpara*assemble(v*((v0-v0*v0*v0))*dx)
        return -bnl.array()[ininds]

    def schloegl_nonl_ptw(funval):
        return -nonlpara*(funval - funval**3)

    def plotit(sol=None, tmesh=None, sfile=None, sname='slgl',
               sfilestr='nn.pvd'):
        if sfile is None:
            sfile = dolfin.File(sfilestr)
        for k, t in enumerate(tmesh):
            dou.output_paraview(VS=V, invinds=ininds, diribcs=[diribc],
                                sc=sol[k, :], sname='slgl', t=t,
                                sfile=sfile)

    if retfemdict:
        return (M, A, B, C, rhsa, schloegl_nonl, schloegl_nonl_ptw, plotit,
                dict(V=V, diribc=diribc, ininds=ininds))
    else:
        return M, A, B, C, rhsa, schloegl_nonl, schloegl_nonl_ptw, plotit


def schloegl_bwd_spacedisc(V=None, diribc=None, ininds=None, nonlpara=None):
    """
    wrapper for dolfin functions that assemble

    `nonl(v).dv` and a functional
    `M` and `A` are the same as in the forward problem

    Notes
    ---
    Only implemented for homogeneous Dirichlet boundary conditions
    """
    u = dolfin.TrialFunction(V)
    v = dolfin.TestFunction(V)

    def nldvop(vvec):
        v0 = expandvfunc(vvec, V=V, ininds=ininds)
        bnl = assemble(v*nonlpara*(3.*v0*v0 - 1.)*u*dx)
        Bnl = dts.mat_dolfin2sparse(bnl)
        Bnl, rhsa = dts.condense_velmatsbybcs(Bnl, [diribc])
        return Bnl

    def fnctnl(vvec):
        v0 = expandvfunc(vvec, V=V, ininds=ininds)
        fncnl = assemble((v*v0)*dx)
        return fncnl.array()[ininds]

    def nldvop_ptw(vcomp):
        return nonlpara*(3.*vcomp**2 - 1.)

    return nldvop, fnctnl, nldvop_ptw


if __name__ == '__main__':
    N = 40
    M, A, B, C, rhsa, schloegl_nonl, schloegl_nonl_ptw, plotit, femp =\
        schloegl_spacedisc(N=N, mu=1e-2, retfemdict=True)
    V, diribc, ininds = femp['V'], femp['diribc'], femp['ininds']

    inivalexp = dolfin.\
        Expression('.5*(1+cos(pi*(2*x[0]-1)))*.5*(1+cos(pi*(2*x[1]-1)))',
                   element=femp['V'].ufl_element())
    inivalfunc = dolfin.interpolate(inivalexp, V)
    inivalvec = inivalfunc.vector().array()[ininds].reshape((ininds.size, 1))

    t0, tE, Nts = 0., 1.0, 100  # Nts is only where to save the sols
    tmesh = np.linspace(t0, tE, Nts+1)

    def rhs(t):
        return rhsa - 0*B

    with dou.Timer('Forward simu'):
        sol, infodict = gpu.\
            time_int_semil(tmesh=tmesh, full_output=True, M=M, A=A,
                           rhs=rhs, nfunc=schloegl_nonl, iniv=inivalvec)
    sfile = dolfin.File('plots/schloegl.pvd')
    plotit(sol=sol, tmesh=tmesh, sfile=sfile)

    cvt = C.dot(sol.T)
    plt.figure(101)
    plt.plot(tmesh, cvt.flatten())
    plt.show(block=False)
    theonesol = 0*inivalvec + 1.
    print('`C*1`: {0}'.format(C.dot(theonesol)))

    sfile = dolfin.File('plots/schloegl_tos.pvd')
    plotit(sol=theonesol.T, tmesh=[0], sfile=sfile)
