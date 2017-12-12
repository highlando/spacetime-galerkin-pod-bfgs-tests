import dolfin

from space_time_galerkin_pod.ldfnp_ext_cholmod import SparseFactorMassmat

import dolfin_navier_scipy.problem_setups as dnsps
import dolfin_navier_scipy.data_output_utils as dou
import dolfin_navier_scipy.dolfin_to_sparrays as dts

# ## 2D-Cylinder
N, Re, scheme, ppin = 3, 50, 'TH', None

femp, stokesmatsc, rhsd_vfrc \
    = dnsps.get_sysmats(problem='cylinderwake', N=N, Re=Re, scheme=scheme,
                        mergerhs=True)

Mc, Ac = stokesmatsc['M'], stokesmatsc['A']

with dou.Timer('factorizing mass mat'):
    facmc = SparseFactorMassmat(Mc)

print('nnz of M: {0:.1e}; nnz of F: {1:.1e}; fillingfac: {2:.2f}\n'.format
      (Mc.nnz, facmc.F.nnz, facmc.F.nnz/Mc.nnz))

# ## 3D-unit cube
N = 20
cmesh = dolfin.UnitCubeMesh(N, N, N)
V = dolfin.FunctionSpace(cmesh, 'CG', 2)
v = dolfin.TestFunction(V)
u = dolfin.TrialFunction(V)
massform = dolfin.assemble(v*u*dolfin.dx)
cubem = dts.mat_dolfin2sparse(massform)

with dou.Timer('factorizing mass mat'):
    facmcb = SparseFactorMassmat(cubem)

print('nnz of M: {0:.1e}; nnz of F: {1:.1e}; fillingfac: {2:.2f}'.format
      (cubem.nnz, facmcb.F.nnz, facmcb.F.nnz/cubem.nnz))
