import unittest
import numpy as np
import scipy.sparse as sps

from numpy.linalg import norm

import dolfin_burgers_scipy as dbs
import burgers_genpod_utils as bgu
import spacetime_pod_utils as spu
import gen_pod_utils as gpu


class BurgerGenpodUtils(unittest.TestCase):

    def setUp(self):

        self.Nq = 50  # dimension of the spatial discretization
        self.Ns = 9  # dimension of the temporal discretization
        self.Nts = 60  # number of time sampling points
        self.nu = 1e-2
        self.t0, self.tE = 0., 1.
        self.mockUky = np.eye(self.Nq)
        self.mockUks = np.eye(self.Ns)
        (M, A, rhs, nfunc, femp) = dbs.\
            burgers_spacedisc(N=self.Nq+1, nu=self.nu, retfemdict=True)
        # +1 bc of boundary conditions that are eliminated
        self.femp, self.ay, self.my = femp, A, M
        self.tmesh = np.linspace(self.t0, self.tE, self.Nts)
        # constructing test solutions
        self.contimsol = np.ones((self.Ns, 1))
        self.conspasol = np.ones((self.Nq, 1))
        self.linspasol = np.linspace(0, 1, self.Nq).reshape((self.Nq, 1))
        self.lintimsol = np.linspace(0, 1, self.Ns).reshape((self.Ns, 1))
        self.dms = gpu.get_dms(sdim=self.Ns, tmesh=self.tmesh, basfuntype='pl')
        self.ms = gpu.get_ms(sdim=self.Ns, tmesh=self.tmesh, basfuntype='pl')

    def test_genpod_burger_quadtensor(self):
        """consistency tests for the assembling of the burger tensor

        of the time space galerkin scheme"""

        Uky = self.mockUky
        Uks = self.mockUks
        uvvdxl = dbs.get_burgertensor_spacecomp(podmat=Uky, **self.femp)
        htittl = bgu.get_burgertensor_timecomp(podmat=Uks, sdim=self.Ns,
                                               tmesh=self.tmesh,
                                               basfuntype='pl')

        # for constant in space burgertensor is zero (at the inner nodes)
        lintimconspa = np.kron(self.lintimsol, self.conspasol)
        ebq = bgu.eva_burger_quadratic(tvvec=lintimconspa, htittl=htittl,
                                       uvvdxl=uvvdxl, iniv=None)
        innernode = np.r_[0, np.ones((self.Nq-2, )), 0]
        inds = np.tile(innernode, self.Ns).astype(bool)
        self.assertTrue(np.allclose(0*lintimconspa[inds], ebq[inds]))
        # for linear in space burgertensor not zero (at the inner nodes)
        contimlinspa = np.kron(self.contimsol, self.linspasol)
        ebq = bgu.eva_burger_quadratic(tvvec=contimlinspa, htittl=htittl,
                                       uvvdxl=uvvdxl, iniv=None)
        innernode = np.r_[0, np.ones((self.Nq-2, )), 0]
        inds = np.tile(innernode, self.Ns).astype(bool)
        self.assertTrue(norm(ebq[inds]) > 1e-8)

    def test_genpod_burger_quadtensor_linearized(self):
        """consistency tests for the assembling of the linearization

        of the burger tensor space galerkin scheme"""

        Uky = self.mockUky
        Uks = self.mockUks
        uvvdxl = dbs.get_burgertensor_spacecomp(podmat=Uky, **self.femp)
        htittl = bgu.get_burgertensor_timecomp(podmat=Uks, sdim=self.Ns,
                                               tmesh=self.tmesh,
                                               basfuntype='pl')

        lintimlinspa = np.kron(self.lintimsol, self.linspasol)
        lintimlinspatwo = np.kron(self.lintimsol, 0.5*self.linspasol)
        eba = bgu.eva_burger_quadratic(tvvec=lintimlinspa+lintimlinspatwo,
                                       htittl=htittl, uvvdxl=uvvdxl, iniv=None)
        ebo = bgu.eva_burger_quadratic(tvvec=lintimlinspa,
                                       htittl=htittl, uvvdxl=uvvdxl, iniv=None)
        ebt = bgu.eva_burger_quadratic(tvvec=lintimlinspatwo,
                                       htittl=htittl, uvvdxl=uvvdxl, iniv=None)
        ebmat = bgu.eva_burger_quadratic(tvvec=lintimlinspa, htittl=htittl,
                                         uvvdxl=uvvdxl, iniv=None,
                                         retjacobian=True)
        ebmvtwo = np.dot(ebmat, lintimlinspatwo)
        print ebo.shape, ebmvtwo.shape
        self.assertTrue(norm(eba - (ebo + ebmvtwo + ebt)) < 1e-12)
        self.assertTrue(norm(eba) > 1e-5)
        self.assertTrue(norm(ebt) > 1e-5)
        self.assertTrue(norm(ebo) > 1e-5)
        self.assertTrue(norm(ebmvtwo) > 1e-5)

    def test_spacetime_evals(self):
        """consistency tests of the space time discretization

        """

        ay = self.ay
        my = self.my
        zsol = np.zeros((self.Nq, self.Nts))
        contimsol = self.contimsol
        linspasol = self.linspasol
        dms = gpu.get_dms(sdim=self.Ns, tmesh=self.tmesh, basfuntype='pl')
        (msx, ms) = gpu.get_genmeasuremat(sol=zsol, tmesh=self.tmesh,
                                          sdim=self.Ns)
        self.assertTrue(np.allclose(ms, self.ms))
        contimlinspa = np.kron(contimsol, linspasol)
        lintimlinspa = np.kron(self.lintimsol, linspasol)
        self.assertTrue(norm(contimlinspa) > 1e-8)
        self.assertTrue(np.allclose(0*contimsol, np.dot(dms, contimsol)))
        # check the dtpart to be zero for const time funcs and nonzero for lin
        dtpart = spu.apply_time_space_kronprod(tvvec=contimlinspa, smat=dms,
                                               qmat=np.array(my.todense()))
        self.assertTrue(np.allclose(0*contimlinspa, dtpart))
        dtpart = spu.apply_time_space_kronprod(tvvec=lintimlinspa, smat=dms,
                                               qmat=np.array(my.todense()))
        self.assertTrue(norm(dtpart) > 1e-8)

        # check the a-part to be zero for lin (except from the boundary)
        # and nonzero for quad
        lintimlinspaqua = np.kron(self.lintimsol, linspasol*linspasol)
        dxxpart = spu.apply_time_space_kronprod(tvvec=lintimlinspa, smat=ms,
                                                qmat=np.array(ay.todense()))
        innernode = np.r_[0, np.ones((self.Nq-2, )), 0]
        inds = np.tile(innernode, self.Ns).astype(bool)
        self.assertTrue(np.allclose(0*contimlinspa[inds], dxxpart[inds]))
        dxxpart = spu.\
            apply_time_space_kronprod(tvvec=lintimlinspaqua, smat=ms,
                                      qmat=np.array(ay.todense()))
        self.assertTrue(norm(dxxpart[1:-1]) > 1e-8)

    def test_check_residuals(self):
        from spacetime_pod_utils import get_spacetimepodres
        iniv = np.r_[np.ones(((self.Nq-1)/2, 1)), np.zeros(((self.Nq)/2+1, 1))]
        ttvvec = np.random.randn(self.Nq*(self.Ns-1), 1)
        # define the residual manually
        dmsz = self.dms[1:, :1]
        msz = self.ms[1:, :1]
        dmsI = self.dms[1:, 1:]
        msI = self.ms[1:, 1:]
        mydense = np.array(self.my.todense())
        aydense = np.array(self.ay.todense())
        solmat = np.kron(dmsI, mydense) + np.kron(msI, aydense)
        rhs = -np.kron(dmsz, np.dot(mydense, iniv)) \
            - np.kron(msz, np.dot(aydense, iniv))
        mres = np.dot(solmat, ttvvec) - rhs
        # define to the residual in burgers
        bres = get_spacetimepodres(tvvec=ttvvec, dms=self.dms, ms=self.ms,
                                   my=mydense, ared=aydense, nfunc=None,
                                   rhs=None, retnorm=False, iniv=iniv)
        self.assertTrue(np.allclose(mres, bres))
        self.assertTrue(norm(mres) > 1e-8)

    def test_check_masssqrts(self):
        from ext_cholmod import SparseFactorMassmat
        my = self.my
        myfac = SparseFactorMassmat(sps.csc_matrix(my))
        # My=myfac.F()*myfac.F().T

        testrhs = np.random.randn(self.Nq, self.Ns)

        lytitestrhs = myfac.solve_Ft(testrhs)

        self.assertTrue(np.allclose(my.todense(),
                        (myfac.F*myfac.Ft).todense()))
        self.assertTrue(np.allclose(myfac.Ft.todense(), myfac.F.T.todense()))
        self.assertTrue(np.allclose(testrhs, myfac.Ft*lytitestrhs))

        lyitestrhs = myfac.solve_F(testrhs)
        self.assertTrue(np.allclose(testrhs, myfac.F*lyitestrhs))

        # lsims = msfac.solve_L(ms)
        # lstilsims = msfac.solve_Lt(lsims)
        # self.assertTrue(np.allclose(lstilsims*ms, ms))

if __name__ == '__main__':
    unittest.main()
