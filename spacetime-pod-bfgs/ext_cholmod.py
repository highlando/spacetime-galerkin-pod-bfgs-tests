import scipy.sparse as sps
import numpy as np
from sksparse.cholmod import cholesky

""" A wrapper for the cholmod module that let's you work with

    `F*F.T = M` rather than `L*D*L.T = P*M*P.T`

    Note that F are as sparse as L but no more triangular """


class SparseFactorMassmat:

    def __init__(self, massmat):
        self.cmfac = cholesky(sps.csc_matrix(massmat))
        # from `P*massmat*P.T = myfac.L()*D*myfac.L().T`
        self.D = self.cmfac.D()
        self.F = self.cmfac.apply_Pt(self.cmfac.L())
        self.Ft = (self.F).T

    def solve_Ft(self, rhs):
        diagmat = sps.diags(np.sqrt(1./self.D))
        litptrhs = self.cmfac.apply_Pt(self.cmfac.solve_Lt(diagmat*rhs))
        return litptrhs

    def solve_F(self, rhs):
        diagmat = sps.diags(np.sqrt(1./self.D))
        liprhs = diagmat*self.cmfac.solve_L(self.cmfac.apply_P(rhs))
        return liprhs


if __name__ == '__main__':
    N, k, alpha, density = 100, 5, 1e-2, 0.2
    E = sps.eye(N)
    V = sps.rand(N, k, density=density)
    mockmy = E + alpha*sps.csc_matrix(V*V.T)

    testrhs = np.random.randn(N, k)

    facmy = SparseFactorMassmat(mockmy)
    lytitestrhs = facmy.solve_Ft(testrhs)

    print(np.allclose(mockmy.todense(), (facmy.F*facmy.Ft).todense()))
    print(np.allclose(testrhs, facmy.Ft*lytitestrhs))
