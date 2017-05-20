import dolfin
import numpy as np
N = 10
a, b = 0., 1.
mesh = dolfin.IntervalMesh(N, a, b)
V = dolfin.FunctionSpace(mesh, 'CG', 1)

linexp = dolfin.Expression('x[0]', degree=1)

linfun = dolfin.interpolate(linexp, V)

print linfun.vector().array()


class MyExpression0(dolfin.Expression):
    def eval(self, value, x):
        if x[0] < 0.5:
            value[0] = 0
        else:
            value[0] = 1

stpexp = MyExpression0(degree=1)
stpfun = dolfin.interpolate(stpexp, V)
print stpfun.vector().array()

iniv = np.r_[np.zeros(((N+1)/2, 1)), np.ones(((N+2)/2, 1))]
v = dolfin.Function(V)
v.vector().set_local(iniv)
print v.vector().array()
