#! /usr/bin/env python3

from nutils import mesh, function, plot, util, log, numpy
import functools


def newton(f, x0, maxiter, restol, lintol):
    '''find x such that f(x) = 0, starting at x0'''

    for i in log.range('newton', maxiter+1):
        if i > 0:
            # solve system linearised around `x`, solution in `direction`
            J = f(x, jacobian=True)
            direction = -J.solve(residual, tol=lintol)
            xprev = x
            residual_norm_prev = residual_norm
            # line search, stop if there is convergence w.r.t. iteration `i-1`
            for j in range(4):
                x = xprev + (0.5)**j * direction
                residual = f(x)
                residual_norm = numpy.linalg.norm(residual, 2)
                if residual_norm < residual_norm_prev:
                    break
                # else: no convergence, repeat and reduce step size by one half
            else:
                log.warning('divergence')
        else:
            # before first iteration, compute initial residual
            x = x0
            residual = f(x)
            residual_norm = numpy.linalg.norm(residual, 2)
        log.info('residual norm: {:.5e}'.format(residual_norm))
        if residual_norm < restol:
            break
    else:
        raise ValueError('newton did not converge in {} iterations'.format(maxiter))
    return x


# residual of a weak formulation of Burgers' equations
def residual(u_hat, *, jacobian=False, domain, geom, φ, diffusion, **int_kwargs):

    # Burgers' equation:
    #
    #   (v  u)   - ε   u    = 0,                                         (1)
    #     j   ,j    jk  ,jk
    #
    # with v = [1, u] and ε a constant diffusion matrix with zeros on the time
    # axis.
    #
    # Multiplying (1) with a basis function φ_i and integrating over the domain
    # Ω gives
    #
    #   ∫  φ  (v  u - ε   u  )   = 0.
    #    Ω  i   j      jk  ,k ,j
    #
    # Applying integration by parts to both terms in the integrand gives
    #
    #   ∫  φ    (-v  u + ε   u  ) + ∫   φ  (v  u n  - ε   u  ) n  = 0.
    #    Ω  i,j    j      jk  ,k     ∂Ω  i   j    j    jk  ,k   j
    #
    # It this point we replace u, not u_,k, with the initial and boundary
    # conditions, u_initial at t=0 and u=0 at the space boundaries, and add the
    # term
    #
    #   ∫   -ε   φ    u n
    #    ∂Ω   jk  i,k    j
    #
    # to make the weak formulation symmetric.  Note that either u is zero or
    # ε_jk n_j is zero, hence this boundary integral equals zero.  This gives
    #
    #   ∫  φ    (-v  u + ε   u  ) + ∫     φ  v  u n  + ...
    #    Ω  i,j    j      jk  ,k     Γ     i  j    j
    #                                 t=T
    #
    #   ... ∫     φ  v  u        n  - ∫         ε   (φ  u   + φ    u) n  = 0.
    #        Γ     i  j  initial  j    Γ         kl   i  ,k    i,k     l
    #         t=0                       x∈{0,1}
    #
    # NOTE: It would be better to use [Nitsche], but this works.
    #
    # [Nitsche]: http://scicomp.stackexchange.com/questions/19910/what-is-the-general-idea-of-nitsches-method-in-numerical-analysis

    # NOTE:  Temporary solution to compute jacobians.  Soon there will be a
    # nice and clean way to compute jacobians of (decorated) python functions
    # without these silly `J`'s appearing in the integrands below.
    if jacobian:
        u_hat, u_hat_ = function.DerivativeTarget(u_hat.shape), u_hat
        replace = lambda f: u_hat_ if f is u_hat else function.edit( f, replace )
        axes = tuple(range(u_hat_.ndim))
        J = lambda arg: replace(function.derivative(arg.unwrap(geom), u_hat, axes))
    else:
        J = lambda arg: arg

    u = φ.dot(u_hat)
    v = function.stack([1, u/2])
    ε = function.diagonalize(function.stack([0, diffusion]))
    n = geom.normal()
    u_initial = function.heaviside(0.5 - geom[1])

    return (
        domain.integrate(J(
            φ['i,k']*(-v['k']*u + ε['kl']*u[',l'])
            ), **int_kwargs)
        # t-initial boundary
        + domain.boundary[0][0].integrate(J(
            φ['i']*v['k']*u_initial*n['k']
            ), **int_kwargs)
        # t-end boundary
        + domain.boundary[0][1].integrate(J(
            φ['i']*v['k']*u*n['k']
            ), **int_kwargs)
        # x boundaries
        + domain.boundary[1].integrate(J(
            -ε['kl']*(φ['i']*u[',k'] + φ['i,k']*u)*n['l']
            ), **int_kwargs)
    )


def main(nelems=8, degree=2, diffusion_over_h=0.5):

    # construct geometry and initial topology
    verts = numpy.linspace(0, 1, nelems+1)
    domain, geom = mesh.rectilinear([verts, verts])

    # common integration arguments
    int_kwargs = dict(geometry=geom, ischeme='gauss{}'.format(degree*2+1))

    for i in log.range('refinement', 4):

        # create domain and basis for global refinement step `i`
        if i > 0:
            domain = domain.refined
        φ = domain.basis('spline', degree=degree)

        # compute an initial guess based on the previous iteration, if any
        if i > 0:
            u_hat = domain.project(u, onto=φ, **int_kwargs)
        else:
            u_hat = numpy.zeros(φ.shape)

        # solve
        R = functools.partial(
            residual, diffusion=diffusion_over_h/(nelems*2**i), geom=geom,
            domain=domain, φ=φ, **int_kwargs)
        u_hat = newton(R, x0=u_hat, maxiter=20, restol=1e-8, lintol=0)

        # expand solution
        u = φ.dot(u_hat)

        # plot
        points, colors = domain.elem_eval([geom, u], ischeme='bezier3', separate=True)
        with plot.PyPlot('solution') as plt:
            plt.title('nelems={}'.format(nelems*2**i))
            plt.mesh(points, colors)
            plt.colorbar()
            plt.clim(0, 1)
            plt.xlabel('t')
            plt.ylabel('x')

if __name__ == '__main__':
    util.run(main)

# vim: sts=4:sw=4:et
