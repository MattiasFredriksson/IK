import numpy as np


def solve_GN(JE, x0, max_iter=100, eps=1e-8, seps=1e-11, rcond=1e-11, **kwargs):
    ''' Solve the non-linear system using Gauss-Newton method.

    Args:
    ----
    JE: Callable computing Jacobian and residual vector pair for given state x. Function should return the pair:
            1) Jacobian matrix for a given x, on form (M, N).
            2) Residual vector for a given x, on form (N, 1).
    x0:         Initial condition.
    max_iter:   Maximum number of iterations to perform.
    eps:        Epsilon, termination threshold for the L2-norm of the error residual (to be less then epsilon).
    seps:       Step epsilon, termination threshold for the L2-norm of the configuration delta/change.
    Returns:
        Optimal x in a least-square sence.
    '''

    x = x0

    def res(y):
        return np.linalg.norm(y)

    for i in range(max_iter):
        # Compute residual error
        A, y = JE(x)
        r = res(y)
        if r < eps:  # Euclidean norm
            break

        # Solve in lstsq sence:
        delta, __, __, __ = np.linalg.lstsq(A, -y, rcond=rcond)
        # Update
        x = x + delta
        if res(delta) < seps:
            # Recompute residual and terminate
            __, y = JE(x)
            r = np.linalg.norm(y)
            break
    return x, r
