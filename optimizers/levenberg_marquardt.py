import numpy as np




def solve_LM(Jr, x0, regu=1e-4, max_iter=100, eps=1e-8, seps=1e-11, rcond=1e-11, scale_invariant=False, **kwargs):
    ''' Solve the non-linear system using Gauss-Newton method.

    Args:
    ----
    Jr: Callable computing Jacobian and residual vector pair for given state x. Function should return the pair:
            1) Jacobian matrix for a given x, on form (M, N).
            2) Residual vector for a given x, on form (N, 1).
    x0:         Initial condition.
    max_iter:   Maximum number of iterations to perform.
    eps:        Epsilon, termination condition for the L2-norm of the error residual (to be less then epsilon).
    seps:       Step epsilon, termination condition for the L2-norm of the configuration delta/change.
    rcond:      Numerical threshold for considering values to be 0 (when solving the lstsq problem, see numpy docs).
    Returns:
        Optimal x in a least-square sence.
    '''

    x = x0
    def res(y):
        return np.linalg.norm(y)

    for i in range(max_iter):
        # Compute residual error
        A, y = Jr(x)
        r = res(y)
        if r < eps: # Euclidean norm
            break

        # Solve in lstsq sence:
        ATA = A.T @ A
        if scale_invariant: # Fletcher modification for scale invariance
            D = ATA + regu * np.diag(ATA)
        else: # Normal
            D = ATA + regu * np.eye(A.shape[-1])
        delta, __, __, __ = np.linalg.lstsq(D, A.T @ y, rcond=rcond)
        # Update
        x = x - delta
        if res(delta) < seps:
            # Recompute residual and terminate
            __, y = Jr(x)
            r = res(y)
            break
    return x, r
