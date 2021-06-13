import numpy as np

def forw(q, F, i=1, h=1e-7):
    ''' Compute the numerical forward differential.

    Args:
    ----
    q:      Parameterization of the system DoF on form Nx1.
    F:      Function computing the dependent variable with respect to the independent variable q.
    i:      Vector of shape Nx1 masking the DoF under change, containing zeros except for a one in the i:th element.
             Determines the change in q as: q + i * h.
    h:      Delta step used to compute the change in F() with respect to q numerically.
    Returns:
        Numerical differential of F with respect to the i:th element of q: dF(q)/dq_i.
    '''
    return (F(q + h * i) - F(q)) / h


def cent(q, F, i=1, h=1e-6):
    ''' Compute the numerical central/symetric differential.

    Args:
    ----
    q:      Parameterization of the system DoF on form Nx1 or (1,).
    F:      Function computing the dependent variable with respect to the independent variable q.
    i:      Vector of shape Nx1 masking the DoF under change, containing zeros except for a one in the i:th element.
             Determines the change in q as: q +- i * h.
    h:      Delta step used to compute the change in F() with respect to q numerically.
    Returns:
        Numerical differential of F with respect to the i:th element of q: dF(q)/dq_i.
    '''
    return (F(q + h * i) - F(q - h * i)) / (2 * h)

def cent2(q, F, i, j, h=1e-6):
    ''' Compute the second order (F'') numerical central/symetric differential.

    Args:
    ----
    q:      Parameterization of the system DoF on form Nx1.
    F:      Function computing the dependent variable with respect to the independent variable q.
    i:      Vector of shape Nx1 masking the first dependent var, contains zeros except for one in the i:th element.
             Determines the change in q as: q +- i * h.
    j:      Vector of shape Nx1 masking the second dependent var.
    h:      Delta step used to compute the change in F() with respect to q numerically.
    Returns:
        Second order numerical differential of F with respect to change in i:th and j:th element of q:
         dF(q)^2/(dq_i*dq_j).
    '''
    return (cent(q + h * i, F, j, h) - cent(q - h * i, F, j, h)) / (2 * h)


def Jacobian(F, q, N=cent):
    ''' Compute the Jacobian numerically.

    Args:
    ----
    F:  Residual function computing the error vector e as: e = F(q).
    q:  Generalized coordinates for the system defined in a column vector on form Nx1 where N is the DoF.
    N:  Callable on form N(F, q, i, h) defining the numerical method for computing the derivate.
         Defaults to numerical.cent().
    Returns:
        Jacobian matrix [de/dq_1, de/dq_2, ...], with columns containing the error differentials
         with respect to change in each independent DoF in q.
    '''

    J = [None] * len(q)                     # Store columns in list
    i = np.zeros(q.shape)
    for j in range(len(q)):
        i[j] = 1.0
        J[j] = N(q, F, i).reshape(-1, 1)    # mx? -> kx1
        i[j] = 0.0

    return np.concatenate(J, axis=-1)       # list -> matrix: |dF(q)/dq_0, dF(q)/dq_1, ..|

def Hessian(F, q, N=cent2, h=1e-6):
    ''' Compute the Hessian numerically.

    Args:
    ----
    F:  Residual function computing the error vector e as: e = F(q).
    q:  Generalized coordinates for the system defined in a column vector on form Nx1 where N is the DoF.
    N:  Callable on form N(F, q, i, j, h) defining the numerical method for computing the derivate.
         Defaults to numerical.cent2().
    Returns:
        Hessian matrix dF^2 / (dq @ dq^T).
    '''
    n = len(q)
    m = len(F(q).reshape(-1, 1))    # Number of matrices.
    H = np.zeros((m, n, n))
    ii, jj = np.zeros(q.shape), np.zeros(q.shape)
    for i in range(n):
        ii[i] = 1.0
        for j in range(n):
            jj[j] = 1.0
            H[:, i, j] = N(q, F, ii, jj, h=h).ravel()    # (ox?) -> (m, )
            jj[j] = 0.0
        ii[i] = 0.0

    if m == 1:
        return H[0]
    return H

def PDH(H):
    ''' Uses Sylvester's criterion to determine if a non-singular symmetric real matrix is positive definite.
    '''

    n = np.shape(H)[-1]
    if np.ndim(H) < 3:
        H = H[np.newaxis, :, :]

    D = np.empty((len(H), n))
    for h in range(len(H)):
        for i in range(n):
            D[h, i] = np.linalg.det(H[h, :i, :i])
    return np.all(D > 0)

def issingular(M, eps=1e-7):
    ''' Determine if matrices in M are singular.
    '''
    return np.abs(np.linalg.det(M)) < eps
