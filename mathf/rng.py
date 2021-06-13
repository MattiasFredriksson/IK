import numpy as np
from scipy.spatial.transform import Rotation as rot

rng = np.random.default_rng(123456)


def bool(N, k, s):
    ''' Generate random booleans with True occuring with probability p = k / s.

    Args:
    ----
    N:  Number of random booleans to generate.
    k:  Number of True elements per s samples.
    s:  Number of samples satisfying k / p = s.
    '''
    return rng.integers(1, s, N) <= k


def ints(N, limit, olimit=None):
    ''' Generate random integers from [low, high)

    Args:
    ----
    N:
    limit: Lower limit (included) if olimit is not None, else higher limit.
    olimit: Upper integer limit (excluded), if not specified lower limit defaults to 0.
    '''
    if olimit is None:
        olimit = limit
        limit = 0
    return rng.integers(limit, olimit, N)

def normal(N, mean=0.0, std=1.0):
    ''' Generate N random values drawn from a normal (Gaussian) distribution.
    '''
    return rng.normal(mean, std, N)

def rnd_a(N=0):
    ''' Generate random angles on form (N, 1).
    '''
    if N == 0:
        return rng.uniform(-np.pi, np.pi, (1,))
    return rng.uniform(-np.pi, np.pi, (N, 1))

def angle(N=0):
    return rnd_a(N)

def rnd_norm(N):
    ''' Generate random normals on form (N, 3).
    '''
    R = rng.uniform((-np.pi, 0), (np.pi, np.pi), (N, 2))
    V = np.empty((N, 3))
    c = np.cos(R[:, 1])
    s = np.sin(R[:, 1])
    V[:, 0] = np.cos(R[:, 0]) * s
    V[:, 1] = np.sin(R[:, 0]) * s
    V[:, 2] = c
    assert np.allclose(np.linalg.norm(V, axis=1), 1), 'Norm gen. failed'
    return V

def norm(N):
    return rnd_norm(N)

def rnd_cnorm(N):
    ''' Generate random column vectors on form (N, 3, 3).
    '''
    raise ValueError('Deprecated use "cnorm()"')

def cnorm(N=None):
    if N == None:
        return np.reshape(rnd_norm(1), (3, 1))
    return np.reshape(rnd_norm(N), (-1, 3, 1))



def rnd_R(N=None):
    ''' Generate random rotation matrices on form (N, 3, 3).
    '''
    if N is None:
        return rnd_R(1)[0]
    R = rng.uniform((-np.pi, 0, -np.pi), (np.pi, np.pi, np.pi), (N, 3))
    R = rot.from_euler('ZYX', R)
    return R.as_matrix()

def R(N=None):
    return rnd_R(N)

def rnd_T(N=None):
    ''' Generate random rotation matrices on form (N, 3, 3).
    '''
    if N is None:
        return rnd_T(1)[0]
    T = np.zeros((N, 4, 4))
    T[:, :3, :3] = rnd_R(N)
    T[:, :3, 3] = norm(N)
    T[:, 3, 3] = 1.0
    return T

def T(N=None):
    return rnd_T(N)

VEC_LIMIT_DEF = 10
def rnd_vec(N, limit=VEC_LIMIT_DEF):
    ''' Generate random vectors on form (N, 3).
    '''
    return rng.uniform((-limit, -limit, -limit), (limit, limit, limit), (N, 3))

def rnd_rvec(N=None, limit=VEC_LIMIT_DEF):
    ''' Generate random row vectors on form (N, 1, 3).
    '''
    if N == None:
        return np.reshape(rnd_vec(1, limit), (1, 3))
    return np.reshape(rnd_vec(N, limit), (-1, 1, 3))

def rnd_cvec(N=None, limit=VEC_LIMIT_DEF):
    ''' Generate random column vectors on form (N, 3, 1).
    '''
    if N == None:
        return np.reshape(rnd_vec(1, limit), (3, 1))
    return np.reshape(rnd_vec(N, limit), (-1, 3, 1))

def vec(N):
    return rnd_vec(N)

def rvec(N):
    return rnd_rvec(N)

def cvec(N):
    return rnd_cvec(N)

def unit_axes(N):
    ''' List of repeated unit axes.
    '''
    M = int(np.ceil(N / 3))
    Rt = rnd_R(M)
    A = np.zeros((N, 3, 1))
    A[::3, 0] = 1.0
    A[1::3, 1] = 1.0
    A[2::3, 2] = 1.0

    for i in range(M):
        A[i:i+3] = Rt[i] @ A[i:i+3] # R^T @ [ex, ey, ez]^T
    return A
