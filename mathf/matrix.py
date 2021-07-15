import numpy as np
from scipy.spatial.transform import Rotation as rot
EPS = 1e-7


def makeN(V):
    return np.reshape(V, (-1, ))


def makeN1(V):
    return np.reshape(V, (-1, 1))


def makeN3(V):
    return np.reshape(V, (-1, 3))


def makeN31(V):
    return np.reshape(V, (-1, 3, 1))


def makeN33(V):
    return np.reshape(V, (-1, 3, 3))


def make_col(v):
    sh = np.shape(v)
    if sh[-1] != 1:
        return np.reshape(v, (*sh, 1))
    return v


def element(V, i):
    ''' Fetch the i:th element from vector(s) V.
    '''
    sh = np.shape(V)
    if len(sh) > 1:
        if len(sh) > 2 or sh[-1] > 1:   # (N, ...) or (N, x) where x > 1
            return V[:, i]
        else:
            return V[i]
    return V[i]


def column(M, i):
    ''' Fetch the i:th column from matrix/ces M.
    '''
    if np.ndim(M) > 2:
        return M[:, :, i]
    return M[:, i]


def inv(T):
    return np.linalg.inv(T)


def transpose(R):
    return np.transpose(R, axes=(0, 2, 1))


def normalize(k):
    sh = np.shape(k)
    if sh[-2:] == (3, 1):
        return 1.0 / np.linalg.norm(k, axis=-2, keepdims=True) * k
    else:
        return 1.0 / np.linalg.norm(k, axis=-1, keepdims=True) * k


def cross(a, b):
    sh = np.shape(a)
    shb = np.shape(b)
    assert sh == shb, 'Shapes must match, was %s and %s.' % (str(sh), str(shb))
    if sh[-2:] == (3, 1):
        return transpose(np.cross(a, b, axisa=-2, axisb=-2))
    elif sh[-1] == 3:
        return np.cross(a, b, axisa=-1, axisb=-1)
    else:
        raise ValueError("Can't handle shape %s" % str(sh))


def dot(a, b):
    sh = np.shape(a)
    shb = np.shape(b)
    assert sh == shb, 'Shapes must match, was %s and %s.' % (str(sh), str(shb))
    if sh[-2:] == (3, 1):
        return np.sum(a * b, axis=-2, keepdims=True)
    elif sh[-1] == 3:
        return np.sum(a * b, axis=-1, keepdims=True)
    else:
        raise ValueError("Can't handle shape %s" % str(sh))


def RX(a):
    return rot.from_euler('x', a).as_matrix()


def RY(a):
    return rot.from_euler('y', a).as_matrix()


def RZ(a):
    return rot.from_euler('z', a).as_matrix()


def skew_matrix(k, a=None):
    ''' k -> K
    '''
    sh = np.shape(k)
    k = makeN3(k)
    K = np.zeros((len(k), 3, 3))
    if a is not None:
        k = a * k
    # Z
    K[:, 0, 1] = -k[:, 2]
    K[:, 1, 0] = k[:, 2]
    # Y
    K[:, 0, 2] = k[:, 1]
    K[:, 2, 0] = -k[:, 1]
    # X
    K[:, 2, 1] = k[:, 0]
    K[:, 1, 2] = -k[:, 0]

    if sh[-1] != 3:
        return np.reshape(K, (*sh[:-2], 3, 3))
    return np.reshape(K, (*sh[:-1], 3, 3))


def skew_vector(K):
    ''' K -> k
    '''
    sh = np.shape(K)
    K = makeN33(K)
    k = np.empty((len(K), 3))
    k[:, 0] = K[:, 2, 1]
    k[:, 1] = K[:, 0, 2]
    k[:, 2] = K[:, 1, 0]

    if len(sh) == 2:
        return k[0]
    else:
        return k


def skew_vectorc(K):
    return skew_vector(K)[:, :, np.newaxis]


def logarithm(R, eps=EPS):
    ''' Logarithm of rotation matrix in SO3, returning the skew matrix for ak.
    '''
    sh = np.shape(R)
    assert sh[-2:] == (3, 3), 'Input must be on form (..., 3, 3)'
    R = makeN33(R)
    cosa = np.trace(R, axis1=-2, axis2=-1) * 0.5 - 0.5
    assert np.all(np.abs(cosa) < 1.0 + eps), 'Not a rotation'
    fac = np.empty(cosa.shape)

    mask = 1.0 - eps < np.abs(cosa)
    mask_inv = np.logical_not(mask)

    a = np.arccos(cosa[mask_inv])
    fac = a / (2.0 * np.sin(a))
    fac = np.reshape(fac, (-1, 1, 1))

    # Case: a = 0
    K = np.zeros(R.shape)
    # Case: 0 < a < pi
    M = R[mask_inv]
    K[mask_inv] = (M - transpose(M)) * fac

    # Edge cases of: a = pi
    mask = eps - 1.0 > cosa
    if np.any(mask):
        R = R[mask]
        # Column X
        xmask = np.logical_and(R[:, 0, 0] >= R[:, 1, 1], R[:, 0, 0] >= R[:, 2, 2])
        ymask = np.logical_and(R[:, 1, 1] >= R[:, 2, 2], np.logical_not(xmask))
        zmask = np.logical_and(np.logical_not(xmask), np.logical_not(ymask))
        S = np.empty(R.shape)
        if np.any(xmask):
            fac = np.pi / np.sqrt(2.0 * (R[xmask, 0, 0] + 1))
            w = fac * R[xmask, :, 0]
            w[:, 0] += fac
            S[xmask] = skew_matrix(w)
        # Column Y
        if np.any(ymask):
            fac = np.pi / np.sqrt(2.0 * (R[ymask, 1, 1] + 1))
            w = fac * R[ymask, :, 1]
            w[:, 1] += fac
            S[ymask] = skew_matrix(w)
        # Column Z
        if np.any(zmask):
            fac = np.pi / np.sqrt(2.0 * (R[zmask, 2, 2] + 1))
            w = fac * R[zmask, :, 2]
            w[:, 2] += fac
            S[zmask] = skew_matrix(w)
        K[mask] = S

    return K.reshape(sh)


def log(R):
    return logarithm(R)


def log_so(R):
    return logarithm(R)


def integrate(k, a=None):
    ''' Rodrigues formula.
    '''
    sh = np.shape(k)
    is_skew_mat = sh[-2:] == (3, 3)
    is_skew_vec = sh[-1:] == (3, )
    is_skew_cvec = sh[-2:] == (3, 1)
    assert is_skew_mat or is_skew_vec or is_skew_cvec,\
        'Expected rotation axis on vector => [.., 3] or skew matrix => [.., 3, 3] form.'
    if not is_skew_mat:
        k = makeN3(k)
    if a is None:
        if is_skew_mat:
            assert np.allclose(k[:, 0, 0], 0), 'Input not skew matrix first diagonal entry K_{0,0} is not zero.'
            assert np.allclose(k[:, 1, 1], 0), 'Input not skew matrix first diagonal entry K_{1,1} is not zero.'
            assert np.allclose(k[:, 2, 2], 0), 'Input not skew matrix first diagonal entry K_{2,2} is not zero.'
            # k is a skew matrix with shape (:, 3, 3)
            a = np.linalg.norm(k, axis=(-1, -2)) / np.sqrt(2)
            K = k / a[:, np.newaxis, np.newaxis]
        else:
            # k is rotation vector with shape (:, 3)
            a = np.linalg.norm(k, axis=-1)
            k = k / a[:, np.newaxis]
    a = makeN1(a)
    # Convert rotation axis to skew matrix
    if not is_skew_mat:
        sh_n = sh[:-1] if is_skew_vec else sh[:-2]
        K = skew_matrix(k)
    else:
        sh_n = sh[:-2]
        K = makeN33(k)

    # Rodrigues formula
    KK = np.matmul(K, K)
    if K.ndim > 2:
        a = a[:, :, np.newaxis]
    R = np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * KK
    # Return result in original form
    return np.reshape(R, (*sh_n, 3, 3))


def exp(k, a=None):
    return integrate(k, a)


def make_angle_skew3(rotation):
    sh = np.shape(rotation)
    # Since rotation (3, 3) can't be distinguised from 3 stacked rotation vectors, only convert rotations for ndim > 2
    if len(sh) > 2 and sh[-2:] == (3, 3):
        skewm = log_so(rotation)
        rotvec = skew_vector(skewm)
    else:
        rotvec = makeN3(rotation)
        skewm = skew_matrix(rotvec)
    a = np.linalg.norm(rotvec, axis=-1)[:, np.newaxis, np.newaxis]
    return a, skewm


def angv2dso_M(rotation):
    ''' Generate linear maps for converting the geometric Jacobian to the analytical Jacobian for rotation vectors.

    Args:
    ----
    rotation: Rotations or rotation vector(s) defining the current orientation for which the differentials are computed.
                All values should be defined in the same frame.
    returns:
        Matrix(es) converting geometric Jacobian for the orientation to the numerical Jacobian.
    '''
    sh = np.shape(rotation)
    a, skewm = make_angle_skew3(rotation)

    A = np.zeros(skewm.shape)
    A[:, 0, 0] = 1.0
    A[:, 1, 1] = 1.0
    A[:, 2, 2] = 1.0
    #
    A -= 1 / 2 * skewm
    fac = 1 / (a * a) * (1 - (a * np.sin(a)) / (2 * (1 - np.cos(a))))
    A += fac * skewm @ skewm

    return A


def dso2angv_M(rotation):
    ''' Generate linear maps for converting the analytical Jacobian for rotation vectors to the geometric Jacobian.

    Args:
    ----
    rotation: Rotations or rotation vector(s) defining the current orientation for which the differentials are computed.
                All values should be defined in the same frame.
    returns:
        Matrix(es) converting the analytical differential of a rotation vector to angular velocity for the rotation.
    '''
    sh = np.shape(rotation)
    a, skewm = make_angle_skew3(rotation)

    A = np.zeros(skewm.shape)
    A[:, 0, 0] = 1.0
    A[:, 1, 1] = 1.0
    A[:, 2, 2] = 1.0
    #
    A += (1 - np.cos(a)) / (a * a) * skewm
    A += (a - np.sin(a)) / (a * a * a) * skewm @ skewm

    return A


def dso2angv(rotation, drotvec):
    sh = np.shape(drotvec)
    drotvec = makeN31(drotvec)

    A = dso2angv_M(rotation)
    return np.reshape(A @ drotvec, sh)


def angv2dso(rotvec, angv):
    sh = np.shape(angv)
    angv = makeN31(angv)

    A = angv2dso_M(rotvec)
    return np.reshape(A @ angv, sh)


# SE3
def log_se_Rp(R, p):
    ''' Logarirthm for transforms in SE(3)
    '''
    sh = np.shape(R)
    shp = np.shape(p)
    ndim = len(sh)
    if len(sh) == 2:
        # no leading dims
        v, a = log_se_Rp(R[np.newaxis, :, :], p[np.newaxis, :, :])
        return v[0], a[0]
    # Check dims
    ldims = sh[:ndim - 2]
    assert ndim == 2 or ldims == shp[:ndim - 2], \
        'Mismatch in leading dimensions between inputs. Shapes R: %s, p: %s' % (str(sh), str(shp))
    p = makeN31(p)
    R = makeN33(R)
    skewm = log(R)
    so3 = skew_vector(skewm)
    a = np.linalg.norm(so3, axis=-1)
    skewm /= a

    Ginv = -1 / 2 * skewm
    Ginv += (1 / a - 1 / (2 * np.tan(a / 2))) * (skewm @ skewm)
    v = p / a + Ginv @ p
    v = np.concatenate([so3 / a, v[:, :, 0]], axis=-1)
    return np.reshape(v, [*ldims, 6]), np.reshape(a, [*ldims, 1])


def log_se(T, p=None):
    if p is None:
        return log_se_Rp(T[..., :3, :3], T[..., :3, 3:4])
    return log_se_Rp(T, p)
