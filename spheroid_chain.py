import sys
import numpy as np
import mathf.rng as rng
import mathf.matrix as mat
import mathf.numerical as numer
from other.funcs import verify, make_callable, run_test
from optimizers.gauss_newton import solve_GN
from optimizers.levenberg_marquardt import solve_LM

def spheroid_config(q, B):
    ''' Compute relevant system configuration parameters.

    Args:
    ----
    q:  Vector defining the current system configuration with regard to the generalized coordinates.
    B:  Rigid body parameters defined by the set [[Rc, pc], ...], containing:
         constant rotation offset, and
         constant translation offset (both defined relative to the parent joint).
    Returns:
        P:  Joint positions in task space on form (m, 3).
        W:  Joint rotation axis W in task space on form (m, 3).
        R:  Rotation matrix for the last joint.
        c:  Position of the last joint.
    '''
    m = 3 * len(B)
    P = np.zeros((m, 3))
    W = np.zeros((m, 3))

    # Compute configuration in task space
    R = np.eye(3)
    c = np.zeros((3, 1))
    exeyez = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    j = 0
    for R_c, p_c in B:
        c = R @ p_c + c
        R = R @ R_c
        for i, k in enumerate(exeyez):
            R = R @ mat.exp(k, q[j])
            W[j] = R[:, i]
            P[j] = c.ravel()
            j += 1
    return P, W, R, c

def spheroid_Jres(q, B, O, S):
    ''' Compute both Jacobian J and system residual r for the link chain of spheroid joints.
    '''
    P, W, R, c = spheroid_config(q, B)
    # Unpack EE
    p_e, w_e = O
    p_st, w_st = S

    # Compute residual
    p_se = R @ p_e + c
    d_w = mat.skew_vector(mat.log(mat.exp(w_st) @ mat.exp(-w_e) @ R.T))[:, np.newaxis]
    r = np.concatenate((p_se - p_st, -d_w), axis=0)
    # Jacobian
    J_pt = mat.skew_matrix(W) @ (p_se.T - P)[:, :, np.newaxis]
    J = np.concatenate((J_pt[:, :, 0].T, W.T), axis=0)
    return J, r

def spheroid_E(q, B, O):
    ''' Compute end effector position and orientation.
    '''
    P, W, R, c = spheroid_config(q, B)
    # Unpack EE
    p_e, w_e = O
    return R @ p_e + c, mat.skew_vector(mat.log(R @ mat.exp(w_e)))[:, np.newaxis]

#
#   Testing helper functions
#
def make_test_system(M):
    ''' Create a test system.

    Args:
    M:          Number of system DoF.
    '''
    RRc = rng.R(M)
    Pc = rng.rnd_cvec(M, 0.1)

    # Rigid body parameters.
    B = [[Rc, pc] for Rc, pc in zip(RRc, Pc)]
    # End effector offsets.
    oa = rng.angle()
    O = (rng.rnd_cvec(limit=0.1), rng.cnorm()*oa)
    Qs = rng.rnd_a(M * 3)       # Angle (final configuration).

    S = spheroid_E(Qs, B, O)

    return B, O, S, Qs

#
#   Test: Compare numerical with analytical Jacobian.
#
def test_compare_numerical_J(M, **sys_kwargs):
    ''' Test comparing numerical with analytical Jacobian.

    Args:
    N:  Number of system DoF.
    M:  Number of system end effectors.
    '''
    B, O, S, As = make_test_system(M, **sys_kwargs)
     # Angle (init).
    Q = rng.rnd_a(M * 3)

    P, W, R, c = spheroid_config(Q, B)
    p_e, w_e = O
    R = R @ mat.exp(w_e)


    Ja, r = spheroid_Jres(Q, B, O, S)

    def Jnumer(q):
        ''' Computing our Jacobian of the rotation vector numerically using the Jres algorithm
            does not make sence as it already computed the difference. Instead compute the end effector
            configuration E and adjust the 'geometrical' Jacobian for the rotation vector.
        '''
        return np.concatenate(spheroid_E(q, B, O), axis=0)
    Jn = numer.Jacobian(Jnumer, Q)

    verify(Ja[:3], Jn[:3], 'Mismatch between analytical and numerical position Jacobian in %i rows', None)

    # Convert Jacobian angv2dso_M(R)
    A = mat.angv2dso_M(R[np.newaxis, :, :])[0]
    verify(A @ Ja[3:], Jn[3:],
           emsg='Mismatch between analytical and numerical orientation Jacobian in %i rows',
           smsg=None,
           atol=5e-6)

#
#   Test: Optimization convergence (analytical)
#
def test_optimization(M, angle_std=0.1, **sys_kwargs):
    ''' Test optimization for system with N DoF and M end effectors.

    Args:
    M:  Number of system DoF.
    '''
    B, O, S, Qs = make_test_system(M, **sys_kwargs)
     # Angle (init).
    if angle_std > 0:
        Q = Qs + rng.normal(Qs.shape, std=angle_std)
    else:
        Q = rng.rnd_a(M * 3)

    EPS = 1e-6
    #
    #   Find optimal system parameterizations using analytical Jacobian.
    #
    X, res = solve_LM(
        make_callable(spheroid_Jres, B, O, S),
        Q,
        max_iter=2000,
        eps=EPS)

    if res > EPS:
        assert False, 'Reached local minima or failed for [%i] system with residual %f.' % (M, res)


#
#   Run tests
#
RUN_JAC_TEST = True
RUN_OPTN_TEST= True

# Optimization tests
##
if RUN_JAC_TEST:
    for M in range(2, 6):
        run_test(lambda: test_compare_numerical_J(M), tname='Ja == Jn, %i' % (M), doraise=True)

if RUN_OPTN_TEST:
    estd = 0.2
    # Single serial chain with 6+ DoF (with close initial condition):
    for M in range(2, 6):
        run_test(lambda: test_optimization(M, angle_std=estd), tname='Analytical optimization, %i' % (3 * M))

    estd = -1
    # Single serial chain with 6+ DoF (with random initial condition):
    for M in range(2, 6):
        run_test(lambda: test_optimization(M, angle_std=estd), tname='Analytical optimization, %i' % (3 * M))
