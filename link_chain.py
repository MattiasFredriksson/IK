import sys
import numpy as np
import mathf.rng as rng
import mathf.matrix as mat
import mathf.numerical as numer
from other.funcs import verify, make_callable, run_test
from optimizers.gauss_newton import solve_GN
from optimizers.levenberg_marquardt import solve_LM


#
#   Single connected link chain algorithms
#
def compute_W(q, B):
    ''' Compute the set of joint parameters W in task space.

    Args:
    ----
    q:  Angle vector defining the current state of the link chain.
    B:  Rigid body parameters defined by the set [[k, p], ...], containing:
         rotation axis and translation offset (both defined in the space of previous joint in the link chain).
    Returns:
        The ordered set of joint parameters W = [[w, R, c], ...], containing:
         axis of rotation, rotation matrix and center of rotation for each joint (all in absolute/task space).
    '''
    R = np.eye(3)
    c = np.zeros((3, 1))
    W = []

    for (k, p), a in zip(B, q):
        c = R @ p + c
        R = R @ mat.integrate(k, a)
        W.append([R @ k, R, c])

    return W


def compute_E(W, O):
    ''' Compute end effector positions in task space.

    Args:
    ----
    W:  Joint parameters on form [[w, R, c], ...].
    O:  End effector parameters defined by [[j, e], ...], containing the joint index and relative offset translation.
    Returns:
        End effector positions in task space containined in the rows of matrix E.
    '''
    E = np.empty((len(O), 3))

    for i, (j, e) in enumerate(O):
        w, R, c = W[j]
        E[i] = (R @ e + c).ravel()

    return E


def compute_I(O):
    ''' Compute joint index mask matrix.
    '''
    I = np.empty((len(O), 1))
    for i, (j, e) in enumerate(O):
        I[i] = j
    return I


def compute_J(W, E, I):
    ''' Compute analytical Jacobian.
    Args:
    ----
    W:      Joint parameters on form [[w, R, c], ...].
    E:      End effector positions on form NxM for each of the N end effectors and M joints.
    I:      Index matrix I mapping the joint associated with each end effector.
    '''
    J = np.empty((3 * E.shape[0], len(W)))
    for j, (w, R, c) in enumerate(W):
        X = (E - c.T) @ mat.skew_matrix(-w)   # (E - c.T) @ [w].T
        J[:, j] = (X * (j <= I)).ravel()      # (X * I @ 1^T)
    return J


#
#   Combined (system) algorithms
#
def system(q, B, O):
    ''' Compute end effector configuration for the system parameterization.
    '''
    W = compute_W(q, B)
    E = compute_E(W, O)
    return E


def residual(E, S):
    ''' Compute the residual between sampled end effector positions and end effector positions.

    Args:
    ----
    E:    Current end effector state in absolute/task space.
    S:    Sampled/target end effector state in absolute/task space.

    Return:
        System residual E - S.
    '''
    return E - S


def system_residual(q, B, O, S):
    ''' Compute the system residual vector.
    '''
    return residual(system(q, B, O), S).reshape(-1, 1)


def compute_Jsys(q, B, O):
    ''' Compute Jacobian for the system defined by B, O and state A.
    '''
    W = compute_W(q, B)
    E = compute_E(W, O)
    I = compute_I(O)
    return compute_J(W, E, I)


def compute_Jres(q, B, O, S):
    ''' Combined computation of the Jacobian J and system residual r.
    '''
    W = compute_W(q, B)
    E = compute_E(W, O)
    I = compute_I(O)
    return compute_J(W, E, I), residual(E, S).reshape(-1, 1)


#
#   Testing helper functions
#
def make_test_system(N, M, rnd_axis=False, rnd_eei=False):
    ''' Create a test system.

    Args:
    N:          Number of system DoF.
    M:          Number of system end effectors.
    rnd_axis:   If true, joint rotation axes will be randomized.
                 Otherwise (default), axes will be a repeated sequence of rotated unit axis ie:
                  [R_1 @ ex, R_1 @ ey, R_1 @ ez, R_2 @ ex, ...]
    rnd_ee:     If true, end effectors will be allocated to random joints (DoF) in the chain.
                 Otherwise (default), all end effectors will be associated to the last joint DoF.
    '''
    if rnd_axis:
        rot_axes = rng.rnd_cnorm(N)
    else:
        rot_axes = rng.unit_axes(N)
    if rnd_eei:
        ee_inds = rng.ints(M, N)
    else:
        ee_inds = [N - 1] * M

    B = [[k, p] for k, p in zip(rot_axes, rng.rnd_cvec(N, 0.1))]   # Rigid body parameters.
    O = [[i, e] for i, e in zip(ee_inds, rng.rnd_cvec(N, 0.1))]  # End effector offsets.

    As = rng.rnd_a(N)       # Angle (final configuration).

    S = system(As, B, O)  # Compute end effector targets as defined by Ay.

    return B, O, S, As


#
#   Test: Compare numerical with analytical Jacobian.
#
def test_compare_numerical_J(N, M, **sys_kwargs):
    ''' Test comparing numerical with analytical Jacobian.

    Args:
    N:  Number of system DoF.
    M:  Number of system end effectors.
    '''
    B, O, S, As = make_test_system(N, M, **sys_kwargs)
    A = rng.rnd_a(N)        # Angle (init).

    Ja = compute_Jsys(A, B, O)
    Jn = numer.Jacobian(make_callable(system_residual, B, O, S), A)

    verify(Ja, Jn, 'Mismatch between analytical and numerical Jacobian in %i rows', None)


#
#   Test: Optimization convergence (analytical)
#
def test_optimization(N, M, angle_std=0.5, **sys_kwargs):
    ''' Test optimization for system with N DoF and M end effectors.

    Args:
    N:  Number of system DoF.
    M:  Number of system end effectors.
    '''
    B, O, S, As = make_test_system(N, M, **sys_kwargs)
    # Angle (init).
    if angle_std <= 0:
        A = rng.rnd_a(N)
    else:
        A = As + rng.normal(As.shape, std=angle_std)

    EPS = 1e-6
    #
    #   Find optimal system parameterizations using analytical Jacobian.
    #
    X, res = solve_LM(
        make_callable(compute_Jres, B, O, S),
        A,
        regu=1e-7,
        max_iter=2000,
        eps=EPS)

    if res > EPS:
        sys = make_callable(system_residual, B, O, S)
        iss = np.any(numer.issingular(numer.Hessian(sys, X)))
        assert False,\
            'Reached local minima or failed. [%i, %i] system with residual %f. Has singular Hessian: %s' %\
            (N, M, res, str(iss))


#
#   Run tests
#
JAJN_TEST = True
OPT_TEST = True
if JAJN_TEST:
    for N in range(6, 8):
        for M in range(1, 4):
            run_test(lambda: test_compare_numerical_J(N, M),
                     tname='Ja == Jn, %i, %i' % (N, M))

    for N in range(6, 8):
        for M in range(1, 12):
            run_test(lambda: test_compare_numerical_J(N, M, rnd_eei=True),
                     tname='Ja == Jn, %i, %i, rnd_eei' % (N, M))

    for M in range(1, 12):
        run_test(lambda: test_optimization(6, M), tname='Analytical optimization, %i, %i' % (6, M))

if OPT_TEST:
    print('----')
    print('Optimization 6 DoF + random ee distribution')
    print('----')
    for M in range(1, 12):
        run_test(lambda: test_optimization(6, M, rnd_eei=True),
                 tname='Analytical optimization, %i, %i, rnd_eei' % (6, M))
