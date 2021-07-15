import sys
import numpy as np
import mathf.rng as rng
import mathf.matrix as mat
import mathf.numerical as numer
from other.funcs import verify, make_callable, run_test
from optimizers.gauss_newton import solve_GN
from optimizers.levenberg_marquardt import solve_LM


#
#   Algorithms for the link chain hierarchy
#
def hier_W(q, B):
    ''' Compute the set of joint parameters W in task space.

    Args:
    ----
    q:  Vector defining the current system configuration with regard to the generalized coordinates.
    B:  Rigid body parameters defined by the set [[i, k, d, Rc, pc], ...], containing:
         index to parent joint (previous link in the partial chain),
         rotation axis,
         translation direction,
         constant rotation offset, and
         constant translation offset (parameters are defined relative to the parent joint).
    Returns:
        The ordered set of joint parameters W = [[w, d, R, c], ...], containing:
         axis of rotation, translation direction, rotation matrix and center of rotation for each joint
         (note that all parameters are transformed into absolute/task space).
    '''
    W = [None] * len(B)
    R = np.empty((len(B) + 1, 3, 3))
    c = np.empty((len(B) + 1, 3, 1))
    R[0] = np.eye(3)
    j = 1
    for (i, k, d, Rc, pc), qj in zip(B, q):
        Rc = R[i + 1] @ Rc
        d = Rc @ d
        c[j] = R[i + 1] @ pc + c[i + 1] + d * qj
        R[j] = Rc @ mat.integrate(k, qj)
        W[j - 1] = [Rc @ k, d, R[j], c[j]]
        j += 1
    return W


def hier_E(W, O):
    ''' Compute end effector positions in task space.

    Args:
    ----
    W:  Joint parameters on form [[w, d, R, c], ...].
    O:  End effector parameters defined by [[j, e], ...], containing the joint index and relative offset translation.
    Returns:
        End effector positions in task space containined in the rows of matrix E.
    '''
    E = np.empty((len(O), 3))

    for i, (j, pe) in enumerate(O):
        w, d, R, c = W[j]
        E[i] = (R @ pe + c).ravel()

    return E


def hier_I(B, O):
    ''' Compute joint index mask matrix.

    Forms a dependency matrix X by iterating from the last joint and each joints dependency to its parent.
    '''
    n = len(O)
    m = len(B)
    X = np.zeros((m, m), dtype=np.bool)
    I = np.empty((m, n))
    j = m - 1
    # Form dependency matrix X for the joint hierarchy
    while j >= 1:
        i, k, d, Rc, pc = B[j]
        X[j, j] = 1
        X[i] = np.logical_or(X[i], X[j])
        j = j - 1
    X[0, 0] = 1

    # Form end effector dependency I
    for i, (j, e) in enumerate(O):
        I[:, i] = X[:, j]
    return I


def hier_Jacobian(W, E, I):
    ''' Compute analytical Jacobian.
    Args:
    ----
    W:  Joint parameters on form [[w, d, R, c], ...].
    E:  End effector positions on form Nx3 for each of the N end effectors.
    I:
    '''
    n = E.shape[0]
    J = np.empty((n * 3, len(W)))
    for j, (w, d, R, c) in enumerate(W):
        X = (E - c.T) @ mat.skew_matrix(-w) + d.T  # (E - c.T) @ [w].T
        J[:, j] = (X * I[j:j + 1, :].T).ravel()      # (X * I @ 1^T)

    return J


#
#   Combined (system) algorithms
#
def system(A, B, O):
    ''' Compute end effector configuration for the system parameterization.
    '''
    W = hier_W(A, B)
    E = hier_E(W, O)
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


def compute_Jsys(A, B, O):
    ''' Compute Jacobian for the system defined by B, O and state A.
    '''
    W = hier_W(A, B)
    E = hier_E(W, O)
    I = hier_I(B, O)
    return hier_Jacobian(W, E, I)


def hier_Jres(q, B, O, S):
    ''' Combined computation of the Jacobian J and system residual r.
    '''
    W = hier_W(q, B)
    E = hier_E(W, O)
    I = hier_I(B, O)
    return hier_Jacobian(W, E, I), residual(E, S).reshape(-1, 1)


#
#   Testing helper functions
#
def make_test_system(N, M,
                     rng_jpi=False, rnd_axis=False,
                     rnd_prism=False, rnd_both=False,
                     rnd_eei=False, rnd_rotc=False):
    ''' Create a test system.

    Args:
    N:          Number of system DoF.
    M:          Number of system end effectors.
    rng_jpi:    If true, parent indices (implying hierarchy structure) will be generated randomly.
                 Otherwise, a continous chain will be generated -> [-1, 1, 2, 3, ..] .
    rnd_axis:   If true, joint rotation axes will be randomized.
                 Otherwise (default), axes will be a repeated sequence of rotated unit axis ie:
                  [R_1 @ ex, R_1 @ ey, R_1 @ ez, R_2 @ ex, ...]
    rnd_prism:  Randomly swap revolute joints to prismatic joints.
    rnd_both:   Randomly assign screw joints (only).
    rnd_eei:    If true, end effectors will be allocated to random joints (DoF) in the chain.
                 Otherwise (default), all end effectors will be associated to the last joint DoF.
    rnd_rotc:   If true, each joint will be assigned a random rotation constant.
    '''
    if rnd_axis:
        rot_axes = rng.rnd_cnorm(N)
    else:
        rot_axes = rng.unit_axes(N)
    if rnd_prism or rnd_both:
        # Randomly swap rotation axis to translation axis for prismatic joint
        trans_dir = np.zeros(rot_axes.shape)
        for i, b in enumerate(rng.bool(N, 1, 2)):
            if b:
                trans_dir[i] = rot_axes[i]
                if not rnd_both:
                    rot_axes[i] = 0
    else:
        trans_dir = np.zeros(rot_axes.shape)
    if rnd_rotc:
        RRc = rng.R(N)
    else:
        RRc = np.repeat(np.eye(3)[np.newaxis, :, :], N, axis=0)
    if rnd_eei:
        ee_inds = rng.ints(M, N)
    else:
        ee_inds = [N - 1] * M
    if rng_jpi:
        # Generate random hierarchy
        jp_inds = np.zeros(N, dtype=np.int64)
        jp_inds[0] = -1

        j = 1
        r = (0, 1)
        while j < N:
            # Create random number of leafs [1, remaining + 1) at each iteration:
            n_connect = rng.ints(1, 1, N - j + 1)[0]
            # Randomize the parents among previous leafs:
            p_inds = rng.ints(n_connect, *r)
            # Assign parents by 'appending' to list:
            jp_inds[j:j + n_connect] = p_inds
            # Update iter. parameters:
            r = (j, j + n_connect)
            j += n_connect
    else:
        jp_inds = np.arange(-1, N - 1)

    # Rigid body parameters.
    B = [[i, k, d, Rc, pc] for i, k, d, Rc, pc in zip(jp_inds, rot_axes, trans_dir, RRc, rng.rnd_cvec(N, 0.1))]
    # End effector offsets.
    O = [[i, e] for i, e in zip(ee_inds, rng.rnd_cvec(N, 0.1))]

    As = rng.rnd_a(N)       # Angle (final configuration).

    S = system(As, B, O)    # Compute end effector targets as defined by Ay.

    return B, O, S, As


#
#   Test: Compare hier_I() chain_I()
#
def test_compare_chain_I(N, M):
    ''' Simple test verifying relationships of the index matrix I,
            all end effectors are assigned to the last joint.

    Args:
    N:  Number of system DoF.
    M:  Number of system end effectors.
    '''
    B, O, S, As = make_test_system(N, M, rnd_eei=True)

    Ic = np.empty((len(O), len(B)))
    # Fill each row with the joint index j associated to the i:th end effector
    for i, (j, e) in enumerate(O):
        Ic[i, :] = j

    # All end effectors should be assigned to last joint, just compare.
    for i in range(len(B)):
        Ic[:, i] = i <= Ic[:, i]

    Ih = hier_I(B, O)

    verify(Ic.T, Ih, 'Mismatch between chain_I and hier_I in %i rows.', None)


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
    # Angle (init).
    A = rng.rnd_a(N)

    Ja = compute_Jsys(A, B, O)
    Jn = numer.Jacobian(make_callable(system_residual, B, O, S), A)

    verify(Ja, Jn, 'Mismatch between analytical and numerical Jacobian in %i rows', None)


#
#   Test: Optimization convergence (analytical)
#
def test_optimization(N, M, angle_std=0.1, **sys_kwargs):
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
        make_callable(hier_Jres, B, O, S),
        A,
        max_iter=2000,
        eps=EPS)

    if res > EPS:
        assert False, 'Reached local minima or failed for [%i, %i] system with residual %f.' % (N, M, res)


#
#   Run tests
#
RUN_JAC_TEST = True
RUN_OPT6_TEST = True
RUN_OPTN_TEST = True

# Test I
##
for N in range(6, 8):
    for M in range(1, 4):
        run_test(lambda: test_compare_chain_I(N, M), tname='Ic == Ih, %i, %i' % (N, M))

# Optimization tests
##
if RUN_JAC_TEST:
    for N in range(6, 8):
        for M in range(1, 4):
            run_test(lambda: test_compare_numerical_J(N, M), tname='Ja == Jn, %i, %i' % (N, M))

    for N in range(6, 8):
        for M in range(1, 4):
            run_test(lambda: test_compare_numerical_J(N, M, rnd_prism=True),
                     tname='Ja == Jn, %i, %i, rnd_prism' % (N, M))

    for N in range(6, 8):
        for M in range(1, 4):
            run_test(lambda: test_compare_numerical_J(N, M, rnd_both=True),
                     tname='Ja == Jn, %i, %i, rnd_both' % (N, M))

    for N in range(6, 8):
        for M in range(1, 4):
            run_test(lambda: test_compare_numerical_J(N, M, rnd_prism=True, rnd_eei=True),
                     tname='Ja == Jn, %i, %i, rnd_prism:rnd_eei' % (N, M))


# Optimization tests
##
if RUN_OPT6_TEST:
    estd = 0.4
    # Single serial chain with 6 DoF and all ee at end:
    for M in range(1, 6):
        run_test(lambda: test_optimization(6, M, angle_std=estd),
                 tname='Analytical optimization, %i, %i' % (6, M))

    ##
    # Hierarchy 6 DoF chain with randomly assign ee
    for M in range(3, 12):
        run_test(lambda: test_optimization(6, M, angle_std=estd, rng_jpi=True, rnd_eei=True),
                 tname='Analytical optimization of %i, %i system (rng_jpi:rnd_eei)' % (6, M))

    ##
    # Hierarchy 6 DoF chain with randomly assign ee and prismatic joints
    for M in range(3, 12):
        run_test(lambda: test_optimization(6, M, angle_std=estd, rng_jpi=True, rnd_prism=True, rnd_eei=True),
                 tname='Analytical optimization of %i, %i system (rng_jpi:rnd_prism:rnd_eei)' % (6, M))

    ##
    # Hierarchy 6 DoF chain with randomly assign ee and screw joints
    for M in range(3, 12):
        run_test(lambda: test_optimization(6, M, angle_std=estd, rng_jpi=True, rnd_both=True, rnd_eei=True),
                 tname='Analytical optimization of %i, %i system (rng_jpi:rnd_both:rnd_eei)' % (6, M))


if RUN_OPTN_TEST:
    estd = 0.1
    ##
    # Hierarchy 8-14 DoF chain with randomly assigned 12 ee and prismatic joints
    for N in range(8, 14):
        M = 12
        run_test(lambda: test_optimization(N, M, angle_std=estd, rng_jpi=True, rnd_prism=True, rnd_eei=True),
                 tname='Analytical optimization of %i, %i system (rng_jpi:rnd_prism:rnd_eei)' % (N, M))
