import sys
import os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import mathf.rng as rng
import mathf.matrix as mat
import mathf.numerical as numer
from other.funcs import verify

N = 1000
R = rng.rnd_R(N)

# k
k = rng.rnd_norm(N)

# ex
ex = np.zeros((N, 3))
ex[:, 0] = 1

# ey
ey = np.zeros((N, 3))
ey[:, 1] = 1

# ez
ez = np.zeros((N, 3))
ez[:, 2] = 1

a = rng.rnd_a(N)


RX = mat.RX(a)
RY = mat.RY(a)
RZ = mat.RZ(a)
RK = mat.integrate(k, a)

#
# Check matrix.transpose
#
I = np.eye(3)
I_t = R @ mat.transpose(R)
verify(I, I_t, 'Mismatch in R @ R^t in %i', 'Success: matrix.transpose()!')


#
# Check skew X, Y, Z
#
# Skew X
K = mat.skew_matrix([1, 0, 0])
K_ref = np.array([[0,0,0], [0,0,-1], [0,1,0]])
verify(K, K_ref, smsg='Success: skew X')
# Skew Y
K = mat.skew_matrix([0, 1, 0])
K_ref = np.array([[0,0,1], [0,0,0], [-1,0,0]])
verify(K, K_ref, smsg='Success: skew Y')
# Skew Z
K = mat.skew_matrix([0, 0, 1])
K_ref = np.array([[0,-1,0], [1,0,0], [0,0,0]])
verify(K, K_ref, smsg='Success: skew Z')

#
# Check k -> K -> k
#
k_t = mat.skew_vector(mat.skew_matrix(k))
verify(k, k_t, smsg='Success: skew conversion k -> K -> k')

#
# Check matrix.integrate() + skew_
#
RX_ref = mat.integrate(ex, a)
verify(RX, RX_ref, smsg='Success: matrix.integrate()')

#
# Check log(RX) == aex
#
aex = mat.skew_vector(mat.logarithm(RX))
verify(aex, a * ex, smsg='Success: log(RX) == aex')

#
# Check log(RY) == aey
#
aey = mat.skew_vector(mat.logarithm(RY))
verify(aey, a * ey, smsg='Success: log(RY) == aey')

#
# Check log(RZ) == aez
#
aez = mat.skew_vector(mat.logarithm(RZ))
verify(aez, a * ez, smsg='Success: log(RZ) == aez')


#
# Check exp(RK) == a*k
#
ak = mat.skew_vector(mat.logarithm(RK))
verify(a * k, ak, smsg='Success: log(RK) == a * k')
# Check ||exp(RK)|| / a == 1
a_ak = np.linalg.norm(ak, axis=-1)
verify(np.abs(a_ak / mat.makeN(a)), 1.0, smsg='Success: ||exp(RK)|| / a == 1')
# Check exp(log(RK)) == RK
K = mat.integrate(ak)
verify(K, RK, smsg='Success: exp(log(RK)) == RK')


#
# Check exp(log(R)) == R
#
K = mat.logarithm(R)
K_exp = mat.integrate(mat.skew_vector(K))
verify(R, K_exp, smsg='Success: exp(log(R)) == R')



##
#   Check log() for rotations of pi around unit axes
##
# X
X = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
])
Xi = np.array([
    [1, 0, 0],
    [0, -1, 0],
    [0, 0, -1]
])
verify(mat.log(Xi @ mat.inv(X)), mat.skew_matrix([[np.pi, 0, 0]]))
# Y
Y = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
])
Yi = np.array([
    [-1, 0, 0],
    [0, 1, 0],
    [0, 0, -1]
])
verify(mat.log(Yi @ mat.inv(Y)), mat.skew_matrix([[0, np.pi, 0]]))
# Z
Z = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
])
Zi = np.array([
    [-1, 0, 0],
    [0, -1, 0],
    [0, 0, 1]
])
verify(mat.log(Zi @ mat.inv(Z)), mat.skew_matrix([[0, 0, np.pi]]))



#
#
#
N = 100
h = 4e-7
r = rng.norm(N) * rng.angle(N)
R = mat.exp(r)

a = rng.angle(N)
k = rng.norm(N)
w = k * a

# Central difference:
dr = mat.skew_vectorc((mat.log(mat.exp(w * h) @ R) - mat.log(mat.exp(w * -h) @ R)) / (2 * h))

# Conersion matrices
A_inv = mat.angv2dso_M(r)
A = mat.dso2angv_M(r)
AA_inv = mat.inv(A)

# Test AA^-1 = I
verify(A_inv @ A, np.eye(3))

# Test inverting w -> dr
verify(A_inv @ w[:, :, np.newaxis], dr, atol=h)
verify(AA_inv @ w[:, :, np.newaxis], dr, atol=h)


# Test using rotation matrix for computing the maps
AR_inv = mat.angv2dso_M(R)
AR = mat.dso2angv_M(R)
verify(AR_inv, A_inv)
verify(AR, A)

# Test converting angular -> difference -> angular
cdr = mat.angv2dso(r, w)
cw = mat.dso2angv(r, cdr)
verify(w, cw)
