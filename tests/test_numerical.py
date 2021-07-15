import sys
import os
import numpy as np
try:
    from mathf import numerical as numer
    from other.funcs import verify
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from mathf import numerical as numer
    from other.funcs import verify


#
#   Diagonal Hessian test using: F = q.T @ q
#
def F(q):
    return q.T @ q  # == q1^2 + q2^2 + .. + qn^2


for n in range(20):
    H = numer.Hessian(F, np.zeros((n, 1)))
    verify(H, np.diag(np.full(n, 2)), smsg=None, emsg='Numerical diagonal hessian failed of rank %i' % n)
    assert numer.PDH(H), 'Expected Hessian to be positive definite.'
print('Diagonal Hessian success!')
