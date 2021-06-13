import numpy as np

def make_callable(func, *args):
    ''' Make a callable from args.
    '''
    def func_of_q(q):
        return func(q, *args)
    return func_of_q


def verify(A, B, emsg='Mismatch in %i', smsg='Success!', atol=4e-7):
    ''' Check for equality of A & B.
    Params:
    ----
    A:      Numpy array as LH argument.
    B:      Numpy array as RH argument.
    emsg:   Error message (printed on failure in any).
    smsg:   Successfull message (printed if all are equal).
    atol:   Absolute tolerance in difference.
    '''
    diff = A - B
    diffsum = np.sum(np.abs(diff), axis=tuple(np.arange(1, diff.ndim, dtype=np.int64)))
    is_equal = np.isclose(diffsum, 0, atol=atol)
    all_equal = np.all(is_equal)
    if not all_equal:
        nequal = np.logical_not(is_equal)
        print('Value difference(s):')
        print(diff[nequal][:10])
        print('Elementwise absolute difference(s):')
        print(diffsum[nequal][:10])
    assert all_equal, emsg % (len(is_equal) - np.sum(is_equal))
    if smsg is not None:
        print(smsg)
        print('-------')


def run_test(F, N=10, tname='Test', doraise=False):
    ''' Helper function for running callable F() N times.
    '''
    print('----')
    f = 0
    for i in range(N):
        if doraise:
            F()
        else:
            try:
                F()
            except AssertionError as e:
                print(e)
                f += 1
    print('----')
    print('%s done. Completed %i and %i failures.' % (tname, N - f, f))
