#!/usr/bin/env python

import sys

import sh
import numpy as np
from scipy.sparse import csr_matrix
from scipy.linalg import toeplitz


def export_full_r(filename, x, dtype=np.double):
    with open(filename, 'w') as fid:
        m, n = x.shape

        z = np.array([np.dtype(dtype).itemsize], dtype='int32')
        z.tofile(fid)

        z = np.array([m, n], dtype='int32')
        z.tofile(fid)

        if (x.dtype != dtype):
            x_dtype = np.array(x, dtype=dtype)
            x_dtype.tofile(fid)
        else:
            x.tofile(fid)
    return filename


def map_bytes_to_dtype(b, int_type=False):
    # Should use a dict instead
    if (b == 4) and (not int_type):
        return "float32"
    elif (b == 8) and (not int_type):
        return "float64"
    elif (b == 4) and int_type:
        return "int32"
    elif (b == 8) and int_type:
        return "int64"
    else:
        raise ValueError("bytes must be 4 or 8 but {0} passed to function".format(b))


def import_rcs(filename):
    with open(filename, 'r') as fid:
        b = int(np.fromfile(fid, dtype=np.int32, count=1))
        m = int(np.fromfile(fid, dtype=np.int32, count=1))
        n = int(np.fromfile(fid, dtype=np.int32, count=1))
        N = int(np.fromfile(fid, dtype=np.int32, count=1))

        v = np.fromfile(fid, dtype=map_bytes_to_dtype(b), count=N)
        j = np.fromfile(fid, dtype=np.int32, count=N)
        r = np.fromfile(fid, dtype=np.int32, count=m+1)

    return csr_matrix((v, j, r), shape=(m, n))


def export_rcs(filename, A, dtype=np.double):
    with open(filename, 'w') as fid:
        A = csr_matrix(A, dtype=dtype)

        m, n = A.shape
        N = A.nnz

        v = A.data
        j = A.indices
        r = A.indptr

        z = np.array([np.dtype(dtype).itemsize, m, n, N], dtype='int32')
        z.tofile(fid)

        assert(j.dtype == 'int32')
        assert(r.dtype == 'int32')

        v.tofile(fid)
        j.tofile(fid)
        r.tofile(fid)
    return filename


def export_sb_toe_r(filename, blocks, dtype=np.double):
    rank = len(blocks.shape)

    # only support 1-D filter for now --- more work for general case
    assert rank == 1

    n = blocks.shape
    n_phy = list(map(len, blocks.nonzero()))

    with open(filename, 'w') as fid:
        z = np.array([np.dtype(dtype).itemsize], dtype='int32')
        z.tofile(fid)

        z = np.array([rank], dtype='int32')
        z.tofile(fid)

        z = np.array(n_phy, dtype='int32')
        z.tofile(fid)

        z = np.array(n, dtype='int32')
        z.tofile(fid)

        z = np.array(blocks[blocks.nonzero()], dtype=dtype)
        z.tofile(fid)

    return filename


if __name__ == '__main__':
    N = 100
    L = 10
    M = 20

    #res = N

    P_HT_fname = '/tmp/P_HT'
    e_fname = '/tmp/e'
    H_fname = '/tmp/H'
    C_fname = '/tmp/C'

    e = np.random.randn(N, L)
    export_full_r(e_fname, e)

    H = np.random.randn(M, N)
    export_rcs(H_fname, H)

    #row0 = np.ones((N, ))
    row0 = np.hstack((np.arange(50, 0, -1), np.zeros((50,))))
    blocks = row0
    export_sb_toe_r(C_fname, blocks)

    C = toeplitz(row0)

    # RUN CODE
    compute_P_HT = sh.Command('../compute_P_HT')
    print(compute_P_HT(P_HT_fname,
                       e_fname,
                       H_fname,
                       C_fname,
                       _err_to_out=True,
                       _out=sys.stdout))

    P_HT_c = import_rcs(P_HT_fname)

    P_HT = np.multiply(C, e @ e.T) @ H.T / (L - 1)

    print('P_HT_c == P_HT? {}'.format(np.allclose(P_HT_c.toarray(),
                                                  P_HT)))
