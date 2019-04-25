import numpy as np
import scipy as sp
import scipy.linalg
import matplotlib.pylab as plt

import libmdb_matrix
import pyrsss.signal.convmtx as convmtx
from pyrsss.signal.util import zero_pad


class ShapeFilter:
    def __init__(self, n, shape_function, m, normalized=False):
        self.n = n
        self.shape_function = shape_function
        self.m = m

        self.h = np.array([shape_function(scipy.linalg.norm(index)) for index in np.ndindex(*m)])
        if normalized:
            self.h /= np.sqrt(sum(self.h * self.h))
        self.h.shape = m


    def asmatrix(self):
        return convmtx.Convmtx(self.n, self.h)


class IsotropicFilter:
    def __init__(self, n, sqrt_shape_function, m, normalized=False):
        self.n = n
        self.sqrt_shape_function = sqrt_shape_function
        self.m = m

        self.h_sqrt_sf = ShapeFilter(n, sqrt_shape_function, m, normalized=normalized)
        self.h_sqrt = self.h_sqrt_sf.h

    def row0(self):
        s = tuple(slice(None, m_i, None) for m_i in self.m)
        row0 = zero_pad(np.real(np.fft.ifftn(abs(np.fft.fftn(self.h_sqrt,
                                                             s=2*np.array(self.m)-1))**2))[s],
                        self.n)
        row0.shape = (np.prod(self.n),)
        return row0

    def asmatrix(self):
        H_sqrt = self.sqrt_asmatrix()
        return H_sqrt * H_sqrt.T

    def sqrt_asmatrix(self):
        return self.h_sqrt_sf.asmatrix().T

    def h(self, include_zeros=True):
        row0 = self.row0()
        row0.resize(self.n)
        if include_zeros:
            return row0
        else:
            s = [slice(None, m_i, None) for m_i in self.m]
            return row0[s]

    @staticmethod
    def save_sqrt_static(fname, H, dtype='double'):
        H.save_sqrt(fname, dtype=dtype)

    def save_sqrt(self, fname, dtype='double'):
        # Suitable for the sb_toe_r import routines in libmdb_matrix
        with open(fname, 'wb') as fid:
            z = np.array([np.dtype(dtype).itemsize], dtype='int32')
            z.tofile(fid)

            n_dim = np.array([len(self.n)], dtype='int32')
            n_dim.tofile(fid)

            n_phy = np.asarray(self.m, dtype='int32')
            n_phy.tofile(fid)

            n = np.asarray(self.n, dtype='int32')
            n.tofile(fid)

            self.h_sqrt.tofile(fid)


    @staticmethod
    def load_sqrt(fname):
        with open(fname, 'r') as fid:
            itemsize = np.fromfile(fid, dtype='int32', count=1)
            dtype = libmdb_matrix.map_bytes_to_dtype(itemsize)

            n_dim = np.fromfile(fid, dtype='int32', count=1)[0]
            n_phy = np.fromfile(fid, dtype='int32', count=n_dim)
            n = np.fromfile(fid, dtype='int32', count=n_dim)
            h_sqrt = np.fromfile(fid, dtype=dtype, count=np.prod(n_phy))
            h_sqrt.shape = n_phy

            h_sqrt_sf = ShapeFilter.__new__(ShapeFilter)
            h_sqrt_sf.shape_function = None
            h_sqrt_sf.n = n
            h_sqrt_sf.m = n_phy
            h_sqrt_sf.h = h_sqrt

            H = IsotropicFilter.__new__(IsotropicFilter)
            H.n = n
            H.sqrt_shape_function = None
            H.m = n_phy
            H.h_sqrt_sf = h_sqrt_sf
            H.h_sqrt = h_sqrt

            return H


if __name__ == '__main__':
    sqrt_fname = '/tmp/H_sqrt'

    n = (4, 5)

    m = (3, 2)
    norm_m = sp.linalg.norm(m)

    H = IsotropicFilter(n, lambda r: 1 - float(r)/(norm_m), m)
    H.save_sqrt(sqrt_fname)

    H2 = IsotropicFilter.load_sqrt(sqrt_fname)

    print((H.asmatrix() - H2.asmatrix()).nnz)

    plt.matshow(H.h_sqrt)
    plt.colorbar()

    s = tuple(slice(None, m_i, None) for m_i in H.m)

    plt.matshow(H.h()[s])
    plt.colorbar()

    plt.show()
