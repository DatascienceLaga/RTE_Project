#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import scipy as sp
import scipy.sparse


class bc(object):
    ''' boundary condition object
    '''

    def __init__(self, t1=None, t2=None, pos=None):
        ''' input tuples such that:
                a1 * y(0) + b1 * (dy/dx)(0) * c1 * (d2y/dx2)(0) = d1
                a2 * y(0) + b2 * (dy/dx)(0) * c2 * (d2y/dx2)(0) = d2
        '''
        if t1 is None:
            t1 = (1., 0., 0., 0.)
        if t2 is None:
            t2 = (0., 1., 0., 0.)
        if ((not (isinstance(t1, tuple) or isinstance(t1, list))) or
                (not (isinstance(t2, tuple) or isinstance(t2, list)))):
            raise TypeError('Inputs t1 and t2 must be list or tuples')
        if len(t1) != 4 or len(t2) != 4:
            raise ValueError('Inputs t1 and t2 must have 4 elements')
        if pos == 'left':
            self.pp = -1.
        elif pos == 'right':
            self.pp = +1.
        else:
            raise ValueError('Input pos must be either \'left\' or \'right\'')
        self.t1 = t1
        self.t2 = t2
        return

    def set(self, ds):
        ''' compute coefficients given a space discretization
        '''
        if not isinstance(ds, float):
            raise TypeError('Input ds must be a float')
        if ds <= 0.:
            raise ValueError('Input ds must be positive')
        a1, b1, c1, d1 = self.t1
        a2, b2, c2, d2 = self.t2
        A1 = a1 - self.pp * b1 / ds - 2. * c1 / ds**2
        A2 = a2 - self.pp * b2 / ds - 2. * c2 / ds**2
        B1 = self.pp * b1 / ds + c1 / ds**2
        B2 = self.pp * b2 / ds + c2 / ds**2
        C1 = -c1 / ds**2
        C2 = -c2 / ds**2
        det = np.linalg.det(np.array([[A1, B1], [A2, B2]]))
        if det == 0.:
            raise ValueError('Matrix is singular')
        self.c1c = (B2 * d1 - B1 * d2) / det
        self.c1q = (B2 * C1 - B1 * C2) / det
        self.c2c = (A1 * d2 - A2 * d1) / det
        self.c2q = (A1 * C2 - A2 * C1) / det

        return


def bc_rot_free(pos, y=0., d2y=0.):
    ''' boundary condition with free rotation and constrained value
    '''
    return bc(t1=(1., 0., 0., y), t2=(0., 0., 1., d2y), pos=pos)


def bc_rot_none(pos, y=0., dy=0.):
    ''' boundary condition with constrained value and derivative
    '''
    return bc(t1=(1., 0., 0., y), t2=(0., 1., 0., dy), pos=pos)


def d2M_cst(n, ds, bcl, bcr):
    ''' matrix and boundary condition vector for finite-difference space second
        derivative computation; n is matrix size, ds is grid step (constant),
        bcl and bcr are boundary condition objects (left and right); return
        values are a matrix 'A' and a vector 'a' such that
        d2y/dx**2 = A * y + a
    '''
    bcl.set(ds)
    bcr.set(ds)

    dinf = +1. / ds**2 * np.ones((n - 1,))
    diag = -2. / ds**2 * np.ones((n - 0,))
    dsup = +1. / ds**2 * np.ones((n - 1,))

    diag[0] += bcl.c1q / ds**2
    diag[-1] += bcr.c1q / ds**2

    A = sp.sparse.diags([dinf, diag, dsup], [-1, 0, 1])
    a = np.concatenate(([bcl.c1c], np.zeros((n - 2)), [bcr.c1c])) / ds**2

    return A, a


def d4M_cst(n, ds, bcl, bcr):
    ''' matrix and boundary condition vector for finite-difference space fourth
        derivative computation; n is matrix size, ds is grid step (constant),
        bcl and bcr are boundary condition objects (left and right); return
        values are a matrix 'D' and a vector 'd' such that
        d4y/dx**4 = A * y + a
    '''
    bcl.set(ds)
    bcr.set(ds)

    dm2 = +1. / ds**4 * np.ones((n - 2,))
    dm1 = -4. / ds**4 * np.ones((n - 1,))
    d0 = +6. / ds**4 * np.ones((n - 0,))
    dp1 = -4. / ds**4 * np.ones((n - 1,))
    dp2 = +1. / ds**4 * np.ones((n - 2,))

    d0[0] += (1. * bcl.c2q - 4. * bcl.c1q) / ds**4
    d0[-1] += (1. * bcl.c2q - 4. * bcl.c1q) / ds**4
    dm1[0] += (1. * bcl.c1q) / ds**4
    dp1[-1] += (1. * bcr.c1q) / ds**4

    D = sp.sparse.diags([dm2, dm1, d0, dp1, dp2], [-2, -1, 0, 1, 2])
    d = np.concatenate(([1. * bcl.c2c - 4. * bcl.c1c,
                         1. * bcl.c1c],
                        np.zeros((n - 4)),
                        [1. * bcr.c1c,
                         1. * bcr.c2c - 4. * bcr.c1c])) / ds**4
    return D, d


def d2M(ds):
    ''' Matrix for finite-difference space second derivative with homogenous
        Dirichlet boundary conditions; ds is a grid step vector
    '''
    h1 = ds[:-1]
    h2 = ds[1:]
    dinf = +2. / (h1 * (h1 + h2))
    dsup = +2. / (h2 * (h1 + h2))
    diag = -1. * (dinf + dsup)
    return sp.sparse.diags([dinf[1:], diag, dsup[:-1]], [-1, 0, 1])


def d1M(ds):
    ''' Matrix for finite-difference space first derivative with homogenous
        Dirichlet boundary conditions; ds is a grid step vector
    '''
    h1 = ds[:-1]
    h2 = ds[1:]
    dinf = -h2 / (h1 * (h1 + h2))
    dsup = +h1 / (h2 * (h1 + h2))
    diag = -1. * (dinf + dsup)
    return sp.sparse.diags([np.concatenate((dinf, [-1. / h2[-1]])),
                            np.concatenate(([-1. / h1[0]], diag, [+1. / h2[-1]])),
                            np.concatenate(([+1. / h1[0]], dsup))
                            ], [-1, 0, 1])


class ZeroForce(object):
    ''' Zero force for beam/cable solve
    '''

    def __init__(self):
        return

    def __call__(self, s, t):
        fl = np.zeros_like(s)
        fd = np.zeros_like(s)
        return fl, fd
