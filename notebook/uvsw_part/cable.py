#!/usr/bin/env python
# -*- coding: utf-8 -*-
import matplotlib.pyplot as mpl
import numpy as np
import scipy as sp
from scipy import linalg
from scipy.sparse import linalg

from uvsw_part import fdm_utils as fdmu
from uvsw_part import simtools


class cable(object):
    ''' a cable object
    '''
    g = 9.81  # gravity (m/s^2)

    def __init__(self, mass=None, diameter=None, EA=None, length=None, tension=None, h=0.):
        '''
        '''
        self.m = mass      # mass per length unit (kg/m)
        self.d = diameter  # conductor diameter (m)
        self.EA = EA       # axial stiffeness (N)
        self.Lp = length   # span length (m)
        self.H = tension   # tension (N)
        self.h = h         # pole altitude difference (2nd minus 1st, m)
        #
        self.a = None   # mechanical parameter (m)
        self.L = None   # conductor length (m)
        self.lm = None  # Irvine number (no unit)
        self.q = None   # usefull value (no unit)
        #
        self.a = self.H / (self.m * self.g)
        self.L = np.sqrt(self.h**2 + (2. * self.a * np.sinh(0.5 * self.Lp / self.a))**2)
        self.lm = np.sqrt((self.Lp / self.a)**2 * (self.Lp * self.EA / (self.L * self.H)))
        self.q = np.log((self.L + self.h) / (self.L - self.h))
        #
        return

    def vsFreq(self, n=1):
        ''' Give the n-th mode of vibrating string (in Hz, default is first)
        '''
        return 0.5 * n / self.L * np.sqrt(self.H / self.m)

    def freq1(self):
        ''' an approximation of a beam natural frequency (in Hz)
        '''
        return self.vsFreq(n=1)

    def tFreq(self, n):
        ''' conductor frequencies / transcendental equation; returns an array
            with n firsts nonzeros solutions (in Hz)
        '''
        epsilon = 1.0E-09
        maxiter = 64
        omega = np.zeros((n,))
        err = np.zeros((n,))
        for k in range(n):
            xm = (2 * k + 1) * np.pi
            xM = (2 * k + 3) * np.pi
            count = 1
            while (xM - xm) > epsilon and count <= maxiter:
                x = 0.5 * (xm + xM)
                y = np.tan(0.5 * x) - 0.5 * x + 0.5 * x**3 / self.lm**2
                if y > 0:
                    xM = x
                elif y < 0:
                    xm = x
                count = count + 1
            omega[k] = 0.5 * (xm + xM)
            err[k] = xM - xm
        return self.vsFreq(n=1) * omega / np.pi

    def altitude_1s(self, s):
        ''' get cable altitude difference regarding 1st pole given curvilinear
            abscissa along span (~ catenary  equation); return value in meters
        '''
        xm = (self.a * self.q - self.Lp)
        return 2. * self.a * np.sinh(0.5 * (self.Lp * s + xm) / self.a
                                     ) * np.sinh(0.5 * self.Lp * s / self.a)

    def altitude_2s(self, s):
        ''' same as altitude_1s but regarding 2nd pole
        '''
        return self.altitude_1s(s) - self.h

    def altitude_1c(self, s):
        ''' same as altitude_1s but with curvilinear abcissa along cable
        '''
        xm = 0.5 * (self.a * self.q - self.Lp)
        return self.a * (np.sqrt(1. + (self.L / self.a * s + np.sinh(xm / self.a))**2
                                 ) - np.sqrt(1. + np.sinh(xm / self.a)**2))

    def altitude_2c(self, s):
        ''' Same as altitude_1s but regarding 2nd pole
        '''
        return self.altitude_1c(s) - self.h

    def cable2span(self, s):
        ''' convert curvilinear abcissa along cable to curvilinear abcissa a
            long span
        '''
        xm = 0.5 * (self.a * self.q - self.Lp)
        return (self.a * np.arcsinh(self.L / self.a * s + np.sinh(xm / self.a))
                - xm) / self.Lp

    def span2cable(self, s):
        ''' convert curvilinear abcissa along span to curvilinear abcissa along
            cable
        '''
        return (2. * self.a / self.L * np.sinh(0.5 * self.Lp / self.a * s) *
                np.cosh(0.5 * self.Lp / self.a * (s - 1.) + 0.5 * self.q))


def solve(cb, pm, shape=None, speed=None, force=None, remove_cat=False, rel_spd=False):
    ''' finite difference solver for cable
    args:
      - cb    : a cable object
      - pm    : a parameters object (from simtools.py)
      - shape : initial position
      - speed : initial speed
      - force : ext force
      - remove_cat : bool to indicate wether catenary should be removed from
        shape
      - rel_spd : bool to indicate wether relative speed should be considerated
    '''
    ns = pm.ns
    s = np.linspace(0., 1., ns)
    ds = np.diff(s)
    N = len(s)
    n = N - 2

    vt2 = cb.H / (cb.m * cb.g * cb.L)
    vl2 = cb.EA / (cb.m * cb.g * cb.L)

    C = fdmu.d1M(ds)
    A = fdmu.d2M(ds)
    J = np.ones((n,))

    # time
    tAd = np.sqrt(cb.L / cb.g)
    uAd = cb.L / tAd
    t = 0.
    tf = pm.tf / tAd
    dt = tf / pm.nt
    ht = 0.5 * dt
    ht2 = ht**2

    # compute ut, ef from un, ub
    def utef(un, ub):
        h = -1. / vt2 * un + 0.5 * ((C * un)**2 + (C * ub)**2)
        H = 0.5 * (h[:-1] + h[1:]) * ds
        ut = np.sum(H) * s - np.cumsum(np.concatenate(([0.], H)))
        e = (C * ut) + 0.5 * ((C * ut)**2 + (C * un)**2 + (C * ub)**2)
        ef = np.log(np.sqrt(1.0 + 2.0 * e))
        return ut, ef

    # init
    if shape is None:
        un = np.zeros((N,))
        ub = np.zeros((N,))
    else:
        if shape[0] is None:
            un = np.zeros((N,))
        else:
            un = np.interp(s, np.linspace(0., 1., len(shape[0])), shape[0])
        if shape[1] is None:
            ub = np.zeros((N,))
        else:
            ub = np.interp(s, np.linspace(0., 1., len(shape[1])), shape[1])
        if remove_cat:
            un -= cb.altitude_1s(s)
        un /= cb.L
        ub /= cb.L

    ut, ef = utef(un, ub)

    if speed is None:
        vn = np.zeros((N,))
        vb = np.zeros((N,))
    else:
        if speed[0] is None:
            vn = np.zeros((N,))
        else:
            vn = np.interp(s, np.linspace(0., 1., len(speed[0])), speed[0])
        if speed[1] is None:
            vb = np.zeros((N,))
        else:
            vb = np.interp(s, np.linspace(0., 1., len(speed[1])), speed[1])
        vn /= uAd
        vb /= uAd

    lov = ['ut', 'un', 'ub', 'ef']
    res = simtools.results(n=pm.nr+1, lov=lov, los=pm.si)
    res.update(0, t, s, lov, [ut, un, ub, ef])

    if force is None:
        force = fdmu.ZeroForce()

    # loop
    res.tstart()
    if pm.pp:
        print('-- running %5.2f%%' % (0.), end='')
    for k in range(pm.nt):

        h = -1. / vt2 * un + 0.5 * ((C * un)**2 + (C * ub)**2)
        e = 0.5 * np.sum((h[:-1] + h[1:]) * ds)
        B = A.multiply(vt2 + vl2 * e)

        if rel_spd:
            fn1, fb1 = force(s, t * tAd, uc=(vn * uAd, vb * uAd))
            fn2, fb2 = force(s, (t + dt) * tAd, uc=(vn * uAd, vb * uAd))
        else:
            fn1, fb1 = force(s, t * tAd)
            fn2, fb2 = force(s, (t + dt) * tAd)

        fn1 /= (cb.m * cable.g)
        fb1 /= (cb.m * cable.g)
        fn2 /= (cb.m * cable.g)
        fb2 /= (cb.m * cable.g)

        phi_n = B * un[1:-1] + fn1[1:-1] + (vl2 / vt2 * e) * J
        phi_b = B * ub[1:-1] + fb1[1:-1]

        Run = un[1:-1] + ht * vn[1:-1]
        Rub = ub[1:-1] + ht * vb[1:-1]
        Rvn = vn[1:-1] + ht * (phi_n + fn2[1:-1] + (vl2 / vt2 * e) * J)
        Rvb = vb[1:-1] + ht * (phi_b + fb2[1:-1])

        D = sp.sparse.eye(n) - ht2 * (vt2 + vl2 * e) * A
        Db = np.zeros((3, n))
        Db[0, +1:] = D.diagonal(k=1)
        Db[1, :] = D.diagonal(k=0)
        Db[2, :-1] = D.diagonal(k=-1)

        Rhs = np.column_stack((Run + ht * Rvn,
                               Rub + ht * Rvb))
        unb = sp.linalg.solve_banded((1, 1), Db, Rhs)

        un = np.concatenate(([0.], unb[:, 0], [0.]))
        ub = np.concatenate(([0.], unb[:, 1], [0.]))
        vn = np.concatenate(([0.], 2. / dt * (unb[:, 0] - Run), [0.]))
        vb = np.concatenate(([0.], 2. / dt * (unb[:, 1] - Rub), [0.]))
        t += dt

        ut, ef = utef(un, ub)

        if (k + 1) % pm.rr == 0:
            res.update((k // pm.rr) + 1, t, s, lov, [ut, un, ub, ef])
            if pm.pp:
                print('\r -- running %5.2f%%' % (100. * (1. + k) / pm.nt), end='')

    # END FOR
    if pm.pp:
        print('\r', end='')
    res.tstop()

    res.t *= tAd
    res.dat[res.idx[lov[0]]] *= cb.L
    res.dat[res.idx[lov[1]]] *= cb.L
    res.dat[res.idx[lov[2]]] *= cb.L
    res.dat[res.idx[lov[3]]] *= cb.EA

    return res


def solve_wo(cb, pm, wo, shape=None, speed=None, remove_cat=False, fast=True):
    ''' ** BETA **
    '''
    M = wo.rho * cb.d**2 * wo.cl0 / (16. * np.pi**2 * wo.st**2 * cb.m)
    w = 2. * np.pi * wo.u * wo.st / cb.d

    al = wo.al / w**2
    bt = wo.bt / w**2
    gm = wo.gm / w**2

    ns = pm.ns
    s = np.linspace(0., 1., ns)
    ds = np.diff(s)
    N = len(s)
    n = N - 2

    C = fdmu.d1M(ds)
    A = fdmu.d2M(ds)
    I = sp.sparse.eye(n)
    J = np.ones((n,))

    # time
    tAd = 1. / w
    uAd = cb.d / tAd
    t = 0.
    tf = pm.tf / tAd
    dt = tf / pm.nt
    ht = 0.5 * dt
    ht2 = ht**2

    # compute ut, ef from un, ub
    def utef(un, ub):
        h = -1. * cb.m * cb.g * cb.d / cb.H * un + 0.5 * (cb.d / cb.L)**2 * ((C * un)**2 + (C * ub)**2)
        H = 0.5 * (h[:-1] + h[1:]) * ds
        ut = np.sum(H) * s - np.cumsum(np.concatenate(([0.], H)))
        e = (C * ut) + 0.5 * ((C * ut)**2 + (C * un)**2 + (C * ub)**2)
        ef = np.log(np.sqrt(1.0 + 2.0 * e))
        return ut, ef

    # init
    if shape is None:
        un = np.zeros((N,))
        ub = np.zeros((N,))
        q = np.zeros((N,))
    else:
        if shape[0] is None:
            un = np.zeros((N,))
        else:
            un = np.interp(s, np.linspace(0., 1., len(shape[0])), shape[0])
        if shape[1] is None:
            ub = np.zeros((N,))
        else:
            ub = np.interp(s, np.linspace(0., 1., len(shape[1])), shape[1])
        if remove_cat:
            un -= cb.altitude_1s(s)
        un /= cb.d
        ub /= cb.d
        if shape[2] is None:
            q = np.zeros((N,))
        else:
            q = np.interp(s, np.linspace(0., 1., len(shape[2])), shape[2])

    ut, ef = utef(un, ub)

    if speed is None:
        vn = np.zeros((N,))
        vb = np.zeros((N,))
        r = np.zeros((N,))
    else:
        if speed[0] is None:
            vn = np.zeros((N,))
        else:
            vn = np.interp(s, np.linspace(0., 1., len(speed[0])), speed[0])
        if speed[1] is None:
            vb = np.zeros((N,))
        else:
            vb = np.interp(s, np.linspace(0., 1., len(speed[1])), speed[1])
        vn /= uAd
        vb /= uAd
        if speed[2] is None:
            r = np.zeros((N,))
        else:
            r = np.interp(s, np.linspace(0., 1., len(speed[2])), speed[2])

    lov = ['un', 'q']
    res = simtools.results(n=pm.nr+1, lov=lov, los=pm.si)

    res.update(0, t, s, lov, [un, q])

    if fast:
        Is1 = I.multiply(1. + ht2)
        Is2 = I.multiply(1. / M)
        Is3 = I.multiply(ht * bt)
    else:
        A25 = I.multiply(M)
        A61 = I.multiply(wo.gm / w**2)
        A62 = I.multiply(wo.bt / w**2)

    # loop
    res.tstart()
    if pm.pp:
        print('-- running %5.2f%%' % (0.), end='')
    for k in range(pm.nt):

        h = -1. * cb.m * cb.g * cb.d / cb.H * un + 0.5 * (cb.d / cb.L)**2 * ((C * un)**2 + (C * ub)**2)
        e = 0.5 * np.sum((h[:-1] + h[1:]) * ds)

        if fast:
            Aa = A.multiply((cb.H + cb.EA * e) / (cb.m * cb.L**2 * w**2) * ht)
            Bb = (cb.EA * cb.g * e) / (cb.H * cb.d * w**2)
            A21 = Aa
            A41 = Aa.multiply(al) + I.multiply(ht * gm)
            A44 = sp.sparse.diags([(-1. * ht * wo.eps) * (q[1:-1]**2 - 1.)], [0])
            R1 = un[1:-1] + ht * vn[1:-1]
            R2 = vn[1:-1] + A21 * un[1:-1] + (ht * M) * q[1:-1] + (dt * Bb) * J
            R3 = q[1:-1] + ht * r[1:-1]
            R4 = r[1:-1] + A41 * un[1:-1] + (ht * bt) * vn[1:-1] - ht * q[1:-1] + A44 * r[1:-1] + (dt * Bb * al) * J
            if M == 0.:
                Xx = I - A21.multiply(ht)
                Xb = np.zeros((3, n))
                Xb[0, +1:] = Xx.diagonal(k=1)
                Xb[1, :] = Xx.diagonal(k=0)
                Xb[2, :-1] = Xx.diagonal(k=-1)
                Yy = Is1 - A44
                Yb = np.zeros((3, n))
                Yb[0, +1:] = Yy.diagonal(k=1)
                Yb[1, :] = Yy.diagonal(k=0)
                Yb[2, :-1] = Yy.diagonal(k=-1)
                vn[1:-1] = sp.linalg.solve_banded((1, 1), Xb, A21 * R1 + R2)
                un[1:-1] = ht * vn[1:-1] + R1
                r[1:-1] = sp.linalg.solve_banded((1, 1), Yb, R4 + A41 * un[1:-1] + (ht * bt) * vn[1:-1] - ht * R3)
                q[1:-1] = ht * r[1:-1] + R3
            else:
                Xx = ((I - A44).multiply(1. / (M * ht2)) + Is2) * (I - A21.multiply(ht)) - A41.multiply(ht) - Is3
                Rs = (A21 * R1 + R2) / M
                Rx = R4 + A41 * R1 + Rs + (I - A44) * (Rs / ht2 + R3 / ht)
                Xb = np.zeros((3, n))
                Xb[0, +1:] = Xx.diagonal(k=1)
                Xb[1, :] = Xx.diagonal(k=0)
                Xb[2, :-1] = Xx.diagonal(k=-1)
                vn[1:-1] = sp.linalg.solve_banded((1, 1), Xb, Rx)
                un[1:-1] = ht * vn[1:-1] + R1
                q[1:-1] = (vn[1:-1] - A21 * un[1:-1] - R2) / (M * ht)
                r[1:-1] = (q[1:-1] - R3) / ht
        else:
            A21 = (cb.H + cb.EA * e) / (cb.m * cb.L**2 * w**2) * A
            A43 = A21
            A66 = sp.sparse.diags([q[1:-1]**2 - 1.], [0]).multiply(-1. * wo.eps)
            AA = sp.sparse.bmat([[None, I, None, None, None, None],
                                 [A21, None, None, None, A25, None],
                                 [None, None, None, I, None, None],
                                 [None, None, A43, None, None, None],
                                 [None, None, None, None, None, I],
                                 [A61 + A21.multiply(wo.al / w**2), A62, None, None, A25.multiply(wo.al / w**2) - I, A66]])
            X = np.concatenate((un[1:-1], vn[1:-1], ub[1:-1], vb[1:-1], q[1:-1], r[1:-1]))
            rhs = cb.EA * cb.g / (cb.H * cb.d * w**2) * e * np.ones(n)
            R = np.hstack((np.zeros(n), rhs, np.zeros(3 * n), wo.al / w**2 * rhs))
            Ma = sp.sparse.eye(6 * n) - AA.multiply(ht)
            Mb = sp.sparse.eye(6 * n) + AA.multiply(ht)
            Xn = sp.sparse.linalg.spsolve(Ma, Mb * X + dt * R)
            un = np.concatenate(([0.], Xn[0 * n: 1 * n], [0.]))
            vn = np.concatenate(([0.], Xn[1 * n: 2 * n], [0.]))
            ub = np.concatenate(([0.], Xn[2 * n: 3 * n], [0.]))
            vb = np.concatenate(([0.], Xn[3 * n: 4 * n], [0.]))
            q = np.concatenate(([0.], Xn[4 * n: 5 * n], [0.]))
            r = np.concatenate(([0.], Xn[5 * n: 6 * n], [0.]))

        t += dt
        ut, ef = utef(un, ub)

        if (k + 1) % pm.rr == 0:
            res.update((k // pm.rr) + 1, t, s, lov, [un, q])
            if pm.pp:
                print('\r -- running %5.2f%%' % (100. * (1. + k) / pm.nt), end='')

    # END FOR
    if pm.pp:
        print('\r', end='')
    res.tstop()

    res.t *= tAd
    res.dat[res.idx[lov[0]]] *= cb.d

    return res


if __name__ == '__main__':

    # caternary check
    if False:
        cbl = cable(mass=1.57, diameter=0.031, EA=3.76E+07, length=400., tension=3.7E+04, h=10.)
        s = np.linspace(0., 1., 501)
        mpl.figure()
        mpl.plot(s, cbl.altitude_1s(s), label='label')
        mpl.title('Cable shape')
        mpl.xlabel('span curvilinear abcissa (normalized)')
        mpl.ylabel('altitude (m)')
        mpl.legend()
        mpl.grid(True)

    # dynamic test
    if False:
        cb = cable(mass=1.57, diameter=0.031, EA=3.76E+07, length=400., tension=3.7E+04, h=0.)
        pm = simtools.parameters(ns=501, tf=15., nt=15000, nr=3000, si=[0.1, 0.25, 0.5], pp=True)
        s = np.linspace(0, 1, 501)
        y = 0.05 * cb.altitude_1s(s)
        res = solve(cb, pm, shape=(y, None))

        simtools.multiplot([res], lb=['cable test'], Lref=400.)
