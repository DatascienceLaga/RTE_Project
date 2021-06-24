#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np


def air_volumic_mass(T=293.15, p=1.013E+05, phi=0.):
    ''' compute air volumic mass (kg/m^3) in function of temperature (T,
        in K), pressure (p, in Pa) and relative humidity (phi, no units)
    '''
    return 1. / (287.06 * T) * (
        p - 230.617 * phi * np.exp(17.5043 * (T - 273.15) / (T - 31.95)))


def air_density(T=293.15, p=1.013E+05, phi=0.):
    ''' compute air density (no units) in function of temperature (T,
        in K), pressure (p, in Pa) and relative humidity (phi, no units)
    '''
    return 1.0E-03 * air_volumic_mass(T, p, phi)


class BH(object):
    ''' Bishop & Hassan model for wind force over a circular cylinder (constant
        wind speed, no relative velocity)
    '''

    def __init__(self, u=None, St=None, cd=None, cd0=None, cl=None, cl0=None, d=None):
        ''' u  : wind speed (m/s)
            St : strouhal number
            cd, cd0, cl, cl0 : Bishop and Hassan coefficients
            d  : cylinder diameter (m)
        '''
        self.u = u
        self.St = St
        self.cd = cd
        self.cd0 = cd0
        self.cl = cl
        self.cl0 = cl0
        self.wfc = 0.5 * air_volumic_mass() * d * self.u * np.abs(self.u)
        fl = self.St / d * u
        fd = 2. * fl
        self.omd = 2. * np.pi * fd
        self.oml = 2. * np.pi * fl
        return

    def __call__(self, s, t, uc=None):
        ''' force computation; uc is a correction terms in case we want to take
            a relative speed into account (int that case it is a tuple with two
            elements for normal and binormal direction; return value in Newton
            per meter
        '''
        if uc is not None:
            return self.force_rel(s, t, uc[0], uc[1])
        else:
            wfl = self.wfc * (self.cl + self.cl0 * np.sin(self.oml * t)) * np.ones_like(s)
            wfd = self.wfc * (self.cd + self.cd0 * np.sin(self.omd * t)) * np.ones_like(s)
            return wfl, wfd

    def force_rel(self, s, t, vn, vb):
        ''' force with relative speed taken into account
        '''
        al = self.wfc / (self.u * np.abs(self.u))
        sq = np.sqrt((0. - vn)**2 + (self.u - vb)**2)
        cl = (self.cl + self.cl0 * np.sin(self.oml * t))
        cd = (self.cd + self.cd0 * np.sin(self.omd * t))
        fn = al * sq * ((0. - vn) * cd + (self.u - vb) * cl)
        fb = al * sq * ((self.u - vb) * cd - (0. - vn) * cl)
        return fn, fb


class BHg(object):
    ''' Bishop & Hassan model with gravity
    '''

    def __init__(self, bh, m, g=9.81):
        ''' bh : a BH object
            m : mass per length unit
            g : gravity constant
        '''
        self.bh = bh
        self.m = m
        self.g = g
        return

    def __call__(self, s, t, uc=None):
        if uc is not None:
            raise NotImplementedError()
        else:
            wfl, wfd = self.bh(s, t, uc=None)
            return wfl - self.g * self.m, wfd


class WO(object):
    ''' [Work in progress] Parameters for wake oscillator
    '''

    def __init__(self, u=None, st=None, cl0=None, eps=None, al=None, bt=None, gm=None):
        self.u = u
        self.st = st
        self.cl0 = cl0
        self.eps = eps
        self.al = al
        self.bt = bt
        self.gm = gm
        self.rho = air_volumic_mass()
        return
