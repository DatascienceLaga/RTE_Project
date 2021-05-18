#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from uvsw_part import cable
from uvsw_part import simtools
from uvsw_part import wind


def run_cable_wakeosc(cf):

    rk = ['cable', 'conductor', 'simulation', 'wakeosc']
    for k in rk:
        print(k)
        if k not in cf.keys():
            raise RuntimeError('missing key in input config')

    # cable/conductor

    ca = cf['cable']
    if not isinstance(ca['length'], float) or ca['length'] < 0.:
        raise RuntimeError('input cable.length must be a positive float')
    if not isinstance(ca['tension'], float) or ca['tension'] < 0.:
        raise RuntimeError('input cable.tension must be a positive float')
    if not isinstance(ca['h'], float):
        raise RuntimeError('input cable.h must be float')

    co = cf['conductor']
    ks = ['m', 'd', 'EA']
    for k in ks:
        if not isinstance(co[k], float) or co[k] < 0.:
            raise RuntimeError('input conductor.%s must be a positive float' % (k,))

    cb = cable.cable(mass=co['m'], diameter=co['d'], EA=co['EA'],
                     length=ca['length'], tension=ca['tension'], h=ca['h'])

    # simulation

    cc = cf['simulation']
    ns = cc['ns']
    tf = cc['tf']
    dt = cc['dt']
    dr = cc['dr']
    si = cc['si']
    pp = cc['pp']

    if not isinstance(ns, int) or ns < 11:
        raise RuntimeError('input simulation.ns must be a int larger than 10')
    if not isinstance(tf, float) or tf < 0.:
        raise RuntimeError('input simulation.tf must be a positive float')
    if not isinstance(dt, float) or dt < 0. or dt >= 0.5 * tf:
        raise RuntimeError('input simulation.dt must be a positive float and '
                           'much smaller than simulation.tf')
    if not isinstance(dr, float) or dr < 0. or dr < dt:
        raise RuntimeError('input simulation.dr must be a positive float and '
                           'larger than (or equal to) dt')
    if isinstance(si, int):
        if si < 1:
            si = [0.5]
        else:
            si = np.linspace(0., 1., si + 2)[1:-1].tolist()
    if isinstance(si, list):
        if len(si) < 1:
            raise RuntimeError('input simulation.si must be a list with at '
                               'least one element')
        for i in si:
            if not isinstance(i, float) or i <= 0. or i >= 1.:
                raise RuntimeError('elements of simulation.si must be floats and '
                                   'range in ]0, 1[')
    if not isinstance(pp, bool):
        raise RuntimeError('input simulation.pp must be a bool')

    si = sorted(si)
    nt = int(round(tf / dt))
    nr = int(round(tf / dr))
    pm = simtools.parameters(ns=ns, tf=tf, nt=nt, nr=nr, si=si, pp=pp)

    # wake oscillator

    cw = cf['wakeosc']
    ks = ['u', 'st', 'cl0', 'eps']
    for k in ks:
        if not isinstance(cw[k], float) or cw[k] < 0.:
            raise RuntimeError('input wakeos.%s must be a positive float' % (k,))
    ks = ['al', 'bt', 'gm', 'y0', 'q0']
    for k in ks:
        if not isinstance(cw[k], float):
            raise RuntimeError('input wakeos.%s must be a float' % (k,))
    if not isinstance(cw['md'], int) or cw['md'] < 1:
        raise RuntimeError('input simulation.ns must be a positive int')

    md = cw.pop('md')
    y0 = cw.pop('y0')
    q0 = cw.pop('q0')
    wo = wind.WO(**cw)

    # run
    s = np.linspace(0, 1, pm.ns)
    y = y0 * np.sin(md * np.pi * s)
    q = q0 * np.sin(md * np.pi * s)
    r = cable.solve_wo(cb, pm, wo, shape=(y, None, q), speed=None, fast=True)

    cl = ['s=%.3f' % (s,) for s in pm.si]
    dy = pd.DataFrame(r.dat[r.idx['un']], columns=cl, index=r.t)
    dq = pd.DataFrame(r.dat[r.idx['q']], columns=cl, index=r.t)

    return dy, dq
