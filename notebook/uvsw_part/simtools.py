#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time

import matplotlib.pyplot as mpl
import numpy as np
from matplotlib import cm


# figure font sizes
TTS = 12
LBS = 10
TKS = 10


class parameters(object):
    ''' class to handle simulation parameters
    '''

    def __init__(self, ns=None, tf=None, nt=None, nr=None, si=[], pp=False):
        self.ns = ns  # number of elements
        self.tf = tf  # final time (s)
        self.nt = nt  # number of time steps
        self.nr = nr  # number of (time) outputs
        self.si = si  # curvilinear abc. of interest (list)
        self.pp = pp  # print progress (or not)
        #
        self.nt = int(self.nt)
        if self.nr is None or self.nr > self.nt or self.nr < 1:
            self.nr = self.nt
            self.rr = 1  # output rate (computed from nr)
        else:
            self.nr = int(self.nr)
            self.rr = self.nt // self.nr
            self.nr = self.nt // self.rr
        return


class results(object):
    ''' class to handle simulation results
    '''

    def __init__(self, n=0, lov=[], los=[]):
        '''
        '''
        self.time = None  # simulation compute time
        self.vars = lov  # list of variables to store
        self.s = los  # positions of interest (adim)
        self.t = np.zeros((n,))  # phys. time (s)
        self.dat = [np.zeros((n, len(los))) for v in lov]  # data
        self.idx = {}
        for i in range(len(self.vars)):
            self.idx[self.vars[i]] = i
        return

    def tstart(self):
        ''' start timer
        '''
        self.time = time.time()
        return

    def tstop(self):
        ''' end timer and compute simulation time
        '''
        self.time = time.time() - self.time
        return

    def update(self, k, t, s, lov, lod):
        ''' get a snapshot
        '''
        self.t[k] = t
        for i in range(len(lov)):
            j = self.idx[lov[i]]
            self.dat[j][k, :] = np.interp(self.s, s, lod[i])
        return

    def export_snapshots(self, filename):
        ''' export content in text format
        '''
        hdr = 'vars=[t, '
        for v in self.vars[:-1]:
            hdr += v + '@s, '
        hdr += self.vars[-1] + ']'
        hdr += '@s, s=['
        for v in self.s[:-1]:
            hdr += str(v) + ', '
        hdr += str(self.s[-1]) + ']'
        dat = np.column_stack([self.t] + self.dat)
        np.savetxt(filename, dat, delimiter='; ', fmt='%+13.6E',
                   newline='\n', header=hdr, comments='# ')
        return

    def import_snapshots(self, filename, lov=[], los=[], t0=0., tf=np.inf):
        ''' import a snapshot file
        '''
        tmp = np.loadtxt(filename, comments='#', delimiter=';')
        self.time = np.nan
        self.vars = lov
        self.s = los
        tx = np.where((tmp[:, 0] >= t0) * (tmp[:, 0] <= tf))[0]
        self.t = tmp[tx, 0]
        self.dat = []
        n = len(self.s)
        for i in range(len(self.vars)):
            self.idx[self.vars[i]] = i
            self.dat.append(tmp[tx, (1 + i * n):(1 + (i + 1) * n)])
        return

    def keep_only(self, lov=None, los=None, t0=0., tf=np.inf):
        ''' use this to keep only some variables or positions of interest from
            a results object
        '''
        if lov is None:
            lov = self.vars
        if los is None:
            los = self.s
        #
        tx = np.where((self.t >= t0) * (self.t <= tf))[0]
        self.t = self.t[tx]
        #
        dat = []
        idx = {}
        k = 0
        for i in range(len(self.vars)):
            if self.vars[i] in lov:
                j = self.idx[self.vars[i]]
                dat.append(self.dat[j][tx, :])
                idx[self.vars[i]] = k
                k += 1
        self.vars = lov
        self.dat = dat
        self.idx = idx
        #
        il = []
        for i in range(len(los)):
            q = np.argwhere(np.array(self.s) == los[i])
            if len(q) > 0:
                il.append(q[0][0])
        if len(il) == 0:
            il = [i for i in self.s]
        for i in range(len(self.vars)):
            dat[i] = dat[i][:, np.array(il)]
        self.s = np.array(self.s)[il].tolist()
        #
        return


def multiplot(res, lb=None, Lref=0., stl='-', log=False, t0=0., tf=np.inf, fst=TTS, fsl=LBS):
    ''' plot on a same figure a list of results objects (res); these results
        objects should have the same list of variables and positions of
        interest; lb is a list of labels (optional); Lref is the refderence
        length (m); stl is default line style; use log=True for logarithmic
        axes; use t0 and tf to restrain time axis
    '''
    nr = len(res[0].vars)
    nc = len(res[0].s)
    # put default label if empty
    if lb is None:
        lb = [str(i) for i in range(len(res))]
    # colormap
    if len(res) == 1:
        cmap = ['royalblue']
    else:
        cmap = cm.viridis(np.linspace(0., 1., len(res) + 2))[1:-1]
    #
    fig, ax = mpl.subplots(nrows=nr, ncols=nc)  # figsize=(18, 9),
    if nc == 1 and nr == 1:
        ax = np.array([[ax]])
    elif nc == 1:
        ax = np.array([[ax[i]] for i in range(len(ax))])
    elif nr == 1:
        ax = np.array([ax])
    for k in range(len(res)):
        for i in range(nr):
            for j in range(nc):
                if not log:
                    tx = np.where((res[k].t >= t0) * (res[k].t <= tf))[0]
                    ax[i, j].plot(res[k].t[tx], res[k].dat[i][tx, j], stl, c=cmap[k], label=lb[k])
                else:
                    ax[i, j].loglog(res[k].t, res[k].dat[i][:, j], stl, c=cmap[k], label=lb[k])
    #
    for i in range(nr):
        for j in range(nc):
            ax[i, j].grid(True)
    for i in range(nr):
        ax[i, 0].set_ylabel(res[0].vars[i], fontsize=fsl)
    for j in range(nc):
        ax[0, j].set_title('@ x=%.1E m (%.1f %%) of the span'
                           % (res[0].s[j] * Lref, res[0].s[j] * 100.), fontsize=fst)
        if not log:
            ax[-1, j].set_xlabel('Time (s)', fontsize=fsl)
        else:
            ax[-1, j].set_xlabel('Freq (Hz)', fontsize=fsl)
    ax[-1, -1].legend()
    #
    return fig, ax


def spectrum(res):
    ''' from a results object compute related spectra using fft; return value
        is also a results object
    '''
    T = res.t[1] - res.t[0]
    n = len(res.t)
    N = n // 2
    f = np.fft.fftfreq(n, d=T)
    f = f[:N]
    spc = results(n=N, lov=res.vars, los=res.s)
    spc.t = f
    for v in res.vars:
        for k in range(len(res.s)):
            spc.dat[spc.idx[v]][:, k] = abs(np.fft.fft(res.dat[res.idx[v]][:, k]) / n)[:N]
    return spc
