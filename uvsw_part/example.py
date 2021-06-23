#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import yaml
from uvsw_part import simulation
if __name__ == '__main__':
    cfg = yaml.safe_load(open('example.in.yaml', 'r'))
    dfy, _ = simulation.run_cable_wakeosc(cfg)
    sb.heatmap(dfy)
    plt.figure()
    plt.plot(dfy.index, dfy['s=0.250']/0.025)
    plt.xlabel('time (s)')
    plt.ylabel('y/d')
    plt.show()
