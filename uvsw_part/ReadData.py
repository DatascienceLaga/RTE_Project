#!/usr/bin/env python3
# -*- coding: utf-8 -*-

### Read data from strip theory reference dataset
### Folder must contain List*.txt and Data*.*.bin files

from array import array
import pandas as pd
import matplotlib.pyplot as mpl
import os.path as path # to check either .csv file exists or not on disk

Ncfd = 2 # No. of CFD strips; same as the number of modes = 1 or 2
plot_data = True # plot data

data = {'nc':[],'md':[],'U':[],'d':[],'m':[],'L':[],'H':[],'Nt':[],'Dt':[],'tf':[],'ymax':[],'time':[],'y/d':[]} # dict object to create dataframe

# open List*.txt file
ftxt = open("List%d.txt" % Ncfd, "r")
line = ftxt.readline()
print(line[:-1])
for n in range(1000):
    line = ftxt.readline()
    if len(line) == 0:
        break
    print(line[:-1])
    tmp = line.split()
    nc = int(tmp[0]) # case number in List file
    md = int(tmp[1]) # mode number
    U = float(tmp[2]) # wind/air velocity [m/s]
    d = float(tmp[3]) # cable diameter [m]
    m = float(tmp[4]) # cable mass per unit length [kg/m]
    L = float(tmp[5]) # cable length [m]
    H = float(tmp[6]) # cable tension [N]
    Nt = int(tmp[7]) # number of timesteps
    Dt = float(tmp[8]) # timestep length [s]
    tf = float(tmp[9]) # total time [s]
    ymax = float(tmp[10]) # max(y) value [m]
    filename = tmp[11] # = "Data%d.%d.bin" % (Ncfd, nc) # data file name

    # open Data*.*.bin file
    fdat=open(filename,"rb")

    float_array = array('d')
    float_array.fromfile(fdat, Nt)
    time = float_array.tolist()

    Fx = [[] for ncfd in range(Ncfd)]
    Fy = [[] for ncfd in range(Ncfd)]
    y = [[] for ncfd in range(Ncfd)]
    for ncfd in range(Ncfd):
        float_array = array('d')
        float_array.fromfile(fdat, Nt)
        Fx[ncfd] = float_array.tolist()

        float_array = array('d')
        float_array.fromfile(fdat, Nt)
        Fy[ncfd] = float_array.tolist()

        float_array = array('d')
        float_array.fromfile(fdat, Nt)
        y[ncfd] = float_array.tolist()

    fdat.close()

    # plot data
    if plot_data:
        fig3, axs = mpl.subplots(1)
        for ncfd in range(Ncfd):
            axs.plot(time, [y[ncfd][nt] / d for nt in range(Nt)])
            for i,j in zip(time, [y[ncfd][nt] / d for nt in range(Nt)]):
                # appending required data into dict
                data['time'].append(i)
                data['y/d'].append(j)
                data['nc'].append(tmp[0])
                data['md'].append(tmp[1])
                data['U'].append(tmp[2])
                data['d'].append(tmp[3])
                data['m'].append(tmp[4])
                data['L'].append(tmp[5])
                data['H'].append(tmp[6])
                data['Nt'].append(tmp[7])
                data['Dt'].append(tmp[8])
                data['tf'].append(tmp[9])
                data['ymax'].append(tmp[10])
                
        # can plot Fx and Fy in the same way
        axs.set_xlabel('time [s]')
        axs.set_ylabel('y / d')
        axs.set_title('md = %d, H = %gN, U = %gm/s' % (md, H, U))
        mpl.show()

ftxt.close()
pd.DataFrame(data).to_csv('data'+str(Ncfd)+'.csv',index=False) # Store dataframe on disk for current case

if path.exists('data1.csv') and path.exists('data2.csv'): # check if files exist on this path or not
    df1 = pd.read_csv('data1.csv') # read case 1 dataframe
    df2 = pd.read_csv('data2.csv') # read case 2 dataframe
    df = pd.concat([df1,df2]) # concate both dataframes
    df.to_csv('combined_data.csv',index=False)
