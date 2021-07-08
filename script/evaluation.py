#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pandas as pd


def root_mean_squared_error(y_test,y_pred):
    """
    Calcule le root_mean_squared_error (RMSE)
    
    Paramètres
    ----------
    y_test
    y_pred
    
    Return
    ------
    
    RMSE
    

    """
    return np.sqrt(mean_squared_error(y_test,y_pred))


def evaluate_ts(dfy,ref):
 
    mse = mean_squared_error(ref['y/d'],(dfy['s=0.250']/0.025).values[:-1])
    rmse = root_mean_squared_error(ref['y/d'],(dfy['s=0.250']/0.025).values[:-1])
    mea = mean_absolute_error(ref['y/d'],(dfy['s=0.250']/0.025).values[:-1])
    r2 = r2_score(ref['y/d'],(dfy['s=0.250']/0.025).values[:-1])

    return mse,rmse,mea,r2

def evaluate_model(y_pred,y_test):
 
    mse = mean_squared_error(y_test,y_pred)
    rmse = root_mean_squared_error(y_test,y_pred)
    mea = mean_absolute_error(y_test,y_pred)
    r2 = r2_score(y_test,y_pred)

    return mse,rmse,mea,r2
