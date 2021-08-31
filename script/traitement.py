#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error



def preprocessing(dfy,ref):
    """
    Traite les données issus de la simulation et de la reference pour les rendre utilisable pour l'entraînement et les test
    On sépare les données du le sim et le ref avec 80% des données pour l'entrainement et 20% pour les test

    Paramètres
    ----------
    dfy : numpy.ndarray
    ref : pandas.core.series.Series  

    Returns

    X_train : numpy.ndarray , sortie du modèle de simulation, training data set
    X_test  : numpy.ndarray , sortie du modèle de simulation, test data set
    y_train : numpy.ndarray , sortie  du modèle de reference dont on doit se rapprocher,
    y_test  : numpy.ndarray , sortiedu modèle de reference dont on doit se rapprocher
    """



    sim_data = dfy['s=0.250'].iloc[1:,]/0.025
    sim_data.head()
    ref_data = ref.set_index("time").sort_index()['y/d']
    ref_data.head(10)

    sim_ref_data = pd.concat([
    pd.DataFrame(sim_data.values,columns = ["sim"]),
    pd.DataFrame(ref_data.values[ref_data.values >= 0], columns = ["ref_pos"]),
    pd.DataFrame(ref_data.values[ref_data.values < 0], columns = ["ref_neg"])   
    ],axis = 1)

    #sim_ref_data.head(50)

    #sim_ref_data.to_csv("../data/csv/sim_ref_data_7.csv", header = True, index = None)
    pd.DataFrame(sim_data.values,columns = ["sim"])

    sim_df = sim_ref_data
    sim_pos = sim_df[sim_df.sim >= 0].sim.values
    ref_pos = sim_df["ref_pos"].dropna().values

    ref_pos = ref_pos[:sim_pos.shape[0]]

    train_index = min(int(len(sim_pos) * 0.8),int(len(ref_pos) * 0.8))
    test_index = min(len(sim_pos),len(ref_pos))

    X_train = sim_pos[:train_index]
    X_test = sim_pos[train_index:test_index]
    y_train = ref_pos[:train_index]
    y_test = ref_pos[train_index:test_index]

    """
    X_train = sim_pos[:int(len(sim_pos) * 0.8)]
    X_test = sim_pos[int(len(sim_pos) * 0.8):]
    y_train = ref_pos[:int(len(ref_pos) * 0.8)]
    y_test = ref_pos[int(len(ref_pos) * 0.8):]


    """
    X_train = X_train.reshape(-1, 1)
    X_test = X_test.reshape(-1, 1)

    return X_train,X_test,y_train,y_test
