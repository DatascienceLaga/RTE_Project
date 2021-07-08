#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import yaml
from uvsw_part import simulation
from script import traitement
from script import entrainement
from script import evaluation
from script import visualisation

def generation_eps(h,tension,u,clo,eps_list,TS_INDEX,LIST_PATH):
    """
    Pipeline entre les paramètres d'entrées et les modèles de prédiction
    
    Paramètres
    ----------
    h_list       : list  : Liste des valeurs de h à tester (pole altitude difference (2nd minus 1st, m))
    tension      : float : Valeur de tension à tester
    u            : float : Valeur de u à tester
    clo          : float : Valeur de cl0 à tester
    TS_INDEX : int   : Numéro de la time-series à exécuter (commence à 0)
    """

    data_list = pd.read_csv(LIST_PATH, delim_whitespace=True)
    set_params = data_list.iloc[TS_INDEX,:]
    ref = pd.read_csv("../data/csv/Liste1/graph{}.csv".format(set_params["nc"]))

    for eps_value in eps_list: 

        print("################ eps value: {} ################".format(eps_value))
        cfg = yaml.safe_load(open('../data/config/example.in.yaml', 'r'))


        cfg["cable"]["h"] = float(h)

        cfg["simulation"]["tf"] = float(set_params["tf[s]"])

        cfg["cable"]["tension"] = float(tension)

        cfg["wakeosc"]["u"] = u

        cfg["wakeosc"]["cl0"] = clo
        
        cfg["wakeosc"]["eps"] = eps_value



        cfg["simulation"]["dt"] = cfg["simulation"]["tf"] / len(ref) # MODIF DT
        cfg["simulation"]["dr"] = cfg["simulation"]["tf"] / len(ref) # MODIF DR



        print("h value: ", cfg["cable"]["h"], " u value: ", cfg["wakeosc"]["u"]," tension value: ",cfg["cable"]["tension"],
             "clo value ",cfg["wakeosc"]["cl0"])
        print("tf value ", cfg["simulation"]["tf"])
        dfy, _ = simulation.run_cable_wakeosc(cfg)
        
        X_train,X_test,y_train,y_test = traitement.preprocessing(dfy,ref)
        print("processing ok")
        
        y_pred = entrainement.train(X_train,X_test,y_train,y_test)
        print("train ok")
        
        eps = eps_value
        print("evaluate ok")
        
        #visualisation.plt_ts(dfy,ref,h,tension,u,clo,eps)
        visualisation.plotly_ts(dfy,ref,h,tension,u,clo,eps)
        
        #visualisation.plt_model(y_pred,y_test,h,tension,u,clo,eps)
        visualisation.plotly_model(y_pred,y_test,h,tension,u,clo,eps)
        
        print("plot ok")




def generation_ann_eps(h,tension,u,clo,eps_list,TS_INDEX,LIST_PATH):
    """
    Pipeline entre les paramètres d'entrées et les modèles de prédiction
    
    Paramètres
    ----------
    h_list       : list  : Liste des valeurs de h à tester (pole altitude difference (2nd minus 1st, m))
    tension      : float : Valeur de tension à tester
    u            : float : Valeur de u à tester
    clo          : float : Valeur de cl0 à tester
    TS_INDEX : int   : Numéro de la time-series à exécuter (commence à 0)
    """

    data_list = pd.read_csv(LIST_PATH, delim_whitespace=True)
    set_params = data_list.iloc[TS_INDEX,:]
    ref = pd.read_csv("../data/csv/Liste1/graph{}.csv".format(set_params["nc"]))

    for eps_value in eps_list: 

        print("################ eps value: {} ################".format(eps_value))
        cfg = yaml.safe_load(open('../data/config/example.in.yaml', 'r'))


        cfg["cable"]["h"] = float(h)

        cfg["simulation"]["tf"] = float(set_params["tf[s]"])

        cfg["cable"]["tension"] = float(tension)

        cfg["wakeosc"]["u"] = u

        cfg["wakeosc"]["cl0"] = clo
        
        cfg["wakeosc"]["eps"] = eps_value



        cfg["simulation"]["dt"] = cfg["simulation"]["tf"] / len(ref) # MODIF DT
        cfg["simulation"]["dr"] = cfg["simulation"]["tf"] / len(ref) # MODIF DR



        print("h value: ", cfg["cable"]["h"], " u value: ", cfg["wakeosc"]["u"]," tension value: ",cfg["cable"]["tension"],
             "clo value ",cfg["wakeosc"]["cl0"])
        print("tf value ", cfg["simulation"]["tf"])
        dfy, _ = simulation.run_cable_wakeosc(cfg)
        
        X_train,X_test,y_train,y_test = traitement.preprocessing(dfy,ref)
        print("processing ok")
        
        y_pred = entrainement.train_ann(X_train,X_test,y_train,y_test)
        print("train ok")
        
        eps = eps_value
        print("evaluate ok")
        
        #visualisation.plt_ts(dfy,ref,h,tension,u,clo,eps)
        visualisation.plotly_ts(dfy,ref,h,tension,u,clo,eps)
        
        #visualisation.plt_model(y_pred,y_test,h,tension,u,clo,eps)
        visualisation.plotly_model(y_pred,y_test,h,tension,u,clo,eps)
        
        print("plot ok")
