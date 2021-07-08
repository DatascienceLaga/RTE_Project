#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.ensemble import RandomForestRegressor


from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import Adam

def train(X_train,X_test,y_train,y_test):
    """
    Entraîne le modèle avec les sorties du modèle de sim (X_train) et de ref (y_train)
    Fais une prédiction (y_pred) sur les sorties test du modèle de sim(Xtest)
    La prédiction doit se rapprocher des sorties test du modèle de reference (y_test)
    
    Paramètres
    ----------
    X_train :  numpy.ndarray : sortie du modèle de simulation, training data set
    X_test  :  numpy.ndarray : sortie  du modèle de simulation, test data set
    
    y_train :  numpy.ndarray : sortie du modèle de reference dont on doit se rapprocher,
    y_test  :  numpy.ndarray : sortie du modèle de reference dont on doit se rapprocher
    
    Returns
    --------
    y_pred  :  numpy.ndarray : Prédiction du modèle sur les données test du simulateur

    """
    reg = RandomForestRegressor()
    reg.fit(X_train,y_train)
    y_pred = reg.predict(X_test)
    
    return y_pred

def train_ann(X_train,X_test,y_train,y_test):
    """
    Entraîne le modèle avec les sorties du modèle de sim (X_train) et de ref (y_train)
    Fais une prédiction (y_pred) sur les sorties test du modèle de sim(Xtest)
    La prédiction doit se rapprocher des sorties test du modèle de reference (y_test)
    
    Paramètres
    ----------
    X_train :  numpy.ndarray : sortie du modèle de simulation, training data set
    X_test  :  numpy.ndarray : sortie  du modèle de simulation, test data set
    
    y_train :  numpy.ndarray : sortie du modèle de reference dont on doit se rapprocher,
    y_test  :  numpy.ndarray : sortie du modèle de reference dont on doit se rapprocher
    
    Returns
    --------
    y_pred  :  numpy.ndarray : Prédiction du modèle sur les données test du simulateur

    """
 

    model = Sequential()
    #model.add(Dense(128,input_dim=1, activation='relu'))
    #model.add(Dropout(0.3))
    model.add(Dense(30,input_dim=1,activation='relu'))
    model.add(Dense(15,activation='relu'))
    model.add(Dense(1,))
    model.add(Activation('sigmoid'))

    opt = Adam(learning_rate=0.0001)

    model.compile(loss='mean_squared_error', optimizer= opt)

    model.fit(X_train,y_train,batch_size = 10, epochs=50,verbose=1)
    y_pred = model.predict(X_test)
    
    return y_pred