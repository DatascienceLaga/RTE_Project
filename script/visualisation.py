#!/usr/bin/env python
# -*- coding: utf-8 -*-


import plotly.graph_objects as go
import matplotlib.pyplot as plt
from script import evaluation


def plt_ts(dfy,ref,h,tension,u,clo,eps):

    plt.figure(figsize = (20,8))
    plt.plot(ref['time'], ref['y/d'], label = "Signal de reference")
    plt.plot(dfy.index, dfy['s=0.250']/0.025, label = "Signal du simulateur")
    plt.xlabel('Time (s)',fontsize=18)
    plt.ylabel('y/d',fontsize=18)

    mse,rmse,mea,r2_score = evaluation.evaluate_ts(dfy,ref)

    mse_text = "MSE = %s " % mse
    rmse_text = "RMSE = %s " % rmse
    mea_text = "MEA = %s " % mea
    r2_text = "R2_SCORE = %s " % r2_score 

    plt.figtext(0.5, 0.00, mse_text, ha="center", fontsize=18, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})
    plt.figtext(0.5, -0.05, rmse_text, ha="center", fontsize=18, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})
    plt.figtext(0.5, -0.10, mea_text, ha="center", fontsize=18, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})
    plt.figtext(0.5, -0.15, r2_text, ha="center", fontsize=18, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})


    plt.figtext(0.5, -0.20, "h = '{0}', tension = '{1}', u = '{2}', clo = '{3}', eps = '{4}'".format(h,tension,u,clo,eps), ha="center", fontsize=18, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})

    title = " Comparaison du signal de simulation avec avec le signal de reference" 
    plt.title(title,fontsize=18)

    plt.legend()
    plt.show()




def plt_model(y_pred,y_test,h,tension,u,clo,eps):
    """
    Plot la reference et la prédiction côte à côte  avec résultats des indicateurs 
    
    y_pred  :  numpy.ndarray  : Prédiction du modèle sur les données test du simulateur
    y_test  :  numpy.ndarray  : sortie du modèle de reference dont on doit se rapprocher
    
    
    """
    plt.figure(figsize = (20,8))
    plt.plot(y_test, label = "Signal du modèle reférence")
    plt.plot(y_pred, label = "Signal de la prédiction")

    mse,rmse,mea,r2_score = evaluation.evaluate_model(y_pred,y_test)
    mse_text = "MSE = %s " % mse
    rmse_text = "RMSE = %s " % rmse
    mea_text = "MEA = %s " % mea
    r2_text = "R2_SCORE = %s " % r2_score
  
    """
    plt.annotate(mse, xy=(0.05, 0.85), xycoords='axes fraction')
    plt.annotate(rmse, xy=(0.05, 0.80), xycoords='axes fraction')
    plt.annotate(mea, xy=(0.05, 0.75), xycoords='axes fraction')
    plt.annotate(p_reussite_precis, xy=(0.02, 0.70), xycoords='axes fraction')
    plt.annotate(p_reussite_approx, xy=(0.02, 0.65), xycoords='axes fraction')
    """
    plt.figtext(0.5, 0.00, mse_text, ha="center", fontsize=18, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})
    plt.figtext(0.5, -0.05, rmse_text, ha="center", fontsize=18, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})
    plt.figtext(0.5, -0.10, mea_text, ha="center", fontsize=18, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})
    plt.figtext(0.5, -0.15, r2_text, ha="center", fontsize=18, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})
        

    plt.figtext(0.5, -0.20, "h = '{0}', tension = '{1}', u = '{2}', clo = '{3}', eps = '{4}'".format(h,tension,u,clo,eps), ha="center", fontsize=18, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})

    
    
    title = " Comparaison entre le modèle de réference et la prédiction du modèle d'apprentissage Random Forest, h = '{0}', tension = '{1}', u = '{2}', clo = '{3}', eps = '{4}'".format(h,tension,u,clo,eps)
    plt.title(title,fontsize = 18)
    plt.xlabel("Timesteps")
    plt.ylabel("Signal")
    plt.legend()
    plt.show()


def plotly_ts(dfy,ref,h,tension,u,clo,eps):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ref['time'], y=ref['y/d'],
                    name='Signal de reference'))
    fig.add_trace(go.Scatter(x=dfy.index, y=dfy['s=0.250']/0.025,
                    name='Signal de simulation'))
    
    fig.update_layout(
        title="Comparaison du signal de simulation avec le signal de reference",
        xaxis_title="Timesteps",
        yaxis_title="Signal",
        legend_title="Signaux",
    )


    fig.show()
    #fig.write_html("../data/ts.html")



def plotly_model(y_pred,y_test,h,tension,u,clo,eps):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y = y_test,
                    name='Signal de reference'))
    fig.add_trace(go.Scatter(y = y_pred,
                    name='Signal de simulation'))
    
    fig.update_layout(
        title=" Comparaison entre le modèle de réference et la prédiction du modèle d'apprentissage Random Forest",
        xaxis_title="Timesteps",
        yaxis_title="Signal",
        legend_title="Signaux",       
    )


    fig.show()
    #fig.write_html("../data/model.html")
