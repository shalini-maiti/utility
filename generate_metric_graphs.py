#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 13:19:47 2020

@author: shalini

Read metrics from an excel sheet and plot the values as a bar and line graphs
"""

import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation
import pandas as pd

colour_dict = {0: 'darkgray', 1: 'turquoise',
               2: 'tomato', 3: 'darkolivegreen',
               4: 'deepskyblue', 5:'crimson',
               6: 'mediumslateblue', 7: 'lightseagreen'}
metric_dict = {'2DKPE': 0, 'AU2D': 1}
excel_sheet_path = r'/home/shalini/Downloads/AllMetrics.xlsx'
sheet_number = 1

def plot_bar_graph(overall_metric, models, datasets, metric_name):
    fig, ax = plt.subplots()

    xaxis = np.arange(len(models))
    ax = plt.subplot(1,1,1)
    width = 1/len(models) - 0.005

    ### BAR PLOTS
    for index_ds, ds in enumerate(datasets):
        ax.bar(xaxis+(index_ds-3)*width, overall_metric[:, index_ds, metric_dict[metric_name]], width=width, label=metric_name+ '['+ds+']', color=colour_dict[index_ds])
        #ax.bar(xaxis-width, report1_models_freibg_acc, width=width, label='% Accuracy[Freihands With Bg]', color='r')

    print(len(models))
    plt.xticks(np.arange(len(models)), models, ha="right", rotation=7)
    ax.set_xlabel('Model ID')
    ax.set_ylabel(metric_name)
    ax.set_title(metric_name +' of prediction of Models on different test Datasets.')


    ax.legend(loc="best")
    plt.show()

    pass

def plot_line_graph(overall_metric, models, datasets, metric_name):
    fig, ax = plt.subplots()

    for index_ds, ds in enumerate(datasets):

        ax.plot(models, overall_metric[:, index_ds, metric_dict[metric_name]], label=metric_name+ '['+ds+']', color=colour_dict[index_ds])
        ax.scatter(models, overall_metric[:, index_ds, metric_dict[metric_name]], color=colour_dict[index_ds])

    plt.xticks(models, rotation=25)
    ax.set_xlabel('Model ID')
    ax.set_ylabel(metric_name + '(in pixels)')
    ax.set_title(metric_name +' of prediction of Models on real test Datasets.')

    ax.legend()
    plt.show()
    pass

def main():
    df = pd.read_excel (excel_sheet_path,
                        sheet_name=sheet_number)
    models = df["Succesful Training Datasets"].drop_duplicates().values[1:] # Remove Nan
    length_models = len(models)
    test_datasets = df["Test Dataset"].drop_duplicates().values[1:]
    length_td = len(test_datasets)
    overall_metric = np.empty((length_models, length_td, 2)) # kpe, au2d
    for index_model, model in enumerate(models):
        for index_td, td in enumerate(test_datasets):
            overall_metric[index_model, index_td, 0] = df[(df["Succesful Training Datasets"] == model)&
                                               (df["Test Dataset"] == td)]["2DKPE"].item()
            overall_metric[index_model, index_td, 1] = df[(df["Succesful Training Datasets"] == model)&
                                               (df["Test Dataset"] == td)]["AU2D"].item()

    plot_bar_graph(overall_metric, models, test_datasets, "2DKPE")
    #plot_bar_graph(overall_metric, models, test_datasets, "AU2D")
    #plot_line_graph(overall_metric, models, test_datasets, "2DKPE")
    #plot_line_graph(overall_metric, models, test_datasets, "AU2D")



if __name__ == '__main__':
    main()