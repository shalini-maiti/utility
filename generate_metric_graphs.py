#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 13:19:47 2020

@author: shalini
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
excel_sheet_path = r'/home/shalini/Downloads/MastersThesisDetails.xlsx'
sheet_number = 4

def ex_code():
    pca = [6, 12]
    model_type = ["hand", "part_arm", "full_arm", "body"]
    test_data = ["gt_bg_cleaned_frei_unresized", "gt_bg_cleaned_frei_resized",
                 "deeplab_bg_cleaned_frei_unresized"]
    metrics = ["mean2dkpError", "Au2d"]
    
    model_parameters = ["pca", "model_type", "rotation_range", "jpg_noise", "hand_colour"
                        "hand_texture", "background", "blur", "hand_size", "online_scale_range", 
                        "hist_equalized_tex", "varying_light", "spherical_harmonics"]
    
    model_hand_id = ['1', '2b', '3b', '4', '5', '6', '16a', '20a', 
                '21', '22', '23_prev', '23_new',  '23b_prev', '23b_new', 
                '23c_prev', '23c_new', '24b', '24c'] # 16, 20b, 24, 27, 27b,
    
    model_arm_id = ['8b', '9b', '10b', '11', '12'] # 8, 13, 14, 25, 000, 001, 002
    
    scaled_pix = 300*300
    unscaled_pix = 140*140
    # Only the scaled ones, 300x300
    kps2derr_hand = [72.016988, 79.683604, 226.646871, 64.045289, 90.544912, 84.799453,
                     90.852851,  50.744274,  46.954067,  50.680035,  187.310081, 165.463248,
                     46.304172, 53.545566, 52.786450, 50.738678, 35.469803, 35.393376]#Check 16 
    
    kps2derr_arm =[20.80004782008411, 70.60799330596552, 69.38629656120101, 
                   44.02539278011741,  69.35909239082723]
    
    #report1_models_id = ['1', '2', '4', '20', '21', '22', '23b', '23c']
    report1_models_id = ['1', '2', '3', '4', '5', '6', '16a', '20', '21', '22', '23', '23b', '23c', 
                         '24b', '24c', '24e', '24f', '24g', '24h', '24i', '27b', '28c', '24b->24', '27b->27', '24e->24d',
                         
                         '24b->24c', '24b->24->24c', '24b->24c->24', '24e->24f', '24e->24d->24f',
                         '24e->24f->24d', '24e->24d->24f->24j', '24e->24f->24d->24j', '24e->24f->24d->24j->24k',
                         '24e->24d->24f->24j->24k', '27b->27c', '27b->27c->27', '27b->27->27c', '27b->27c->27->27d',
                         '27b->27c->27->27e', '27b->27c->27->27e->27d', '27b->27c->27->27d->27e', '24e->24j->24f',
                         '24e->24j->24d']
    	
    report1_models_frei_err = [72.755949, 79.126185, 63.462633, 50.524404, 47.5, 49.07, 56.76, 50.508,
                          36.62, 38.34, 36.28, 37.07, 37.09, 42.00, 83.95, 35.31, 45.493427, 39.829781,
                          ] # not for 27b->27
    report1_models_frei_acc = [10.28, 2.28, 0,  8.76, 0.13, 1.14, 0, 39.85, 42.89, 38.83, 3.42, 31.60, 33.76,
                               60.15, 57.36, 61.04, 58.63, 59.01, 49.87, 16.49, 60.53, 48.35, 53.68, 55.58, 39.054839,
                               ]
    		
    report1_models_shds_err = [30.94, 37.11, 31.88, 34.54, 27.58, 37.04, 64.30, 37.16, 
                               22.63, 22.93, 21.48, 21.81, 22.12, 30.23, 32.11, 23.42, 72.32, 27.210701, 30.066526, 26.385601,
                               ]
    report1_models_shds_acc = [40.88, 15.72, 0, 30.81, 0, 18.86, 1.26, 55.97, 63.52, 37.74, 10.69, 40.88, 32.70,
                               77.36, 72.96, 80.50, 76.73, 78.61, 57.23, 49.69, 76.72, 24.790202, 61.63,58.49, 67.29,
                               ]
    
    report1_models_ho3d_err = [35.780065, 102.239722, 202.537585, 37.539264, 51.40264, 33.71, 130.545997, 48.659944,
                               42.076274, 97.829698, 50.538221, 56.260066, 94.139645, 48.148979, 87.412994, 41.137381,
                               84.842382, 48.415408, 90.84236, 28.80927954, 53.821026, 47.422442, 21.244759, 20.405136, 19.681736,
                               ]
    report1_models_ho3d_acc = [44.94, 0, 0, 25.09, 7.12, 39.556549, 3, 33.33, 
                               32.2, 0, 23.22, 14.61, 0, 23.59, 0, 35.96, 
                               0, 25.09, 0, 60.3,  22.47, 25.84, 78.27, 79.78, 84.64,
                               
                               12.73, 28.84, 84.27, 20.97, 41.57, 84.27, 88.76, 92.13, 40.45, 38.58, 25.47, 85.39, 50.56, 
                               83.52, 35.96, 64.04, 28.46, 23.6, 82.39]
    
    report1_models_freibg_err = [112.9, 117, 323.41, 64.98, 111.279327, 102.346685, 177.693191, 107.514247, 96.100016,
                                 174.942576, 147.919674, 151.02992, 158.182, 99.859871, 134.980697, 103.728882, 138.465171,
                                 103.383507, 122.639376, 91.181472, 115.7083201, 124.460283, 43.656597, 37.38597, 44.872923, 
                                 ]
    report1_models_freibg_acc = [1.01, 0, 0, 7.6, 0.38, 3.55, 3.04, 2.41, 5.71, 
                                 0, 3.17, 0.12, 0, 5.076, 0, 4.06, 0.25, 
                                 5.08, 0.12, 18.4, 2.03, 0.76, 49.87, 57.87, 50.38, 
                                 
                                 2.16, 26.77, 55.46, 3.3, 31.09, 52.92, 54.7, 54.82, 9.9, 8.38, 4.7, 56.98, 25.51, 
                                 52.53, 3.29, 40.6, 0.89, 3.81, 48.1]
    
    fig, ax = plt.subplots()
    
    xaxis = np.arange(len(report1_models_freibg_acc))
    ax = plt.subplot(1,1,1)
    width = 0.25
    
    ### BAR PLOTS
    
    ax.bar(xaxis-2*width, report1_models_ho3d_acc, width=width, label='% Accuracy[Ho3d varied]', color='b')
    ax.bar(xaxis-width, report1_models_freibg_acc, width=width, label='% Accuracy[Freihands With Bg]', color='r')
    #ax.bar(xaxis+width, report1_models_frei_acc, width=0.2, label='% Accuracy[Freihands Processed]', color='g') #'Mean Keypoints Error'
    #ax.bar(xaxis+2*width, report1_models_shds_acc, width=0.2, label='% Accuracy[ShreyasDS Processed]', color='c')
    #ax.axhline(y=40, color='r', linestyle= '--', label='Error Threshold')
    #ax.set_ylabel('Mean 2d Keypoints Error(in px)')
    #ax.set_title('Mean Keypoint Error of Models on preprocessed Freihands Dataset.')
    
    
    ### SCATTER PLOTS
    #ax.plot(report1_models_id, report1_models_frei_acc, label='% Accuracy[Freihands Processed]', color='g')
    #ax.scatter(report1_models_id, report1_models_frei_acc, color='g')
    
    #ax.plot(report1_models_id, report1_models_shds_acc, label='% Accuracy[ShreyasDS Processed]', color='c')
    #ax.scatter(report1_models_id, report1_models_shds_acc, color='c')
    
    #ax.plot(report1_models_id, report1_models_ho3d_acc, label='% Accuracy[Ho3d varied]')
    #ax.scatter(report1_models_id, report1_models_ho3d_acc, color='b')
    
    #ax.plot(report1_models_id, report1_models_freibg_acc, label='% Accuracy[Freihands With Bg]', color='r')
    #ax.scatter(report1_models_id, report1_models_freibg_acc, color='r')
    
    
    plt.xticks(xaxis + width /2, report1_models_id, rotation='vertical')
    ax.set_xlabel('Model ID')
    ax.set_ylabel('Percentage of images where Mean Keypoints Error < predefined threshold')
    ax.set_title('Accuracy of prediction of Models on different test Datasets.')
    
    
    ax.legend()
    plt.show()
    
    '''
    f, (ax0, ax1) = plt.subplots(1, 2)
     
    ax0.bar(model_hand_id, kps2derr_hand, color='r')
    ax1.bar(model_arm_id, kps2derr_arm, color='b')
    ax2.scatter(est_re[0, :, 0], est_re[0, :, 1], c='b', marker='o')
    plt.show()    
    '''
    
    # A new scatter plot for only finetuned tests
    
    report_fine_models_id = ['24b->24', '27b->27', '24e->24d', '24b->24c', '24b->24->24c', 
                             '24b->24c->24', '24e->24f', '24e->24d->24f', '24e->24f->24d', 
                             '24e->24d->24f->24j', '24e->24f->24d->24j', '24e->24f->24d->24j->24k',
                             '24e->24d->24f->24j->24k', '27b->27c', '27b->27c->27', '27b->27->27c', 
                             '27b->27c->27->27d', '27b->27c->27->27e', '27b->27c->27->27e->27d', 
                             '27b->27c->27->27d->27e', '24e->24j->24f', '24e->24j->24d']
    	
    report_finet_models_frei_err = [] 
    report_fine_models_frei_acc = []
    		
    report_finet_models_shds_err = []
    report_finet_models_shds_acc = []
    
    report_finet_models_ho3d_err = []
    report_finet_models_ho3d_acc = []
    
    report_finet_models_freibg_err = []
    report_finet_models_freibg_acc = []
    
    fig, ax = plt.subplots()
    #ax.bar(report1_models_id, report1_models_frei_err, 0.5, label='Mean Keypoints Error')
    #ax.axhline(y=40, color='r', linestyle= '--', label='Error Threshold')
    #ax.set_ylabel('Mean 2d Keypoints Error(in px)')
    #ax.set_title('Mean Keypoint Error of Models on preprocessed Freihands Dataset.')
    
    #ax.bar(report1_models_id, report1_models_acc, 0.5, label='Percentage Accuracy')
    
    #ax.plot(report1_models_id, report1_models_frei_acc, label='% Accuracy[Freihands Processed]', color='g')
    #ax.scatter(report1_models_id, report1_models_frei_acc, color='g')
    
    #ax.plot(report1_models_id, report1_models_shds_acc, label='% Accuracy[ShreyasDS Processed]', color='y')
    #ax.scatter(report1_models_id, report1_models_shds_acc, color='y')
    
    ax.plot(report1_models_id, report1_models_ho3d_acc, label='% Accuracy[Ho3d varied]')
    ax.scatter(report1_models_id, report1_models_ho3d_acc, color='b')
    
    ax.plot(report1_models_id, report1_models_freibg_acc, label='% Accuracy[Freihands With Bg]', color='r')
    ax.scatter(report1_models_id, report1_models_freibg_acc, color='r')
    
    plt.xticks(report1_models_id, rotation='vertical')
    ax.set_xlabel('Model ID')
    ax.set_ylabel('Percentage of images where Mean Keypoints Error < predefined threshold')
    ax.set_title('Accuracy of prediction of Models on different test Datasets.')
    
    
    ax.legend()
    plt.show()

def plot_bar_graph(overall_metric, models, datasets, metric_name):
    fig, ax = plt.subplots()
    
    xaxis = np.arange(len(models))
    ax = plt.subplot(1,1,1)
    width = 1/len(models)
    
    ### BAR PLOTS
    for index_ds, ds in enumerate(datasets):
        ax.bar(xaxis+index_ds*width, overall_metric[:, index_ds, metric_dict[metric_name]], width=width, label=metric_name+ '['+ds+']', color=colour_dict[index_ds])
        #ax.bar(xaxis-width, report1_models_freibg_acc, width=width, label='% Accuracy[Freihands With Bg]', color='r')
        
    plt.xticks(xaxis + width /2, models, rotation='vertical')
    ax.set_xlabel('Model ID')
    ax.set_ylabel(metric_name)
    ax.set_title(metric_name +' of prediction of Models on different test Datasets.')
    
    
    ax.legend()
    plt.show()

    pass
    
def plot_line_graph(overall_metric, models, datasets, metric_name):
    fig, ax = plt.subplots()
    
    for index_ds, ds in enumerate(datasets):
        
        ax.plot(models, overall_metric[:, index_ds, metric_dict[metric_name]], label=metric_name+ '['+ds+']', color=colour_dict[index_ds])
        ax.scatter(models, overall_metric[:, index_ds, metric_dict[metric_name]], color=colour_dict[index_ds])
    
    plt.xticks(models, rotation='vertical')
    ax.set_xlabel('Model ID')
    ax.set_ylabel(metric_name)
    ax.set_title(metric_name +' of prediction of Models on different test Datasets.')    
    
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
    plot_bar_graph(overall_metric, models, test_datasets, "AU2D")
    plot_line_graph(overall_metric, models, test_datasets, "2DKPE")
    plot_line_graph(overall_metric, models, test_datasets, "AU2D")
    
    
    
if __name__ == '__main__':
    main()