# -*- coding: utf-8 -*-
"""
Created on Thu Sep 8 13:40:59 2022

@author: RLaje, ASilva
"""

import numpy as np
#import matplotlib.pyplot as plt
import glob
import pandas as pd
import json
import sys
from plotnine import ggplot, aes, geom_line, geom_errorbar, geom_point, scale_linetype_manual, scale_shape_manual
from plotnine import scale_x_continuous, theme, scale_color_manual, facet_grid
from plotnine import xlab, ylab, theme_bw, element_rect, ggtitle, themes, element_text, element_blank
from plotnine import scale_color_grey, scale_size_manual
import patchworklib as pw
from matplotlib import cm
from matplotlib.colors import rgb2hex


#import patchworklib as pw
#import matplotlib.gridspec as gridspec
from scipy import stats as st
import statsmodels.stats.multitest as sm
# Setting path.
sys.path.append('../experiment')
# Importing.
import tappingduino as tp


#%% Load Data.


#%% Load_SubjectExpMetadata.
# Function to load experiment subject metadata. Return a dataframe.
# subject_number --> int (ej: 0). path --> string (ej: '../data/Experiment_PS_SC/'). total_blocks --> int (ej: 1).
def Load_SubjectMetadata(subject_number, path, total_blocks):
    s_number = '{0:0>3}'.format(subject_number)
    conc_block_conditions_df = pd.DataFrame()
    for i in range(total_blocks):
        file_to_load = glob.glob(path + 'S'+s_number+"*-block"+str(i)+"-trials.csv")
        if (len(file_to_load) != 0):
            file_to_load = glob.glob(path + 'S'+s_number+"*-block"+str(i)+"-trials.csv")[0]
            file_to_load_df = pd.read_csv(file_to_load)
            conc_block_conditions_df = pd.concat([conc_block_conditions_df, file_to_load_df]).reset_index(drop = True)
    return conc_block_conditions_df


#%% Load_ExpMetadata.
# Function to load experiment metadata. Return a dataframe.
# path --> string (ej: '../data/Experiment_PS_SC/'). n_blocks --> int (ej: 1).
def Load_ExpMetadata(path, n_blocks):
    filename_names = path + 'Dic_names.dat'
    with open(filename_names) as f_names:
       total_subjects = sum(1 for line in f_names) - 1
    f_names.close()
    conc_blocks_conditions_df = pd.DataFrame()
    for i in range(total_subjects):
        blocks_conditions_df = Load_SubjectMetadata(i, path, n_blocks)    
        conc_blocks_conditions_df = pd.concat([conc_blocks_conditions_df, blocks_conditions_df]).reset_index(drop = True)
    conc_blocks_conditions_df.to_csv(path + 'ExpMetaData.csv',na_rep = np.NaN)
    return conc_blocks_conditions_df


#%% Load_SingleTrial.
# Function to load trial data from trial data file. Return a dataframe.
# subject_number --> int (ej: 0). path --> string (ej: '../data/Experiment_PS_SC/'). block --> int (ej: 1). trial --> int (ej: 2).
def Load_SingleTrial(subject_number, path, block, trial):
    s_number = '{0:0>3}'.format(subject_number)
    file_to_load = glob.glob(path + 'S'+s_number+"*-block"+str(block)+"-trial"+str(trial)+".dat")[0]
    f_to_load = open(file_to_load,"r")
    content = f_to_load.read()
    f_to_load.close()
    content = json.loads(content)

    trialData_df = pd.DataFrame(columns=['Subject', 'Block', 'Trial', 'Event', 'Event_Order', 'Assigned_Stim', 'Time'], index=range(len(content['Stim_time'])))
    trialData_df['Subject'] = subject_number
    trialData_df['Block'] = block
    trialData_df['Trial'] = trial
    trialData_df['Event'] = 'S'
    trialData_df['Event_Order'] = range(0,len(content['Stim_time']))
    trialData_df['Time'] = content['Stim_time']
    
    trialDataAux_df = pd.DataFrame(columns=['Subject', 'Block', 'Trial', 'Event', 'Event_Order', 'Assigned_Stim', 'Time'], index=range(len(content['Resp_time'])))
    trialDataAux_df['Subject'] = subject_number
    trialDataAux_df['Block'] = block
    trialDataAux_df['Trial'] = trial
    trialDataAux_df['Event'] = 'R'
    trialDataAux_df['Event_Order'] = range(0,len(content['Resp_time']))
    trialDataAux_df['Time'] = content['Resp_time']

    trialData_df = pd.concat([trialData_df, trialDataAux_df]).reset_index(drop = True)

    trialDataAux_df = pd.DataFrame(columns=['Subject', 'Block', 'Trial', 'Event', 'Event_Order', 'Assigned_Stim', 'Time'], index=range(len(content['Asynchrony'])))
    trialDataAux_df['Subject'] = subject_number
    trialDataAux_df['Block'] = block
    trialDataAux_df['Trial'] = trial
    trialDataAux_df['Event'] = 'A'
    trialDataAux_df['Assigned_Stim'] = content['Stim_assigned_to_asyn']
    trialDataAux_df['Time'] = content['Asynchrony']
    
    trialData_df = pd.concat([trialData_df, trialDataAux_df]).reset_index(drop = True)
    return trialData_df


#%% Load_TrialsData.
# Function to load trials data. Return a dataframe.
# path --> string (ej: '../data/Experiment_PS_SC/'). n_blocks --> int (ej: 1).
def Load_TrialsData(path, n_blocks):
    expMetadata_df = Load_ExpMetadata(path, n_blocks)
    conc_trialData_df = pd.DataFrame()
    for i in range(len(expMetadata_df.index)): 
        subject_number = expMetadata_df['Subject'][i]
        block = expMetadata_df['Block'][i]
        trial = expMetadata_df['Trial'][i]
        trialData_df = Load_SingleTrial(subject_number, path, block, trial)
        conc_trialData_df = pd.concat([conc_trialData_df, trialData_df]).reset_index(drop = True)
    return conc_trialData_df


#%% Load_AllAsynchronies_AllValidTrials.
# Function to load asynchronies from valid trials. Return a dataframe.
# path --> string (ej: '../data/Experiment_PS_SC/'). n_blocks --> int (ej: 1).
def Load_AllAsynchronies_AllValidTrials(path, n_blocks):
    expMetaData_df = Load_ExpMetadata(path, n_blocks)
    trialsData_df = Load_TrialsData(path, n_blocks)
    result_df = pd.merge(expMetaData_df, trialsData_df, on=["Subject", "Block", "Trial"])
    result_df = result_df[(result_df['Valid_trial'] == 1) & (result_df['Event'] == 'A')].reset_index().drop(columns = ['index'])
    result_df['Relative_beep'] = result_df['Assigned_Stim'] - result_df['Perturb_bip']
    result_df['Relative_beep'] = result_df['Relative_beep'].astype('int')
    result_df.drop(columns = ['Original_trial', 'Perturb_type','Perturb_size', 'Valid_trial', 'Message', 'Error', 'Event', 'Event_Order'], inplace = True)
    result_df = result_df.reindex(columns=['Subject', 'Block', 'Trial', 'Condition', 'Assigned_Stim', 'Relative_beep', 'Perturb_bip', 'Time'])
    return result_df


#%% Experiments_Parameters
# Function to load all experiments parameters. Return a nested dictionary.
# path --> string (ej: '../analysis/'). 
def Experiments_Parameters(path):
    
    # Define path.
    path_list = glob.glob(path + 'Experiment' + '*')

    # Experiment Parameters.
    file_to_load = []
    content_dict = {}
    for i in range(len(path_list)):
        try:
            file_to_load = path_list[i] + '/Experiment_parameters.dat'
            f_to_load = open(file_to_load,"r")
            content = f_to_load.read()
            f_to_load.close()
            content_dict[str(i)] = json.loads(content)
        except:
            print('Warning: Para ' + file_to_load + ' no existe el archivo Experiment_parameters.dat')
    return content_dict


#%%










#%% Preprocessing Data.


#%% Asyn_Zeroed
# Complementary function of the function Preprocessing_Data to bring asynchrony to zero. Return a dataframe.
# data_df --> dataframe. n_subjects --> int (ej: 2). n_blocks --> int (ej: 3). 
def Asyn_Zeroed(data_df):
    
    data_mean_aux_df = pd.DataFrame()
    n_experiments = sorted((pd.unique(data_df['Experiment'])).tolist())
    for experiment in n_experiments:
        data_aux_df = data_df[(data_df['Experiment'] == experiment)]
        n_subjects = sorted((pd.unique(data_df['Subject'])).tolist())
        for subject in n_subjects:
            data_aux2_df = data_aux_df[(data_aux_df['Subject'] == subject)]
            n_blocks = sorted((pd.unique(data_aux2_df['Block'])).tolist())
            for block in n_blocks:
                data_aux3_df = data_aux2_df[(data_aux2_df['Block'] == block)]
                n_trials = sorted((pd.unique(data_aux3_df['Trial'])).tolist())
                for trial in n_trials:
                    data_aux4_df = data_aux3_df[(data_aux3_df['Trial'] == trial)].reset_index(drop = True)
                    perturb_bip = int(data_aux4_df['Perturb_bip'][0])
                    data_aux5_df = data_aux4_df[(data_aux4_df['Assigned_Stim'] < perturb_bip)]
                    data_mean_aux_df = pd.concat([data_mean_aux_df, data_aux5_df]).reset_index(drop = True)
    
    data_mean_df = data_mean_aux_df.groupby(["Experiment", "Subject", "Block", "Trial", "Condition"], as_index = False)["Time"].mean()
    data_mean_df.rename(columns={"Time": "Asyn_mean"}, inplace = True)
    
    result_df = pd.merge(data_df, data_mean_df, on=["Experiment", "Subject", "Block", "Trial", "Condition"])
    result_df["Asyn_zeroed"] = result_df["Time"] - result_df["Asyn_mean"]
    result_df = result_df.drop(columns = ['Perturb_bip', 'Asyn_mean'])
        
    return result_df


#%% Preprocessing_Data
# Function to process data for all experiments and general conditions dictionary, 
# considering the transient and the beeps out of range. Return tuple with two dataframes.
# path --> string (ej: '../analysis/'). transient_dur --> int (ej: 1).
def Preprocessing_Data(path, transient_dur): 

    # Experiments Parameters.
    experiments_parameters_dict = Experiments_Parameters(path)
    exp_index_list = list(experiments_parameters_dict.keys())

    general_cond_dict_df = pd.DataFrame()
    general_proc_data_df = pd.DataFrame()
    for i in exp_index_list:
        # Experiment Parameters.
        n_blocks = experiments_parameters_dict[i]['n_blocks']                                                       # Number of blocks.
        perturb_type_dictionary = experiments_parameters_dict[i]['perturb_type_dictionary']				            # Perturbation type dictionary. 1--> Step change. 2--> Phase shift.                                     
        perturb_size_dictionary = experiments_parameters_dict[i]['perturb_size_dictionary']				            # Perturbation size dictionary.
        data_path = experiments_parameters_dict[i]['path']                                                          # Data path.
        
        # General conditions dictionary.
        exp_name_index = data_path.find('Experiment')
        experiment_name = data_path[exp_name_index:-1]
        condition_dictionary_df = tp.Condition_Dictionary(perturb_type_dictionary, perturb_size_dictionary)         # Possible conditions dictionary per experiment.
        condition_dictionary_df = condition_dictionary_df.assign(Experiment = int(i), Exp_name = experiment_name)
        general_cond_dict_df = pd.concat([general_cond_dict_df, condition_dictionary_df]).reset_index(drop = True)
        
        # General process data.
        processing_data_df = Load_AllAsynchronies_AllValidTrials(data_path, n_blocks)                                    # Process data per experiment.
        processing_data_df = processing_data_df.assign(Experiment = int(i))
        general_proc_data_df = pd.concat([general_proc_data_df, processing_data_df]).reset_index(drop = True)

    general_cond_dict_df.index.name = "General_condition"
    general_cond_dict_df.reset_index(inplace = True)

    result_df = pd.DataFrame()
    result_df = pd.merge(general_cond_dict_df, general_proc_data_df, on=["Experiment", "Condition"])
    result_df = result_df.reindex(columns=['Exp_name', 'Experiment', 'Subject', 'Block', 'Trial', 'General_condition', 'Condition', 'Assigned_Stim', 'Relative_beep', 'Time', 'Perturb_bip'])
    result_df.sort_values(['Experiment', 'Subject', 'Block', 'Trial'], inplace = True)
    result_df.reset_index(drop = True, inplace = True)

    # Find first and last stim of every trial.
    result_df["First_beep"] = result_df.groupby(["Experiment", "Subject", "Condition", "Trial"])["Relative_beep"].transform(min)
    result_df["Last_beep"] = result_df.groupby(["Experiment", "Subject", "Condition", "Trial"])["Relative_beep"].transform(max)
    #result_df["First_beep"] = result_df["First_beep"].astype('int')
    #result_df["Last_beep"] = result_df["Last_beep"].astype('int')
    
    
    # Keep beeps after transient only.
    result_df.query("Relative_beep >= First_beep + @transient_dur", inplace=True)
    
    # Find first and last COMMON stim for all trials. The first beeps of the trial are discarded.
    result_df.drop(columns=["First_beep"], inplace=True)
    result_df["First_beep"] = result_df.groupby(["Experiment", "Subject", "Condition", "Trial"])["Relative_beep"].transform(min)
    result_df["First_common_beep"] = result_df["First_beep"].agg(max)
    result_df["Last_common_beep"] = result_df["Last_beep"].agg(min)

    # Keep common beeps only. If the relative beep falls outside the range of common beeps, the asynchrony is discarded.
    result_df.query("Relative_beep >= First_common_beep & Relative_beep <= Last_common_beep", inplace=True)
    
    # Remove unused columns.
    result_df.drop(columns=["First_beep","Last_beep","First_common_beep","Last_common_beep"], inplace=True)
    result_df = result_df.reset_index(drop=True)
    
    # Asynchronies zeroed.
    data_df = Asyn_Zeroed(result_df)

    return general_cond_dict_df, data_df


#%% Preprocessing_Data_AllExperiments_MarkingTrialOutliers
# Function to process data for all experiments and general conditions dictionary, marking outlier trials. Return a tuple with three dataframes.
# path --> string (ej: '../analysis/'). data_df --> dataframe. postPerturb_bip --> int (ej: 5).
def Preprocessing_Data_AllExperiments_MarkingTrialOutliers(path, data_df, postPerturb_bip):

    # Experiments dictionary and data
    general_cond_dict_df = data_df[0]
    general_proc_data_df = data_df[1]

    # Keep beeps after postPerturb_bip only
    postPerturb_bip_df = general_proc_data_df.query("Relative_beep >= @postPerturb_bip")

    # Data mean per each trial across beeps
    data_mean_df = (postPerturb_bip_df
                          # First average across trials.
                          .groupby(["Experiment", "Subject", "Condition", "Block", "Trial"], as_index=False)
                          .agg(mean_asyn=("Asyn_zeroed","mean"),std_asyn=("Asyn_zeroed","std")))

    # Applying quantile criteria
    data_mean_df["Outlier_trial_meanAsyn"] = 0
    data_mean_df["Outlier_trial_std"] = 0   
    data_mean_outlier_df = pd.DataFrame()
    n_experiments = sorted((pd.unique(general_proc_data_df['Experiment'])).tolist())
    for experiment in n_experiments:
        quantile_data_aux_df = general_proc_data_df[general_proc_data_df['Experiment'] == experiment]
        n_subjects = sorted((pd.unique(quantile_data_aux_df['Subject'])).tolist())
        for subject in n_subjects:
            n_conditions = sorted((pd.unique(quantile_data_aux_df['Condition'])).tolist())
            for condition in n_conditions:
                quantile_data_df = data_mean_df[(data_mean_df['Experiment'] == experiment) & (data_mean_df['Subject'] == subject)
                                                   & (data_mean_df['Condition'] == condition)].reset_index(drop = True)
                
                quantile_data_df["Outlier_trial_meanAsyn"] = 0
                quantile_data_df["Outlier_trial_std"] = 0
                                
                quantile = quantile_data_df.mean_asyn.quantile([0.25,0.5,0.75])
                Q1 = quantile[0.25]
                Q3 = quantile[0.75]
                IQR = Q3 - Q1             
                quantile_data_df.loc[quantile_data_df.mean_asyn < (Q1 - 1.5 * IQR),'Outlier_trial_meanAsyn'] = 1
                quantile_data_df.loc[quantile_data_df.mean_asyn > (Q3 + 1.5 * IQR),'Outlier_trial_meanAsyn'] = 1
                
                quantile = quantile_data_df.std_asyn.quantile([0.25,0.5,0.75])
                Q1 = quantile[0.25]
                Q3 = quantile[0.75]
                IQR = Q3 - Q1             
                quantile_data_df.loc[quantile_data_df.std_asyn > (Q3 + 1.5 * IQR),'Outlier_trial_std'] = 1
                
                # Trial mean
                data_mean_outlier_df = pd.concat([data_mean_outlier_df, quantile_data_df]).reset_index(drop = True)
    data_mean_outlier_df.drop(columns = ["mean_asyn", "std_asyn"], inplace = True)
    
    # Experiments data with meanAsyn and STD outlier trials marked
    result_df = pd.merge(general_proc_data_df, data_mean_outlier_df, on=["Experiment", "Subject", "Block", "Trial", "Condition"])
 
    # Data Search trials outliers
    data_aux_df = data_mean_outlier_df[(data_mean_outlier_df['Outlier_trial_meanAsyn'] == 1) | (data_mean_outlier_df['Outlier_trial_std'] == 1)].reset_index(drop = True)
    data_aux_df['Outlier'] = 1
    data_aux_df = (data_aux_df.groupby(["Experiment", "Subject", "Condition"], as_index=False).agg(Total_outlier_trial_mean = ("Outlier_trial_meanAsyn", "sum"), Total_outlier_trial_std = ("Outlier_trial_std", "sum"), Total_outlier = ("Outlier", "sum")))
    
    # Total trials per experiment, per subject and per condition
    data_aux2_df = data_mean_outlier_df.reset_index(drop = True)
    data_aux2_df['Trials'] = 1 
    data_aux2_df = (data_aux2_df.groupby(["Experiment", "Subject", "Condition"], as_index=False).agg(Total_trials = ("Trials", "sum")))

    # Porcentual outlier trials per experiment, per subject and per condition
    data_porcOutTrials_df = pd.merge(data_aux_df, data_aux2_df, on=["Experiment", "Subject", "Condition"])
    data_porcOutTrials_df['Total_outlier_trial_porc'] = (data_porcOutTrials_df['Total_outlier'] * 100) / data_porcOutTrials_df['Total_trials']
    
    return general_cond_dict_df, result_df, data_porcOutTrials_df


#%% Outliers_Trials_Cuantification
# Function to know outlier trials information.
# path --> string (ej: '../analysis/'). data_OutTrials_df --> dataframe.
def Outliers_Trials_Cuantification(path, data_OutTrials_df):
    
    # Experiments dictionary and data
    general_proc_data_df = data_OutTrials_df[1]
    data_porcOutTrials_df = data_OutTrials_df[2]
        
    # Total trials per experiment and per subject
    data_aux_df = general_proc_data_df.reset_index(drop = True)
    data_aux_df['Trials'] = 1
    data_aux_df = (data_aux_df.groupby(["Experiment", "Subject", "Condition", "Block", "Trial"], as_index=False).agg(Trials = ("Trials", "max")))
    data_aux_df = (data_aux_df.groupby(["Experiment", "Subject"], as_index=False).agg(Total_trials = ("Trials", "sum")))
            
    # Outlier trials per experiment and per subject
    data_aux2_df = data_porcOutTrials_df.reset_index(drop = True)
    data_aux2_df = (data_aux2_df.groupby(["Experiment", "Subject"], as_index=False).agg(Total_outliers = ("Total_outlier", "sum")))

    # Porcentual outlier trials per experiment and per subject
    data_aux3_df = pd.merge(data_aux_df, data_aux2_df, on=["Experiment", "Subject"])
    data_aux3_df['Total_outlier_trials_porc'] = (data_aux3_df['Total_outliers'] * 100) / data_aux3_df['Total_trials']
    
    # Total trials per experiment
    data_aux4_df = (data_aux_df.groupby(["Experiment"], as_index=False).agg(Total_trials = ("Total_trials", "sum")))

    # Outlier trials per experiment
    data_aux5_df = (data_aux2_df.groupby(["Experiment"], as_index=False).agg(Total_outliers = ("Total_outliers", "sum")))
        
    # Porcentual outlier trials per experiment
    data_aux6_df = pd.merge(data_aux4_df, data_aux5_df, on=["Experiment"])
    data_aux6_df['Total_outlier_trial_porc'] = (data_aux6_df['Total_outliers'] * 100) / data_aux6_df['Total_trials']
    
    # Total subjects with outlier trials    
    data_aux7_df = pd.DataFrame({'Total_subects': [len(data_aux2_df['Subject'])]})
        
    # Subjects with more outlier trials
    data_aux8_df = data_aux3_df[(data_aux3_df['Total_outliers'] == data_aux3_df.Total_outliers.max())].reset_index(drop = True)
    
    # Porcentual subjects with more outlier trials %
    data_aux9_df = data_aux3_df[(data_aux3_df['Total_outlier_trials_porc'] == data_aux3_df.Total_outlier_trials_porc.max())].reset_index(drop = True)

    # Save files
    data_porcOutTrials_df.to_csv('./outlier_metrics/' + "porc_outlier_trials_perExp_perSubj_perCond.csv", na_rep = np.NaN)
    data_aux3_df.to_csv('./outlier_metrics/' + "porc_outlier_trials_perExp_perSubj.csv", na_rep = np.NaN)
    data_aux6_df.to_csv('./outlier_metrics/' + "porc_outlier_trials_perExp.csv", na_rep = np.NaN)
    data_aux7_df.to_csv('./outlier_metrics/' + "total_subj_with_outlier_trials.csv", na_rep = np.NaN)
    data_aux8_df.to_csv('./outlier_metrics/' + "subj_with_more_outlier_trials.csv", na_rep = np.NaN)
    data_aux9_df.to_csv('./outlier_metrics/' + "subj_with_more_outlier_trials_porc.csv", na_rep = np.NaN)
    
    return


#%% Preprocessing_Data_AllExperiments_MarkingSubjCondOutliers
# Function to process data for all experiments and general conditions dictionary, marking outlier subject conditions. Return a tuple with four dataframes.
# path --> string (ej: '../analysis/'). data_OutTrials_df--> dataframe. porcTrialPrevCond --> int (ej: 10). postPerturb_bip --> int (ej: 5).
def Preprocessing_Data_AllExperiments_MarkingSubjCondOutliers(path, data_OutTrials_df, porcTrialPrevCond, postPerturb_bip):

    # Experiments dictionary and data
    general_cond_dict_df = data_OutTrials_df[0]
    general_proc_data_df = data_OutTrials_df[1]
    data_porcOutTrials_df = data_OutTrials_df[2]
  
    # Applying porcentual previous outlier trials conditions criteria
    general_proc_data2_aux_df = data_porcOutTrials_df.reset_index(drop = True) 
    general_proc_data2_aux_df["Outlier_trial_porcPrevCond"] = 0
    general_proc_data2_aux_df.loc[general_proc_data2_aux_df.Total_outlier_trial_porc > porcTrialPrevCond,'Outlier_trial_porcPrevCond'] = 1
    general_proc_data2_aux_df = general_proc_data2_aux_df.reset_index(drop = True).drop(columns = ['Total_outlier_trial_mean', 'Total_outlier_trial_std', 'Total_outlier', 'Total_trials', 'Total_outlier_trial_porc'])
    
    # Experiments data with meanAsyn, STD, procentual outlier trials marked
    general_proc_data2_aux2_df = general_proc_data_df.groupby(["Experiment", "Subject", "Condition"], as_index=False).agg(Outlier_trial_meanAsyn = ("Outlier_trial_meanAsyn", "max"), Outlier_trial_std = ("Outlier_trial_std", "max"))
    general_proc_data2_aux2_df = general_proc_data2_aux2_df[(general_proc_data2_aux2_df['Outlier_trial_meanAsyn'] == 0) & (general_proc_data2_aux2_df['Outlier_trial_std'] == 0)]
    general_proc_data2_aux2_df["Outlier_trial_porcPrevCond"] = 0
    general_proc_data2_aux2_df = general_proc_data2_aux2_df.reset_index().drop(columns = ['Outlier_trial_meanAsyn', 'Outlier_trial_std'])
    general_proc_data2_aux3_df = pd.concat([general_proc_data2_aux_df, general_proc_data2_aux2_df]).reset_index(drop = True)
    general_proc_data2_aux3_df = general_proc_data2_aux3_df.reset_index().drop(columns = ['index', 'level_0'])
    general_proc_data2_df = pd.merge(general_proc_data_df, general_proc_data2_aux3_df, on=["Experiment", "Subject", "Condition"])

    # Searching data without outlier trials
    particular_exp_data_df = general_proc_data2_df[(general_proc_data2_df['Outlier_trial_meanAsyn'] == 0) & (general_proc_data2_df['Outlier_trial_std'] == 0) & 
                                               (general_proc_data2_df['Outlier_trial_porcPrevCond'] == 0)].reset_index(drop = True)
    perSubjAve_df = (particular_exp_data_df
                          # Average across trials without outlier trials and outlier subj cond.
                          .groupby(["Experiment", "Subject", "General_condition", "Condition", "Relative_beep"], as_index=False)
                          .agg(mean_asyn=("Asyn_zeroed","mean")))

    # Keep beeps after postPerturb_bip only
    postPerturb_bip_df = perSubjAve_df.query("Relative_beep >= @postPerturb_bip")
        
    # Mean valid trials asyn all subjects per conditions and mean of the last ones
    data_mean_df = (postPerturb_bip_df
                          # First average across trials.
                          .groupby(["Experiment", "Subject", "General_condition", "Condition"], as_index=False)
                          .agg(mean_asyn=("mean_asyn","mean"),std_asyn=("mean_asyn","std")))

    # Applying quantile criteria
    data_mean_df["Outlier_subj_meanAsyn"] = 0
    data_mean_df["Outlier_subj_std"] = 0   
    data_mean_outlier_df = pd.DataFrame()
    n_experiments = sorted((pd.unique(data_mean_df['Experiment'])).tolist())
    for experiment in n_experiments:
        quantile_data_aux_df = data_mean_df[data_mean_df['Experiment'] == experiment]
        n_conditions = sorted((pd.unique(quantile_data_aux_df['Condition'])).tolist())
        for condition in n_conditions:
            quantile_data_df = data_mean_df[(data_mean_df['Experiment'] == experiment) & (data_mean_df['Condition'] == condition)].reset_index(drop = True)
                        
            quantile = quantile_data_df.mean_asyn.quantile([0.25,0.5,0.75])
            Q1 = quantile[0.25]
            Q3 = quantile[0.75]
            IQR = Q3 - Q1             
            quantile_data_df.loc[quantile_data_df.mean_asyn < (Q1 - 1.5 * IQR),'Outlier_subj_meanAsyn'] = 1
            quantile_data_df.loc[quantile_data_df.mean_asyn > (Q3 + 1.5 * IQR),'Outlier_subj_meanAsyn'] = 1
                
            quantile = quantile_data_df.std_asyn.quantile([0.25,0.5,0.75])
            Q1 = quantile[0.25]
            Q3 = quantile[0.75]
            IQR = Q3 - Q1             
            quantile_data_df.loc[quantile_data_df.std_asyn > (Q3 + 1.5 * IQR),'Outlier_subj_std'] = 1
            
            # Exp Subj Cond mean
            data_mean_outlier_df = pd.concat([data_mean_outlier_df, quantile_data_df]).reset_index(drop = True)
    
    # General processing data with meanAsyn and STD outlier subj cond marked
    data_mean_outlier2_df = data_mean_outlier_df.drop(columns = ['General_condition', 'mean_asyn', 'std_asyn'])
    data_mean_outlier3_df = general_proc_data2_aux_df[(general_proc_data2_aux_df['Outlier_trial_porcPrevCond'] == 1)].reset_index(drop = True)
    data_mean_outlier3B_df = data_mean_outlier3_df.drop(columns = ["Outlier_trial_porcPrevCond"])
    data_mean_outlier3B_df["Outlier_subj_meanAsyn"] = 0
    data_mean_outlier3B_df["Outlier_subj_std"] = 0
    data_mean_outlier4_df = pd.concat([data_mean_outlier2_df, data_mean_outlier3B_df]).reset_index(drop = True)
    general_proc_data3_df = pd.merge(general_proc_data2_df, data_mean_outlier4_df, on=["Experiment", "Subject", "Condition"]).reset_index(drop = True)

    # General processing data without outlier trials.  
    general_proc_WithoutOutlierTrials_df = general_proc_data3_df[(general_proc_data3_df['Outlier_trial_meanAsyn'] == 0) & (general_proc_data3_df['Outlier_trial_std'] == 0)].reset_index(drop = True)

    # Total conditions per experiment.
    gen_cond_df = general_cond_dict_df.reset_index(drop = True)
    gen_cond_df["Total_conditions"] = 1
    gen_cond_df = (gen_cond_df.groupby(["Experiment"], as_index=False).agg(Total_conditions = ("Total_conditions", "sum")))

    # Total outlier conditions per trials analysis.
    result_df = data_mean_outlier3_df.groupby(["Experiment", "Subject"], as_index = False).agg(Total_outlier_trial_conditions = ("Outlier_trial_porcPrevCond", "sum"))
    result_df = pd.merge(gen_cond_df, result_df, on=["Experiment"]).reset_index(drop = True)

    # Total outlier conditions per subj cond per meanAsyn and STD analysis.
    result2_df = data_mean_outlier_df[(data_mean_outlier_df['Outlier_subj_meanAsyn'] == 1) | (data_mean_outlier_df['Outlier_subj_std'] == 1)].reset_index(drop = True)    
    result2_df['Total_outlier_subj_conditions'] = 1
    result2_df = result2_df.groupby(["Experiment", "Subject", "Condition"], as_index = False).agg(Total_outlier_subj_conditions = ("Total_outlier_subj_conditions", "max"))
    result2_df = result2_df.groupby(["Experiment", "Subject"], as_index = False).agg(Total_outlier_subj_conditions = ("Total_outlier_subj_conditions", "sum"))
    result2_df = pd.merge(result2_df, gen_cond_df, on=["Experiment"]).reset_index(drop = True)
    result_df["Total_outlier_subj_conditions"] = 0
    result2_df["Total_outlier_trial_conditions"] = 0
    result3_df = pd.concat([result_df, result2_df]).reset_index(drop = True)
    result3_df["Total_outlier_conditions"] = result3_df["Total_outlier_trial_conditions"] + result3_df["Total_outlier_subj_conditions"]
    result4_df = result3_df.groupby(["Experiment", "Subject", "Total_conditions"], as_index = False).agg(Total_outlier_conditions = ("Total_outlier_conditions", "sum"))

    # Total outlier conditions porcentual
    data_porcOutSubjCond_df = result4_df.reset_index(drop = True)
    data_porcOutSubjCond_df["Total_outlier_conditions_porc"] = data_porcOutSubjCond_df["Total_outlier_conditions"] * 100 / data_porcOutSubjCond_df["Total_conditions"]
    
    return general_cond_dict_df, general_proc_data3_df, general_proc_WithoutOutlierTrials_df, data_porcOutSubjCond_df


#%% Outliers_SubjCond_Cuantification
# Function to know outlier subject conditions cuantification.
# path --> string (ej: '../analysis/'). data_OutSubjCond_df --> dataframe.
def Outliers_SubjCond_Cuantification(path, data_OutSubjCond_df):

    # Experiments dictionary and data
    general_proc_data_df = data_OutSubjCond_df[1]
    data_porcOutSubjCond_df = data_OutSubjCond_df[3] 

    # Total subjects per experiment
    n_subjects_df = general_proc_data_df.drop(columns = ['Exp_name', 'General_condition', "Condition", "Assigned_Stim", 
                                                         "Relative_beep", "Time", "Asyn_zeroed", "Outlier_trial_meanAsyn", 
                                                         "Outlier_trial_std", 'Outlier_trial_porcPrevCond', 'Outlier_subj_meanAsyn', 'Outlier_subj_std'])
    n_subjects_df["Total_subjects"] = 1
    n_subjects_df = n_subjects_df.groupby(["Experiment", "Subject"], as_index = False).agg(Total_subjects = ("Total_subjects", "max"))
    n_subjects_df = n_subjects_df.groupby(["Experiment"], as_index = False).agg(Total_subjects = ("Total_subjects", "sum"))

    # Total outlier conditions and total subject conditions per experiment. 
    n_subjCond_df = data_porcOutSubjCond_df.drop(columns = ['Subject', 'Total_outlier_conditions_porc'])
    n_subjCond_df = n_subjCond_df.groupby(["Experiment"], as_index = False).agg(Total_conditions = ("Total_conditions", "max"), Total_outlier_conditions = ("Total_outlier_conditions", "sum"))
    n_subjCond_df = pd.merge(n_subjCond_df, n_subjects_df, on=["Experiment"]).reset_index(drop = True)
    n_subjCond_df["Total_subjCond"] = n_subjCond_df["Total_conditions"] * n_subjCond_df["Total_subjects"] 
    n_subjCond_df.drop(columns = ['Total_conditions', 'Total_subjects'], inplace = True)
    
    # Porcentual total outlier conditions per total subject conditions per experiment.
    data_porcTotalOutCondPerTotalSubjCond_df = n_subjCond_df.reset_index(drop = True)
    data_porcTotalOutCondPerTotalSubjCond_df["Total_out_cond_perSubjCond_porc"] = data_porcTotalOutCondPerTotalSubjCond_df["Total_outlier_conditions"] * 100 / data_porcTotalOutCondPerTotalSubjCond_df["Total_subjCond"]

    # Save Files.
    data_porcOutSubjCond_df.to_csv('./outlier_metrics/' + "porc_outlier_conditions_perExp_perSub.csv", na_rep = np.NaN)
    data_porcTotalOutCondPerTotalSubjCond_df.to_csv('./outlier_metrics/' + "porc_outlier_conditions_perExp_perSubjCond.csv", na_rep = np.NaN)

    return


#%% Preprocessing_Data_AllExperiments_MarkingSubjOutliers
# Function to process data for all experiments and general conditions dictionary, marking outlier subjects. Return a tuple with four dataframes.
# path --> string (ej: '../analysis/'). data_OutSubjCond_df --> dataframe. porcSubjCondPrevCond --> int (ej: 10).
def Preprocessing_Data_AllExperiments_MarkingSubjOutliers(path, data_OutSubjCond_df, porcSubjCondPrevCond):

    # Experiments dictionary and data
    general_cond_dict_df = data_OutSubjCond_df[0]
    general_proc_data_df = data_OutSubjCond_df[1]
    general_proc_WithoutOutlierTrials_df = data_OutSubjCond_df[2]
    data_porcOutSubjCond_df = data_OutSubjCond_df[3]

    # Applying porcentual previous outlier subject conditions criteria
    general_proc_data2_aux_df = data_porcOutSubjCond_df.reset_index(drop = True) 
    general_proc_data2_aux_df["Outlier_subj_porcPrevCond"] = 0
    general_proc_data2_aux_df.loc[general_proc_data2_aux_df.Total_outlier_conditions_porc > porcSubjCondPrevCond,'Outlier_subj_porcPrevCond'] = 1
    general_proc_data2_aux_df = general_proc_data2_aux_df.reset_index(drop = True).drop(columns = ['Total_outlier_conditions', 'Total_conditions', 'Total_outlier_conditions_porc'])

    # General processing data with meanAsyn, STD, procentual outlier trials, subj cond and subj marked
    general_proc_data2_aux2_df = general_proc_WithoutOutlierTrials_df.groupby(["Experiment", "Subject"], as_index=False).agg(Outlier_trial_porcPrevCond = ("Outlier_trial_porcPrevCond", "max"), 
                                                                                                                             Outlier_subj_meanAsyn = ("Outlier_subj_meanAsyn", "max"), Outlier_subj_std = ("Outlier_subj_std", "max"))                       
    general_proc_data2_aux2_df = general_proc_data2_aux2_df[(general_proc_data2_aux2_df['Outlier_trial_porcPrevCond'] == 0) & (general_proc_data2_aux2_df['Outlier_subj_meanAsyn'] == 0) & (general_proc_data2_aux2_df['Outlier_subj_std'] == 0)]
    general_proc_data2_aux2_df["Outlier_subj_porcPrevCond"] = 0
    general_proc_data2_aux2_df = general_proc_data2_aux2_df.reset_index(drop = True).drop(columns = ['Outlier_trial_porcPrevCond', 'Outlier_subj_meanAsyn', 'Outlier_subj_std'])
    general_proc_data2_aux3_df = pd.concat([general_proc_data2_aux_df, general_proc_data2_aux2_df]).reset_index(drop = True)
    general_proc_data2_df = pd.merge(general_proc_data_df, general_proc_data2_aux3_df, on=["Experiment", "Subject"])

    # General processing data without outlier trials and subj cond.  
    general_proc_WithoutOutlierTrialsAndSubjCond_df = general_proc_data2_df[(general_proc_data2_df['Outlier_trial_meanAsyn'] == 0) & (general_proc_data2_df['Outlier_trial_std'] == 0) 
                                                                            & (general_proc_data2_df['Outlier_trial_porcPrevCond'] == 0) & (general_proc_data2_df['Outlier_subj_meanAsyn'] == 0) 
                                                                            & (general_proc_data2_df['Outlier_subj_std'] == 0)].reset_index(drop = True)
    
    # General processing data without outlier trials and subj cond and subject.  
    general_proc_WithoutOutlierTrialsAndSubjCondAndSubj_df = general_proc_WithoutOutlierTrialsAndSubjCond_df[(general_proc_WithoutOutlierTrialsAndSubjCond_df['Outlier_subj_porcPrevCond'] == 0)].reset_index(drop = True)

    # Save files
    general_cond_dict_df.to_csv(path + "general_cond_dict.csv", na_rep = np.NaN)
    general_proc_data2_df.to_csv(path + "general_proc_data.csv", na_rep = np.NaN)

    return general_cond_dict_df, general_proc_data2_df, general_proc_WithoutOutlierTrialsAndSubjCond_df, general_proc_WithoutOutlierTrialsAndSubjCondAndSubj_df


#%% Outliers_Subj_Cuantification
# Function to know outlier subjects cuantification.
# path --> string (ej: '../analysis/'). data_OutSubj_df --> dataframe.
def Outliers_Subj_Cuantification(path, data_OutSubj_df):

    # Experiments dictionary and data
    general_proc_data_df = data_OutSubj_df[1]
    
    # Preprocessing data
    data2_df = general_proc_data_df.drop(columns = ['Exp_name', 'Block', 'Trial', 'Assigned_Stim', 
                                                    'Relative_beep', 'Time', 'Asyn_zeroed', 'Outlier_trial_meanAsyn', 
                                                    'Outlier_trial_std'])

    # Total subjects per experiment
    n_subjects_df = data2_df.drop(columns = ['General_condition', "Condition", 'Outlier_trial_porcPrevCond', 'Outlier_subj_meanAsyn', 'Outlier_subj_std', 'Outlier_subj_porcPrevCond'])
    n_subjects_df["Total_subjects"] = 1
    n_subjects_df = n_subjects_df.groupby(["Experiment", "Subject"], as_index = False).agg(Total_subjects = ("Total_subjects", "max"))
    n_subjects_df = n_subjects_df.groupby(["Experiment"], as_index = False).agg(Total_subjects = ("Total_subjects", "sum"))

    # Per group condition, percentage of complete subjects discarded
    data3_df = data2_df[(data2_df['Outlier_subj_porcPrevCond'] == 1)]
    data3_df = data3_df.drop(columns = ['Outlier_trial_porcPrevCond', 'Outlier_subj_meanAsyn', 'Outlier_subj_std']).reset_index(drop = True)    
    result_df = data3_df.groupby(["Experiment", "Condition", "Subject"], as_index = False).agg(N_subjects = ("Outlier_subj_porcPrevCond", "max"))
    result_df = result_df.groupby(["Experiment", "Condition"], as_index = False).agg(Total_outlier_subjects = ("N_subjects", "sum"))
    result_df = pd.merge(n_subjects_df, result_df, on=["Experiment"]).reset_index(drop = True)
    result_df["Total_outlier_subjects_porc"] = result_df["Total_outlier_subjects"] * 100 / result_df["Total_subjects"]
    result_df = result_df.reindex(columns=['Experiment', 'Condition', 'Total_subjects', 'Total_outlier_subjects', 'Total_outlier_subjects_porc'])

    # Per group subject, conditions removed.
    data4_df = data2_df[(data2_df['Outlier_trial_porcPrevCond'] == 1) | (data2_df['Outlier_subj_meanAsyn'] == 1) | (data2_df['Outlier_subj_std'] == 1) | 
                        (data2_df['Outlier_subj_porcPrevCond'] == 1)].reset_index(drop = True)
    result2_df = data4_df.groupby(["Experiment", "Subject", "Condition"], as_index = False).agg(Outlier_trial_porcPrevCond = ("Outlier_trial_porcPrevCond", "max"),
                                                                                                Outlier_subj_meanAsyn = ("Outlier_subj_meanAsyn", "max"),
                                                                                                Outlier_subj_std = ("Outlier_subj_std", "max"),
                                                                                                Outlier_subj_porcPrevCond = ("Outlier_subj_porcPrevCond", "max"))

    # Save Files.
    result_df.to_csv('./outlier_metrics/' + "porc_outlier_subjects_perExp_perCond.csv", na_rep = np.NaN)
    result2_df.to_csv('./outlier_metrics/' + "perExp_perSubj_perCond_What_is_the_outlier_condition_criteria.csv", na_rep = np.NaN)


#%% Group_Subject_Condition_Outlier_Subject
# Function to obtain meanasyn and stdasyn for each group subject condition.
def Group_Subject_Condition_Outlier_Subject(data_OutSubj_df):

    # Experiments dictionary and data
    general_cond_dict_df = data_OutSubj_df[0]
    general_proc_WithoutOutlierTrialsAndSubjCond_df = data_OutSubj_df[2]

    # Searching data for experiment_type
    particular_exp_data2_df = general_proc_WithoutOutlierTrialsAndSubjCond_df[(general_proc_WithoutOutlierTrialsAndSubjCond_df['Outlier_subj_porcPrevCond'] == 0)].reset_index(drop = True)
    particular_exp_data3_df = general_proc_WithoutOutlierTrialsAndSubjCond_df[(general_proc_WithoutOutlierTrialsAndSubjCond_df['Outlier_subj_porcPrevCond'] == 1)].reset_index(drop = True)     

    # Mean valid trials asyn all subjects per conditions and mean of the last ones
    perSubjAve2_df = (particular_exp_data2_df
                          # Average across trials without outlier trials and outlier subj cond.
                          .groupby(["Exp_name", "Experiment", "Subject", "General_condition", "Condition", "Relative_beep"], as_index=False)
                          .agg(mean_asyn=("Asyn_zeroed","mean"),std_asyn=("Asyn_zeroed","std"), sem_asyn=("Asyn_zeroed","sem"), n_asyn=("Asyn_zeroed","size"), 
                               ci_asyn=("Asyn_zeroed", lambda value: 1.96 * st.sem(value, axis=None))))
    
    perSubjAve3_df = (particular_exp_data3_df
                          # Average across trials, only outlier subj cond.
                          .groupby(["Exp_name", "Experiment", "Subject", "General_condition", "Condition", "Relative_beep"], as_index=False)
                          #.ass(mean_asyn=np.mean(particular_exp_data3_df["Asyn_zeroed"]), n_asyn=np.size(particular_exp_data3_df["Asyn_zeroed"])))  #, sem_asyn=np.std(particular_exp_data3_df["Asyn_zeroed"])/np.sqrt(n_asyn)))
                          # .agg(mean_asyn=("Asyn_zeroed","mean"),sem_asyn=("Asyn_zeroed","1.96 * sem")))
                          #.agg(mean_asyn=("Asyn_zeroed","mean"),sem_asyn=("Asyn_zeroed","sem")))
                          #.agg(mean_asyn=("Asyn_zeroed","mean"),sem_asyn=("Asyn_zeroed", "sem"), CI=("Asyn_zeroed", lambda value: st.sem(value, axis=None))))
                          .agg(mean_asyn=("Asyn_zeroed","mean"),std_asyn=("Asyn_zeroed","std"), sem_asyn=("Asyn_zeroed","sem"), n_asyn=("Asyn_zeroed","size"), 
                               ci_asyn=("Asyn_zeroed", lambda value: 1.96 * st.sem(value, axis=None))))


    return general_cond_dict_df, perSubjAve2_df, perSubjAve3_df
    

#%%










#%% Calculating Difference.


#%% Difference_Between_Same_Condition_Different_Experiments_BootstrappingPerSubject
# Function to get difference between same condition different experiments calculating p-value using bootstrapping per subject.
# path --> string (ej: '../analysis/'). data_GroupSubjCond_OS_df --> dataframe. experiment_name_1 --> string (ej: 'Experiment_SC'). 
# experiment_name_2 --> string (ej: 'Experiment_PS_SC'). experiment_condition --> string (ej: 'SCneg'). perturb_size --> int (ej: 50). 
# relative_beep_ini --> int (ej: 1). relative_beep_final --> int (ej: 6). figure_number --> int (ej: 1). histogram --> boolean (ej: True).
def Difference_Between_Same_Condition_Different_Experiments_BootstrappingPerSubject(path, data_GroupSubjCond_OS_df, experiment_name_1, experiment_name_2, experiment_condition, perturb_size, relative_beep_ini, relative_beep_final, figure_number):

    # Experiments dictionary and data
    experiment_dictionary_df = data_GroupSubjCond_OS_df[0]
    experiment_data_df = data_GroupSubjCond_OS_df[1]

    # Search dictionay for experiment_type    
    particular_exp_dic_df = experiment_dictionary_df[(experiment_dictionary_df['Exp_name'] == experiment_name_1) | 
                                                     (experiment_dictionary_df['Exp_name'] == experiment_name_2)]
    particular_exp_dic_df = particular_exp_dic_df[particular_exp_dic_df.Name.str.contains(experiment_condition)]
    particular_exp_dic_df = particular_exp_dic_df[(particular_exp_dic_df['Perturb_size'] == perturb_size) | 
                                                  (particular_exp_dic_df['Perturb_size'] == -perturb_size)]
    particular_exp_dic_df.reset_index(drop = True, inplace = True)

    # Searching data for experiment_type
    n_gen_cond = particular_exp_dic_df['General_condition'].tolist()
    particular_exp_data_df = pd.DataFrame()
    for gen_cond in n_gen_cond:
        particular_exp_data_aux_df = (experiment_data_df[experiment_data_df['General_condition'] == gen_cond]).reset_index(drop = True)
        particular_exp_data_df = pd.concat([particular_exp_data_df, particular_exp_data_aux_df])
    particular_exp_data_df.reset_index(drop = True, inplace = True)

    # Mean data across subjects
    groupCond_data_df = (particular_exp_data_df 
                              # then average across subjects
                              .groupby(["Exp_name", "Experiment", "General_condition", "Condition", "Relative_beep"], as_index=False)
                              .agg(Asyn_zeroed = ("mean_asyn", "mean"), std_asyn=("mean_asyn","std"), sem_asyn = ("mean_asyn", "sem"), 
                                   n_asyn=("Subject","size"), ci_asyn=("mean_asyn", lambda value: 1.96 * st.sem(value, axis=None))))

    # Difference real value
    condition_min_df = particular_exp_dic_df[particular_exp_dic_df['Exp_name'] == experiment_name_1].reset_index(drop = True)
    condition_min = condition_min_df['General_condition'][0]
    minuendo_df = groupCond_data_df[groupCond_data_df['General_condition'] == condition_min].reset_index(drop = True)
    condition_sub_df = particular_exp_dic_df[particular_exp_dic_df['Exp_name'] == experiment_name_2].reset_index(drop = True)
    condition_sub = condition_sub_df['General_condition'][0]
    subtrahend_df = groupCond_data_df[groupCond_data_df['General_condition'] == condition_sub].reset_index(drop = True)
    difference_df = pd.DataFrame()
    difference_df['Relative_beep'] = minuendo_df['Relative_beep']
    difference_df['Asyn_zeroed'] = minuendo_df['Asyn_zeroed'] - subtrahend_df['Asyn_zeroed']
    #difference_df['sem_asyn'] = minuendo_df['sem_asyn'] + subtrahend_df['sem_asyn']
    ci_asyn_list = []
    for i in range(len(minuendo_df)):
        n1 = minuendo_df['n_asyn'][i]
        S1 = minuendo_df['std_asyn'][i]
        n2 = subtrahend_df['n_asyn'][i]
        S2 = subtrahend_df['std_asyn'][i]
        #Sdiff = (((S1**2)/n1) + ((S2**2)/n2))**0.5
        Sdiff = ((((n1-1)*(S1**2))+((n2-1)*(S2**2))) / ((n1-1)+(n2-1)))**0.5
        #ndiff = (((((S1**2)/n1) + ((S2**2)/n2))**2) / (((((S1**2)/n1)**2) / (n1-1)) + ((((S2**2)/n2)**2) / (n2-1)))) + 2    
        ndiff = ((((S1**2)/n1) + ((S2**2)/n2))**2) / (((((S1**2)/n1)**2) / (n1-1)) + ((((S2**2)/n2)**2) / (n2-1)))
        ci_asyn = 1.96 * (Sdiff / (ndiff**0.5))
        ci_asyn_list.append(ci_asyn)
    difference_df['ci_asyn'] = ci_asyn_list
        
    # Particular data per experiment
    particular_exp1_data_df = particular_exp_data_df[particular_exp_data_df['General_condition'] == condition_min]
    particular_exp2_data_df = particular_exp_data_df[particular_exp_data_df['General_condition'] == condition_sub]

    # Number of subjects
    n_subjects_exp1 = len((pd.unique(particular_exp1_data_df['Subject'])).tolist())
    n_subjects_exp2 = len((pd.unique(particular_exp2_data_df['Subject'])).tolist())
    n_subjects = n_subjects_exp1 + n_subjects_exp2

    # Subject unique
    particular_exp_data2_df = particular_exp_data_df.reset_index(drop = True)
    particular_exp_data2_df["Subject_unique"] = 0
    particular_exp_data2_df = particular_exp_data2_df.groupby(["Experiment", "General_condition", "Subject"], as_index=False).agg(Subject_unique = ("Subject_unique", "min"))
    particular_exp_data2_df["Subject_unique"] = list(range(n_subjects))
    particular_exp_data2_df.drop(columns = ['General_condition'], inplace = True)
    particular_exp_data3_df = pd.merge(particular_exp_data_df, particular_exp_data2_df, on=["Experiment", "Subject"]).reset_index(drop = True)

    # Bootstrapping
    n_iterations = 5000
    difference_totalFake_df = pd.DataFrame()
    for i in range(n_iterations):
        # Random data
        data_df = particular_exp_data3_df.reset_index(drop = True)
        data_df.set_index(['Subject_unique'], inplace = True) 
        data_df['General_condition'] = data_df.loc[np.random.permutation(data_df.index.unique())][['General_condition']].values

        # Mean fake data across subjects
        groupCond_fake_data_df = (data_df.groupby(["General_condition", "Relative_beep"], as_index=False)
               .agg(Asyn_zeroed = ("mean_asyn", "mean"), sem_asyn = ("mean_asyn", "sem")))
    
        # Difference fake value
        minuendo_fake_df = groupCond_fake_data_df[groupCond_fake_data_df['General_condition'] == condition_min].reset_index(drop = True)
        subtrahend_fake_df = groupCond_fake_data_df[groupCond_fake_data_df['General_condition'] == condition_sub].reset_index(drop = True)
        difference_fake_df = pd.DataFrame()
        difference_fake_df['Relative_beep'] = minuendo_fake_df['Relative_beep']
        difference_fake_df['Asyn_zeroed'] = minuendo_fake_df['Asyn_zeroed'] - subtrahend_fake_df['Asyn_zeroed']
        difference_fake_df['sem_asyn'] = minuendo_fake_df['sem_asyn'] + subtrahend_fake_df['sem_asyn']
        difference_fake_df['Number'] = i 
        difference_totalFake_df = pd.concat([difference_totalFake_df, difference_fake_df]).reset_index(drop = True)  

    # Merge real values and fake values
    data4_df = difference_df.reset_index(drop = True)
    #data4_df.drop(columns = ['sem_asyn'], inplace = True)
    data4_df.drop(columns = ['ci_asyn'], inplace = True)
    data4_df.rename(columns={"Asyn_zeroed": "Real_asyn_zeroed"}, inplace = True)    
    data5_df = pd.merge(data4_df, difference_totalFake_df, on=["Relative_beep"]).reset_index(drop = True)
    data5B_df = difference_totalFake_df.groupby(["Relative_beep"], as_index=False).agg(Mean_asyn_zeroed = ("Asyn_zeroed", "mean"))
    data5_df = pd.merge(data5_df, data5B_df, on=["Relative_beep"]).reset_index(drop = True)

    # P-value
    data6_df = data5_df.reset_index(drop = True)
    data6_df["False_pos_diff"] = 0
    data6_df.loc[(data6_df.Real_asyn_zeroed > data6_df.Mean_asyn_zeroed) & (data6_df.Asyn_zeroed > data6_df.Real_asyn_zeroed),'False_pos_diff'] = 1 
    data6_df.loc[(data6_df.Real_asyn_zeroed < data6_df.Mean_asyn_zeroed) & (data6_df.Asyn_zeroed < data6_df.Real_asyn_zeroed),'False_pos_diff'] = 1  
    data7_df = data6_df.groupby(["Relative_beep"], as_index=False).agg(False_pos_diff = ("False_pos_diff", "sum"))
    data7_df["n_iterations"] = n_iterations
    data7_df["p_value"] = data7_df["False_pos_diff"] / data7_df["n_iterations"]
    
    # Merge p_values
    data8_df = data7_df.reset_index(drop = True)
    data8_df.drop(columns = ['False_pos_diff', 'n_iterations'], inplace = True)

    # Applying FDR correction ((Benjamini/Hochberg (non-negative))
    data10_df = data7_df[(data7_df['Relative_beep'] >= relative_beep_ini) & (data7_df['Relative_beep'] <= relative_beep_final)].reset_index(drop = True)    
    pvalues_list = data10_df['p_value'].tolist()
    rejected_array, pvalue_corrected_array = sm.fdrcorrection(pvalues_list, alpha = 0.05, method = 'indep', is_sorted = False)
    data10_df['pvalue_corrected'] = pvalue_corrected_array
    data10_df['rejected'] = rejected_array
    data10_df.loc[data10_df.rejected == False, 'rejected'] = 0
    data10_df.loc[data10_df.rejected == True, 'rejected'] = 1
    data10_df['rejected'] = data10_df['rejected'].astype('int')

    # P_values state
    data11_df = data10_df.drop(columns = ['False_pos_diff', 'n_iterations', 'p_value', 'pvalue_corrected'])
    n_relativeBeep = sorted((pd.unique(data7_df['Relative_beep'])).tolist())
    data12_df = pd.DataFrame({"Relative_beep" : n_relativeBeep})
    data12_df = data12_df[(data12_df['Relative_beep'] < relative_beep_ini) | (data12_df['Relative_beep'] > relative_beep_final)]
    data12_df['rejected'] = 0
    data13_df = pd.concat([data11_df, data12_df]).reset_index(drop = True)  

    # Difference with p_values state
    data14_df = pd.merge(difference_df, data13_df, on=["Relative_beep"]).reset_index(drop = True)
    
    # Difference for plots
    data15_df = data14_df.reset_index(drop = True)
    data15_df['Experiment_type'] = experiment_condition[0:2]
    data15_df['Experiment_sign'] = experiment_condition[2:5]
    data15_df['Title'] = experiment_condition[0:2] + ' ' + experiment_condition[2:5]
    data15_df = data15_df.reindex(columns=['Experiment_type', 'Experiment_sign', 'Title', 'Relative_beep', 'Asyn_zeroed', 'ci_asyn', 'rejected'])
    
    # Difference min and subs for plots
    data16_df = minuendo_df.reset_index(drop = True)
    data17_df = pd.merge(data16_df, data13_df, on=["Relative_beep"]).reset_index(drop = True)
    if experiment_name_1 == 'Experiment_PS':
        experiment_context = 'Pure'
    elif experiment_name_1 == 'Experiment_SC':
        experiment_context = 'Pure'
    elif experiment_name_1 == 'Experiment_PS_SC': 
        experiment_context = 'Combined'
    elif experiment_name_1 == 'Experiment2_PS_SC': 
        experiment_context = 'Combined'
    data17_df['Experiment_context'] = experiment_context
    data17_df['Experiment_type'] = experiment_condition[0:2]
    data17_df['Experiment_sign'] = experiment_condition[2:5]
    data17_df['Title'] = experiment_condition[0:2] + ' ' + experiment_condition[2:5]
    data17_df = data17_df.reindex(columns=['Experiment_context', 'Experiment_type', 'Experiment_sign', 'Title', 'Relative_beep', 'Asyn_zeroed', 'ci_asyn', 'rejected'])
    data18_df = subtrahend_df.reset_index(drop = True)
    data19_df = pd.merge(data18_df, data13_df, on=["Relative_beep"]).reset_index(drop = True)
    if experiment_name_2 == 'Experiment_PS':
        experiment_context = 'Pure'
    elif experiment_name_2 == 'Experiment_SC':
        experiment_context = 'Pure'
    elif experiment_name_2 == 'Experiment_PS_SC': 
        experiment_context = 'Combined'
    elif experiment_name_2 == 'Experiment2_PS_SC': 
        experiment_context = 'Combined'
    data19_df['Experiment_context'] = experiment_context
    data19_df['Experiment_type'] = experiment_condition[0:2]
    data19_df['Experiment_sign'] = experiment_condition[2:5]
    data19_df['Title'] = experiment_condition[0:2] + ' ' + experiment_condition[2:5]
    data19_df = data19_df.reindex(columns=['Experiment_context', 'Experiment_type', 'Experiment_sign', 'Title', 'Relative_beep', 'Asyn_zeroed', 'ci_asyn', 'rejected'])
    data20_df = pd.concat([data17_df, data19_df]).reset_index(drop = True)

    return data15_df, data20_df


#%% Difference
# Function to get difference between same condition different experiments.
# path --> string (ej: '../analysis/'). data_GroupSubjCond_OS_df --> dataframe. difference_list --> list (ej: [['Experiment_PS', 'Experiment_PS_SC', 'PSneg']]). 
# perturb_size --> int (ej: 50). relative_beep_ini --> int (ej: 1). relative_beep_final --> int (ej: 6). figure_number --> int (ej: 1).
def Difference(path, data_GroupSubjCond_OS_df, difference_list, perturb_size, relative_beep_ini, relative_beep_final, figure_number):

    # Creating difference dataframe for all difference_list.
    difference_df = pd.DataFrame()
    differenceMinSub_df = pd.DataFrame()
    for i in difference_list:
        # Parameters assigment. 
        experiment_name_1 = i[0]
        experiment_name_2 = i[1]
        experiment_condition = i[2]

        # Calling function to create auxiliar difference dataframe for each pair of elements of the difference_list.
        difference_aux_df, differenceMinSub_aux_df = Difference_Between_Same_Condition_Different_Experiments_BootstrappingPerSubject(path, data_GroupSubjCond_OS_df, experiment_name_1, experiment_name_2, experiment_condition, perturb_size, relative_beep_ini, relative_beep_final, figure_number)

        # Joing auxiliar difference dataframes. 
        difference_df = pd.concat([difference_df, difference_aux_df]).reset_index(drop = True)
        differenceMinSub_df = pd.concat([differenceMinSub_df, differenceMinSub_aux_df]).reset_index(drop = True)
        
    #difference_df.to_csv(path + "difference_df.csv", na_rep = np.NaN)
    #differenceMinSub_df.to_csv(path + "differenceMinSub_df.csv", na_rep = np.NaN)

    return difference_df, differenceMinSub_df


#%%










#%% Calculating Asymmetry.


#%% Asymmetry_Between_Opposite_Conditions_Same_Experiments_BootstrappingPerSubject
# Function to get asymmetry between opposite conditions from same experiment calculating p-value using bootstrapping per subject.
# path --> string (ej: '../analysis/'). data_GroupSubjCond_OS_df --> dataframe. experiment_name --> string (ej: 'Experiment_SC'). 
# experiment_type --> string (ej: 'SC'). perturb_size --> int (ej: 50). relative_beep_ini --> int (ej: 1). relative_beep_final --> int (ej: 6). 
# figure_number --> int (ej: 1). histogram --> boolean (ej: True).
def Asymmetry_Between_Opposite_Conditions_Same_Experiments_BootstrappingPerSubject(path, data_GroupSubjCond_OS_df, experiment_name, experiment_type, perturb_size, relative_beep_ini, relative_beep_final, figure_number):

    # Experiments dictionary and data
    experiment_dictionary_df = data_GroupSubjCond_OS_df[0]
    experiment_data_df = data_GroupSubjCond_OS_df[1]
    
    # Search dictionay for experiment_type    
    particular_exp_dic_df = experiment_dictionary_df[experiment_dictionary_df['Exp_name'] == experiment_name]
    particular_exp_dic_df = particular_exp_dic_df[particular_exp_dic_df.Name.str.contains(experiment_type)]
    particular_exp_dic_df = particular_exp_dic_df[(particular_exp_dic_df['Perturb_size'] == perturb_size) | 
                                                  (particular_exp_dic_df['Perturb_size'] == -perturb_size)]
    particular_exp_dic_df.reset_index(drop = True, inplace = True)
    
    # Searching data for experiment_type
    n_gen_cond = particular_exp_dic_df['General_condition'].tolist()
    particular_exp_data_df = pd.DataFrame()
    for gen_cond in n_gen_cond:
        particular_exp_data_aux_df = (experiment_data_df[experiment_data_df['General_condition'] == gen_cond]).reset_index(drop = True)
        particular_exp_data_df = pd.concat([particular_exp_data_df, particular_exp_data_aux_df])
    particular_exp_data_df.reset_index(drop = True, inplace = True) 

    # Mean data across subjects
    groupCond_data_df = (particular_exp_data_df 
                              # then average across subjects
                              .groupby(["Exp_name", "Experiment", "General_condition", "Condition", "Relative_beep"], as_index=False)
                              #.agg(Asyn_zeroed = ("mean_asyn", "mean"), sem_asyn = ("mean_asyn", "sem")))
                              .agg(Asyn_zeroed = ("mean_asyn", "mean"), std_asyn=("mean_asyn","std"), sem_asyn = ("mean_asyn", "sem"), 
                                   n_asyn=("Subject","size"), ci_asyn=("mean_asyn", lambda value: 1.96 * st.sem(value, axis=None))))

    # Asymmetry
    condition_addUp1_df = particular_exp_dic_df[particular_exp_dic_df.Name.str.contains("neg")].reset_index(drop = True)
    condition_addUp1 = condition_addUp1_df['General_condition'][0]
    condition_addUp2_df = particular_exp_dic_df[particular_exp_dic_df.Name.str.contains("pos")].reset_index(drop = True)
    condition_addUp2 = condition_addUp2_df['General_condition'][0]  
    addingUp1_df = groupCond_data_df[groupCond_data_df['General_condition'] == condition_addUp1].reset_index(drop = True)
    addingUp2_df = groupCond_data_df[groupCond_data_df['General_condition'] == condition_addUp2].reset_index(drop = True)
    asymmetry_df = pd.DataFrame()
    asymmetry_df['Relative_beep'] = addingUp1_df['Relative_beep'] 
    asymmetry_df['Asyn_zeroed'] = addingUp1_df['Asyn_zeroed'] + addingUp2_df['Asyn_zeroed']
    #asymmetry_df['sem_asyn'] = addingUp1_df['sem_asyn'] + addingUp2_df['sem_asyn']
    ci_asyn_list = []
    for i in range(len(addingUp1_df)):
        n1 = addingUp1_df['n_asyn'][i]
        S1 = addingUp1_df['std_asyn'][i]
        n2 = addingUp2_df['n_asyn'][i]
        S2 = addingUp2_df['std_asyn'][i]
        #Sdiff = (((S1**2)/n1) + ((S2**2)/n2))**0.5
        Sdiff = ((((n1-1)*(S1**2))+((n2-1)*(S2**2))) / ((n1-1)+(n2-1)))**0.5
        #ndiff = (((((S1**2)/n1) + ((S2**2)/n2))**2) / (((((S1**2)/n1)**2) / (n1-1)) + ((((S2**2)/n2)**2) / (n2-1)))) + 2    
        ndiff = ((((S1**2)/n1) + ((S2**2)/n2))**2) / (((((S1**2)/n1)**2) / (n1-1)) + ((((S2**2)/n2)**2) / (n2-1)))
        ci_asyn = 1.96 * (Sdiff / (ndiff**0.5))
        ci_asyn_list.append(ci_asyn)
    asymmetry_df['ci_asyn'] = ci_asyn_list
   
    # Particular data per experiment
    particular_exp1_data_df = particular_exp_data_df[particular_exp_data_df['General_condition'] == condition_addUp1]
    particular_exp2_data_df = particular_exp_data_df[particular_exp_data_df['General_condition'] == condition_addUp2]

    # Number of subjects per experiment
    n_subjects_exp1 = len((pd.unique(particular_exp1_data_df['Subject'])).tolist())
    n_subjects_exp2 = len((pd.unique(particular_exp2_data_df['Subject'])).tolist())
    n_subjects = n_subjects_exp1 + n_subjects_exp2

    # Invert the sign to the second group
    particular_exp2B_data_df = particular_exp2_data_df.reset_index(drop = True)
    particular_exp2B_data_df['mean_asyn'] = - particular_exp2B_data_df['mean_asyn']         
    particular_exp_data2_df = pd.concat([particular_exp1_data_df, particular_exp2B_data_df]).reset_index(drop = True)
    
    # Subject unique
    particular_exp_data3_df = particular_exp_data2_df.reset_index(drop = True)
    particular_exp_data3_df["Subject_unique"] = 0
    particular_exp_data3_df = particular_exp_data3_df.groupby(["Experiment", "General_condition", "Subject"], as_index=False).agg(Subject_unique = ("Subject_unique", "min"))
    particular_exp_data3_df["Subject_unique"] = list(range(n_subjects))
    particular_exp_data3_df.drop(columns = ['Experiment'], inplace = True)
    particular_exp_data4_df = pd.merge(particular_exp_data2_df, particular_exp_data3_df, on=["General_condition", "Subject"]).reset_index(drop = True)
    
    # Bootstrapping
    n_iterations = 5000
    asymmetry_totalFake_df = pd.DataFrame()
    for i in range(n_iterations):
        # Random data
        data_df = particular_exp_data4_df.reset_index(drop = True)
        data_df.set_index(['Subject_unique'], inplace = True) 
        data_df['General_condition'] = data_df.loc[np.random.permutation(data_df.index.unique())][['General_condition']].values
        
        # Mean fake data across subjects
        groupCond_fake_data_df = (data_df.groupby(["General_condition", "Relative_beep"], as_index=False)
               .agg(Asyn_zeroed = ("mean_asyn", "mean"), sem_asyn = ("mean_asyn", "sem")))
        
        # Asymmetry fake value
        addingUp1_fake_df = groupCond_fake_data_df[groupCond_fake_data_df['General_condition'] == condition_addUp1].reset_index(drop = True)
        addingUp2_fake_df = groupCond_fake_data_df[groupCond_fake_data_df['General_condition'] == condition_addUp2].reset_index(drop = True)
        asymmetry_fake_df = pd.DataFrame()
        asymmetry_fake_df['Relative_beep'] = addingUp1_fake_df['Relative_beep'] 
        asymmetry_fake_df['Asyn_zeroed'] = addingUp1_fake_df['Asyn_zeroed'] - addingUp2_fake_df['Asyn_zeroed']
        asymmetry_fake_df['sem_asyn'] = addingUp1_fake_df['sem_asyn'] + addingUp2_fake_df['sem_asyn']
        asymmetry_fake_df['Number'] = i
        asymmetry_totalFake_df = pd.concat([asymmetry_totalFake_df, asymmetry_fake_df]).reset_index(drop = True)
        
    # Merge real values and fake values
    data4_df = asymmetry_df.reset_index(drop = True)
    data4_df.drop(columns = ['ci_asyn'], inplace = True)
    data4_df.rename(columns={"Asyn_zeroed": "Real_asyn_zeroed"}, inplace = True)
    data5_df = pd.merge(data4_df, asymmetry_totalFake_df, on=["Relative_beep"]).reset_index(drop = True)
    data5B_df = asymmetry_totalFake_df.groupby(["Relative_beep"], as_index=False).agg(Mean_asyn_zeroed = ("Asyn_zeroed", "mean"))
    data5_df = pd.merge(data5_df, data5B_df, on=["Relative_beep"]).reset_index(drop = True)

    # P-value
    data6_df = data5_df.reset_index(drop = True)
    data6_df["False_pos_diff"] = 0
    data6_df.loc[(data6_df.Real_asyn_zeroed > data6_df.Mean_asyn_zeroed) & (data6_df.Asyn_zeroed > data6_df.Real_asyn_zeroed),'False_pos_diff'] = 1 
    data6_df.loc[(data6_df.Real_asyn_zeroed < data6_df.Mean_asyn_zeroed) & (data6_df.Asyn_zeroed < data6_df.Real_asyn_zeroed),'False_pos_diff'] = 1 
    data7_df = data6_df.groupby(["Relative_beep"], as_index=False).agg(False_pos_diff = ("False_pos_diff", "sum"))
    data7_df["n_iterations"] = n_iterations
    data7_df["p_value"] = data7_df["False_pos_diff"] / data7_df["n_iterations"]

    # Merge p_values
    data8_df = data7_df.reset_index(drop = True)
    data8_df.drop(columns = ['False_pos_diff', 'n_iterations'], inplace = True)

    # Applying FDR correction ((Benjamini/Hochberg (non-negative))
    data10_df = data7_df[(data7_df['Relative_beep'] >= relative_beep_ini) & (data7_df['Relative_beep'] <= relative_beep_final)].reset_index(drop = True)
    pvalues_list = data10_df['p_value'].tolist()
    rejected_array, pvalue_corrected_array = sm.fdrcorrection(pvalues_list, alpha = 0.05, method = 'indep', is_sorted = False)
    data10_df['pvalue_corrected'] = pvalue_corrected_array
    data10_df['rejected'] = rejected_array
    data10_df.loc[data10_df.rejected == False, 'rejected'] = 0
    data10_df.loc[data10_df.rejected == True, 'rejected'] = 1
    data10_df['rejected'] = data10_df['rejected'].astype('int')

    # P_values state
    data11_df = data10_df.drop(columns = ['False_pos_diff', 'n_iterations', 'p_value', 'pvalue_corrected'])
    n_relativeBeep = sorted((pd.unique(data7_df['Relative_beep'])).tolist())
    data12_df = pd.DataFrame({"Relative_beep" : n_relativeBeep})
    data12_df = data12_df[(data12_df['Relative_beep'] < relative_beep_ini) | (data12_df['Relative_beep'] > relative_beep_final)]
    data12_df['rejected'] = 0
    data13_df = pd.concat([data11_df, data12_df]).reset_index(drop = True)  

    # Asymmetry with p_values state
    data14_df = pd.merge(asymmetry_df, data13_df, on=["Relative_beep"]).reset_index(drop = True)

    # Asymmetry for plots
    if experiment_name == 'Experiment_PS':
        experiment_context = 'Pure'
    elif experiment_name == 'Experiment_SC':
        experiment_context = 'Pure'
    elif experiment_name == 'Experiment_PS_SC': 
        experiment_context = 'Combined'
    elif experiment_name == 'Experiment2_PS_SC': 
        experiment_context = 'Combined'
    data15_df = data14_df.reset_index(drop = True)
    data15_df['Experiment_context'] = experiment_context
    data15_df['Experiment_type'] = experiment_type
    data15_df['Title'] = experiment_context + ' ' + experiment_type
    data15_df = data15_df.reindex(columns=['Experiment_context', 'Experiment_type', 'Title', 'Relative_beep', 'Asyn_zeroed', 'ci_asyn', 'rejected'])

    # Asymmetry add1 and add2 for plots
    data16_df = addingUp1_df.reset_index(drop = True)
    data17_df = pd.merge(data16_df, data13_df, on=["Relative_beep"]).reset_index(drop = True)
    if experiment_name == 'Experiment_PS':
        experiment_context = 'Pure'
    elif experiment_name == 'Experiment_SC':
        experiment_context = 'Pure'
    elif experiment_name == 'Experiment_PS_SC': 
        experiment_context = 'Combined'
    elif experiment_name == 'Experiment2_PS_SC': 
        experiment_context = 'Combined'
    data17_df['Experiment_context'] = experiment_context
    data17_df['Experiment_type'] = experiment_type
    data17_df['Experiment_sign'] = "neg"
    data17_df['Title'] = experiment_context + ' ' + experiment_type
    data17_df = data17_df.reindex(columns=['Experiment_context', 'Experiment_type', 'Experiment_sign', 'Title', 'Relative_beep', 'Asyn_zeroed', 'ci_asyn', 'rejected'])
    data18_df = addingUp2_df.reset_index(drop = True)
    data19_df = pd.merge(data18_df, data13_df, on=["Relative_beep"]).reset_index(drop = True)
    if experiment_name == 'Experiment_PS':
        experiment_context = 'Pure'
    elif experiment_name == 'Experiment_SC':
        experiment_context = 'Pure'
    elif experiment_name == 'Experiment_PS_SC': 
        experiment_context = 'Combined'
    elif experiment_name == 'Experiment2_PS_SC': 
        experiment_context = 'Combined'
    data19_df['Experiment_context'] = experiment_context
    data19_df['Experiment_type'] = experiment_type
    data19_df['Experiment_sign'] = "pos"
    data19_df['Title'] = experiment_context + ' ' + experiment_type
    data19_df = data19_df.reindex(columns=['Experiment_context', 'Experiment_type', 'Experiment_sign', 'Title', 'Relative_beep', 'Asyn_zeroed', 'ci_asyn', 'rejected'])
    data20_df = pd.concat([data17_df, data19_df]).reset_index(drop = True)

    return data15_df, data20_df


#%% Asymmetry
# Function to get asymmetry between opposite conditions from same experiment.
# path --> string (ej: '../analysis/'). data_GroupSubjCond_OS_df --> dataframe. asymmetry_list --> list (ej: [['Experiment_PS', 'PS']]).
# perturb_size --> int (ej: 50). relative_beep_ini --> int (ej: 1). relative_beep_final --> int (ej: 6). figure_number --> int (ej: 1).
def Asymmetry(path, data_GroupSubjCond_OS_df, asymmetry_list, perturb_size, relative_beep_ini, relative_beep_final, figure_number):

    # Creating asymmetry dataframe for all difference_list.
    asymmetry_df = pd.DataFrame()
    asymmetryAdd1Add2_df = pd.DataFrame()
    for i in asymmetry_list:
        # Parameters assigment. 
        experiment_name = i[0]
        experiment_type = i[1]

        # Calling function to create auxiliar difference dataframe for each pair of elements of the difference_list.
        asymmetry_aux_df, asymmetryAdd1Add2_aux_df = Asymmetry_Between_Opposite_Conditions_Same_Experiments_BootstrappingPerSubject(path, data_GroupSubjCond_OS_df, experiment_name, experiment_type, perturb_size, relative_beep_ini, relative_beep_final, figure_number)

        # Joing auxiliar difference dataframes. 
        asymmetry_df = pd.concat([asymmetry_df, asymmetry_aux_df]).reset_index(drop = True)
        asymmetryAdd1Add2_df = pd.concat([asymmetryAdd1Add2_df, asymmetryAdd1Add2_aux_df]).reset_index(drop = True)
    
    #asymmetry_df.to_csv(path + "asymmetry_df.csv", na_rep = np.NaN)
    #asymmetryAdd1Add2_df.to_csv(path + "asymmetryAdd1Add2_df.csv", na_rep = np.NaN)
    
    return asymmetry_df, asymmetryAdd1Add2_df


#%% Group_Condition_Mean_Postperturb_Transient
# Function to obtain mean_asyn and ci_asyn for each group condition in postperturb resynchronization transient (RelativeBeep 1-6).
# Group_Subject_Condition_Outlier_Subject --> data_GroupSubjCond_OS_df. perturb_size --> int (ej: 50).
def Group_Condition_Mean_Postperturb_Transient(data_GroupSubjCond_OS_df, perturb_size):

    # General experiment dictionary
    data_GSC_OS_dict_df = data_GroupSubjCond_OS_df[0].reset_index(drop = True)
    data_GSC_OS_dict_df = data_GSC_OS_dict_df[['General_condition', 'Perturb_type', 'Perturb_size']]

    # Experiment data (Group - Subject - Condition)
    data_GSC_OS_df = data_GroupSubjCond_OS_df[1].reset_index(drop = True)
    data_GSC_OS_df = pd.merge(data_GSC_OS_df, data_GSC_OS_dict_df, on=["General_condition"]).reset_index(drop = True)
    data_GSC_OS_df.query("(Relative_beep >= 1 & Relative_beep <= 6) & (Perturb_size == -@perturb_size | Perturb_size == @perturb_size)", inplace=True)
    data_GSC_OS_df.insert(0, 'Experiment_context', np.select([data_GSC_OS_df['Exp_name']=='Experiment_PS', 
                                                             data_GSC_OS_df['Exp_name']=='Experiment_SC', 
                                                             data_GSC_OS_df['Exp_name']=='Experiment_PS_SC',
                                                             data_GSC_OS_df['Exp_name']=='Experiment2_PS_SC'], 
                                                            ['Pure', 'Pure', 'Combined', 'Combined']))
    data_GSC_OS_df.insert(1, 'Experiment_type', np.where(data_GSC_OS_df['Perturb_type']==0,'SC','PS'))
    data_GSC_OS_df.insert(2, 'Experiment_sign', np.where(data_GSC_OS_df['Perturb_size']==perturb_size,'pos','neg'))
    data_GSC_OS_df.drop(columns = ['Exp_name', 'Perturb_type', 'Perturb_size'], inplace = True)
    #data_GSC_OS_df.to_csv(path + "data_GSC_OS_df.csv", na_rep = np.NaN)
    
    # Mean data across subjects-relativebeeps
    group_data_df = (data_GSC_OS_df 
                               # then average across subjects
                               .groupby(["Experiment_context", "Experiment_type", "Experiment_sign"], as_index=False)
                               .agg(Asyn_zeroed = ("mean_asyn", "mean"), std_asyn=("mean_asyn","std"), sem_asyn = ("mean_asyn", "sem"), 
                                    n_asyn=("Subject","size"), ci_asyn=("mean_asyn", lambda value: 1.96 * st.sem(value, axis=None))))
    #group_data_df.to_csv(path + "group_data_df.csv", na_rep = np.NaN)
    
    #Mean data across subject-sign-relativebeeps
    group_data2_df = (data_GSC_OS_df 
                               # then average across subjects
                               .groupby(["Experiment_context", "Experiment_type"], as_index=False)
                               .agg(Asyn_zeroed = ("mean_asyn", "mean"), std_asyn=("mean_asyn","std"), sem_asyn = ("mean_asyn", "sem"), 
                                    n_asyn=("Subject","size"), ci_asyn=("mean_asyn", lambda value: 1.96 * st.sem(value, axis=None))))
    #group_data2_df.to_csv(path + "group_data2_df.csv", na_rep = np.NaN)
    
    # Mean data across subjects-type-relativebeeps
    group_data3_df = (data_GSC_OS_df 
                               # then average across subjects
                               .groupby(["Experiment_context", "Experiment_sign"], as_index=False)
                               .agg(Asyn_zeroed = ("mean_asyn", "mean"), std_asyn=("mean_asyn","std"), sem_asyn = ("mean_asyn", "sem"), 
                                    n_asyn=("Subject","size"), ci_asyn=("mean_asyn", lambda value: 1.96 * st.sem(value, axis=None))))
    #group_data3_df.to_csv(path + "group_data3_df.csv", na_rep = np.NaN)

    #Mean data across subject-context-relativebeeps pos-inverted
    data_GSC_OS_inv_df = (data_GSC_OS_df
                          # switch sign of asynchrony in positive perturbations
                          .assign(mean_asyn = np.select([data_GSC_OS_df['Experiment_sign']=='pos', data_GSC_OS_df['Experiment_sign']=='neg'],
                                                          [-data_GSC_OS_df['mean_asyn'], data_GSC_OS_df['mean_asyn']]))
                          )
    group_data4_df = (data_GSC_OS_inv_df 
                               # then average across subjects
                               .groupby(["Experiment_sign", "Experiment_type"], as_index=False)
                               .agg(Asyn_zeroed = ("mean_asyn", "mean"), std_asyn=("mean_asyn","std"), sem_asyn = ("mean_asyn", "sem"), 
                                    n_asyn=("Subject","size"), ci_asyn=("mean_asyn", lambda value: 1.96 * st.sem(value, axis=None))))
    #group_data4_df.to_csv(path + "group_data4_df.csv", na_rep = np.NaN)

    return group_data_df, group_data2_df, group_data3_df, group_data4_df 


#%%










#%% Plotting


#%% Plot_Differences
# Function to plot difference results.
# difference_df --> dataframe.
def Plot_Differences(data_GroupCond_MPT_df, difference_df):

    # Parameters
    line_map = ["solid","dashed"]
    shape_map = ["s","D"]
    marker_size = 2
    error_width = 0.1
    fig_xsize = 15 * 0.393701   # centimeter to inch
    fig_ysize = 9 * 0.393701   # centimeter to inch
    fig_xsize2 = 5 * 0.393701   # centimeter to inch
    fig_ysize2 = 9 * 0.393701   # centimeter to inch
    x_lims = [-3,11]

    # Filtering information
    difference_df = difference_df[(difference_df['Relative_beep'] >= x_lims[0]) & (difference_df['Relative_beep'] <= x_lims[1])]
    difference_df.rename(columns={"Experiment_sign": "Sign", "Experiment_type": "Type"}, inplace=True)

    # Plotting
    plot_d = (ggplot(difference_df, aes(x = 'Relative_beep', y = 'Asyn_zeroed', group = 'Title', 
                                    linetype = 'Sign', shape = 'Type'))
              + geom_line()
              + geom_errorbar(aes(x = 'Relative_beep', ymin = "Asyn_zeroed-ci_asyn", ymax = "Asyn_zeroed+ci_asyn", width = error_width))
              + geom_point(size = marker_size)
              + scale_linetype_manual(values = line_map, guide=False)
              + scale_shape_manual(values = shape_map, guide=False)
              + scale_x_continuous(breaks=range(x_lims[0]+1,x_lims[1],2))
              + theme_bw(base_size=14)
              + theme(legend_key=element_rect(fill = "white", color = 'white'), figure_size = (fig_xsize, fig_ysize))  
              + themes.theme(
                  axis_title_y = themes.element_text(angle = 90, va = 'center', size = 12),
                  axis_title_x = themes.element_text(va = 'center', size = 12))
              + xlab("Beep $n$ relative to perturbation")
              + ylab("Difference (ms)")
              #+ ggtitle("(a)") #For 50ms
              + ggtitle("(b)") #For 20ms
              )
    #print(plot_d)
    #plot_d.save('../analysis/' + 'plot_d.pdf')

    # Difference in postperturb resynchronization transient (RelativeBeep 1-6).
    data_df = data_GroupCond_MPT_df.reset_index(drop=True)
    
    data_df.loc[data_df.Experiment_context == "Combined", 'Asyn_zeroed'] *= -1 
    data2_df = (data_df.groupby(["Experiment_type", "Experiment_sign"], as_index=False)
                #.apply(lambda df: pd.Series({'S_pooled': (((df.std_asyn**2)/df.n_asyn).sum())**0.5,
                .apply(lambda df: pd.Series({'S_pooled': ((((df.n_asyn-1)*df.std_asyn**2).sum())/((df.n_asyn-1).sum()))**0.5,
                                             #'n_pooled': (((((df.std_asyn**2)/df.n_asyn).sum())**2) / (((((df.std_asyn**2)/df.n_asyn)**2) / (df.n_asyn-1)).sum())) + 2,
                                             'n_pooled': ((df.std_asyn**2/df.n_asyn).sum())**2/(((df.std_asyn**2/df.n_asyn)**2/(df.n_asyn-1)).sum()),
                                             'Asyn_zeroed': (df.Asyn_zeroed).sum()})))
    data2_df["ci_asyn"] = 1.96 * (data2_df["S_pooled"] / (data2_df["n_pooled"]**0.5))
    
    # Defining title and categories
    data2_df['Title'] = data2_df['Experiment_type'] + ' ' + data2_df['Experiment_sign']
    data2_df.loc[data2_df.Title == "PS pos", 'Title'] = "PS\npos"
    data2_df.loc[data2_df.Title == "SC neg", 'Title'] = "SC\nneg"
    data2_df.loc[data2_df.Title == "SC pos", 'Title'] = "SC\npos"
    data2_df.loc[data2_df.Title == "PS neg", 'Title'] = "PS\nneg"
    data2_cat = pd.Categorical(data2_df['Title'], categories=['PS\npos', 'SC\nneg', 'SC\npos','PS\nneg'])
    data2_df = data2_df.assign(Title = data2_cat)
    data2_df.rename(columns={"Experiment_sign": "Sign", "Experiment_type": "Type"}, inplace=True)
    
    # Plotting
    plot_e = (
                ggplot(data2_df, aes(x = 'Title', y = 'Asyn_zeroed', linetype = 'Sign', shape = 'Type'))
              + geom_point()
              + geom_errorbar(aes(x = 'Title', ymin = "Asyn_zeroed-ci_asyn", ymax = "Asyn_zeroed+ci_asyn", width = error_width))
              + scale_linetype_manual(values = line_map)
              + scale_shape_manual(values = shape_map)
              + theme_bw(base_size=14)
              + theme(legend_key=element_rect(fill = "white", color = 'white'), figure_size = (fig_xsize2, fig_ysize2))  
              + themes.theme(
                  axis_title_y = themes.element_text(angle = 90, va = 'center', size = 12),
                  axis_title_x = themes.element_text(va = 'center', size = 12))
              + xlab("Condition")
              + ylab("Average difference (ms) (beeps 1 through 6)")
              #+ ggtitle("(b)")
              )
    #print(plot_e)
    #plot_e.save('../analysis/' + 'plot_e.pdf')

    # Plotting
    #plot_d2 = pw.load_ggplot(plot_d)
    #plot_e2 = pw.load_ggplot(plot_e)
    #plot_de = plot_d2|plot_e2
    #plot_de.savefig('../analysis/' + 'S1.pdf')
    
    return plot_d, plot_e


#%% Plot_Asymmetries
# Function to plot asymmetry results.
# asymmetry_df --> dataframe.
def Plot_Asymmetries(data_GroupCond_MPT_df, asymmetry_df):
    
    # Parameters
    #color_map = ["blue","magenta"]
    lower_color = 0
    upper_color = 3
    num_colors = 5
    color_map_rgba = cm.get_cmap('plasma')(np.linspace(lower_color,upper_color,num_colors))
    color_map_hex = [rgb2hex(color, keep_alpha=True) for color in color_map_rgba.tolist()]    
    shape_map = ["s","D"]
    marker_size = 2
    error_width = 0.1
    fig_xsize = 15 * 0.393701   # centimeter to inch
    fig_ysize = 9 * 0.393701   # centimeter to inch
    fig_xsize2 = 5 * 0.393701   # centimeter to inch
    fig_ysize2 = 9 * 0.393701   # centimeter to inch
    x_lims = [-3,11]

    # Filtering information
    asymmetry_df = asymmetry_df[(asymmetry_df['Relative_beep'] >= x_lims[0]) & (asymmetry_df['Relative_beep'] <= x_lims[1])]
    asymmetry_df.rename(columns={"Experiment_context": "Context", "Experiment_type": "Type"}, inplace=True)
    #asymmetry_df.to_csv('../analysis/' + "asymmetry_df.csv", na_rep = np.NaN)
    
    # Plotting
    plot_a = (ggplot(asymmetry_df, aes(x = 'Relative_beep', y = 'Asyn_zeroed', group = 'Title', 
                                   color = 'Context', shape = 'Type'))
              + geom_line()
              + geom_errorbar(aes(x = 'Relative_beep', ymin = "Asyn_zeroed-ci_asyn", ymax = "Asyn_zeroed+ci_asyn", width = error_width))
              + geom_point(size = marker_size)
              + scale_color_manual(values = color_map_hex, guide=False)
              + scale_shape_manual(values = shape_map, guide=False)
              + scale_x_continuous(breaks=range(x_lims[0]+1,x_lims[1],2))
              + theme_bw(base_size=14)
              + theme(legend_key=element_rect(fill = "white", color = 'white'), figure_size = (fig_xsize, fig_ysize))
              + themes.theme(
                  axis_title_y = themes.element_text(angle = 90, va = 'center', size = 12),
                  axis_title_x = themes.element_text(va = 'center', size = 12))
              + xlab("Beep $n$ relative to perturbation")
              + ylab("Asymmetry (ms)")
              + ggtitle("(a)")
              )
    #print(plot_a)
    #plot_a.save('../analysis/' + 'plot_a.pdf')
    
    # Asynchrony in postperturb resynchronization transient (RelativeBeep 1-6).
    data_df = data_GroupCond_MPT_df.reset_index(drop=True)
    data2_df = (data_df.groupby(["Experiment_context", "Experiment_type"], as_index=False)
                #.apply(lambda df: pd.Series({'S_pooled': (((df.std_asyn**2)/df.n_asyn).sum())**0.5,
                .apply(lambda df: pd.Series({'S_pooled': ((((df.n_asyn-1)*df.std_asyn**2).sum())/((df.n_asyn-1).sum()))**0.5,
                                             #'n_pooled': (((((df.std_asyn**2)/df.n_asyn).sum())**2) / (((((df.std_asyn**2)/df.n_asyn)**2) / (df.n_asyn-1)).sum())) + 2,
                                             'n_pooled': ((df.std_asyn**2/df.n_asyn).sum())**2/(((df.std_asyn**2/df.n_asyn)**2/(df.n_asyn-1)).sum()),
                                             'Asyn_zeroed': (df.Asyn_zeroed).sum()})))
    data2_df["ci_asyn"] = 1.96 * (data2_df["S_pooled"] / (data2_df["n_pooled"]**0.5))

    # Defining title and categories
    data2_df['Title'] = data2_df['Experiment_context'] + ' ' + data2_df['Experiment_type']
    data2_df.rename(columns={"Experiment_context": "Context", "Experiment_type": "Type"}, inplace=True)
    data2_df.loc[data2_df.Title == "Pure PS", 'Title'] = "pure\nPS"
    data2_df.loc[data2_df.Title == "Combined PS", 'Title'] = "comb\nPS"
    data2_df.loc[data2_df.Title == "Combined SC", 'Title'] = "comb\nSC"
    data2_df.loc[data2_df.Title == "Pure SC", 'Title'] = "pure\nSC"
    data2_cat = pd.Categorical(data2_df['Title'], categories=['pure\nPS', 'comb\nPS', 'comb\nSC','pure\nSC'])
    data2_df = data2_df.assign(Title = data2_cat)
    data2_df.loc[data2_df.Context == "Combined", 'Context'] = "comb"
    data2_df.loc[data2_df.Context == "Pure", 'Context'] = "pure"
    
    # Plotting
    plot_b = (
                ggplot(data2_df, aes(x = 'Title', y = 'Asyn_zeroed', color = 'Context', shape = 'Type'))
              + geom_point()
              + geom_errorbar(aes(x = 'Title', ymin = "Asyn_zeroed-ci_asyn", ymax = "Asyn_zeroed+ci_asyn", width = error_width))
              + scale_color_manual(values = color_map_hex)
              + scale_shape_manual(values = shape_map)
              + theme_bw(base_size=14)
              + theme(legend_key=element_rect(fill = "white", color = 'white'), figure_size = (fig_xsize2, fig_ysize2))
              + themes.theme(
                  axis_title_y = themes.element_text(angle = 90, va = 'center', size = 12),
                  axis_title_x = themes.element_text(va = 'center', size = 12))
              + xlab("Condition")
              + ylab("Average asymmetry (ms) (beeps 1 through 6)")
              + ggtitle("(b)")
              )
    #print(plot_b)
    #plot_b.save('../analysis/' + 'plot_b.pdf')

    # Plotting
    plot_a2 = pw.load_ggplot(plot_a)
    plot_b2 = pw.load_ggplot(plot_b)
    plot_ab = plot_a2|plot_b2
    #plot_ab.savefig('../analysis/' + 'figure_4.pdf')
    
    return plot_ab


#%% Plot_Mean_Across_Subjects_to_Calculate_Difference
# Function to plot mean across subject data to calculate difference.
# differenceMinSub_df --> dataframe.
def Plot_Mean_Across_Subjects_to_Calculate_Difference(differenceMinSub_df):

    # Parameters
    #color_map = ["blue","magenta"]
    lower_color = 0
    upper_color = 3
    num_colors = 5
    color_map_rgba = cm.get_cmap('plasma')(np.linspace(lower_color,upper_color,num_colors))
    color_map_hex = [rgb2hex(color, keep_alpha=True) for color in color_map_rgba.tolist()]  
    line_map = ["solid","dashed"]
    shape_map = ["s","D"]
    marker_size = 1
    ast_size = 1
    error_width = 0.1
    fig_xsize = 20 * 0.393701   # centimeter to inch
    fig_ysize = 12 * 0.393701   # centimeter to inch
    x_lims = [-3,11]
    rejected_scale = 40

    # Filtering and ordering information
    differenceMinSub_df = differenceMinSub_df[(differenceMinSub_df['Relative_beep'] >= x_lims[0]) & 
                                              (differenceMinSub_df['Relative_beep'] <= x_lims[1])]
    differenceMinSub_df.rejected = differenceMinSub_df.rejected.multiply(rejected_scale)
    differenceMinSub_df.rename(columns={"Experiment_context": "Context", "Experiment_sign": "Sign", "Experiment_type": "Type"}, inplace=True)
    differenceMinSub_df.loc[differenceMinSub_df.Context == "Combined", 'Context'] = "comb"
    differenceMinSub_df.loc[differenceMinSub_df.Context == "Pure", 'Context'] = "pure"
    #differenceMinSub_df.to_csv('../analysis/' + "differenceMinSub_df.csv", na_rep = np.NaN)
    
    # Plotting
    plot_f = (
                ggplot(differenceMinSub_df) 
                + aes(x = 'Relative_beep', y = 'Asyn_zeroed',
                      color = 'Context',
                      linetype = 'Sign',
                      shape = 'Type')
                + facet_grid(facets="Sign~Type")
                + geom_line()
                + geom_errorbar(aes(x = 'Relative_beep', ymin = "Asyn_zeroed-ci_asyn", ymax = "Asyn_zeroed+ci_asyn", width = error_width))
                + geom_point(size = marker_size)
                + geom_point(differenceMinSub_df[differenceMinSub_df['rejected'] == rejected_scale], 
                             aes(x='Relative_beep', y="rejected"),
                             shape="*", size=ast_size, color="black")
                + scale_color_manual(values = color_map_hex)
                + scale_linetype_manual(values = line_map)
                + scale_shape_manual(values = shape_map)
                + scale_x_continuous(breaks=range(x_lims[0]+1,x_lims[1],2))
                + theme_bw(base_size=12)
                + theme(legend_title = element_text(size=9),
                        legend_text=element_text(size=9),
                        legend_key=element_rect(fill = "white", color = 'white'), 
                        figure_size = (fig_xsize, fig_ysize))
                + themes.theme(
                    axis_title_y = themes.element_text(angle = 90, va = 'center', size = 10),
                    axis_title_x = themes.element_text(va = 'center', size = 10))
                #+ theme(strip_background = element_rect(fill = "white", color = "white"))
                + theme(strip_background = element_blank())
                + xlab("Beep $n$ relative to perturbation")
                + ylab("Asynchrony (ms)")
                + ggtitle("(a)")   # For 20ms
                )
    #print(plot_f)
    #plot_f.save('../analysis/' + 'figure_2.pdf')
    
    return plot_f


#%% Plot_Mean_Across_Subjects_to_Calculate_Asymmetry
# Function to plot mean across subject data to calculate asymmetry.
# asymmetryAdd1Add2_df --> dataframe.
def Plot_Mean_Across_Subjects_to_Calculate_Asymmetry(asymmetryAdd1Add2_df):

    # Parameters
    #color_map = ["blue","magenta"]
    lower_color = 0
    upper_color = 3
    num_colors = 5
    color_map_rgba = cm.get_cmap('plasma')(np.linspace(lower_color,upper_color,num_colors))
    color_map_hex = [rgb2hex(color, keep_alpha=True) for color in color_map_rgba.tolist()]  
    line_map = ["solid","dashed"]
    shape_map = ["s","D"]
    marker_size = 1
    ast_size = 1
    error_width = 0.1
    fig_xsize = 20 * 0.393701   # centimeter to inch
    fig_ysize = 12 * 0.393701   # centimeter to inch
    x_lims = [-3,11]
    rejected_scale = 40

    # Filtering and ordering information
    asymmetryAdd1Add2_df = asymmetryAdd1Add2_df[(asymmetryAdd1Add2_df['Relative_beep'] >= x_lims[0]) & (asymmetryAdd1Add2_df['Relative_beep'] <= x_lims[1])]
    asymmetryAdd1Add2_df.rejected = asymmetryAdd1Add2_df.rejected.multiply(rejected_scale)
    asymmetryAdd1Add2_df.rename(columns={"Experiment_context": "Context", "Experiment_sign": "Sign", "Experiment_type": "Type"}, inplace=True)
    asymmetryAdd1Add2_df.loc[asymmetryAdd1Add2_df.Context == "Combined", 'Context'] = "comb"
    asymmetryAdd1Add2_df.loc[asymmetryAdd1Add2_df.Context == "Pure", 'Context'] = "pure"
    #asymmetryAdd1Add2_df.to_csv('../analysis/' + "asymmetryAdd1Add2_df.csv", na_rep = np.NaN)

    # Plotting
    plot_g = (
                ggplot(asymmetryAdd1Add2_df) 
                + aes(x = 'Relative_beep', y = 'Asyn_zeroed',
                      color = 'Context',
                      linetype = 'Sign',
                      shape = 'Type')
                + facet_grid(facets="Context~Type")
                + geom_line()
                + geom_errorbar(aes(x = 'Relative_beep', ymin = "Asyn_zeroed-ci_asyn", ymax = "Asyn_zeroed+ci_asyn", width = error_width))
                + geom_point(size = marker_size)
                + geom_point(asymmetryAdd1Add2_df[asymmetryAdd1Add2_df['rejected'] == rejected_scale],
                             aes(x='Relative_beep', y="rejected"), 
                             shape="*", size=ast_size, color="black") 
                + scale_color_manual(values = color_map_hex)
                + scale_linetype_manual(values = line_map)
                + scale_shape_manual(values = shape_map)
                + scale_x_continuous(breaks=range(x_lims[0]+1,x_lims[1],2))
                + theme_bw(base_size=12)
                + theme(legend_title = element_text(size=9),
                        legend_text=element_text(size=9),
                        legend_key=element_rect(fill = "white", color = 'white'), 
                        figure_size = (fig_xsize, fig_ysize))
                + themes.theme(
                    axis_title_y = themes.element_text(angle = 90, va = 'center', size = 10),
                    axis_title_x = themes.element_text(va = 'center', size = 10))
                #+ theme(strip_background = element_rect(fill = "white", color = "white"))
                + theme(strip_background = element_blank())
                + xlab("Beep $n$ relative to perturbation")
                + ylab("Asynchrony (ms)")
                )
    #print(plot_g)
    #plot_g.save('../analysis/' + 'plot_g.pdf')


#%% Plot_Mean_Across_Subjects_to_Calculate_Asymmetry_PosInverted
# Function to plot mean across subject data to calculate asymmetry.
# asymmetryAdd1Add2_df --> dataframe.
def Plot_Mean_Across_Subjects_to_Calculate_Asymmetry_PosInverted(asymmetryAdd1Add2_df):

    # Parameters
    #color_map = ["blue","magenta"]
    lower_color = 0
    upper_color = 3
    num_colors = 5
    color_map_rgba = cm.get_cmap('plasma')(np.linspace(lower_color,upper_color,num_colors))
    color_map_hex = [rgb2hex(color, keep_alpha=True) for color in color_map_rgba.tolist()]  
    line_map = ["solid","dashed"]
    shape_map = ["s","D"]
    marker_size = 1
    ast_size = 1
    error_width = 0.1
    fig_xsize = 20 * 0.393701   # centimeter to inch
    fig_ysize = 12 * 0.393701   # centimeter to inch
    x_lims = [-3,11]
    rejected_scale = 40

    # Filtering and ordering information
    asymmetryAdd1Add2_df = asymmetryAdd1Add2_df[(asymmetryAdd1Add2_df['Relative_beep'] >= x_lims[0]) & (asymmetryAdd1Add2_df['Relative_beep'] <= x_lims[1])]
    asymmetryAdd1Add2_df.rejected = asymmetryAdd1Add2_df.rejected.multiply(rejected_scale)
    asymmetryAdd1Add2_df.loc[asymmetryAdd1Add2_df.Experiment_sign == "pos", 'Asyn_zeroed'] *= -1
    asymmetryAdd1Add2_df.loc[asymmetryAdd1Add2_df.Experiment_sign == "pos", 'Experiment_sign'] = "(-)pos"
    asymmetryAdd1Add2_df.rename(columns={"Experiment_context": "Context", "Experiment_sign": "Sign", "Experiment_type": "Type"}, inplace=True)
    asymmetryAdd1Add2_cat = pd.Categorical(asymmetryAdd1Add2_df['Sign'], categories=['neg', '(-)pos'], ordered=True)
    asymmetryAdd1Add2_df = asymmetryAdd1Add2_df.assign(Sign = asymmetryAdd1Add2_cat)
    asymmetryAdd1Add2_df.loc[asymmetryAdd1Add2_df.Context == "Combined", 'Context'] = "comb"
    asymmetryAdd1Add2_df.loc[asymmetryAdd1Add2_df.Context == "Pure", 'Context'] = "pure"
    #asymmetryAdd1Add2_df.to_csv('../analysis/' + "asymmetryAdd1Add2_df.csv", na_rep = np.NaN)

    # Plotting
    plot_h = (
                ggplot(asymmetryAdd1Add2_df) 
                + aes(x = 'Relative_beep', y = 'Asyn_zeroed',
                      color = 'Context',
                      linetype = 'Sign',
                      shape = 'Type')
                + facet_grid(facets="Context~Type")
                + geom_line()
                + geom_errorbar(aes(x = 'Relative_beep', ymin = "Asyn_zeroed-ci_asyn", ymax = "Asyn_zeroed+ci_asyn", width = error_width))
                + geom_point(size = marker_size)
                + geom_point(asymmetryAdd1Add2_df[asymmetryAdd1Add2_df['rejected'] == rejected_scale],
                             aes(x='Relative_beep', y="rejected"), 
                             shape="*", size=ast_size, color="black")
                + scale_color_manual(values = color_map_hex)
                + scale_linetype_manual(values = line_map)
                + scale_shape_manual(values = shape_map)
                + scale_x_continuous(breaks=range(x_lims[0]+1,x_lims[1],2))
                + theme_bw(base_size=12)
                + theme(legend_title = element_text(size=9),
                        legend_text=element_text(size=9),
                        legend_key=element_rect(fill = "white", color = 'white'), 
                        figure_size = (fig_xsize, fig_ysize))
                + themes.theme(
                    axis_title_y = themes.element_text(angle = 90, va = 'center', size = 10),
                    axis_title_x = themes.element_text(va = 'center', size = 10))
                #+ theme(strip_background = element_rect(fill = "white", color = "white"))
                + theme(strip_background = element_blank())
                + xlab("Beep $n$ relative to perturbation")
                + ylab("Asynchrony (ms)")
                + ggtitle("(b)")   # For 20ms                      
                )
    #print(plot_h)
    #plot_h.save('../analysis/' + 'figure_3.pdf')
    
    return plot_h


#%% Plot_Mean_Across_Subjects
# Function to plot mean across subject data.
# differenceMinSub_df --> dataframe.
def Plot_Mean_Across_Subjects(differenceMinSub_df):
        
    # Parameters
    #color_map = ["blue","magenta"]
    lower_color = 0
    upper_color = 3
    num_colors = 5
    color_map_rgba = cm.get_cmap('plasma')(np.linspace(lower_color,upper_color,num_colors))
    color_map_hex = [rgb2hex(color, keep_alpha=True) for color in color_map_rgba.tolist()]  
    line_map = ["solid","dashed"]
    shape_map = ["s","D"]
    marker_size = 1
    error_width = 0.1
    fig_xsize = 20 * 0.393701   # centimeter to inch
    fig_ysize = 12 * 0.393701   # centimeter to inch
    x_lims = [-3,11]

    # Filtering and ordering information
    differenceMinSub_df = differenceMinSub_df[(differenceMinSub_df['Relative_beep'] >= x_lims[0]) & (differenceMinSub_df['Relative_beep'] <= x_lims[1])]
    differenceMinSub_df.rename(columns={"Experiment_context": "Context", "Experiment_sign": "Sign", "Experiment_type": "Type"}, inplace=True)
    differenceMinSub_df.loc[differenceMinSub_df.Context == "Combined", 'Context'] = "comb"
    differenceMinSub_df.loc[differenceMinSub_df.Context == "Pure", 'Context'] = "pure"

    # Plotting
    plot_i = (
                ggplot(differenceMinSub_df)
                + aes(x = 'Relative_beep', y = 'Asyn_zeroed', 
                      color = 'Context', 
                      linetype = 'Sign', 
                      shape = 'Type')
              + geom_line()
              + geom_errorbar(aes(x = 'Relative_beep', ymin = "Asyn_zeroed-ci_asyn", ymax = "Asyn_zeroed+ci_asyn", width = error_width))
              + geom_point(size = marker_size)
              + scale_color_manual(values = color_map_hex)
              + scale_linetype_manual(values = line_map)
              + scale_shape_manual(values = shape_map)
              + scale_x_continuous(breaks=range(x_lims[0]+1,x_lims[1],2))
              + theme_bw(base_size=14)
              + theme(legend_title = element_text(size=11),
                      legend_text=element_text(size=11),
                      legend_key=element_rect(fill = "white", color = 'white'), 
                      figure_size = (fig_xsize, fig_ysize))
              + themes.theme(
                  axis_title_y = themes.element_text(angle = 90, va = 'center', size = 12),
                  axis_title_x = themes.element_text(va = 'center', size = 12))
              + xlab("Beep $n$ relative to perturbation")
              + ylab("Asynchrony (ms)")
            )
    #print(plot_i)
    #plot_i.save('../analysis/' + 'plot_i.pdf')    


#%% Plot_Mean_Across_Subjects_for_Type_Sign_and_Context
# Function to plot mean across subject data Relative beep 1 to 6 for Type, Sign and Context.
def Plot_Mean_Across_Subjects_for_Type_Sign_and_Context(data_GroupCond_MPT_df, data_GroupCond2_MPT_df, data_GroupCond3_MPT_df, data_GroupCond4_MPT_df):

    # Parameters
    lower_color = 0
    upper_color = 3
    num_colors = 5
    color_map_rgba = cm.get_cmap('plasma')(np.linspace(lower_color,upper_color,num_colors))
    color_map_hex = [rgb2hex(color, keep_alpha=True) for color in color_map_rgba.tolist()]
    line_map = ["solid","dashed"]
    shape_map = ["s","D"]
    marker_size = 1
    error_width = 0.02
    fig_xsize = 6 * 0.393701   # centimeter to inch
    fig_ysize = 9 * 0.393701   # centimeter to inch


    # DIFFERENCE
    # Data for plotting x = 'Context-Sign-Type'
    data_df = data_GroupCond_MPT_df.reset_index(drop=True)
    data_df.rename(columns={"Experiment_context": "Context", "Experiment_sign": "Sign", "Experiment_type": "Type"}, inplace=True)
    data_df['Title'] = data_df['Type'] + ' ' + data_df['Sign']
    data_df.loc[data_df.Context == "Combined", 'Context'] = "comb"
    data_df.loc[data_df.Context == "Pure", 'Context'] = "pure"
 
    # Plotting 'Context-Sign-Type'
    plot_l = (
                ggplot(data_df)
                + aes(x = 'Context', y = 'Asyn_zeroed', 
                      group = 'Title',
                      color = 'Context', 
                      linetype = 'Sign', 
                      shape = 'Type')
                + geom_line(color="black")
                + geom_point(size = marker_size)
                + geom_errorbar(aes(x = 'Context', ymin = "Asyn_zeroed-ci_asyn", ymax = "Asyn_zeroed+ci_asyn", width = error_width))
                + scale_color_manual(values = color_map_hex)
                + scale_linetype_manual(values = line_map)
                + scale_shape_manual(values = shape_map)
                + theme_bw(base_size=14)
                + theme(legend_title = element_text(size=11),
                        legend_text=element_text(size=11),
                        legend_key=element_rect(fill = "white", color = 'white'), 
                        figure_size = (fig_xsize, fig_ysize))
                + themes.theme(
                    axis_title_y = themes.element_text(angle = 90, va = 'center', size = 12),
                    axis_title_x = themes.element_text(va = 'center', size = 12))
                + xlab("Context")
                + ylab("Average asynchrony (ms) (beeps 1 through 6)")
                + ggtitle("(c)")
                )
    #print(plot_l)
    #plot_l.save('../analysis/' + 'plot_l.pdf')
        
    # Data for plotting x = 'Context-Type'
    data_df = data_GroupCond2_MPT_df.reset_index(drop=True)
    data_df.rename(columns={"Experiment_context": "Context", "Experiment_type": "Type"}, inplace=True)
    data_df.loc[data_df.Context == "Combined", 'Context'] = "comb"
    data_df.loc[data_df.Context == "Pure", 'Context'] = "pure"
    #data_df.to_csv(path + "data_df.csv", na_rep = np.NaN)

    # Plotting 'Context-Type'
    plot_m = (
                ggplot(data_df)
                + aes(x = 'Context', y = 'Asyn_zeroed', 
                      group = 'Type',
                      color = 'Context', 
                      shape = 'Type')
                + geom_line(color="black", linetype="dotted")
                + geom_point(size = marker_size)
                + geom_errorbar(aes(x = 'Context', ymin = "Asyn_zeroed-ci_asyn", ymax = "Asyn_zeroed+ci_asyn", width = error_width))
                + scale_color_manual(values = color_map_hex)
                + scale_shape_manual(values = shape_map)
                + theme_bw(base_size=14)
                + theme(legend_title = element_text(size=11),
                        legend_text=element_text(size=11),
                        legend_key=element_rect(fill = "white", color = 'white'), 
                        figure_size = (fig_xsize, fig_ysize))
                + themes.theme(
                    axis_title_y = themes.element_text(angle = 90, va = 'center', size = 12),
                    axis_title_x = themes.element_text(va = 'center', size = 12))
                + xlab("Context")
                + ylab("Average asynchrony (ms) (beeps 1 through 6)")
                + ggtitle("(a)")
                )
    #print(plot_m)
    #plot_m.save('../analysis/' + 'plot_m.pdf')
    
    # Data for plotting x = 'Context-Sign'
    data_df = data_GroupCond3_MPT_df.reset_index(drop=True)
    data_df.rename(columns={"Experiment_context": "Context", "Experiment_sign": "Sign"}, inplace=True)
    data_df.loc[data_df.Context == "Combined", 'Context'] = "comb"
    data_df.loc[data_df.Context == "Pure", 'Context'] = "pure"
    #data_df.to_csv(path + "data_df.csv", na_rep = np.NaN)

    # Plotting 'Context-Sign'
    plot_n = (
                ggplot(data_df)
                + aes(x = 'Context', y = 'Asyn_zeroed', 
                      group = 'Sign',
                      color = 'Context', 
                      linetype = 'Sign') 
                + geom_line(color="black")
                + geom_point(size = marker_size)
                + geom_errorbar(aes(x = 'Context', ymin = "Asyn_zeroed-ci_asyn", ymax = "Asyn_zeroed+ci_asyn", width = error_width))
                + scale_color_manual(values = color_map_hex)
                + scale_linetype_manual(values = line_map)
                + theme_bw(base_size=14)
                + theme(legend_title = element_text(size=11),
                        legend_text=element_text(size=11),
                        legend_key=element_rect(fill = "white", color = 'white'), 
                        figure_size = (fig_xsize, fig_ysize))
                + themes.theme(
                    axis_title_y = themes.element_text(angle = 90, va = 'center', size = 12),
                    axis_title_x = themes.element_text(va = 'center', size = 12))
                + xlab("Context")
                + ylab("Average asynchrony (ms) (beeps 1 through 6)")
                + ggtitle("(b)")
                )
    #print(plot_n)
    #plot_n.save('../analysis/' + 'plot_n.pdf')


    #ASYMMETRY
    data_inv_df = (data_GroupCond_MPT_df
                   # switch sign of asynchrony in positive perturbations
                   .assign(Asyn_zeroed = np.select([data_GroupCond_MPT_df['Experiment_sign']=='pos', data_GroupCond_MPT_df['Experiment_sign']=='neg'],
											 [-data_GroupCond_MPT_df['Asyn_zeroed'], data_GroupCond_MPT_df['Asyn_zeroed']]))
                   )
    #data_inv_df.to_csv(path + "data_inv_df.csv", na_rep = np.NaN)
    
    # Data for plotting x = 'Sign-Context-Type'
    data_df = data_inv_df.reset_index(drop=True)
    data_df.rename(columns={"Experiment_context": "Context", "Experiment_sign": "Sign", "Experiment_type": "Type"}, inplace=True)
    data_df['Title'] = data_df['Context'] + ' ' + data_df['Type']
    data_df.loc[data_df.Context == "Combined", 'Context'] = "comb"
    data_df.loc[data_df.Context == "Pure", 'Context'] = "pure"
    data_df.loc[data_df.Sign == "pos", 'Sign'] = "(-)pos"
    data_cat = pd.Categorical(data_df['Sign'], categories=['neg', '(-)pos'], ordered=True)
    data_df = data_df.assign(Sign = data_cat)
    
    # Plotting 'Sign-Context-Type'
    plot_o = (
                ggplot(data_df)
                + aes(x = 'Sign', y = 'Asyn_zeroed',
                      group = 'Title',
                      color = 'Context', 
                      linetype = 'Sign', 
                      shape = 'Type')
                + geom_line(color="black", linetype="dotted")
                + geom_point(size = marker_size)
                + geom_errorbar(aes(x = 'Sign', ymin = "Asyn_zeroed-ci_asyn", ymax = "Asyn_zeroed+ci_asyn", width = error_width))
                + scale_color_manual(values = color_map_hex)
                + scale_linetype_manual(values = line_map)
                + scale_shape_manual(values = shape_map)
                + theme_bw(base_size=14)
                + theme(legend_title = element_text(size=11),
                        legend_text=element_text(size=11),
                        legend_key=element_rect(fill = "white", color = 'white'), 
                        figure_size = (fig_xsize, fig_ysize))
                + themes.theme(
                    axis_title_y = themes.element_text(angle = 90, va = 'center', size = 12),
                    axis_title_x = themes.element_text(va = 'center', size = 12))
                + xlab("Sign")
                + ylab("Average asynchrony (ms) (beeps 1 through 6)")
                + ggtitle("(c)")
                )
    #print(plot_o)
    #plot_o.save('../analysis/' + 'plot_o.pdf')

    # Data for plotting x = 'Sign-Type'
    data_df = data_GroupCond4_MPT_df.reset_index(drop=True)
    data_df.rename(columns={"Experiment_sign": "Sign", "Experiment_type": "Type"}, inplace=True)

    # Plotting 'Sign-Type'
    plot_p = (
                ggplot(data_df)
                + aes(x = 'Sign', y = 'Asyn_zeroed', 
                      group = 'Type',
                      linetype = 'Sign', 
                      shape = 'Type')
                + geom_line(color="black", linetype="dotted")
                + geom_point(size = marker_size)
                + geom_errorbar(aes(x = 'Sign', ymin = "Asyn_zeroed-ci_asyn", ymax = "Asyn_zeroed+ci_asyn", width = error_width))
                + scale_linetype_manual(values = line_map)
                + scale_shape_manual(values = shape_map)
                + theme_bw(base_size=14)
                + theme(legend_title = element_text(size=11),
                        legend_text=element_text(size=11),
                        legend_key=element_rect(fill = "white", color = 'white'), 
                        figure_size = (fig_xsize, fig_ysize))
                + themes.theme(
                    axis_title_y = themes.element_text(angle = 90, va = 'center', size = 12),
                    axis_title_x = themes.element_text(va = 'center', size = 12))
                + xlab("Sign")
                + ylab("Average asynchrony (ms) (beeps 1 through 6)")
                + ggtitle("(a)")
                )
#    print(plot_p)
#    plot_p.save('../analysis/' + 'plot_p.pdf')

    data_inv_df = (data_GroupCond3_MPT_df
                   # switch sign of asynchrony in positive perturbations
                   .assign(Asyn_zeroed = np.select([data_GroupCond3_MPT_df['Experiment_sign']=='pos', data_GroupCond3_MPT_df['Experiment_sign']=='neg'],
											 [-data_GroupCond3_MPT_df['Asyn_zeroed'], data_GroupCond3_MPT_df['Asyn_zeroed']]))
                   )
    #data_inv_df.to_csv(path + "data_inv_df.csv", na_rep = np.NaN)

    # Data for plotting x = 'Sign-Context'
    data_df = data_inv_df.reset_index(drop=True)
    data_df.rename(columns={"Experiment_context": "Context", "Experiment_sign": "Sign"}, inplace=True)
    data_df.loc[data_df.Context == "Combined", 'Context'] = "comb"
    data_df.loc[data_df.Context == "Pure", 'Context'] = "pure"
    data_df.loc[data_df.Sign == "pos", 'Sign'] = "(-)pos"
    data_cat = pd.Categorical(data_df['Sign'], categories=['neg', '(-)pos'], ordered=True)
    data_df = data_df.assign(Sign = data_cat)
    #data_df.to_csv(path + "data_df.csv", na_rep = np.NaN)

    # Plotting 'Sign-Context'
    plot_q = (
                ggplot(data_df)
                + aes(x = 'Sign', y = 'Asyn_zeroed', 
                      group = 'Context',
                      color = 'Context', 
                      linetype = 'Sign') 
                + geom_line(color="black", linetype="dotted")
                + geom_point(size = marker_size)
                + geom_errorbar(aes(x = 'Sign', ymin = "Asyn_zeroed-ci_asyn", ymax = "Asyn_zeroed+ci_asyn", width = error_width))
                + scale_color_manual(values = color_map_hex)
                + scale_linetype_manual(values = line_map)
                + theme_bw(base_size=14)
                + theme(legend_title = element_text(size=11),
                        legend_text=element_text(size=11),
                        legend_key=element_rect(fill = "white", color = 'white'), 
                        figure_size = (fig_xsize, fig_ysize))
                + themes.theme(
                    axis_title_y = themes.element_text(angle = 90, va = 'center', size = 12),
                    axis_title_x = themes.element_text(va = 'center', size = 12))
                + xlab("Sign")
                + ylab("Average asynchrony (ms) (beeps 1 through 6)")
                + ggtitle("(b)")
                )
    #print(plot_q)
    #plot_q.save('../analysis/' + 'plot_q.pdf')

    # Plotting
    plot_l2 = pw.load_ggplot(plot_l)
    plot_m2 = pw.load_ggplot(plot_m)
    plot_n2 = pw.load_ggplot(plot_n)
    plot_o2 = pw.load_ggplot(plot_o)
    plot_p2 = pw.load_ggplot(plot_p)
    plot_q2 = pw.load_ggplot(plot_q)
    plot_mnl = plot_m2|plot_n2|plot_l2
    #plot_mnl.savefig('../analysis/' + 'plot_mnl.pdf')
    plot_pqo = plot_p2|plot_q2|plot_o2
    #plot_pqo.savefig('../analysis/' + 'plot_pqo.pdf')


#%%










#%% Difference across subject between context per sign, type and block
# data_plot_df --> dataframe. perturb_size --> int (ej: 50).
def Difference_across_subject_between_context_per_sign_type_and_block(data_plot_df, perturb_size):

    data_plot3_df = data_plot_df.reset_index(drop=True)

    # Filtering Perturb_size
    data_plot3_df.query("Perturb_size == -@perturb_size | Perturb_size == @perturb_size", inplace=True)
    data_plot3_df.drop(columns = ['Perturb_size'], inplace = True)

    # Difference between contexts
    data_contextdiff_df = (data_plot3_df.groupby(['Type','Sign','Block','Relative_beep'], as_index=False).
                           apply(lambda df: pd.Series({'diff': np.diff(df.mean_asyn).item(),
                                                       'var_pooled': ((df.n_subj-1)*df.std_asyn**2).sum()/(df.n_subj-1).sum(),
                                                       'n_pooled': (df.std_asyn**2/df.n_subj).sum()**2/((df.std_asyn**2/df.n_subj)**2/(df.n_subj-1)).sum()
                                                       })))
    data_contextdiff_df = (data_contextdiff_df
    					   # confidence intervals
    					   .assign(ci_diff = lambda df: 1.96*np.sqrt(df.var_pooled)/np.sqrt(df.n_pooled))
    					   # create label for plot grouping
    					   .assign(Title = lambda df: df.Type + df.Sign)
    					   )

    # Parameters
    line_map = ["solid","dashed"]
    shape_map = ["s","D"]
    marker_size = 2
    error_width = 0.1
    fig_xsize = 15 * 0.393701   # centimeter to inch
    fig_ysize = 9 * 0.393701   # centimeter to inch
    fig_xsize2 = 5 * 0.393701   # centimeter to inch
    fig_ysize2 = 9 * 0.393701   # centimeter to inch
    x_lims = [-3,11]

    # Filtering and ordering information
    data_contextdiff_df = data_contextdiff_df[(data_contextdiff_df['Relative_beep'] >= x_lims[0]) & (data_contextdiff_df['Relative_beep'] <= x_lims[1])]
    data_contextdiff_df.rename(columns={"diff": "Asyn_zeroed"}, inplace=True)
    data_contextdiff_df.rename(columns={"ci_diff": "ci_asyn"}, inplace=True)
    data_contextdiff_df.rename(columns={"Block": "Block_old"}, inplace=True)
    data_contextdiff_df.insert(0, 'Block', np.select([data_contextdiff_df['Block_old']==0,
                                                      data_contextdiff_df['Block_old']==1,
                                                      data_contextdiff_df['Block_old']==2],
                                                     [1,2,3]))
    data_contextdiff_df["Block"] = data_contextdiff_df["Block"].astype('string')

    # Plotting
    plot2 = (ggplot(data_contextdiff_df, aes(x = 'Relative_beep', 
                                             y = 'Asyn_zeroed', 
                                             linetype = 'Sign', 
                                             shape = 'Type',
                                             color = 'Block'))
             + geom_line()
             + geom_errorbar(aes(x = 'Relative_beep', ymin = "Asyn_zeroed-ci_asyn", ymax = "Asyn_zeroed+ci_asyn", width = error_width))
             + geom_point(size = marker_size)
             + scale_linetype_manual(values = line_map, guide=False)
             + scale_shape_manual(values = shape_map, guide=False)
             + scale_color_grey(guide=False)
             + scale_x_continuous(breaks=range(x_lims[0]+1,x_lims[1],2))
             + theme_bw(base_size=16)
             + theme(legend_key=element_rect(fill = "white", color = 'white'), figure_size = (fig_xsize, fig_ysize))  
             + themes.theme(
                 axis_title_y = themes.element_text(angle = 90, va = 'center', size = 14),
                 axis_title_x = themes.element_text(va = 'center', size = 14))
             + xlab("Beep $n$ relative to perturbation")
             + ylab("Difference (ms)")
             + ggtitle("(a)")
             )
    #print(plot2)
    #plot2.save('../analysis/' + 'plot2.pdf')

    # Difference across subject between context per sign, type and block (mean asyn resynchronization phase)
    data_plot4_df = data_plot_df.reset_index(drop=True)

    # Filtering Perturb_size == 50
    data_plot4_df.query("Perturb_size == -@perturb_size | Perturb_size == @perturb_size", inplace=True)
    data_plot4_df.drop(columns = ['Perturb_size'], inplace = True)

    # Select resynchronization phase
    resynch_start = 1	# bip number
    resynch_end = 6
    data_plot4_df.query("(Type=='SC' and (Relative_beep>=@resynch_start and Relative_beep<=@resynch_end)) or (Type=='PS' and (Relative_beep>=@resynch_start+1 and Relative_beep<=@resynch_end))", inplace=True)

    # Difference between contexts, resynchronization phase only, after averaging across subjects and beeps
    data_contextdiff_resynch_df = (data_plot4_df
                                   # first average across subjects and beeps
                                   .groupby(['Context','Type','Sign','Block'], as_index=False)
                                   .apply(lambda df: pd.Series({
								   'mean_asyn': df.mean_asyn.mean(),
								   'std_asyn': df.mean_asyn.std(),
								   'n_asyn': df.mean_asyn.count()
								   }))
                                   # then compute diff
                                   .groupby(['Type','Sign','Block'], as_index=False)
                                   .apply(lambda df: pd.Series({
                                       'diff': np.diff(df.mean_asyn).item(),
                                       'var_pooled': (((df.n_asyn-1)*df.std_asyn**2).sum())/((df.n_asyn-1).sum()),
                                       'n_pooled': ((df.std_asyn**2/df.n_asyn).sum())**2/(((df.std_asyn**2/df.n_asyn)**2/(df.n_asyn-1)).sum())
								   }))
                                   )
    data_contextdiff_resynch_df = (data_contextdiff_resynch_df
                                   # confidence intervals
                                   .assign(ci_diff = lambda df: 1.96*np.sqrt(df['var_pooled'])/np.sqrt(df['n_pooled']))
                                   # create label for plot grouping
                                   .assign(Title = lambda df: df.Type + df.Sign)
                                   )

    # Filtering and ordering information
    data_contextdiff_resynch_df.rename(columns={"diff": "Asyn_zeroed"}, inplace=True)
    data_contextdiff_resynch_df.rename(columns={"ci_diff": "ci_asyn"}, inplace=True)
    data_contextdiff_resynch_df.rename(columns={"Block": "Block_old"}, inplace=True)
    data_contextdiff_resynch_df.insert(0, 'Block', np.select([data_contextdiff_resynch_df['Block_old']==0,
                                                              data_contextdiff_resynch_df['Block_old']==1,
                                                              data_contextdiff_resynch_df['Block_old']==2],
                                                             [1,2,3]))
    data_contextdiff_resynch_df["Block"] = data_contextdiff_resynch_df["Block"].astype('string')

    # Modifying title
    data_contextdiff_resynch_df.loc[data_contextdiff_resynch_df.Title == "PSpos", 'Title'] = "PS\npos"
    data_contextdiff_resynch_df.loc[data_contextdiff_resynch_df.Title == "SCneg", 'Title'] = "SC\nneg"
    data_contextdiff_resynch_df.loc[data_contextdiff_resynch_df.Title == "SCpos", 'Title'] = "SC\npos"
    data_contextdiff_resynch_df.loc[data_contextdiff_resynch_df.Title == "PSneg", 'Title'] = "PS\nneg"

    # Set level order
    levels_TS = ['PS\npos','SC\nneg','SC\npos','PS\nneg']
    data_contextdiff_resynch_df['Title'] = data_contextdiff_resynch_df['Title'].astype("category").cat.set_categories(levels_TS, ordered=True)

    # Plotting
    plot3 = (
                ggplot(data_contextdiff_resynch_df, aes(x = 'Title', 
                                                        y = 'Asyn_zeroed', 
                                                        linetype = 'Sign', 
                                                        shape = 'Type',
                                                        color = 'Block'))
                + geom_line(data_contextdiff_resynch_df,
                            aes(x = 'Title', 
                                y = 'Asyn_zeroed',
                                group = 'Block'), 
                                linetype="dashdot")
                + geom_point(size = marker_size)
                + geom_errorbar(aes(x = 'Title', ymin = "Asyn_zeroed-ci_asyn", ymax = "Asyn_zeroed+ci_asyn", width = error_width))
                + scale_linetype_manual(values = line_map)
                + scale_shape_manual(values = shape_map)
                + scale_color_grey()
                + theme_bw(base_size=16)
                + theme(legend_key=element_rect(fill = "white", color = 'white'), figure_size = (fig_xsize2, fig_ysize2))
                + themes.theme(
                    axis_title_y = themes.element_text(angle = 90, va = 'center', size = 14),
                    axis_title_x = themes.element_text(va = 'center', size = 14))
                + xlab("Condition")
                + ylab("Average difference (ms) (beeps 1 through 6)")
                #+ ggtitle("(b)")
                )
    #print(plot3)
    #plot3.save('../analysis/' + 'plot3.pdf')

    # Plotting
    #plot2 = pw.load_ggplot(plot2)
    #plot3 = pw.load_ggplot(plot3)
    #plot23 = plot2|plot3
    #plot23.savefig('../analysis/' + 'plot23.pdf')

    return plot2, plot3 


#%% Plotting across subject per context, size and sign (block 0, first half and second half)
# path --> string (ej: '../data_aux/'). data_GroupSubjCond_OS_df[0] --> dataframe. data_OutSubj_df[3] --> dataframe. perturb_size --> int (ej: 50).
def Plotting_across_subject_per_context_size_and_sign_block0_first_half_and_second_half(path, data_GroupSubjCond_OS_0_df, data_OutSubj_3_df, perturb_size):

    general_cond_dict_df = data_GroupSubjCond_OS_0_df
    general_cond_dict_df = general_cond_dict_df.drop(columns = ['Condition', 'Experiment', 'Exp_name'])

    general_proc_data_withoutOutliers_df = data_OutSubj_3_df
    general_proc_data_withoutOutliers_df = pd.merge(general_cond_dict_df, general_proc_data_withoutOutliers_df,
                                                    on=["General_condition"]).reset_index(drop = True)

    # Filtering per blocks
    general_proc_data_withoutOutliers_df.query("Block == 0", inplace=True)

    # Filtering per experiment
    dataA_df = pd.DataFrame()
    dataB_df = pd.DataFrame()
    n_experiments = sorted((pd.unique(general_proc_data_withoutOutliers_df['Experiment'])).tolist())
    for experiment in n_experiments:
        data_exp_df = general_proc_data_withoutOutliers_df[general_proc_data_withoutOutliers_df['Experiment'] == experiment]
        # Filtering per subject
        n_subjects = sorted((pd.unique(data_exp_df['Subject'])).tolist())
        #data_exp_df.to_csv(path + 'data_exp_df',na_rep = np.NaN)
        for subject in n_subjects:
            data_subj_df = data_exp_df[data_exp_df['Subject'] == subject]
            # Filtering per trials
            n_trials = sorted((pd.unique(data_subj_df['Trial'])).tolist())
            #data_subj_df.to_csv(path + 'data_subj_df',na_rep = np.NaN)
            a=0
            for trial in n_trials:
                if a < len(n_trials)//2:
                    data_aux_df = data_subj_df[data_subj_df['Trial'] == trial]
                    dataA_df = pd.concat([dataA_df, data_aux_df]).reset_index(drop = True)
                else:
                    data_aux_df = data_subj_df[data_subj_df['Trial'] == trial]
                    dataB_df = pd.concat([dataB_df, data_aux_df]).reset_index(drop = True)
                a=a+1
 
    # Filtering Perturb_size
    dataA_df.query("Perturb_size == -@perturb_size | Perturb_size == @perturb_size", inplace=True)
    dataB_df.query("Perturb_size == -@perturb_size | Perturb_size == @perturb_size", inplace=True)
    dataA_df.insert(0, 'Half_Block', 0)
    dataB_df.insert(0, 'Half_Block', 1)
    dataAB_df = pd.concat([dataA_df, dataB_df]).reset_index(drop = True)
    dataAB_df.to_csv(path + 'data_Block0_halves_df',na_rep = np.NaN)

    # Mean across trials
    dataA_across_trials_df = (dataA_df.groupby(["Perturb_type", "Perturb_size", "Exp_name", "Experiment", "Subject", "Block", "General_condition", "Condition", "Relative_beep"], as_index=False).
                              agg(mean_asyn=("Asyn_zeroed","mean"),std_asyn=("Asyn_zeroed","std"), sem_asyn=("Asyn_zeroed","sem"),
                                  n_asyn=("Asyn_zeroed","size"), ci_asyn=("Asyn_zeroed", lambda value: 1.96 * st.sem(value, axis=None))))
    dataB_across_trials_df = (dataB_df.groupby(["Perturb_type", "Perturb_size", "Exp_name", "Experiment", "Subject", "Block", "General_condition", "Condition", "Relative_beep"], as_index=False).
                              agg(mean_asyn=("Asyn_zeroed","mean"),std_asyn=("Asyn_zeroed","std"), sem_asyn=("Asyn_zeroed","sem"),
                                  n_asyn=("Asyn_zeroed","size"), ci_asyn=("Asyn_zeroed", lambda value: 1.96 * st.sem(value, axis=None))))
    dataA_across_trials_df.insert(0, 'Half_Block', 0)
    dataB_across_trials_df.insert(0, 'Half_Block', 1)
    dataAB_across_trials_df = pd.concat([dataA_across_trials_df, dataB_across_trials_df]).reset_index(drop = True)
    dataAB_across_trials_df.to_csv(path + 'data_Block0_across_trials_halves_df',na_rep = np.NaN)

    # Mean across subjects
    dataA_across_subjects_df = (dataA_across_trials_df.
                                groupby(["Perturb_type", "Perturb_size", "Exp_name", "Experiment", "Block", "General_condition", "Condition", "Relative_beep"], as_index=False).
                                agg(mean_asyn = ("mean_asyn", "mean"), std_asyn=("mean_asyn","std"), sem_asyn = ("mean_asyn", "sem"),
                                    n_subj=("Subject","size"), ci_asyn=("mean_asyn", lambda value: 1.96 * st.sem(value, axis=None))))
    dataB_across_subjects_df = (dataB_across_trials_df.
                                groupby(["Perturb_type", "Perturb_size", "Exp_name", "Experiment", "Block", "General_condition", "Condition", "Relative_beep"], as_index=False).
                                agg(mean_asyn = ("mean_asyn", "mean"), std_asyn=("mean_asyn","std"), sem_asyn = ("mean_asyn", "sem"),
                                    n_subj=("Subject","size"), ci_asyn=("mean_asyn", lambda value: 1.96 * st.sem(value, axis=None))))

    # Ordering data for plotting
    dataA_across_subjects_df.insert(0, 'Context', np.select([dataA_across_subjects_df['Exp_name']=='Experiment_PS',
                                                             dataA_across_subjects_df['Exp_name']=='Experiment_SC',
                                                             dataA_across_subjects_df['Exp_name']=='Experiment_PS_SC',
                                                             dataA_across_subjects_df['Exp_name']=='Experiment2_PS_SC'],
                                                            ['pure', 'pure', 'comb', 'comb']))
    dataA_across_subjects_df.insert(1, 'Type', np.where(dataA_across_subjects_df['Perturb_type']==0,'SC','PS'))
    dataA_across_subjects_df.insert(2, 'Sign', np.where(dataA_across_subjects_df['Perturb_size']>0,'pos', 'neg'))
    dataA_across_subjects_df.drop(columns = ["Experiment", 'Exp_name', 'Perturb_type', 'Perturb_size', 'General_condition', 'Condition', 'Block'], inplace = True)
    dataB_across_subjects_df.insert(0, 'Context', np.select([dataB_across_subjects_df['Exp_name']=='Experiment_PS',
                                                             dataB_across_subjects_df['Exp_name']=='Experiment_SC',
                                                             dataB_across_subjects_df['Exp_name']=='Experiment_PS_SC',
                                                             dataB_across_subjects_df['Exp_name']=='Experiment2_PS_SC'],
                                                            ['pure', 'pure', 'comb', 'comb']))
    dataB_across_subjects_df.insert(1, 'Type', np.where(dataB_across_subjects_df['Perturb_type']==0,'SC','PS'))
    dataB_across_subjects_df.insert(2, 'Sign', np.where(dataB_across_subjects_df['Perturb_size']>0,'pos', 'neg'))
    dataB_across_subjects_df.drop(columns = ["Experiment", 'Exp_name', 'Perturb_type', 'Perturb_size', 'General_condition', 'Condition', 'Block'], inplace = True)

    # Parameters
    lower_color = 0
    upper_color = 3
    num_colors = 5
    color_map_rgba = cm.get_cmap('plasma')(np.linspace(lower_color,upper_color,num_colors))
    color_map_hex = [rgb2hex(color, keep_alpha=True) for color in color_map_rgba.tolist()]  
    line_map = ["solid","dashed"]
    shape_map = ["s","D"]
    color_map_block = {'first half':'black', 'second half':'grey'}
    size_map = (0.2,0.8)
    marker_size = 2
    error_width = 0.1
    fig_xsize = 20 * 0.393701   # centimeter to inch
    fig_ysize = 12 * 0.393701   # centimeter to inch
    fig_xsize2 = 5 * 0.393701   # centimeter to inch
    fig_ysize2 = 9 * 0.393701   # centimeter to inch
    x_lims = [-3,11]

    # Filtering and ordering information
    dataA_across_subjects_df = dataA_across_subjects_df[(dataA_across_subjects_df['Relative_beep'] >= x_lims[0]) 
                                                        & (dataA_across_subjects_df['Relative_beep'] <= x_lims[1])]
    dataA_across_subjects_df.rename(columns={"mean_asyn": "Asyn_zeroed"}, inplace=True)
    dataB_across_subjects_df = dataB_across_subjects_df[(dataB_across_subjects_df['Relative_beep'] >= x_lims[0]) 
                                                        & (dataB_across_subjects_df['Relative_beep'] <= x_lims[1])]
    dataB_across_subjects_df.rename(columns={"mean_asyn": "Asyn_zeroed"}, inplace=True)

    # Creating column "Half Block" and concatenating part A and part B
    dataA_across_subjects_df.insert(0, 'Half_Block', 0)
    dataB_across_subjects_df.insert(0, 'Half_Block', 1)
    dataAB_across_subjects_df = pd.concat([dataA_across_subjects_df, dataB_across_subjects_df]).reset_index(drop = True)
    dataAB_across_subjects_df["Half_Block"] = dataAB_across_subjects_df["Half_Block"].astype('string')
    dataAB_across_subjects2_df = dataAB_across_subjects_df.reset_index(drop=True)

    # Plotting
    plotAB = (
            ggplot(dataAB_across_subjects_df) 
            + aes(x = 'Relative_beep', y = 'Asyn_zeroed',
                  color = 'Context',
                  linetype = 'Sign',
                  shape = 'Type',
                  size = 'Half_Block')
            + facet_grid(facets="Sign~Type")
            + geom_line()
            + geom_point(size = marker_size)
            + scale_color_manual(values = color_map_hex)
            + scale_linetype_manual(values = line_map)
            + scale_shape_manual(values = shape_map)
            + scale_size_manual(values=size_map)
            + scale_x_continuous(breaks=range(x_lims[0]+1,x_lims[1],2))
            + theme_bw(base_size=12)
            + theme(legend_title = element_text(size=9),
                    legend_text=element_text(size=9),
                    legend_key=element_rect(fill = "white", color = 'white'), 
                    figure_size = (fig_xsize, fig_ysize))
            + themes.theme(
                axis_title_y = themes.element_text(angle = 90, va = 'center', size = 10),
                axis_title_x = themes.element_text(va = 'center', size = 10))           
            + theme(strip_background = element_blank())
            + xlab("Beep $n$ relative to perturbation")
            + ylab("Asynchrony (ms)")
            )
    #print(plotAB)
    #plotAB.save('../analysis/' + 'plot_Block0_Half_Block.pdf')




    # Difference between contexts
    dataAB_contextdiff_df = (dataAB_across_subjects_df.groupby(['Type','Sign','Half_Block','Relative_beep'], as_index=False).
                             apply(lambda df: pd.Series({'diff': np.diff(df.Asyn_zeroed).item(),
                                                         'var_pooled': ((df.n_subj-1)*df.std_asyn**2).sum()/(df.n_subj-1).sum(),
                                                         'n_pooled': (df.std_asyn**2/df.n_subj).sum()**2/((df.std_asyn**2/df.n_subj)**2/(df.n_subj-1)).sum()
                                                         })))
    dataAB_contextdiff_df = (dataAB_contextdiff_df
                             # confidence intervals
                             .assign(ci_diff = lambda df: 1.96*np.sqrt(df.var_pooled)/np.sqrt(df.n_pooled))
                             # create label for plot grouping
                             .assign(Title = lambda df: df.Type + df.Sign)
                             )

    # Filtering and ordering information
    dataAB_contextdiff_df = dataAB_contextdiff_df[(dataAB_contextdiff_df['Relative_beep'] >= x_lims[0]) & (dataAB_contextdiff_df['Relative_beep'] <= x_lims[1])]
    dataAB_contextdiff_df.rename(columns={"diff": "Asyn_zeroed"}, inplace=True)
    dataAB_contextdiff_df.rename(columns={"ci_diff": "ci_asyn"}, inplace=True)
    dataAB_contextdiff_df["Half_Block"] = dataAB_contextdiff_df["Half_Block"].astype('int')
    dataAB_contextdiff_df.insert(0, 'Block1', np.select([dataAB_contextdiff_df['Half_Block']==0,
                                                         dataAB_contextdiff_df['Half_Block']==1],
                                                        ['first half', 'second half']))
    dataAB_contextdiff_df["Block1"] = dataAB_contextdiff_df["Block1"].astype('string')

    # Plotting
    plotAB2 = (ggplot(dataAB_contextdiff_df, aes(x = 'Relative_beep',
                                                 y = 'Asyn_zeroed',
                                                 linetype = 'Sign',
                                                 shape = 'Type',
                                                 color = 'Block1'))
               + geom_line()
               + geom_errorbar(aes(x = 'Relative_beep', ymin = "Asyn_zeroed-ci_asyn", ymax = "Asyn_zeroed+ci_asyn", width = error_width))
               + geom_point(size = marker_size)
               + scale_linetype_manual(values = line_map, guide=False)
               + scale_shape_manual(values = shape_map, guide=False)
               + scale_color_manual(values = color_map_block, guide=False)
               + scale_x_continuous(breaks=range(x_lims[0]+1,x_lims[1],2))
               + theme_bw(base_size=16)
               + theme(legend_key=element_rect(fill = "white", color = 'white'), figure_size = (fig_xsize, fig_ysize))  
               + themes.theme(
                   axis_title_y = themes.element_text(angle = 90, va = 'center', size = 14),
                   axis_title_x = themes.element_text(va = 'center', size = 14))
               + xlab("Beep $n$ relative to perturbation")
               + ylab("Difference (ms)")
               + ggtitle("(b)")
               )
    #print(plotAB2)
    #plotAB2.save('../analysis/' + 'plot_Difference_Block0_Half_Block.pdf')




    # Select resynchronization phase
    resynch_start = 1	# bip number
    resynch_end = 6
    dataAB_across_subjects_df.query("(Type=='SC' and (Relative_beep>=@resynch_start and Relative_beep<=@resynch_end)) or (Type=='PS' and (Relative_beep>=@resynch_start+1 and Relative_beep<=@resynch_end))", inplace=True)

    # Difference between contexts, resynchronization phase only, after averaging across subjects and beeps
    dataAB_contextdiff_resynch_df = (dataAB_across_subjects_df
                                     # first average across subjects and beeps
                                     .groupby(['Context','Type','Sign','Half_Block'], as_index=False)
                                     .apply(lambda df: pd.Series({
                                         'mean_asyn': df.Asyn_zeroed.mean(),
                                         'std_asyn': df.Asyn_zeroed.std(),
                                         'n_asyn': df.Asyn_zeroed.count()
                                         }))
                                     # then compute diff
                                     .groupby(['Type','Sign','Half_Block'], as_index=False)
                                     .apply(lambda df: pd.Series({
                                         'diff': np.diff(df.mean_asyn).item(),
                                         'var_pooled': (((df.n_asyn-1)*df.std_asyn**2).sum())/((df.n_asyn-1).sum()),
                                         'n_pooled': ((df.std_asyn**2/df.n_asyn).sum())**2/(((df.std_asyn**2/df.n_asyn)**2/(df.n_asyn-1)).sum())
                                         }))
                                     )
    dataAB_contextdiff_resynch_df = (dataAB_contextdiff_resynch_df
                                     # confidence intervals
                                     .assign(ci_diff = lambda df: 1.96*np.sqrt(df['var_pooled'])/np.sqrt(df['n_pooled']))
                                     # create label for plot grouping
                                     .assign(Title = lambda df: df.Type + df.Sign)
                                     )

    # Filtering and ordering information
    dataAB_contextdiff_resynch_df.rename(columns={"diff": "Asyn_zeroed"}, inplace=True)
    dataAB_contextdiff_resynch_df.rename(columns={"ci_diff": "ci_asyn"}, inplace=True)

    dataAB_contextdiff_resynch_df["Half_Block"] = dataAB_contextdiff_resynch_df["Half_Block"].astype('int')
    dataAB_contextdiff_resynch_df.insert(0, 'Block1', np.select([dataAB_contextdiff_resynch_df['Half_Block']==0,
                                                                 dataAB_contextdiff_resynch_df['Half_Block']==1],
                                                                ['first half', 'second half']))
    dataAB_contextdiff_resynch_df["Block1"] = dataAB_contextdiff_resynch_df["Block1"].astype('string')

    # Modifying title
    dataAB_contextdiff_resynch_df.loc[dataAB_contextdiff_resynch_df.Title == "PSpos", 'Title'] = "PS\npos"
    dataAB_contextdiff_resynch_df.loc[dataAB_contextdiff_resynch_df.Title == "SCneg", 'Title'] = "SC\nneg"
    dataAB_contextdiff_resynch_df.loc[dataAB_contextdiff_resynch_df.Title == "SCpos", 'Title'] = "SC\npos"
    dataAB_contextdiff_resynch_df.loc[dataAB_contextdiff_resynch_df.Title == "PSneg", 'Title'] = "PS\nneg"

    # Set level order
    levels_TS = ['PS\npos','SC\nneg','SC\npos','PS\nneg']
    dataAB_contextdiff_resynch_df['Title'] = dataAB_contextdiff_resynch_df['Title'].astype("category").cat.set_categories(levels_TS, ordered=True)

    # Plotting
    plotAB3 = (
                ggplot(dataAB_contextdiff_resynch_df, aes(x = 'Title', 
                                                          y = 'Asyn_zeroed', 
                                                          linetype = 'Sign', 
                                                          shape = 'Type',
                                                          color = 'Block1'))
                + geom_line(dataAB_contextdiff_resynch_df,
                            aes(x = 'Title', 
                                y = 'Asyn_zeroed',
                                group = 'Block1'), 
                                linetype="dashdot")
                + geom_point(size = marker_size)
                + geom_errorbar(aes(x = 'Title', ymin = "Asyn_zeroed-ci_asyn", ymax = "Asyn_zeroed+ci_asyn", width = error_width))
                + scale_linetype_manual(values = line_map)
                + scale_shape_manual(values = shape_map)
                + scale_color_manual(values = color_map_block)
                + theme_bw(base_size=16)
                + theme(legend_key=element_rect(fill = "white", color = 'white'), figure_size = (fig_xsize2, fig_ysize2))
                + themes.theme(
                    axis_title_y = themes.element_text(angle = 90, va = 'center', size = 14),
                    axis_title_x = themes.element_text(va = 'center', size = 14))
                + xlab("Condition")
                + ylab("Average difference (ms) (beeps 1 through 6)")
                #+ ggtitle("(b)")
                )
    #print(plotAB3)
    #plotAB3.save('../analysis/' + 'plotAB3.pdf')

    # Plotting
    #plotAB2 = pw.load_ggplot(plotAB2)
    #plotAB3 = pw.load_ggplot(plotAB3)
    #plotAB23 = plotAB2|plotAB3
    #plotAB23.savefig('../analysis/' + 'plot_Difference_Block0_Half_Block.pdf')




    # Difference between Half_Block
    dataAB_contextdiff2_df = (dataAB_across_subjects2_df.groupby(['Context','Type','Sign','Relative_beep'], as_index=False).
                              apply(lambda df: pd.Series({'diff': np.diff(df.Asyn_zeroed).item(),
                                                          'var_pooled': ((df.n_subj-1)*df.std_asyn**2).sum()/(df.n_subj-1).sum(),
                                                          'n_pooled': (df.std_asyn**2/df.n_subj).sum()**2/((df.std_asyn**2/df.n_subj)**2/(df.n_subj-1)).sum()
                                                          })))
    dataAB_contextdiff2_df = (dataAB_contextdiff2_df
                              # confidence intervals
                              .assign(ci_diff = lambda df: 1.96*np.sqrt(df.var_pooled)/np.sqrt(df.n_pooled))
                              # create label for plot grouping
                              .assign(Title = lambda df: df.Type + df.Sign)
                              )

    # Filtering and ordering information
    dataAB_contextdiff2_df = dataAB_contextdiff2_df[(dataAB_contextdiff2_df['Relative_beep'] >= x_lims[0]) & (dataAB_contextdiff2_df['Relative_beep'] <= x_lims[1])]
    dataAB_contextdiff2_df.rename(columns={"diff": "Asyn_zeroed"}, inplace=True)
    dataAB_contextdiff2_df.rename(columns={"ci_diff": "ci_asyn"}, inplace=True)

    # Plotting
    plotAB4 = (ggplot(dataAB_contextdiff2_df, aes(x = 'Relative_beep',
                                                  y = 'Asyn_zeroed',
                                                  linetype = 'Sign',
                                                  shape = 'Type',
                                                  color = 'Context'))
               + geom_line()
               + geom_errorbar(aes(x = 'Relative_beep', ymin = "Asyn_zeroed-ci_asyn", ymax = "Asyn_zeroed+ci_asyn", width = error_width))
               + geom_point(size = marker_size)
               + scale_linetype_manual(values = line_map, guide=False)
               + scale_shape_manual(values = shape_map, guide=False)
               + scale_color_manual(values = color_map_hex, guide=False)
               + scale_x_continuous(breaks=range(x_lims[0]+1,x_lims[1],2))
               + theme_bw(base_size=16)
               + theme(legend_key=element_rect(fill = "white", color = 'white'), figure_size = (fig_xsize, fig_ysize))  
               + themes.theme(
                   axis_title_y = themes.element_text(angle = 90, va = 'center', size = 14),
                   axis_title_x = themes.element_text(va = 'center', size = 14))
               + xlab("Beep $n$ relative to perturbation")
               + ylab("Difference between halves of first block (ms)")
               + ggtitle("(b)")
               )
    #print(plotAB4)
    #plotAB4.save('../analysis/' + 'plot_AB4.pdf')




    # Select resynchronization phase
    resynch_start = 1	# bip number
    resynch_end = 6
    dataAB_across_subjects2_df.query("(Type=='SC' and (Relative_beep>=@resynch_start and Relative_beep<=@resynch_end)) or (Type=='PS' and (Relative_beep>=@resynch_start+1 and Relative_beep<=@resynch_end))", inplace=True)

    # Difference Difference between Half_Block, resynchronization phase only, after averaging across subjects and beeps
    dataAB_contextdiff2_resynch_df = (dataAB_across_subjects2_df
                                      # first average across subjects and beeps
                                      .groupby(['Context','Type','Sign','Half_Block'], as_index=False)
                                      .apply(lambda df: pd.Series({
                                          'mean_asyn': df.Asyn_zeroed.mean(),
                                          'std_asyn': df.Asyn_zeroed.std(),
                                          'n_asyn': df.Asyn_zeroed.count()
                                          }))
                                      # then compute diff
                                      .groupby(['Context','Type','Sign'], as_index=False)
                                      .apply(lambda df: pd.Series({
                                          'diff': np.diff(df.mean_asyn).item(),
                                          'var_pooled': (((df.n_asyn-1)*df.std_asyn**2).sum())/((df.n_asyn-1).sum()),
                                          'n_pooled': ((df.std_asyn**2/df.n_asyn).sum())**2/(((df.std_asyn**2/df.n_asyn)**2/(df.n_asyn-1)).sum())
                                          }))
                                      )
    dataAB_contextdiff2_resynch_df = (dataAB_contextdiff2_resynch_df
                                      # confidence intervals
                                      .assign(ci_diff = lambda df: 1.96*np.sqrt(df['var_pooled'])/np.sqrt(df['n_pooled']))
                                      # create label for plot grouping
                                      .assign(Title = lambda df: df.Type + df.Sign)
                                      )

    # Filtering and ordering information
    dataAB_contextdiff2_resynch_df.rename(columns={"diff": "Asyn_zeroed"}, inplace=True)
    dataAB_contextdiff2_resynch_df.rename(columns={"ci_diff": "ci_asyn"}, inplace=True)

    # Modifying title
    dataAB_contextdiff2_resynch_df.loc[dataAB_contextdiff2_resynch_df.Title == "PSpos", 'Title'] = "PS\npos"
    dataAB_contextdiff2_resynch_df.loc[dataAB_contextdiff2_resynch_df.Title == "SCneg", 'Title'] = "SC\nneg"
    dataAB_contextdiff2_resynch_df.loc[dataAB_contextdiff2_resynch_df.Title == "SCpos", 'Title'] = "SC\npos"
    dataAB_contextdiff2_resynch_df.loc[dataAB_contextdiff2_resynch_df.Title == "PSneg", 'Title'] = "PS\nneg"

    # Set level order
    levels_TS = ['PS\npos','SC\nneg','SC\npos','PS\nneg']
    dataAB_contextdiff2_resynch_df['Title'] = dataAB_contextdiff2_resynch_df['Title'].astype("category").cat.set_categories(levels_TS, ordered=True)

    # Plotting
    plotAB5 = (
                ggplot(dataAB_contextdiff2_resynch_df, aes(x = 'Title',
                                                           y = 'Asyn_zeroed',
                                                           linetype = 'Sign',
                                                           shape = 'Type',
                                                           color = 'Context'))
                + geom_line(dataAB_contextdiff2_resynch_df,
                            aes(x = 'Title', 
                                y = 'Asyn_zeroed',
                                group = 'Context'), 
                                linetype="dashdot")
                + geom_point(size = marker_size)
                + geom_errorbar(aes(x = 'Title', ymin = "Asyn_zeroed-ci_asyn", ymax = "Asyn_zeroed+ci_asyn", width = error_width))
                + scale_linetype_manual(values = line_map)
                + scale_shape_manual(values = shape_map)
                + scale_color_manual(values = color_map_hex)
                + theme_bw(base_size=16)
                + theme(legend_key=element_rect(fill = "white", color = 'white'), figure_size = (fig_xsize2, fig_ysize2))
                + themes.theme(
                    axis_title_y = themes.element_text(angle = 90, va = 'center', size = 14),
                    axis_title_x = themes.element_text(va = 'center', size = 14))
                + xlab("Condition")
                + ylab("Average difference (ms) (beeps 1 through 6)")
                #+ ggtitle("(b)")
                )
    #print(plotAB5)
    #plotAB5.save('../analysis/' + 'plot_AB5.pdf')
    
    return plotAB2, plotAB3, plotAB4, plotAB5


#%% Difference across subject between blocks 0 and 1 per context, sign and type
# perturb_size --> int (ej: 50). data_plot_df --> dataframe.
def Difference_across_subject_between_blocks_0_and_1_per_context_sign_and_type(perturb_size, data_plot_df):

    data_plot5_df = data_plot_df.reset_index(drop=True)

    # Filtering Perturb_size
    data_plot5_df.query("Perturb_size == -@perturb_size | Perturb_size == @perturb_size", inplace=True)
    data_plot5_df.drop(columns = ['Perturb_size'], inplace = True)

    # Filtering blocks 0 and 1
    data_plot_blocks01_df = data_plot5_df.query("Block == 0 | Block == 1")

    # Difference between blocks 0 and 1
    data_blocksdiff01_df = (data_plot_blocks01_df.groupby(['Context','Type','Sign','Relative_beep'], as_index=False).
                            apply(lambda df: pd.Series({'diff': np.diff(df.mean_asyn).item(),
                                                        'var_pooled': ((df.n_subj-1)*df.std_asyn**2).sum()/(df.n_subj-1).sum(),
                                                        'n_pooled': (df.std_asyn**2/df.n_subj).sum()**2/((df.std_asyn**2/df.n_subj)**2/(df.n_subj-1)).sum()
                                                        })))
    data_blocksdiff01_df = (data_blocksdiff01_df
                            # confidence intervals
                            .assign(ci_diff = lambda df: 1.96*np.sqrt(df.var_pooled)/np.sqrt(df.n_pooled))
                            # create label for plot grouping
                            .assign(Title = lambda df: df.Type + df.Sign)
                            )

    # Filtering and ordering information
    data_blocksdiff01_df.rename(columns={"diff": "Asyn_zeroed"}, inplace=True)
    data_blocksdiff01_df.rename(columns={"ci_diff": "ci_asyn"}, inplace=True)
    data_blocksdiff01_df["BlockDiff"] = "2-1"




    # Difference across subject between blocks 1 and 2 per context, sign and type
    data_plot7_df = data_plot_df.reset_index(drop=True)

    # Filtering Perturb_size
    data_plot7_df.query("Perturb_size == -@perturb_size | Perturb_size == @perturb_size", inplace=True)
    data_plot7_df.drop(columns = ['Perturb_size'], inplace = True)

    # Filtering blocks 1 and 2
    data_plot_blocks12_df = data_plot7_df.query("Block == 1 | Block == 2")

    # Difference between blocks 1 and 2
    data_blocksdiff12_df = (data_plot_blocks12_df.groupby(['Context','Type','Sign','Relative_beep'], as_index=False).
                            apply(lambda df: pd.Series({'diff': np.diff(df.mean_asyn).item(),
                                                        'var_pooled': ((df.n_subj-1)*df.std_asyn**2).sum()/(df.n_subj-1).sum(),
                                                        'n_pooled': (df.std_asyn**2/df.n_subj).sum()**2/((df.std_asyn**2/df.n_subj)**2/(df.n_subj-1)).sum()
                                                        })))
    data_blocksdiff12_df = (data_blocksdiff12_df
                            # confidence intervals
                            .assign(ci_diff = lambda df: 1.96*np.sqrt(df.var_pooled)/np.sqrt(df.n_pooled))
                            # create label for plot grouping
                            .assign(Title = lambda df: df.Type + df.Sign)
                            )

    # Filtering and ordering information
    data_blocksdiff12_df.rename(columns={"diff": "Asyn_zeroed"}, inplace=True)
    data_blocksdiff12_df.rename(columns={"ci_diff": "ci_asyn"}, inplace=True)
    data_blocksdiff12_df["BlockDiff"] = "3-2"




    # Concatening diff information
    data_blocksdiff_df = pd.concat([data_blocksdiff01_df, data_blocksdiff12_df]).reset_index(drop = True)




    # Parameters
    lower_color = 0
    upper_color = 3
    num_colors = 5
    color_map_rgba = cm.get_cmap('plasma')(np.linspace(lower_color,upper_color,num_colors))
    color_map_hex = [rgb2hex(color, keep_alpha=True) for color in color_map_rgba.tolist()]  
    line_map = ["solid","dashed"]
    shape_map = ["s","D"]
    size_map = (0.7,1.5)
    marker_size = 2
    error_width = 0.1
    fig_xsize = 15 * 0.393701   # centimeter to inch
    fig_ysize = 9 * 0.393701   # centimeter to inch
    fig_xsize2 = 5 * 0.393701   # centimeter to inch
    fig_ysize2 = 9 * 0.393701   # centimeter to inch
    x_lims = [-3,11]

    # Filtering and ordering information
    data_blocksdiff_plot_df = data_blocksdiff_df[(data_blocksdiff_df['Relative_beep'] >= x_lims[0]) & (data_blocksdiff_df['Relative_beep'] <= x_lims[1])]

    # Plotting
    plot4 = (ggplot(data_blocksdiff_plot_df, aes(x = 'Relative_beep',
                                                 y = 'Asyn_zeroed',
                                                 color = 'Context',
                                                 linetype = 'Sign',
                                                 shape = 'Type',
                                                 size = 'BlockDiff'))
             + geom_line()
             + geom_errorbar(aes(x = 'Relative_beep', ymin = "Asyn_zeroed-ci_asyn", ymax = "Asyn_zeroed+ci_asyn", width = error_width))
             + geom_point(size = marker_size)
             + scale_color_manual(values = color_map_hex, guide=False)
             + scale_linetype_manual(values = line_map, guide=False)
             + scale_shape_manual(values = shape_map, guide=False)
             + scale_size_manual(values=size_map, guide=False)
             + scale_x_continuous(breaks=range(x_lims[0]+1,x_lims[1],2))
             + theme_bw(base_size=16)
             + theme(legend_key=element_rect(fill = "white", color = 'white'), figure_size = (fig_xsize, fig_ysize))  
             + themes.theme(
                 axis_title_y = themes.element_text(angle = 90, va = 'center', size = 14),
                 axis_title_x = themes.element_text(va = 'center', size = 14))
             + xlab("Beep $n$ relative to perturbation")
             + ylab("Difference between blocks (ms)")
             + ggtitle("(a)")
             )
    #print(plot4)
    #plot4.save('../analysis/' + 'plot4.pdf')




    # Difference across subject between blocks 0 and 1 per context, sign and type and (mean asyn resynchronization phase)
    # Select resynchronization phase
    resynch_start = 1	# bip number
    resynch_end = 6
    data_blocksdiff_df.query("(Type=='SC' and (Relative_beep>=@resynch_start and Relative_beep<=@resynch_end)) or (Type=='PS' and (Relative_beep>=@resynch_start+1 and Relative_beep<=@resynch_end))", inplace=True)

    # Difference between blocks 0 and 1, resynchronization phase only, after averaging across subjects and beeps
    data_blocksdiff_resynch_df = (data_blocksdiff_df
                                  # first average across subjects and beeps
                                  .groupby(['Context','Type','Sign','BlockDiff'], as_index=False)
                                  .apply(lambda df: pd.Series({
                                      'mean_asyn': df.Asyn_zeroed.mean(),
                                      'std_asyn': df.Asyn_zeroed.std(),
                                      'n_asyn': df.Asyn_zeroed.count()
                                      })))
    data_blocksdiff_resynch_df = (data_blocksdiff_resynch_df
                                  # confidence intervals
                                  .assign(ci_asyn = lambda df: 1.96*df['std_asyn']/np.sqrt(df['n_asyn']))
                                  # create label for plot grouping
                                  .assign(Title = lambda df: df.Type + df.Sign,
                                          ContextBlockDiff = lambda df: df.Context + df.BlockDiff)
                                  )

    # Filtering and ordering information
    data_blocksdiff_resynch_df.rename(columns={"mean_asyn": "Asyn_zeroed"}, inplace=True)

    # Modifying title
    data_blocksdiff_resynch_df.loc[data_blocksdiff_resynch_df.Title == "PSpos", 'Title'] = "PS\npos"
    data_blocksdiff_resynch_df.loc[data_blocksdiff_resynch_df.Title == "SCneg", 'Title'] = "SC\nneg"
    data_blocksdiff_resynch_df.loc[data_blocksdiff_resynch_df.Title == "SCpos", 'Title'] = "SC\npos"
    data_blocksdiff_resynch_df.loc[data_blocksdiff_resynch_df.Title == "PSneg", 'Title'] = "PS\nneg"

    # Set level order
    levels_TS = ['PS\npos','SC\nneg','SC\npos','PS\nneg']
    data_blocksdiff_resynch_df['Title'] = data_blocksdiff_resynch_df['Title'].astype("category").cat.set_categories(levels_TS, ordered=True)
    levels_CBD = ['pure2-1','pure3-2','comb2-1','comb3-2']
    data_blocksdiff_resynch_df['ContextBlockDiff'] = data_blocksdiff_resynch_df['ContextBlockDiff'].astype("category").cat.set_categories(levels_CBD, ordered=True)

    # Plotting
    plot5 = (
                ggplot(data_blocksdiff_resynch_df, aes(x = 'Title',
                                                       y = 'Asyn_zeroed',
                                                       color = 'Context',
                                                       linetype = 'Sign',
                                                       shape = 'Type',
                                                       size = 'BlockDiff'))
                + geom_line(data_blocksdiff_resynch_df,
                            aes(x = 'Title', 
                                y = 'Asyn_zeroed',
                                group = 'ContextBlockDiff'), 
                                linetype="dashdot")
                + geom_point(size = marker_size)
                + geom_errorbar(aes(x = 'Title', ymin = "Asyn_zeroed-ci_asyn", ymax = "Asyn_zeroed+ci_asyn", width = error_width))
                + scale_color_manual(values = color_map_hex)
                + scale_linetype_manual(values = line_map)
                + scale_shape_manual(values = shape_map)
                + scale_size_manual(values=size_map)
                + theme_bw(base_size=16)
                + theme(legend_key=element_rect(fill = "white", color = 'white'), figure_size = (fig_xsize2, fig_ysize2))
                + themes.theme(
                    axis_title_y = themes.element_text(angle = 90, va = 'center', size = 14),
                    axis_title_x = themes.element_text(va = 'center', size = 14))
                + xlab("Condition")
                + ylab("Average difference (ms) (beeps 1 through 6)")
                #+ ggtitle("(b)")
                )
    #print(plot5)
    #plot5.save('../analysis/' + 'plot5.pdf')




    # Plotting
    #plot4 = pw.load_ggplot(plot4)
    #plot5 = pw.load_ggplot(plot5)
    #plot45 = plot4|plot5
    #plot45.savefig('../analysis/' + 'plot45.pdf')

    return plot4, plot5


#%%

