#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 13:46:34 2022

@author: RLaje, Asilva
"""

import numpy as np
import analysis_aux_2 as aux


#%% Define Python user-defined exceptions.
class Error(Exception):
	"""Base class for other exceptions"""
	pass


#%% DEFINITIONS FOR ALL EXPERIMENTS (version 1.1).
# Preprocessing data version 1.1 (Add outliers per subj cond analysis): Preperturbation zeroed. At the end, outlier cuantification. 


# Define path.
path = '../data/'


#%%










#%% Preprocessing Data

# Preprocessing_Data
# Function to process data for all experiments and general conditions dictionary, 
# considering the transient and the beeps out of range. Return tuple with two dataframes.
# path --> string (ej: '../data_aux/'). transient_dur --> int (ej: 1).
data_df = aux.Preprocessing_Data(path, 7)

# Preprocessing_Data_AllExperiments_MarkingTrialOutliers
# Function to process data for all experiments and general conditions dictionary, marking outlier trials. Return a tuple with three dataframes.
# path --> string (ej: '../data_aux/'). data_df --> dataframe. postPerturb_bip --> int (ej: 5).
data_OutTrials_df = aux.Preprocessing_Data_AllExperiments_MarkingTrialOutliers(path, data_df, 7)

# Preprocessing_Data_AllExperiments_MarkingSubjCondOutliers
# Function to process data for all experiments and general conditions dictionary, marking outlier subject conditions. Return a tuple with four dataframes.
# path --> string (ej: '../data_aux/'). data_OutTrials_df--> dataframe. porcTrialPrevCond --> int (ej: 10). postPerturb_bip --> int (ej: 5).
data_OutSubjCond_df = aux.Preprocessing_Data_AllExperiments_MarkingSubjCondOutliers(path, data_OutTrials_df, 50, 7)

# Preprocessing_Data_AllExperiments_MarkingSubjOutliers
# Function to process data for all experiments and general conditions dictionary, marking outlier subjects. Return a tuple with four dataframes.
# path --> string (ej: '../data_aux/'). data_OutSubjCond_df --> dataframe. porcSubjCondPrevCond --> int (ej: 10).
data_OutSubj_df = aux.Preprocessing_Data_AllExperiments_MarkingSubjOutliers(path, data_OutSubjCond_df, 50)


#%%










#%% Statistics

# Outliers_Trials_Cuantification
# Function to know outlier trials information.
# path --> string (ej: '../data_aux/'). data_OutTrials_df --> dataframe.
aux.Outliers_Trials_Cuantification(path, data_OutTrials_df)

# Outliers_SubjCond_Cuantification
# Function to know outlier subject conditions cuantification.
# path --> string (ej: '../data_aux/'). data_OutSubjCond_df --> dataframe.
aux.Outliers_SubjCond_Cuantification(path, data_OutSubjCond_df)

# Outliers_Subj_Cuantification
# Function to know outlier subjects cuantification.
# path --> string (ej: '../data_aux/'). data_OutSubj_df --> dataframe.
aux.Outliers_Subj_Cuantification(path, data_OutSubj_df)


#%%










#%% Group_Subject_Condition_Outlier_Subject
# Function to obtain meanasyn and stdasyn for each group subject condition.
data_GroupSubjCond_OS_df = aux.Group_Subject_Condition_Outlier_Subject(data_OutSubj_df)
data_GroupSubjCond_OS_df[0].to_csv(path + "data_GroupSubjCond_OS_dict.csv", na_rep = np.NaN)
data_GroupSubjCond_OS_df[1].to_csv(path + "data_GroupSubjCond_OS_df.csv", na_rep = np.NaN)


#%% Difference
# Function to get difference between same condition different experiments.
# path --> string (ej: '../data_aux/'). data_GroupSubjCond_OS_df --> dataframe. difference_list --> list (ej: [['Experiment_PS', 'Experiment_PS_SC', 'PSneg']]). 
# perturb_size --> int (ej: 50). relative_beep_ini --> int (ej: 1). relative_beep_final --> int (ej: 6). figure_number --> int (ej: 1).

# Perturb size 50 ms.
difference_list = [['Experiment_PS', 'Experiment_PS_SC', 'PSneg'],
                   ['Experiment_PS', 'Experiment_PS_SC', 'PSpos'],
                   ['Experiment_SC', 'Experiment_PS_SC', 'SCneg'],
                   ['Experiment_SC', 'Experiment_PS_SC', 'SCpos']]
difference_df, differenceMinSub_df = aux.Difference(path, data_GroupSubjCond_OS_df, difference_list, 50, 1, 6, 1)

# Perturb size 20 ms.
difference2_list = [['Experiment_PS', 'Experiment2_PS_SC', 'PSneg'],
                    ['Experiment_PS', 'Experiment2_PS_SC', 'PSpos'],
                    ['Experiment_SC', 'Experiment2_PS_SC', 'SCneg'],
                    ['Experiment_SC', 'Experiment2_PS_SC', 'SCpos']]
difference2_df, differenceMinSub2_df = aux.Difference(path, data_GroupSubjCond_OS_df, difference2_list, 20, 1, 6, 1)


difference_df.to_csv(path + "difference_df.csv", na_rep = np.NaN)
difference2_df.to_csv(path + "difference2_df.csv", na_rep = np.NaN)
differenceMinSub_df.to_csv(path + "differenceMinSub_df.csv", na_rep = np.NaN)
differenceMinSub2_df.to_csv(path + "differenceMinSub2_df.csv", na_rep = np.NaN)


#%% Asymmetry
# Function to get asymmetry between opposite conditions from same experiment.
# path --> string (ej: '../data_aux/'). data_GroupSubjCond_OS_df --> dataframe. asymmetry_list --> list (ej: [['Experiment_PS', 'PS']]).
# perturb_size --> int (ej: 50). relative_beep_ini --> int (ej: 1). relative_beep_final --> int (ej: 6). figure_number --> int (ej: 1).

# Perturb size 50 ms.
asymmetry_list = [['Experiment_PS', 'PS'],
                  ['Experiment_SC', 'SC'],
                  ['Experiment_PS_SC', 'PS'],
                  ['Experiment_PS_SC', 'SC']]
asymmetry_df, asymmetryAdd1Add2_df = aux.Asymmetry(path, data_GroupSubjCond_OS_df, asymmetry_list, 50, 1, 6, 1)

# Perturb size 20 ms.
asymmetry2_list = [['Experiment_PS', 'PS'],
                   ['Experiment_SC', 'SC'],
                   ['Experiment2_PS_SC', 'PS'],
                   ['Experiment2_PS_SC', 'SC']]
asymmetry2_df, asymmetryAdd1Add22_df = aux.Asymmetry(path, data_GroupSubjCond_OS_df, asymmetry2_list, 20, 1, 6, 1)


#%% Group_Condition_Mean_Postperturb_Transient
# Function to obtain mean_asyn and ci_asyn for each group condition in postperturb resynchronization transient (RelativeBeep 1-6).
# Group_Subject_Condition_Outlier_Subject --> data_GroupSubjCond_OS_df. perturb_size --> int (ej: 50).
# data_GroupCond_MPT_df (across subject). data_GroupCond2_MPT_df (across subject sign). data_GroupCond3_MPT_df (across subject type)  
data_GroupCond_MPT_df, data_GroupCond2_MPT_df, data_GroupCond3_MPT_df, data_GroupCond4_MPT_df = aux.Group_Condition_Mean_Postperturb_Transient(data_GroupSubjCond_OS_df,50)
data_GroupCond_MPT2_df, data_GroupCond2_MPT2_df, data_GroupCond3_MPT2_df, data_GroupCond4_MPT2_df = aux.Group_Condition_Mean_Postperturb_Transient(data_GroupSubjCond_OS_df,20)


#%%










#%% Plotting


#%% Plot_Differences
# Function to plot difference results.
# difference_df --> dataframe.
aux.Plot_Differences(data_GroupCond_MPT_df, difference_df)
aux.Plot_Differences(data_GroupCond_MPT2_df, difference2_df)


#%% Plot_Asymmetries
# Function to plot asymmetry results.
# asymmetry_df --> dataframe.
aux.Plot_Asymmetries(data_GroupCond_MPT_df, asymmetry_df)
aux.Plot_Asymmetries(data_GroupCond_MPT2_df, asymmetry2_df)


#%% Plot_Mean_Across_Subjects_to_Calculate_Difference
# Function to plot mean across subject data to calculate difference.
# differenceMinSub_df --> dataframe.
aux.Plot_Mean_Across_Subjects_to_Calculate_Difference(differenceMinSub_df)
aux.Plot_Mean_Across_Subjects_to_Calculate_Difference(differenceMinSub2_df)


#%% Plot_Mean_Across_Subjects_to_Calculate_Asymmetry
# Function to plot mean across subject data to calculate asymmetry.
# asymmetryAdd1Add2_df --> dataframe.
aux.Plot_Mean_Across_Subjects_to_Calculate_Asymmetry(asymmetryAdd1Add2_df)
aux.Plot_Mean_Across_Subjects_to_Calculate_Asymmetry(asymmetryAdd1Add22_df)


#%% Plot_Mean_Across_Subjects_to_Calculate_Asymmetry_PosInverted
# Function to plot mean across subject data to calculate asymmetry.
# asymmetryAdd1Add2_df --> dataframe.
aux.Plot_Mean_Across_Subjects_to_Calculate_Asymmetry_PosInverted(asymmetryAdd1Add2_df)
aux.Plot_Mean_Across_Subjects_to_Calculate_Asymmetry_PosInverted(asymmetryAdd1Add22_df)


#%% Plot_Mean_Across_Subjects
# Function to plot mean across subject data.
# differenceMinSub_df --> dataframe.
aux.Plot_Mean_Across_Subjects(differenceMinSub_df)
aux.Plot_Mean_Across_Subjects(differenceMinSub2_df)


#%% Plot_Mean_Across_Subjects_for_Type_Sign_and_Context
# Function to plot mean across subject data Relative beep 1 to 6 for Type, Sign and Context.
aux.Plot_Mean_Across_Subjects_for_Type_Sign_and_Context(data_GroupCond_MPT_df, data_GroupCond2_MPT_df, data_GroupCond3_MPT_df, data_GroupCond4_MPT_df)
aux.Plot_Mean_Across_Subjects_for_Type_Sign_and_Context(data_GroupCond_MPT2_df, data_GroupCond2_MPT2_df, data_GroupCond3_MPT2_df, data_GroupCond4_MPT2_df)


#%%

