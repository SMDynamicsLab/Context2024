#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 13:46:34 2022

@author: RLaje, Asilva
"""

import numpy as np
import analysis_aux_3 as aux
from scipy import stats as st
import pandas as pd
from matplotlib import cm
from matplotlib.colors import rgb2hex
from plotnine import ggplot, aes, geom_line, geom_errorbar, geom_point, scale_linetype_manual, scale_shape_manual, scale_size_manual
from plotnine import scale_x_continuous, theme, scale_color_manual, facet_grid
from plotnine import xlab, ylab, theme_bw, element_rect, ggtitle, themes, element_text, element_blank
import patchworklib as pw


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


#difference_df.to_csv(path + "difference_df.csv", na_rep = np.NaN)
#difference2_df.to_csv(path + "difference2_df.csv", na_rep = np.NaN)
#differenceMinSub_df.to_csv(path + "differenceMinSub_df.csv", na_rep = np.NaN)
#differenceMinSub2_df.to_csv(path + "differenceMinSub2_df.csv", na_rep = np.NaN)


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










#%% Mean across trials and subjects
general_cond_dict_df = data_GroupSubjCond_OS_df[0]
general_proc_data_withoutOutliers_df = data_OutSubj_df[3]
#general_proc_data_withoutOutliers_df.to_csv(path + "general_proc_data_withoutOutliers_df.csv", na_rep = np.NaN)

# Mean across trials
data_across_trials_df = (general_proc_data_withoutOutliers_df.
                         groupby(["Exp_name", "Experiment", "Subject", "Block", "General_condition", "Condition", "Relative_beep"], as_index=False).
                         agg(mean_asyn=("Asyn_zeroed","mean"),std_asyn=("Asyn_zeroed","std"), sem_asyn=("Asyn_zeroed","sem"), 
                             n_asyn=("Asyn_zeroed","size"), ci_asyn=("Asyn_zeroed", lambda value: 1.96 * st.sem(value, axis=None))))
#data_across_trials_df.to_csv(path + 'data_across_trials_df',na_rep = np.NaN)

# Mean across subjects
data_across_subjects_df = (data_across_trials_df.
                           groupby(["Exp_name", "Experiment", "Block", "General_condition", "Condition", "Relative_beep"], as_index=False).
                           agg(mean_asyn = ("mean_asyn", "mean"), std_asyn=("mean_asyn","std"), sem_asyn = ("mean_asyn", "sem"), 
                               n_subj=("Subject","size"), ci_asyn=("mean_asyn", lambda value: 1.96 * st.sem(value, axis=None))))
#data_across_subjects_df.to_csv(path + 'data_across_subjects_df',na_rep = np.NaN)


#%% Ordering data for plotting
general_cond_dict_df = general_cond_dict_df.drop(columns = ['Condition', 'Experiment', 'Exp_name'])
data_plot_df = pd.merge(general_cond_dict_df, data_across_subjects_df, on=["General_condition"]).reset_index(drop = True)
data_plot_df.query("Perturb_size != 0", inplace=True)
data_plot_df.insert(0, 'Context', np.select([data_plot_df['Exp_name']=='Experiment_PS',
                                                        data_plot_df['Exp_name']=='Experiment_SC', 
                                                        data_plot_df['Exp_name']=='Experiment_PS_SC',
                                                        data_plot_df['Exp_name']=='Experiment2_PS_SC'],
                                                       ['pure', 'pure', 'comb', 'comb']))
data_plot_df.insert(1, 'Type', np.where(data_plot_df['Perturb_type']==0,'SC','PS'))
data_plot_df.insert(2, 'Sign', np.where(data_plot_df['Perturb_size']>0,'pos', 'neg'))
data_plot_df.drop(columns = ["Experiment", 'Exp_name', 'Perturb_type', 'General_condition', 'Condition', 'Name'], inplace = True)
data_plot_df.to_csv(path + 'data_plot_df',na_rep = np.NaN)


#%%










#%% Plotting across subject per context, size, sign and block
data_plot2_df = data_plot_df.reset_index(drop=True)

# Filtering Perturb_size == 20
#data_plot2_df.query("Perturb_size == -20 | Perturb_size == 20", inplace=True)
# Filtering Perturb_size == 50
data_plot2_df.query("Perturb_size == -50 | Perturb_size == 50", inplace=True)
data_plot2_df.drop(columns = ['Perturb_size'], inplace = True)

# Filtering per blocks
data_plot2_df.query("Block == 0 | Block == 2", inplace=True)
#data_plot2_df.to_csv(path + 'data_plot2_df',na_rep = np.NaN)

# Parameters
#color_map = ["blue","magenta"]
lower_color = 0
upper_color = 3
num_colors = 5
color_map_rgba = cm.get_cmap('plasma')(np.linspace(lower_color,upper_color,num_colors))
color_map_hex = [rgb2hex(color, keep_alpha=True) for color in color_map_rgba.tolist()]  
line_map = ["solid","dashed"]
shape_map = ["s","D"]
size_map = (0.1,0.5,1)
marker_size = 1
ast_size = 1
error_width = 0.1
fig_xsize = 20 * 0.393701   # centimeter to inch
fig_ysize = 12 * 0.393701   # centimeter to inch
x_lims = [-3,11]

# Filtering and ordering information
data_plot2_df = data_plot2_df[(data_plot2_df['Relative_beep'] >= x_lims[0]) & (data_plot2_df['Relative_beep'] <= x_lims[1])]
data_plot2_df.rename(columns={"mean_asyn": "Asyn_zeroed"}, inplace=True)
data_plot2_df["Block"] = data_plot2_df["Block"].astype('string')

# Plotting
plot = (
        ggplot(data_plot2_df) 
        + aes(x = 'Relative_beep', y = 'Asyn_zeroed',
              color = 'Context',
              linetype = 'Sign',
              shape = 'Type',
              size = 'Block')
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
print(plot)
plot.save('../analysis/' + 'plot.pdf')


#%%










#%% Difference across subject between context per sign, type and block
data_plot3_df = data_plot_df.reset_index(drop=True)

# Filtering Perturb_size == 20
#data_plot3_df.query("Perturb_size == -20 | Perturb_size == 20", inplace=True)
# Filtering Perturb_size == 50
data_plot3_df.query("Perturb_size == -50 | Perturb_size == 50", inplace=True)
data_plot3_df.drop(columns = ['Perturb_size'], inplace = True)

# Filtering per blocks
#data_plot3_df.query("Block == 0 | Block == 2", inplace=True)
data_plot3_df.to_csv(path + 'data_plot3_df',na_rep = np.NaN)

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
data_contextdiff_df.to_csv(path + 'data_contextdiff_df',na_rep = np.NaN)

# Parameters
line_map = ["solid","dashed"]
shape_map = ["s","D"]
size_map = (0.2,0.4,0.6)
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
data_contextdiff_df["Block"] = data_contextdiff_df["Block"].astype('string')

# Plotting
plot2 = (ggplot(data_contextdiff_df, aes(x = 'Relative_beep', 
                                         y = 'Asyn_zeroed', 
                                         linetype = 'Sign', 
                                         shape = 'Type',
                                         size = 'Block'))
          + geom_line()
          + geom_errorbar(aes(x = 'Relative_beep', ymin = "Asyn_zeroed-ci_asyn", ymax = "Asyn_zeroed+ci_asyn", width = error_width))
          + geom_point(size = marker_size)
          + scale_linetype_manual(values = line_map, guide=False)
          + scale_shape_manual(values = shape_map, guide=False)
          + scale_size_manual(values=size_map, guide=False)
          + scale_x_continuous(breaks=range(x_lims[0]+1,x_lims[1],2))
          + theme_bw(base_size=14)
          + theme(legend_key=element_rect(fill = "white", color = 'white'), figure_size = (fig_xsize, fig_ysize))  
          + themes.theme(
              axis_title_y = themes.element_text(angle = 90, va = 'center', size = 12),
              axis_title_x = themes.element_text(va = 'center', size = 12))
          + xlab("Beep $n$ relative to perturbation")
          + ylab("Difference (ms)")
          + ggtitle("(a)")
          )
#print(plot2)
#plot2.save('../analysis/' + 'plot2.pdf')


# Difference across subject between context per sign, type and block (mean asyn resynchronization phase)
data_plot4_df = data_plot_df.reset_index(drop=True)

# Filtering Perturb_size == 20
#data_plot4_df.query("Perturb_size == -20 | Perturb_size == 20", inplace=True)
# Filtering Perturb_size == 50
data_plot4_df.query("Perturb_size == -50 | Perturb_size == 50", inplace=True)
data_plot4_df.drop(columns = ['Perturb_size'], inplace = True)

# Filtering per blocks
#data_plot4_df.query("Block == 0 | Block == 2", inplace=True)
#data_plot4_df.to_csv(path + 'data_plot4_df',na_rep = np.NaN)

# Select resynchronization phase
resynch_start = 1	# bip number
resynch_end = 6
data_plot4_df.query("(Type=='SC' and (Relative_beep>=@resynch_start and Relative_beep<=@resynch_end)) or (Type=='PS' and (Relative_beep>=@resynch_start+1 and Relative_beep<=@resynch_end))", inplace=True)
data_plot4_df.to_csv(path + 'data_plot4_df',na_rep = np.NaN)							   

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
data_contextdiff_resynch_df.to_csv(path + 'data_contextdiff_resynch_df',na_rep = np.NaN)

# Filtering and ordering information
data_contextdiff_resynch_df.rename(columns={"diff": "Asyn_zeroed"}, inplace=True)
data_contextdiff_resynch_df.rename(columns={"ci_diff": "ci_asyn"}, inplace=True)
data_contextdiff_resynch_df["Block"] = data_contextdiff_resynch_df["Block"].astype('string')

# Set level order
levels_TS = ['PSpos','SCneg','SCpos','PSneg']
data_contextdiff_resynch_df['Title'] = data_contextdiff_resynch_df['Title'].astype("category").cat.set_categories(levels_TS, ordered=True)

# Plotting
plot3 = (
            ggplot(data_contextdiff_resynch_df, aes(x = 'Title', 
                                                    y = 'Asyn_zeroed', 
                                                    linetype = 'Sign', 
                                                    shape = 'Type',
                                                    size = 'Block'))
            + geom_point()
            + geom_errorbar(aes(x = 'Title', ymin = "Asyn_zeroed-ci_asyn", ymax = "Asyn_zeroed+ci_asyn", width = error_width))
            + scale_linetype_manual(values = line_map)
            + scale_shape_manual(values = shape_map)
            + scale_size_manual(values=size_map)
            + theme_bw(base_size=14)
            + theme(legend_key=element_rect(fill = "white", color = 'white'), figure_size = (fig_xsize2, fig_ysize2))
            + themes.theme(
                axis_title_y = themes.element_text(angle = 90, va = 'center', size = 12),
                axis_title_x = themes.element_text(va = 'center', size = 12))
            + xlab("Condition")
            + ylab("Average asymmetry (ms) (beeps 1 through 6)")
            + ggtitle("(b)")
         )
#print(plot3)
#plot3.save('../analysis/' + 'plot3.pdf')

# Plotting
plot2 = pw.load_ggplot(plot2)
plot3 = pw.load_ggplot(plot3)
plot23 = plot2|plot3
plot23.savefig('../analysis/' + 'plot23.pdf')


#%%










#%% Difference across subject between blocks per context, sign and type










