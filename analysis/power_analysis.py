#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  2 14:18:42 2023

@author: rodrigo, ariel
"""


import pandas as pd
import numpy as np
from math import ceil
from statsmodels.stats.power import TTestIndPower
from scipy import stats as st
from scipy.stats import t #,norm
from plotnine import ggplot, aes, geom_line, geom_errorbar, geom_point, scale_linetype_manual, scale_shape_manual
from plotnine import scale_x_continuous, scale_y_continuous, theme, scale_color_manual, facet_grid
from plotnine import xlab, ylab, theme_bw, element_rect, ggtitle, themes, element_text, element_blank
from matplotlib import cm
from matplotlib.colors import rgb2hex


#%% DATA LARGE

# Load data
dataL = './data_Large_Bavassi_Repp_Thaut/PSSC_Large.csv'
dataL_df = pd.read_csv(dataL)

# Calculating Asyn_zeroed, StdDesv, StdError and ci_asyn 
dataL_df['Sn+1_-Sn'] = dataL_df['IOI'] + dataL_df['Perturb_size']
dataL_df['Asyn_zeroed'] = dataL_df['Asyn_norm'] * dataL_df['Sn+1_-Sn']
dataL_df['StdDesv'] = dataL_df['StdDesv_norm'] * dataL_df['Sn+1_-Sn']
dataL_df['StdError'] = dataL_df['StdDesv'] / np.sqrt(dataL_df['n_subj'])
dataL_df['ci_asyn'] = 1.96 * dataL_df['StdError']
dataL_df = dataL_df[(dataL_df['Relative_beep'] >= 0) & (dataL_df['Relative_beep'] <= 1)]

# Calculating m, stdDesv_pooled and n_pooled for IOI interpolation
dataL4_df = dataL_df.reset_index(drop = True)
dataL4_df = dataL4_df.assign(Asyn_zeroed = np.select([dataL4_df['IOI']==400], [-dataL4_df['Asyn_zeroed']], default = dataL4_df['Asyn_zeroed']))
dataL4_df = dataL4_df.assign(IOI = np.select([dataL4_df['IOI']==400], [-dataL4_df['IOI']], default = dataL4_df['IOI']))
dataL5_df = (dataL4_df
             .groupby(['Relative_beep','Type','Sign','Porcentage'], as_index=False)
             .apply(lambda df: pd.Series({
                 'm': df.Asyn_zeroed.sum() / df.IOI.sum(),
                 'StdDesv_pooled': (((df.n_subj-1)*df.StdDesv**2).sum()/(df.n_subj-1).sum())**0.5,
                 'n_pooled': (df.StdDesv**2/df.n_subj).sum()**2/((df.StdDesv**2/df.n_subj)**2/(df.n_subj-1)).sum(),
                 }))
             )

# IOI Interpolation
dataL6_df = dataL4_df[dataL4_df['IOI']==600]
dataL6_df = dataL6_df.reindex(columns=['Relative_beep','Type','Sign','IOI','Porcentage', 'Asyn_zeroed'])
dataL6_df.rename(columns={"IOI":"x1","Asyn_zeroed":"y1"}, inplace = True)
dataL7_df = pd.merge(dataL5_df, dataL6_df, on=['Relative_beep','Type','Sign','Porcentage'])
dataL7_df['b'] = dataL7_df.y1 - dataL7_df.m * dataL7_df.x1
dataL7_df['IOI'] = 500
dataL7_df = dataL7_df.assign(Perturb_size = np.select([(dataL7_df['Type']=='PS') & (dataL7_df['Relative_beep']==1),
                                                       dataL7_df['Sign']=='neg'],
                                                      [0, 
                                                       -dataL7_df.Porcentage * dataL7_df.IOI / 100],
                                                      default = dataL7_df.Porcentage * dataL7_df.IOI / 100))
dataL7_df['Asyn_zeroed'] = dataL7_df.m * dataL7_df.IOI + dataL7_df.b
dataL7_df['ci_asyn'] = 1.96 * (dataL7_df.StdDesv_pooled / (dataL7_df.n_pooled**0.5))
dataL7_df.drop(columns = ['m','x1','y1','b'], inplace = True)

# Calculating m, stdDesv_pooled and n_pooled for Perturb_size interpolation.
dataL8_df = dataL7_df.reset_index(drop = True)
dataL8_df = dataL8_df.assign(Asyn_zeroed = np.select([dataL8_df['Porcentage']==8], [-dataL8_df['Asyn_zeroed']], default = dataL8_df['Asyn_zeroed']))
dataL8_df = dataL8_df.assign(Porcentage = np.select([dataL8_df['Porcentage']==8], [-dataL8_df['Porcentage']], default = dataL8_df['Porcentage']))
dataL9_df = (dataL8_df
             .groupby(['Relative_beep','Type','Sign','IOI'], as_index=False)
             .apply(lambda df: pd.Series({
                 'm': df.Asyn_zeroed.sum() / df.Porcentage.sum(),
                 'StdDesv_pooled_new': (((df.n_pooled-1)*df.StdDesv_pooled**2).sum()/(df.n_pooled-1).sum())**0.5,
                 'n_pooled_new': (df.StdDesv_pooled**2/df.n_pooled).sum()**2/((df.StdDesv_pooled**2/df.n_pooled)**2/(df.n_pooled-1)).sum(),
                 }))
             )

# Porcentage interpolation
dataL10_df = dataL8_df[dataL8_df['Porcentage']==15]
dataL10_df = dataL10_df.reindex(columns=['Relative_beep','Type','Sign','IOI','Porcentage', 'Asyn_zeroed'])
dataL10_df.rename(columns={"Porcentage":"x1","Asyn_zeroed":"y1"}, inplace = True)
dataL11_df = pd.merge(dataL9_df, dataL10_df, on=['Relative_beep','Type','Sign','IOI'])
dataL11_df['b'] = dataL11_df.y1 - dataL11_df.m * dataL11_df.x1
dataL11_df['Porcentage'] = 10
dataL11_df = dataL11_df.assign(Perturb_size = np.select([(dataL11_df['Type']=='PS') & (dataL11_df['Relative_beep']==1),
                                                       dataL11_df['Sign']=='neg'],
                                                      [0, 
                                                       -dataL11_df.Porcentage * dataL11_df.IOI / 100],
                                                      default = dataL11_df.Porcentage * dataL11_df.IOI / 100))
dataL11_df['Asyn_zeroed'] = dataL11_df.m * dataL11_df.Porcentage + dataL11_df.b
dataL11_df['ci_asyn'] = 1.96 * (dataL11_df.StdDesv_pooled_new / (dataL11_df.n_pooled_new**0.5))
dataL11_df.drop(columns = ['m','x1','y1','b'], inplace = True)
dataL11_df.rename(columns={"StdDesv_pooled_new":"StdDesv","n_pooled_new":"n_subj"}, inplace = True)

# Calculating Virtual_Asyn
dataL11_df['Virtual_Asyn'] = np.NaN
dataL11_df.loc[dataL11_df['Relative_beep'] == 0, 'Virtual_Asyn'] = dataL11_df['Asyn_zeroed']
dataL11_df.loc[(dataL11_df['Relative_beep'] == 1) & (dataL11_df['Type'] == 'PS'), 'Virtual_Asyn'] = dataL11_df['Asyn_zeroed']
dataL11_df.loc[(dataL11_df['Relative_beep'] == 1) & (dataL11_df['Type'] == 'SC'), 'Virtual_Asyn'] = dataL11_df['Perturb_size'] + dataL11_df['Asyn_zeroed']

dataL12_df = dataL11_df.reset_index(drop=True) 
dataL12_df['Type'] = dataL12_df['Type'].astype('str')
dataL12_df['Sign'] = dataL12_df['Sign'].astype('str')
dataL12_df["Condition"] = dataL12_df.Type + dataL12_df.Sign 
dataL12_df['Relative_beep'] = dataL12_df['Relative_beep'].astype('int64')
dataL12_df['Virtual_Asyn'] = dataL12_df['Virtual_Asyn'].astype('float64')
dataL12_df['ci_asyn'] = dataL12_df['ci_asyn'].astype('float64')

dataL_df = dataL12_df.reset_index(drop = True)            
dataL_df = dataL_df.assign(Context = "comb") 
dataL_df = dataL_df.reindex(columns=['Relative_beep', 'Context', 'Type', 'Sign', 'Condition', 'Virtual_Asyn', 'ci_asyn', 'n_subj']).reset_index(drop=True)
#dataL_df.to_csv("./data_Large_Bavassi_Repp_Thaut/dataL_df.csv", na_rep = np.NaN)


#%% DATA BAVASSI

dataB = './data_Large_Bavassi_Repp_Thaut/SC_Bavassi.csv'
dataB_df = pd.read_csv(dataB)
dataB_df['ci_asyn'] = 1.96 * dataB_df['StdError']
dataB_df['Virtual_Asyn'] = np.NaN
dataB_df.loc[dataB_df['Relative_beep'] == 0, 'Virtual_Asyn'] = dataB_df['Asyn_zeroed']
dataB_df.loc[dataB_df['Relative_beep'] == 1, 'Virtual_Asyn'] = dataB_df['Asyn_zeroed'] + dataB_df['Perturb_size'] 
dataB_df = dataB_df[(dataB_df['Relative_beep'] >= 0) & (dataB_df['Relative_beep'] <= 1)]
dataB_df['StdDesv'] = dataB_df['StdError'] * np.sqrt(dataB_df['n_subj'])
dataB_df = dataB_df.reindex(columns=['Relative_beep', 'Type', 'Sign', 'Condition', 'n_subj', 'StdDesv', 'Virtual_Asyn']).reset_index(drop=True)


#%% DATA THAUT

dataT = './data_Large_Bavassi_Repp_Thaut/SC_Thaut.csv'
dataT_df = pd.read_csv(dataT)
dataTaux_df= dataT_df.query("Relative_beep < 0")
dataTaux_df = dataTaux_df.reindex(columns=['Condition', 'Asyn_zeroed'])
dataTaux_df = (dataTaux_df.groupby(['Condition'], as_index=False)
               .agg(mean_asyn = ("Asyn_zeroed", "mean"), std_asyn=("Asyn_zeroed","std"), sem_asyn = ("Asyn_zeroed", "sem"), n_asyn=("Asyn_zeroed", "size"), ci_asyn=("Asyn_zeroed", lambda value: 1.96 * st.sem(value, axis=None))))
dataT_df['StdDesv'] = np.NaN
dataT_df.loc[(dataT_df['Relative_beep'] == 0) & (dataT_df['Condition'] == 'SCneg'), 'StdDesv'] = float(dataTaux_df[dataTaux_df['Condition'] == 'SCneg']['std_asyn'])
dataT_df.loc[(dataT_df['Relative_beep'] == 1) & (dataT_df['Condition'] == 'SCneg'), 'StdDesv'] = float(dataTaux_df[dataTaux_df['Condition'] == 'SCneg']['std_asyn'])
dataT_df.loc[(dataT_df['Relative_beep'] == 0) & (dataT_df['Condition'] == 'SCpos'), 'StdDesv'] = float(dataTaux_df[dataTaux_df['Condition'] == 'SCpos']['std_asyn'])
dataT_df.loc[(dataT_df['Relative_beep'] == 1) & (dataT_df['Condition'] == 'SCpos'), 'StdDesv'] = float(dataTaux_df[dataTaux_df['Condition'] == 'SCpos']['std_asyn'])
dataT_df['StdError'] = np.NaN
dataT_df.loc[(dataT_df['Relative_beep'] == 0) & (dataT_df['Condition'] == 'SCneg'), 'StdError'] = float(dataTaux_df[dataTaux_df['Condition'] == 'SCneg']['sem_asyn'])
dataT_df.loc[(dataT_df['Relative_beep'] == 1) & (dataT_df['Condition'] == 'SCneg'), 'StdError'] = float(dataTaux_df[dataTaux_df['Condition'] == 'SCneg']['sem_asyn'])
dataT_df.loc[(dataT_df['Relative_beep'] == 0) & (dataT_df['Condition'] == 'SCpos'), 'StdError'] = float(dataTaux_df[dataTaux_df['Condition'] == 'SCpos']['sem_asyn'])
dataT_df.loc[(dataT_df['Relative_beep'] == 1) & (dataT_df['Condition'] == 'SCpos'), 'StdError'] = float(dataTaux_df[dataTaux_df['Condition'] == 'SCpos']['sem_asyn'])
dataT_df['ci_asyn'] = np.NaN
dataT_df.loc[(dataT_df['Relative_beep'] == 0) & (dataT_df['Condition'] == 'SCneg'), 'ci_asyn'] = float(dataTaux_df[dataTaux_df['Condition'] == 'SCneg']['ci_asyn'])
dataT_df.loc[(dataT_df['Relative_beep'] == 1) & (dataT_df['Condition'] == 'SCneg'), 'ci_asyn'] = float(dataTaux_df[dataTaux_df['Condition'] == 'SCneg']['ci_asyn'])
dataT_df.loc[(dataT_df['Relative_beep'] == 0) & (dataT_df['Condition'] == 'SCpos'), 'ci_asyn'] = float(dataTaux_df[dataTaux_df['Condition'] == 'SCpos']['ci_asyn'])
dataT_df.loc[(dataT_df['Relative_beep'] == 1) & (dataT_df['Condition'] == 'SCpos'), 'ci_asyn'] = float(dataTaux_df[dataTaux_df['Condition'] == 'SCpos']['ci_asyn'])
dataT_df['Virtual_Asyn'] = np.NaN
dataT_df.loc[(dataT_df['Relative_beep'] == 0) & (dataT_df['Condition'] == 'SCneg'), 'Virtual_Asyn'] = float(dataT_df[(dataT_df['Condition'] == 'SCneg') & (dataT_df['Relative_beep'] == 0)]['Asyn_zeroed'])
dataT_df.loc[(dataT_df['Relative_beep'] == 1) & (dataT_df['Condition'] == 'SCneg'), 'Virtual_Asyn'] = (float(dataT_df[(dataT_df['Condition'] == 'SCneg') & (dataT_df['Relative_beep'] == 1)]['Asyn_zeroed']) + 
                                                                                                          float(dataT_df[(dataT_df['Condition'] == 'SCneg') & (dataT_df['Relative_beep'] == 1)]['Perturb_size']))
dataT_df.loc[(dataT_df['Relative_beep'] == 0) & (dataT_df['Condition'] == 'SCpos'), 'Virtual_Asyn'] = float(dataT_df[(dataT_df['Condition'] == 'SCpos') & (dataT_df['Relative_beep'] == 0)]['Asyn_zeroed'])
dataT_df.loc[(dataT_df['Relative_beep'] == 1) & (dataT_df['Condition'] == 'SCpos'), 'Virtual_Asyn'] = (float(dataT_df[(dataT_df['Condition'] == 'SCpos') & (dataT_df['Relative_beep'] == 1)]['Asyn_zeroed']) + 
                                                                                                          float(dataT_df[(dataT_df['Condition'] == 'SCpos') & (dataT_df['Relative_beep'] == 1)]['Perturb_size']))
dataT_df = dataT_df[(dataT_df['Relative_beep'] >= 0) & (dataT_df['Relative_beep'] <= 1)]
dataT_df.rename(columns={"n_beeps": "n_subj"}, inplace=True)
dataT_df = dataT_df.reindex(columns=['Relative_beep', 'Type', 'Sign', 'Condition', 'n_subj', 'StdDesv', 'Virtual_Asyn']).reset_index(drop = True)


#%% UNIFYING BAVASSI AND THAUT

dataBT_df = pd.concat([dataB_df, dataT_df], ignore_index = True)
dataBT_df = (dataBT_df.groupby(["Relative_beep", "Type", "Sign", "Condition"], as_index=False)
              .apply(lambda df: pd.Series({'S_pooled': (((df.StdDesv**2)/df.n_subj).sum())**0.5,
                                           'n_pooled': (((((df.StdDesv**2)/df.n_subj).sum())**2) / (((((df.StdDesv**2)/df.n_subj)**2) / (df.n_subj-1)).sum())) + 2,
                                           'Virtual_Asyn': (df.Virtual_Asyn).mean()})))
dataBT_df["ci_asyn"] = 1.96 * (dataBT_df["S_pooled"] / (dataBT_df["n_pooled"]**0.5))
dataBT_df["Context"] = "pure" 
dataBT_df = dataBT_df.assign(Context = "pure") 
dataBT_df.rename(columns={"n_pooled": "n_subj"}, inplace=True)
dataBT_df = dataBT_df.reindex(columns=['Relative_beep', 'Context', 'Type', 'Sign', 'Condition', 'Virtual_Asyn', 'ci_asyn', 'n_subj']).reset_index(drop=True)
   

#%% DATA REPP

dataR = './data_Large_Bavassi_Repp_Thaut/PS_Repp.csv'
dataR_df = pd.read_csv(dataR)
dataR_df['ci_asyn'] = 1.96 * dataR_df['StdError'] 
dataR_df['Virtual_Asyn'] = dataR_df['Asyn_zeroed']
dataR_df['StdDesv'] = dataR_df['StdError'] * np.sqrt(dataR_df['n_subj'])
dataR_df = (dataR_df.groupby(["Condition", "Type", "Sign"], as_index=False)
             .apply(lambda df: pd.Series({'S_pooled': (((df.StdDesv**2)/df.n_subj).sum())**0.5,
                                          'n_pooled': (((((df.StdDesv**2)/df.n_subj).sum())**2) / (((((df.StdDesv**2)/df.n_subj)**2) / (df.n_subj-1)).sum())) + 2,
                                          'Virtual_Asyn': (df.Virtual_Asyn).mean()})))
dataR_df["ci_asyn"] = 1.96 * (dataR_df["S_pooled"] / (dataR_df["n_pooled"]**0.5))
dataR_df["Relative_beep"] = 1
dataR2 = {'Condition': ['PSneg', 'PSpos'], 'Type': ['PS', 'PS'], 'Sign': ['neg', 'pos'], 'Virtual_Asyn': [50, -50], 'ci_asyn': [0, 0], 'Relative_beep': [0, 0]}
dataR2_df = pd.DataFrame(dataR2)
dataR_df = pd.concat([dataR_df, dataR2_df], ignore_index = True)
dataR_df = dataR_df.assign(Context = "pure") 
dataR_df.rename(columns={"n_pooled": "n_subj"}, inplace=True)
dataR_df = dataR_df.reindex(columns=['Relative_beep', 'Context', 'Type', 'Sign', 'Condition', 'Virtual_Asyn', 'ci_asyn', 'n_subj']).reset_index(drop=True)


#%% UNIFIED DATA

dataU_df = pd.concat([dataL_df, dataBT_df, dataR_df], ignore_index = True).reset_index(drop=True)
dataU_cat = pd.Categorical(dataU_df['Condition'], categories=['PSneg', 'SCneg', 'PSpos','SCpos'])
dataU_df = dataU_df.assign(Condition = dataU_cat)
dataU_df.to_csv('./data_Large_Bavassi_Repp_Thaut/dataU_df.csv')

# add significance for plotting (see last section)
conditions = [(dataU_df['Relative_beep']==1) & (dataU_df['Type']=='PS') & (dataU_df['Context']=='pure')]
dataU_df['signif'] = np.where(conditions,1,0)[0]


#%% PLOTTING

# Parameters
#color_map = ["blue","magenta"]
lower_color = 0
upper_color = 3
num_colors = 5
color_map_rgba = cm.get_cmap('plasma')(np.linspace(lower_color,upper_color,num_colors))
color_map_hex = [rgb2hex(color, keep_alpha=True) for color in color_map_rgba.tolist()]  
line_map = ["solid","dashed"]
shape_map = ["s","D"]
marker_size = 2
error_width = 0.1
ast_size = 4
fig_xsize = 5
fig_ysize = 6
x_lims = (0,1)
y_lims = (-50,50)


# Plotting
plot_a = (ggplot(dataU_df,aes(x = 'Relative_beep', y = 'Virtual_Asyn',
                color = 'Context', 
                linetype = 'Sign',
                shape = 'Type'))
          + facet_grid(facets="Sign~Type")
          + geom_line()
          + geom_point(size = marker_size)
          + geom_errorbar(aes(x = 'Relative_beep', ymin = "Virtual_Asyn-ci_asyn", ymax = "Virtual_Asyn+ci_asyn", width = error_width))
		  + geom_point(dataU_df[dataU_df['signif']==1], aes(x='Relative_beep',y=55), shape="*", size=ast_size, color="black")
          + scale_color_manual(values = color_map_hex)
          + scale_linetype_manual(values = line_map)
          + scale_shape_manual(values = shape_map)
          + scale_x_continuous(breaks=range(x_lims[0],x_lims[1]+1,1))
          + scale_y_continuous(breaks=range(y_lims[0],y_lims[1]+1,50))
          + theme_bw(base_size=18, base_family="sans-serif")
          + theme(legend_title = element_text(size=16),
                  legend_text=element_text(size=14),
                  legend_key=element_rect(fill = "white", color = 'white'), 
                  figure_size = (fig_xsize, fig_ysize))
          + themes.theme(
              axis_title_y = themes.element_text(angle = 90, va = 'center', size = 16),
              axis_title_x = themes.element_text(va = 'center', size = 16))
          + theme(strip_background = element_blank())
          + xlab("Beep $n$ relative to perturbation")
          + ylab("Asynchrony (ms)") 
          + ggtitle("(b)")
          + theme(plot_title=element_text(face="bold"))
         )
#print(plot_a)
plot_a.save('./figure_1b.pdf')


    #%% POWER ANALYSIS PSneg.
n_estim = 11 # initial estimated number of subjects (to be compared to the computed number of subjects n_min below)

# PSneg comb
psneg_comb_VL = float(dataU_df[(dataU_df['Context'] == 'comb') & (dataU_df['Condition'] == 'PSneg') & (dataU_df['Relative_beep'] == 1)]['Virtual_Asyn'])
psneg_comb_ciAsyn = float(dataU_df[(dataU_df['Context'] == 'comb') & (dataU_df['Condition'] == 'PSneg') & (dataU_df['Relative_beep'] == 1)]['ci_asyn'])
psneg_comb_nSubj = float(dataU_df[(dataU_df['Context'] == 'comb') & (dataU_df['Condition'] == 'PSneg') & (dataU_df['Relative_beep'] == 1)]['n_subj'])
psneg_comb_stdev = (psneg_comb_ciAsyn / 1.96) * np.sqrt(psneg_comb_nSubj)

# PSneg pure
psneg_pure_VL = float(dataU_df[(dataU_df['Context'] == 'pure') & (dataU_df['Condition'] == 'PSneg') & (dataU_df['Relative_beep'] == 1)]['Virtual_Asyn'])
psneg_pure_ciAsyn = float(dataU_df[(dataU_df['Context'] == 'pure') & (dataU_df['Condition'] == 'PSneg') & (dataU_df['Relative_beep'] == 1)]['ci_asyn'])
psneg_pure_nSubj = float(dataU_df[(dataU_df['Context'] == 'pure') & (dataU_df['Condition'] == 'PSneg') & (dataU_df['Relative_beep'] == 1)]['n_subj'])
psneg_pure_stdev = (psneg_pure_ciAsyn / 1.96) * np.sqrt(psneg_pure_nSubj)

# t critical value
alpha = 0.05
# t_critic = 1.64 # <-- approximation
t_critic = t.ppf(1-alpha,2*n_estim-2) # <-- exact

# power of the test (beta = 1 - test_power)
test_power = 0.9
beta = 1 - test_power
# zeta = norm.ppf(beta) # one-tailed <-- aproximation
zeta = t.ppf(beta,2*n_estim-2) # one-tailed <-- exact

# effect size
stdev_pooled = np.sqrt(np.mean(psneg_pure_stdev**2 + psneg_comb_stdev**2)) # pooled standard deviation
effect_size = np.abs(psneg_pure_VL - psneg_comb_VL) / stdev_pooled # standardized effect size


n_min = ceil(2*((t_critic - zeta)/effect_size)**2) # it should be equal to n_estim, otherwise adjust n_estim and re-run


# compute n_min with simplifications
analysis = TTestIndPower()
n_min_alt = analysis.solve_power(effect_size=effect_size, nobs1=None, alpha=0.05, power=0.9, ratio=1.0, alternative='larger')

print("PSneg: n_estim=%d, n_min=%d, n_min_alt=%d" % (n_estim,n_min,n_min_alt))


#%% POWER ANALYSIS PSpos.
n_estim = 13 # initial estimated number of subjects (to be compared to the computed number of subjects n_min below)

# PSpos comb
pspos_comb_VL = float(dataU_df[(dataU_df['Context'] == 'comb') & (dataU_df['Condition'] == 'PSpos') & (dataU_df['Relative_beep'] == 1)]['Virtual_Asyn'])
pspos_comb_ciAsyn = float(dataU_df[(dataU_df['Context'] == 'comb') & (dataU_df['Condition'] == 'PSpos') & (dataU_df['Relative_beep'] == 1)]['ci_asyn'])
pspos_comb_nSubj = float(dataU_df[(dataU_df['Context'] == 'comb') & (dataU_df['Condition'] == 'PSpos') & (dataU_df['Relative_beep'] == 1)]['n_subj'])
pspos_comb_stdev = (pspos_comb_ciAsyn / 1.96) * np.sqrt(pspos_comb_nSubj)

# PSpos pure
pspos_pure_VL = float(dataU_df[(dataU_df['Context'] == 'pure') & (dataU_df['Condition'] == 'PSpos') & (dataU_df['Relative_beep'] == 1)]['Virtual_Asyn'])
pspos_pure_ciAsyn = float(dataU_df[(dataU_df['Context'] == 'pure') & (dataU_df['Condition'] == 'PSpos') & (dataU_df['Relative_beep'] == 1)]['ci_asyn'])
pspos_pure_nSubj = float(dataU_df[(dataU_df['Context'] == 'pure') & (dataU_df['Condition'] == 'PSpos') & (dataU_df['Relative_beep'] == 1)]['n_subj'])
pspos_pure_stdev = (pspos_pure_ciAsyn / 1.96) * np.sqrt(pspos_pure_nSubj)

# t critical value
alpha = 0.05
# t_critic = 1.64 # <-- approximation
t_critic = t.ppf(1-alpha,2*n_estim-2) # <-- exact

# power of the test (beta = 1 - test_power)
test_power = 0.9
beta = 1 - test_power
# zeta = norm.ppf(beta) # one-tailed <-- aproximation
zeta = t.ppf(beta,2*n_estim-2) # one-tailed <-- exact

# effect size
stdev_pooled = np.sqrt(np.mean(pspos_pure_stdev**2 + pspos_comb_stdev**2)) # pooled standard deviation
effect_size = np.abs(pspos_pure_VL - pspos_comb_VL) / stdev_pooled # standardized effect size


n_min = ceil(2*((t_critic - zeta)/effect_size)**2) # it should be equal to n_estim, otherwise adjust n_estim and re-run


# compute n_min with simplifications
analysis = TTestIndPower()
n_min_alt = analysis.solve_power(effect_size=effect_size, nobs1=None, alpha=0.05, power=0.9, ratio=1.0, alternative='larger')

print("PSpos: n_estim=%d, n_min=%d, n_min_alt=%d" % (n_estim,n_min,n_min_alt))


#%% POWER ANALYSIS SCneg.
n_estim = 29 # initial estimated number of subjects (to be compared to the computed number of subjects n_min below)

# SCneg comb
scneg_comb_VL = float(dataU_df[(dataU_df['Context'] == 'comb') & (dataU_df['Condition'] == 'SCneg') & (dataU_df['Relative_beep'] == 1)]['Virtual_Asyn'])
scneg_comb_ciAsyn = float(dataU_df[(dataU_df['Context'] == 'comb') & (dataU_df['Condition'] == 'SCneg') & (dataU_df['Relative_beep'] == 1)]['ci_asyn'])
scneg_comb_nSubj = float(dataU_df[(dataU_df['Context'] == 'comb') & (dataU_df['Condition'] == 'SCneg') & (dataU_df['Relative_beep'] == 1)]['n_subj'])
scneg_comb_stdev = (scneg_comb_ciAsyn / 1.96) * np.sqrt(scneg_comb_nSubj)

# SCneg pure
scneg_pure_VL = float(dataU_df[(dataU_df['Context'] == 'pure') & (dataU_df['Condition'] == 'SCneg') & (dataU_df['Relative_beep'] == 1)]['Virtual_Asyn'])
scneg_pure_ciAsyn = float(dataU_df[(dataU_df['Context'] == 'pure') & (dataU_df['Condition'] == 'SCneg') & (dataU_df['Relative_beep'] == 1)]['ci_asyn'])
scneg_pure_nSubj = float(dataU_df[(dataU_df['Context'] == 'pure') & (dataU_df['Condition'] == 'SCneg') & (dataU_df['Relative_beep'] == 1)]['n_subj'])
scneg_pure_stdev = (scneg_pure_ciAsyn / 1.96) * np.sqrt(scneg_pure_nSubj)

# t critical value
alpha = 0.05
# t_critic = 1.64 # <-- approximation
t_critic = t.ppf(1-alpha,2*n_estim-2) # <-- exact

# power of the test (beta = 1 - test_power)
test_power = 0.9
beta = 1 - test_power
# zeta = norm.ppf(beta) # one-tailed <-- aproximation
zeta = t.ppf(beta,2*n_estim-2) # one-tailed <-- exact

# effect size
stdev_pooled = np.sqrt(np.mean(scneg_pure_stdev**2 + scneg_comb_stdev**2)) # pooled standard deviation
effect_size = np.abs(scneg_pure_VL - scneg_comb_VL) / stdev_pooled # standardized effect size


n_min = ceil(2*((t_critic - zeta)/effect_size)**2) # it should be equal to n_estim, otherwise adjust n_estim and re-run


# compute n_min with simplifications
analysis = TTestIndPower()
n_min_alt = analysis.solve_power(effect_size=effect_size, nobs1=None, alpha=0.05, power=0.9, ratio=1.0, alternative='larger')

print("SCneg: n_estim=%d, n_min=%d, n_min_alt=%d" % (n_estim,n_min,n_min_alt))


#%% POWER ANALYSIS SCpos.
n_estim = 88 # initial estimated number of subjects (to be compared to the computed number of subjects n_min below)

# SCpos comb
scpos_comb_VL = float(dataU_df[(dataU_df['Context'] == 'comb') & (dataU_df['Condition'] == 'SCpos') & (dataU_df['Relative_beep'] == 1)]['Virtual_Asyn'])
scpos_comb_ciAsyn = float(dataU_df[(dataU_df['Context'] == 'comb') & (dataU_df['Condition'] == 'SCpos') & (dataU_df['Relative_beep'] == 1)]['ci_asyn'])
scpos_comb_nSubj = float(dataU_df[(dataU_df['Context'] == 'comb') & (dataU_df['Condition'] == 'SCpos') & (dataU_df['Relative_beep'] == 1)]['n_subj'])
scpos_comb_stdev = (scpos_comb_ciAsyn / 1.96) * np.sqrt(scpos_comb_nSubj)

# SCpos pure
scpos_pure_VL = float(dataU_df[(dataU_df['Context'] == 'pure') & (dataU_df['Condition'] == 'SCpos') & (dataU_df['Relative_beep'] == 1)]['Virtual_Asyn'])
scpos_pure_ciAsyn = float(dataU_df[(dataU_df['Context'] == 'pure') & (dataU_df['Condition'] == 'SCpos') & (dataU_df['Relative_beep'] == 1)]['ci_asyn'])
scpos_pure_nSubj = float(dataU_df[(dataU_df['Context'] == 'pure') & (dataU_df['Condition'] == 'SCpos') & (dataU_df['Relative_beep'] == 1)]['n_subj'])
scpos_pure_stdev = (scpos_pure_ciAsyn / 1.96) * np.sqrt(scpos_pure_nSubj)

# t critical value
alpha = 0.05
# t_critic = 1.64 # <-- approximation
t_critic = t.ppf(1-alpha,2*n_estim-2) # <-- exact

# power of the test (beta = 1 - test_power)
test_power = 0.9
beta = 1 - test_power
# zeta = norm.ppf(beta) # one-tailed <-- aproximation
zeta = t.ppf(beta,2*n_estim-2) # one-tailed <-- exact

# effect size
stdev_pooled = np.sqrt(np.mean(scpos_pure_stdev**2 + scpos_comb_stdev**2)) # pooled standard deviation
effect_size = np.abs(scpos_pure_VL - scpos_comb_VL) / stdev_pooled # standardized effect size


n_min = ceil(2*((t_critic - zeta)/effect_size)**2) # it should be equal to n_estim, otherwise adjust n_estim and re-run


# compute n_min with simplifications
analysis = TTestIndPower()
n_min_alt = analysis.solve_power(effect_size=effect_size, nobs1=None, alpha=0.05, power=0.9, ratio=1.0, alternative='larger')

print("SCpos: n_estim=%d, n_min=%d, n_min_alt=%d" % (n_estim,n_min,n_min_alt))


#%% TEST INDIVIDUAL SIGNIFICANCE AT n=1

# PSneg comparison
psneg_comb_descriptivestats = [psneg_comb_VL,psneg_comb_stdev,psneg_comb_nSubj]
psneg_pure_descriptivestats = [psneg_pure_VL,psneg_pure_stdev,psneg_pure_nSubj]
aux, psneg_pvalue = st.ttest_ind_from_stats(*psneg_comb_descriptivestats,*psneg_pure_descriptivestats) # unpack argument lists

# PSpos comparison
pspos_comb_descriptivestats = [pspos_comb_VL,pspos_comb_stdev,pspos_comb_nSubj]
pspos_pure_descriptivestats = [pspos_pure_VL,pspos_pure_stdev,pspos_pure_nSubj]
aux, pspos_pvalue = st.ttest_ind_from_stats(*pspos_comb_descriptivestats,*pspos_pure_descriptivestats) # unpack argument lists

# SCneg comparison
scneg_comb_descriptivestats = [scneg_comb_VL,scneg_comb_stdev,scneg_comb_nSubj]
scneg_pure_descriptivestats = [scneg_pure_VL,scneg_pure_stdev,scneg_pure_nSubj]
aux, scneg_pvalue = st.ttest_ind_from_stats(*scneg_comb_descriptivestats,*scneg_pure_descriptivestats) # unpack argument lists

# SCpos comparison
scpos_comb_descriptivestats = [scpos_comb_VL,scpos_comb_stdev,scpos_comb_nSubj]
scpos_pure_descriptivestats = [scpos_pure_VL,scpos_pure_stdev,scpos_pure_nSubj]
aux, scpos_pvalue = st.ttest_ind_from_stats(*scpos_comb_descriptivestats,*scpos_pure_descriptivestats) # unpack argument lists

# Bonferroni-corrected pvalues
corrected_pvalues = tuple(4*x for x in (psneg_pvalue,pspos_pvalue,scneg_pvalue,scpos_pvalue))
print("Bonferroni-corrected pvalues for individual comparisons:\n \
	  PSpos: %.2g\n\
	  PSneg: %.2g\n\
	  SCpos: %.2g\n\
	  SCneg: %.2g" % corrected_pvalues)


#%%

