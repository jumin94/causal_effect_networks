#Correlacion entre indices
#1. abre indicies
#2. calcula correlaciones 
import numpy as np
import pandas as pd
import xarray as xr
import os, fnmatch
import glob
import open_data
import regresion
import csv2nc
import plot_sensitivity_maps

path_indices = '/home/julia.mindlin/Tesis/Capitulo3/scripts/CMIP6_storylines/Regresions_across_models/JJA/indices'
#Open indices
gloW  = pd.read_csv(path_indices+'/GW_index_JJA.csv')
gw_index = gloW.iloc[:,2].values
TA = pd.read_csv(path_indices+'/TA_index_JJA.csv')
SV = pd.read_csv(path_indices+'/SV_index_JJA.csv')
SST_C = pd.read_csv(path_indices+'/Nino4_asym_index_JJA.csv')
SST_E = pd.read_csv(path_indices+'/Nino12_asym_index_JJA.csv')
SST_IOD = pd.read_csv(path_indices+'/Indico_asym_index_JJA.csv')
TA = TA.iloc[:,2] / gw_index
SV = SV.iloc[:,2] / gw_index
SST_C = SST_C.iloc[:,2] / gw_index
SST_E = SST_E.iloc[:,2] / gw_index
SST_IOD = SST_IOD.iloc[:,2] / gw_index


indices = [TA,SV,SST_C,SST_E,SST_IOD] #,zonal_SST]
indices_names = ['TA','VB','SST_central','SST_east','SST_Indico'] #,'zonal_SST']
df = {indices_names[0]:indices[0].values,indices_names[1]:indices[1].values,indices_names[2]:indices[2].values,
      indices_names[3]:indices[3].values,indices_names[4]:indices[4].values}

df = pd.DataFrame(df)

import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

def corr_sig(df=None):
    p_matrix = np.zeros(shape=(df.shape[1],df.shape[1]))
    for col in df.columns:
        for col2 in df.drop(col,axis=1).columns:
            _ , p = stats.pearsonr(df[col],df[col2])
            p_matrix[df.columns.to_list().index(col),df.columns.to_list().index(col2)] = p
    return p_matrix

def plot_cor_matrix(corr, mask=None):
    f, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, ax=ax,
                mask=mask,
                # cosmetics
                annot=True, vmin=-1, vmax=1, center=0,
                cmap='coolwarm', linewidths=2, linecolor='black', cbar_kws={'orientation': 'horizontal'})
    return f


corr = df.corr()                            # get correlation
p_values = corr_sig(df)                     # get p-Value
mask = np.invert(np.tril(p_values<0.05))    # mask - only get significant corr
fig = plot_cor_matrix(corr,mask)  
fig.savefig('/home/julia.mindlin/Tesis/Capitulo3/scripts/CMIP6_storylines/Regresions_across_models/JJA/indices/index_corr.png')
