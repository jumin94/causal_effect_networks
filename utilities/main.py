#1. abre datos
#1. abre datos
#2. abre indicies
#3. genera regresion y archivos .csv
#4. plotea los mapas de sensibilidad guarda en SST_regressions_across_models
import numpy as np
import pandas as pd
import xarray as xr
import os, fnmatch
import glob
import open_data
import regresion
import csv2nc
import plot_sensitivity_maps

#Open data--------------------------------------------
ruta = '/pikachu/datos/julia.mindlin/CMIP6_ensambles/preprocesados' #Dropbox/DATOS_CMIP6' #/historical/mon/tas/past'
var = 'mon/psl'
models = ['ACCESS-CM2','ACCESS-ESM1-5','BCC-CSM2-MR','CAMS-CSM1-0','CanESM5','CESM2_','CESM2-WACCM','CMCC-CM2-SR5','CNRM-CM6-1','CNRM-ESM2-1','EC-Earth3','FGOALS-g3','GFDL-ESM4','HadGEM3-GC31-LL','HadGEM3-GC31-MM','IITM-ESM','INM-CM4-8','INM-CM5-0','IPSL-CM6A-LR','KACE-1-0-G','MIROC6','MIROC-ES2L','MPI-ESM1-2-HR','MPI-ESM1-2-LR','MRI-ESM2-0','NESM3','NorESM2-LM','NorESM2-MM','UKESM1-0-LL','TaiESM1']

scenarios = ['historical','ssp585']
os.chdir(ruta)
os.getcwd()

#Create dictionary
dato = open_data.cargo_todo(scenarios,models,ruta,var)

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

#indices_path = '/home/julia.mindlin/Tesis/Capitulo3/scripts/SST_regresions_across_models/indices'
#figure = plot_sensitivity_maps.plot_indices_box(indices,indices_names,indices_path)

#Create regression class
reg = regresion.across_models()

#Generate regression data
reg.regression_data(dato,scenarios,models,gw_index)

#Create sensitivity maps
path_maps = '/home/julia.mindlin/Tesis/Capitulo3/scripts/CMIP6_storylines/Regresions_across_models/JJA/sensitivity_maps/psl'
reg.perform_regression(indices,indices_names,gw_index,path_maps)
file_list = csv2nc.csv_to_nc(path_maps)

#Create plots
path_maps = '/home/julia.mindlin/Tesis/Capitulo3/scripts/CMIP6_storylines/Regresions_across_models/JJA/sensitivity_maps/psl'
GlobalWarming = xr.open_dataset(path_maps+'/Aij.nc')
TropicalWarming = xr.open_dataset(path_maps+'/TAij.nc')
VorBreak_GW = xr.open_dataset(path_maps+'/SVij.nc')
SeaSurfaceTemperature = xr.open_dataset(path_maps+'/SST_Cij.nc')
SeaSurfaceTemperature2 = xr.open_dataset(path_maps+'/SST_Eij.nc')
SeaSurfaceTemperature_Indico = xr.open_dataset(path_maps+'/SST_Indicoij.nc')
maps = [GlobalWarming, TropicalWarming, VorBreak_GW, SeaSurfaceTemperature,SeaSurfaceTemperature2,SeaSurfaceTemperature_Indico]

frac_var = xr.open_dataset(path_maps+'/R2ij.nc')

GlobalWarming_pval = xr.open_dataset(path_maps+'/Apij.nc')
TropicalWarming_pval = xr.open_dataset(path_maps+'/TApij.nc')
VorBreak_GW_pval = xr.open_dataset(path_maps+'/SVpij.nc')
SeaSurfaceTemperature_pval = xr.open_dataset(path_maps+'/SST_Cpij.nc')
SeaSurfaceTemperature2_pval = xr.open_dataset(path_maps+'/SST_Epij.nc')
SeaSurfaceTemperature_Indico_pval = xr.open_dataset(path_maps+'/SST_Indicopij.nc')

maps_pval = [GlobalWarming_pval, TropicalWarming_pval, VorBreak_GW_pval, SeaSurfaceTemperature_pval,SeaSurfaceTemperature2_pval,SeaSurfaceTemperature_Indico_pval]

path_figs = '/home/julia.mindlin/Tesis/Capitulo3/figures/SST_regressions_across_models/JJA/u850'
figure = plot_sensitivity_maps.plot_sensitivity_ua(maps,maps_pval,frac_var,path_figs)

figure = plot_sensitivity_maps.plot_sensitivity_ua_carree(maps,maps_pval,frac_var,path_figs)

path_figs = '/home/julia.mindlin/Tesis/Capitulo3/figures/SST_regressions_across_models/JJA/zg'
figure = plot_sensitivity_maps.plot_sensitivity_zg(maps,maps_pval,frac_var,path_figs)

figure = plot_sensitivity_maps.plot_sensitivity_zg_carree(maps,maps_pval,frac_var,path_figs)


path_figs = '/home/julia.mindlin/Tesis/Capitulo3/figures/SST_regressions_across_models/JJA/pr'
figure = plot_sensitivity_maps.plot_sensitivity_pr(maps,maps_pval,frac_var,path_figs)


path_figs = '/home/julia.mindlin/Tesis/Capitulo3/figures/SST_regressions_across_models/JJA/psl'
figure = plot_sensitivity_maps.plot_sensitivity_psl(maps,maps_pval,frac_var,path_figs)




