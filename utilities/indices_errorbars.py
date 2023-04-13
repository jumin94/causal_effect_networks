#Indices remote drivers errorbars------------------------------------------------------------------------------------------

#This code evaluates all the remote driver indices and their errorbars - needed for a storyline analysis of the SH atmospheric ciruclation in CMIP6. This is: 

#Global Warming - Global mean warming from surface temperature (tas) area weighted 
#Tropical Warming - Tropical upper tropospheric warming zonally averaged and averaged between 15N and 15S. 
#Vortex breakdown delay with respect to 1940 - 1969 evaluated with a linear regression including EESCs
#Asymmetric SST anomalies evaluated as Central and Eastern Pacific Warming patterns 

#Climatological change is defined as the is ensemble mean SSP5-8.8 2070-2099 vs. ensemble mean historical simulations 1940-1969 except for VB delay where 1950-1979 is used. 

#Author: Julia Mindlin

import numpy as np
import pandas as pd
import xarray as xr
import index_module as im
from index_module import cargo_todo
import os

models = ['ACCESS-CM2','ACCESS-ESM1-5','BCC-CSM2-MR','CAMS-CSM1-0','CanESM5','CESM2_','CESM2-WACCM','CMCC-CM2-SR5','CNRM-CM6-1','CNRM-ESM2-1','EC-Earth3','FGOALS-g3','HadGEM3-GC31-LL','HadGEM3-GC31-MM','IITM-ESM','INM-CM4-8','INM-CM5-0','IPSL-CM6A-LR','KACE-1-0-G','MIROC6','MIROC-ES2L','MPI-ESM1-2-HR','MPI-ESM1-2-LR','MRI-ESM2-0','NESM3','NorESM2-LM','NorESM2-MM','TaiESM1','UKESM1-0-LL']
experiments = ['historical','ssp585']

path = '/datos/julia.mindlin/CMIP6_ensambles/preprocesados' #/historical/mon/tas/past'
os.chdir(path)
os.getcwd()

ruta = path

#Global warming - timeseries
var = 'mon/tas'
variables = ['tas']
dato = cargo_todo(ruta,experiments,models,var)
seasons = ['DJF'] #['MAM','JJA','SON']
season_name = 'DJF'
ens_members_hist = np.array([2,10,3,2,25,11,3,2,10,5,2,6,4,4,1,1,10,32,1,50,10,10,10,5,5,3,1,1,14])
ens_members_rcp = np.array([1,3,1,2,25,2,5,1,6,5,4,4,4,4,1,1,1,6,3,3,1,10,10,1,2,1,1,1,2])

ts_hist, ts_rcp = im.global_warming_time_series(dato,season,models,experiments)
gw_mean = np.array([])
gw_error = np.array([])
for i in range(len(models)):
    model = models[i]
    factor_hist = np.sqrt(len(ts_hist[model])*ens_members_hist[i])
    factor_rcp = np.sqrt(len(ts_rcp[model])*ens_members_rcp[i])
    std_hist = np.std(ts_hist[model].values)/factor_hist
    std_rcp = np.std(ts_rcp[model].values)/factor_rcp
    mean = np.mean(ts_rcp[model].values) - np.mean(ts_hist[model].values)
    error = np.sqrt(std_hist**2 + std_rcp**2)
    gw_mean = np.append(gw_mean,mean)
    gw_error = np.append(gw_error,error)

GWerr = {season+'_mean':gw_mean,season+'_error': gw_error}
GWerr = pd.DataFrame(GWerr)
GWerr.insert(0,"Modelo", models,True)
GWerr.to_csv('/home/julia.mindlin/Tesis/BSC/indices/GW_index_errors_'+season_name+'.csv',float_format='%g')

def cross_year_season(month):
    return (month >= 12) | (month <= 2)
    #return (month >= 9) & (month <= 11)
    
#Flexible SEASON! 
season = [10,11]; season_name = 'ON'
ens_members_hist = np.array([2,10,3,2,25,11,3,2,10,5,2,6,4,4,1,1,10,32,1,50,10,10,10,5,5,3,1,1,14])
ens_members_rcp = np.array([1,3,1,2,25,2,5,1,6,5,4,4,4,4,1,1,1,6,3,3,1,10,10,1,2,1,1,1,2])

ts_hist, ts_rcp = im.global_warming_time_series_flex_season(dato,season,models,experiments)
gw_mean = np.array([])
gw_error = np.array([])
for i in range(len(models)):
    model = models[i]
    factor_hist = np.sqrt(len(ts_hist[model])*ens_members_hist[i])
    factor_rcp = np.sqrt(len(ts_rcp[model])*ens_members_rcp[i])
    std_hist = np.std(ts_hist[model].values)/factor_hist
    std_rcp = np.std(ts_rcp[model].values)/factor_rcp
    mean = np.mean(ts_rcp[model].values) - np.mean(ts_hist[model].values)
    error = np.sqrt(std_hist**2 + std_rcp**2)
    gw_mean = np.append(gw_mean,mean)
    gw_error = np.append(gw_error,error)

GWerr = {season+'_mean':gw_mean,season+'_error': gw_error}
GWerr = pd.DataFrame(GWerr)
GWerr.insert(0,"Modelo", models,True)
GWerr.to_csv('/home/julia.mindlin/Tesis/BSC/indices/GW_index_errors_'+season_name+'.csv',float_format='%g')
    
#Tropical warming - timeseries
var = 'mon/ta'
variables = ['ta']
dato = cargo_todo(ruta,experiments,models,var)
seasons = ['DJF'] #['MAM','JJA','SON']
ens_members_hist = np.array([2,10,3,2,25,11,3,2,10,5,2,6,4,4,1,1,10,32,1,50,10,10,10,5,5,3,1,1,14])
ens_members_rcp = np.array([1,3,1,2,25,2,5,1,6,5,4,4,4,4,1,1,1,6,3,3,1,10,10,1,2,1,1,1,2])

for season in seasons:
    ts_hist, ts_rcp = im.tropical_warming_time_series(dato,season,models,experiments)
    tw_mean = np.array([])
    tw_error = np.array([])
    for i in range(len(models)):
        model = models[i]
        factor_hist = np.sqrt(len(ts_hist[model])*ens_members_hist[i])
        factor_rcp = np.sqrt(len(ts_rcp[model])*ens_members_rcp[i])
        std_hist = np.std(ts_hist[model].values)/factor_hist
        std_rcp = np.std(ts_rcp[model].values)/factor_rcp
        mean = np.mean(ts_rcp[model].values) - np.mean(ts_hist[model].values)
        error = np.sqrt(std_hist**2 + std_rcp**2)
        tw_mean = np.append(tw_mean,mean)
        tw_error = np.append(tw_error,error)

    TWerr = {season+'_mean':tw_mean,season+'_error': tw_error}
    TWerr = pd.DataFrame(TWerr)
    TWerr.insert(0,"Modelo", models,True)
    TWerr.to_csv('/home/julia.mindlin/Tesis/Capitulo3/scripts/CMIP6_storylines/DJF/indices/TW_index_errors_'+season+'.csv',float_format='%g')


#Sea Surface Temperature changes - timeseries
var = 'mon/tos'
variables = ['tos']
dato = cargo_todo(ruta,experiments,models,var)
seasons = ['DJF'] #['MAM','JJA','SON']
ens_members_hist = np.array([2,10,3,2,25,11,3,2,10,5,2,6,4,4,1,1,10,32,1,50,10,10,10,5,5,3,1,1,14])
ens_members_rcp = np.array([1,3,1,2,25,2,5,1,6,5,4,4,4,4,1,1,1,6,3,3,1,10,10,1,2,1,1,1,2])
season = 'DJF'
path_indices = '/home/julia.mindlin/Tesis/Capitulo3/scripts/CMIP6_storylines/DJF/indices'
#Open indices
gloW  = pd.read_csv(path_indices+'/GW_index_'+season+'.csv')
gw_index = gloW.iloc[:,2].values

#Central Pacific CMIP6 ensemble standard deviation maximum 
box = [5,-5,180,250]
for season in seasons:
    ts_hist, ts_rcp = im.asym_sst_box_time_series(dato,season,models,experiments,box)
    sst_mean = np.array([])
    sst_error = np.array([])
    for i in range(len(models)):
        model = models[i]
        factor_hist = np.sqrt(len(ts_hist[model])*ens_members_hist[i])
        factor_rcp = np.sqrt(len(ts_rcp[model])*ens_members_rcp[i])
        std_hist = np.std(ts_hist[model].values)/factor_hist
        std_rcp = np.std(ts_rcp[model].values)/factor_rcp
        mean = np.mean(ts_rcp[model].values) - np.mean(ts_hist[model].values)
        error = np.sqrt(std_hist**2 + std_rcp**2)
        sst_mean = np.append(sst_mean,mean)
        sst_error = np.append(sst_error,error)

    SSTerr = {season+'_mean':sst_mean,season+'_error': sst_error,season+'_/K':sst_mean/gw_index}
    SSTerr = pd.DataFrame(SSTerr)
    SSTerr.insert(0,"Modelo", models,True)
    SSTerr.to_csv('/home/julia.mindlin/Tesis/Capitulo3/scripts/CMIP6_storylines/DJF/indices/C_std_asym_index_errors_'+season+'.csv',float_format='%g')


