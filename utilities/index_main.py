#main_indices
import os, fnmatch
import numpy as np
import pandas as pd
import xarray as xr
from index_module import changes_list,components
from index_module import sst_index 

models = ['ACCESS-CM2','ACCESS-ESM1-5','BCC-CSM2-MR','CAMS-CSM1-0','CanESM5','CESM2_','CESM2-WACCM','CMCC-CM2-SR5','CNRM-CM6-1','CNRM-ESM2-1','EC-Earth3','FGOALS-g3','GFDL-ESM4','HadGEM3-GC31-LL','HadGEM3-GC31-MM','IITM-ESM','INM-CM4-8','INM-CM5-0','IPSL-CM6A-LR','KACE-1-0-G','MIROC6','MIROC-ES2L','MPI-ESM1-2-HR','MPI-ESM1-2-LR','MRI-ESM2-0','NESM3','NorESM2-LM','NorESM2-MM','UKESM1-0-LL','TaiESM1']

scenarios = ['historical','ssp585']

path = '/datos/julia.mindlin/CMIP6_ensambles/preprocesados' #/historical/mon/tas/past'
os.chdir(path)
os.getcwd()

ruta = path

#Tropical warming
var = 'mon/ta'
variables = ['ta']
ruta = path
dato = cargo_todo(ruta,scenarios,models,var)
seasons = ['DJF'] #['MAM','JJA','SON']

for season in seasons:
    DT = tropical_warming(dato,season,models,scenarios) 
    TA = {season: DT}
    TA = pd.DataFrame(TA)
    TA.insert(0,"Modelo", models,True)
    TA.to_csv('/home/julia.mindlin/Tesis/Capitulo3/enso_variability/data/processed/indices/TA_index_'+season+'.csv',float_format='%g')

#Global warming

var = 'mon/tas'
variables = ['tas']
dato = cargo_todo(ruta,scenarios,models,var)
seasons = ['DJF'] #['MAM','JJA','SON']

for season in seasons:
    DT = global_warming(dato,season,models,scenarios)
    GW = {season: DT}
    GW = pd.DataFrame(GW)
    GW.insert(0,"Modelo", models,True)
    GW.to_csv('/home/julia.mindlin/Tesis/Capitulo3/enso_variability/data/processed/indices/GW_index_'+season+'.csv',float_format='%g')

#Stratospheric polar vortex
var = 'mon/ua/50'
variables = ['ua']
dato = cargo_todo(ruta,scenarios,models,var)
seasons = ['DJF'] #['MAM','JJA','SON']

#for season in seasons:
#    DV = stratospheric_vortex(dato,season)
#    SV = {season: DV}
#    SV = pd.DataFrame(SV)
#    SV.insert(0,"Modelo", models,True)
#    SV.to_csv('/home/julia.mindlin/Tesis/Capitulo3/enso_variability/data/processed/indices/SV_index_'+season+'.csv',float_format='%g')


#SST changes
var = 'mon/tos'
variables = ['tos']
dato = cargo_todo(ruta,scenarios,models,var)
seasons = ['SON'] #['MAM','JJA','SON']seasons = ['MAM','JJA','SON']

#Indico
box_w = [10,-10,50,70]
box_e = [0,-10,90,110]
for season in seasons:
    ssts = changes_list(dato,scenarios,models,season)
    D_SST1 = sst_index(models,box_w,ssts)
    D_SST2 = sst_index(models,box_e,ssts)
    D_SST = D_SST1 - D_SST2
    D_SST = {season: D_SST}
    D_SST = pd.DataFrame(D_SST)
    D_SST.insert(0,"Modelo", models,True)
    D_SST.to_csv('/home/julia.mindlin/Tesis/Capitulo3/scripts/CMIP6_storylines/Regresions_across_models/SON/indices/Indico_full_index_'+season+'.csv',float_format='%g')

#Niño3.4
box = [5,-5, 190,240]
for season in seasons:
    ssts = changes_list(dato,scenarios,models,season)
    D_SST = sst_index(models,box,ssts)
    D_SST = {season: D_SST}
    D_SST = pd.DataFrame(D_SST)
    D_SST.insert(0,"Modelo", models,True)
    D_SST.to_csv('/home/julia.mindlin/Tesis/Capitulo3/scripts/CMIP6_storylines/Regresions_across_models/SON/indices/Nino34_full_index_'+season+'.csv',float_format='%g')


#Niño1.2
box = [0,-10,270,280]
for season in seasons:
    ssts = changes_list(dato,scenarios,models,season)
    D_SST = sst_index(models,box,ssts)
    D_SST = {season: D_SST}
    D_SST = pd.DataFrame(D_SST)
    D_SST.insert(0,"Modelo", models,True)
    D_SST.to_csv('/home/julia.mindlin/Tesis/Capitulo3/scripts/CMIP6_storylines/Regresions_across_models/SON/indices/Nino12_full_index_'+season+'.csv',float_format='%g')

#Niño4
box = [5,-5,160,210]
for season in seasons:
    ssts = changes_list(dato,scenarios,models,season)
    D_SST = sst_index(models,box,ssts)
    D_SST = {season: D_SST}
    D_SST = pd.DataFrame(D_SST)
    D_SST.insert(0,"Modelo", models,True)
    D_SST.to_csv('/home/julia.mindlin/Tesis/Capitulo3/scripts/CMIP6_storylines/Regresions_across_models/SON/indices/Nino4_full_index_'+season+'.csv',float_format='%g')

#C index = 1.7*niño4 - 0.1*niño12
#E index = niño12 - 0.5*nino4

