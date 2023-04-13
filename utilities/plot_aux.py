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

#Create plots
path_maps = '/home/julia.mindlin/Tesis/Capitulo3/scripts/CMIP6_storylines/Regresions_across_models/JJA/sensitivity_maps/zg'
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

#Create plots
path_maps = '/home/julia.mindlin/Tesis/Capitulo3/scripts/CMIP6_storylines/Regresions_across_models/JJA/sensitivity_maps/zg'
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


path_figs = '/home/julia.mindlin/Tesis/Capitulo3/figures/SST_regressions_across_models/JJA/zg'
#figure = plot_sensitivity_maps.plot_sensitivity_ua_carree(maps,maps_pval,frac_var,path_figs)
figure = plot_sensitivity_maps.plot_sensitivity_zg(maps,maps_pval,frac_var,path_figs)

