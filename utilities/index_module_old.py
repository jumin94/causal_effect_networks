#Climate indices for storylines OND
#Lo primero que hace es cargar todos los datos de temperatura para el historical y RDP8.5
#Luego calcula la media de cada uno, y hace la diferencia. 
#Debería entregar un array con todos los valores del índice. 

import numpy as np
import pandas as pd
import xarray as xr
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point
import glob
import os, fnmatch


#Clase diccionarios --------------------
class my_dictionary(dict):
    # __init__ function 
    def __init__(self):
        self = dict()
    # Function to add key:value 
    def add(self, key, value):
        self[key] = value

def cargo_todo(ruta,scenarios,models,var):
    dic = {}
    dic['historical'] = {}
    dic['ssp585'] = {}
    for scenario in dic.keys():
        listOfFiles = os.listdir(ruta+'/'+scenario+'/'+var)
        for model in models:
            dic[scenario][model] = {}
            if scenario == 'ssp585':
                periods = ['2070-2099']
            else:
                periods = ['1940-1969']
            for period in periods:
                dic[scenario][model][period] = []
                pattern = "*"+model+"*"+scenario+"*"+period+"*T42*"
                for entry in listOfFiles:
                    if fnmatch.fnmatch(entry,pattern):
                        #dato = xr.open_dataset(ruta+'/'+scenario+'/'+var+'/'+entry) 
                        dic[scenario][model][period].append(ruta+'/'+scenario+'/'+var+'/'+entry)
    return dic


#Tropical Warming -------------------------------------------------------------------------------------

def tropical_warming(dato,season,models,scenarios):
    DT_SONi = np.array([])
    for model in models:
        print(model)
        ta_hist = xr.open_dataset(dato[scenarios[0]][model]['1940-1969'][0])
        ta_rcp5 = xr.open_dataset(dato[scenarios[1]][model]['2070-2099'][0])
        #Selecciono las variables y calculo los indices
        ta_h = ta_hist.ta
        ta_h.attrs = ta_hist.ta.attrs
        ta_rcp = ta_rcp5.ta
        ta_rcp.attrs = ta_rcp5.ta.attrs
        seasonalrcp = ta_rcp.groupby('time.season').mean(dim='time')
        seasonalh = ta_h.groupby('time.season').mean(dim='time')
        ta_hSON = seasonalh.sel(season=season)
        ta_hMEAN_SON = ta_hSON.mean(dim='lon')
        ta_hMEAN_SON = ta_hMEAN_SON.isel(plev=0)
        ta_hMEAN_SON = ta_hMEAN_SON.sel(lat=slice(15,-15))
        ta_hMEAN_SON = ta_hMEAN_SON.mean(dim='lat')
        ta_rcpSON = seasonalrcp.sel(season=season)
        ta_rcpMEAN_SON = ta_rcpSON.mean(dim='lon')
        ta_rcpMEAN_SON = ta_rcpMEAN_SON.isel(plev=0)
        ta_rcpMEAN_SON = ta_rcpMEAN_SON.sel(lat=slice(15,-15))
        ta_rcpMEAN_SON = ta_rcpMEAN_SON.mean(dim='lat')
        DT_SON = ta_rcpMEAN_SON - ta_hMEAN_SON
        print(DT_SON)
        DT_SONi = np.append(DT_SONi,DT_SON)
        del ta_h
        del ta_rcp
    
    return DT_SONi

#Global Warming--------------------------------------------------------------------------------------------
#Defino los índices que quiero, DT_DJF



def global_warming(dato,season,models,scenarios):
    DT_SONi = np.array([])
    for model in models:
        print(model)
        ta_hist = xr.open_dataset(dato[scenarios[0]][model]['1940-1969'][0])
        ta_rcp5 = xr.open_dataset(dato[scenarios[1]][model]['2070-2099'][0])
        #Selecciono las variables y calculo los indices
        ta_h = ta_hist.tas
        ta_h.attrs = ta_hist.tas.attrs
        ta_rcp = ta_rcp5.tas
        ta_rcp.attrs = ta_rcp5.tas.attrs
        
        seasonalrcp = ta_rcp.groupby('time.season').mean(dim='time')
        seasonalh = ta_h.groupby('time.season').mean(dim='time')
        ta_hSON = seasonalh.sel(season='SON')
        ta_hMEAN_SON = ta_hSON.mean(dim='lon')
        ta_hMEAN_SON = ta_hMEAN_SON.fillna(ta_hMEAN_SON[-1]-1)
        lats = np.cos(ta_hMEAN_SON.lat.values*np.pi/180)
        s = sum(lats)
        ta_hMEAN_SON = sum(ta_hMEAN_SON*lats)/s
        
        ta_rcpSON = seasonalrcp.sel(season='SON')
        ta_rcpMEAN_SON = ta_rcpSON.mean(dim='lon')
        ta_rcpMEAN_SON = ta_rcpMEAN_SON.fillna(ta_rcpMEAN_SON[-1]-1)
        lats = np.cos(ta_rcpMEAN_SON.lat.values*np.pi/180)
        s = sum(lats)
        ta_rcpMEAN_SON = sum(ta_rcpMEAN_SON*lats)/s
        DT_SON = ta_rcpMEAN_SON - ta_hMEAN_SON
        print(DT_SON)
        DT_SONi = np.append(DT_SONi,DT_SON)
        del ta_h
        del ta_rcp
    return DT_SONi


#Stratospheric Polar Vortex---------------------------------------------------------------------------------

def stratospheric_vortex(dato,season,models,scenarios):
    DU_JJAi = np.array([])
    for model in models:
        print(model)
        ua_hist = xr.open_dataset(dato[scenarios[0]][model]['1940-1969'][0])
        ua_rcp5 = xr.open_dataset(dato[scenarios[1]][model]['2070-2099'][0])
        #Selecciono las variables y calculo los indices
        ua_h = ua_hist.ua
        ua_h.attrs = ua_hist.ua.attrs
        ua_rcp = ua_rcp5.ua
        ua_rcp.attrs = ua_rcp5.ua.attrs
        seasonalrcp = ua_rcp.groupby('time.season').mean(dim='time')
        seasonalh = ua_h.groupby('time.season').mean(dim='time')

        ua_hJJA = seasonalh.sel(season=season)
        ua_hu_JJA = ua_hJJA.isel(plev=0)
        ua_hu_JJA = ua_hu_JJA.mean(dim='lon')
        ua_hu_JJA = ua_hu_JJA.sel(lat=slice(-50,-60))
        lats = np.cos(ua_hu_JJA.lat.values*np.pi/180)
        s = sum(lats)
        ua_hu_JJA = sum(ua_hu_JJA*lats)/s

        ua_rcpJJA = seasonalrcp.sel(season=season)
        ua_rcpu_JJA = ua_rcpJJA.isel(plev=0)
        ua_rcpu_JJA = ua_rcpu_JJA.mean(dim='lon')
        ua_rcpu_JJA = ua_rcpu_JJA.sel(lat=slice(-50,-60))
        lats = np.cos(ua_rcpu_JJA.lat.values*np.pi/180)
        s = sum(lats)
        ua_rcpu_JJA = sum(ua_rcpu_JJA*lats)/s

        DU_JJA = ua_rcpu_JJA - ua_hu_JJA
        DU_JJAi = np.append(DU_JJAi,DU_JJA)

        del ua_rcpu_JJA, ua_hu_JJA
        
    return DU_JJAi

def sst_change(datos,scenarios,models,season):
    D_SSTi = np.array([])
    for i in range(len(models)):
        print(models[i])
        tos_hist = xr.open_dataset(datos[scenarios[0]][models[i]]['1940-1969'][0])
        tos_rcp5 = xr.open_dataset(datos[scenarios[1]][models[i]]['2070-2099'][0])
        tos_h = tos_hist.tos
        tos_h.attrs = tos_hist.tos.attrs
        tos_rcp = tos_rcp5.tos
        tos_rcp.attrs = tos_rcp5.tos.attrs
        seasonal = tos_h.groupby('time.season').mean(dim='time')
        tosDJF = seasonal.sel(season=season)
        tosDJF.attrs = tos_h.attrs
        seasonal_r = tos_rcp.groupby('time.season').mean(dim='time')
        tosDJF_r = seasonal_r.sel(season=season)
        tosDJF_r.attrs = tos_rcp.attrs
        sst_change = tosDJF_r - tosDJF
        sst_change.sel
        D_SSTi = np.append(S_SSTi,sst_change)

    return SST

def changes_list(datos,scenarios,models,season):
    SST = {}
    for i in range(len(models)):
        print(models[i])
        tos_hist = xr.open_dataset(datos[scenarios[0]][models[i]]['1940-1969'][0])
        tos_rcp5 = xr.open_dataset(datos[scenarios[1]][models[i]]['2070-2099'][0])
        tos_h = tos_hist.tos
        tos_h.attrs = tos_hist.tos.attrs
        tos_rcp = tos_rcp5.tos
        tos_rcp.attrs = tos_rcp5.tos.attrs
        seasonal = tos_h.groupby('time.season').mean(dim='time')
        tosDJF = seasonal.sel(season=season)
        tosDJF.attrs = tos_h.attrs
        seasonal_r = tos_rcp.groupby('time.season').mean(dim='time')
        tosDJF_r = seasonal_r.sel(season=season)
        tosDJF_r.attrs = tos_rcp.attrs
        sst_change = tosDJF_r - tosDJF
        SST[models[i]] = []
        SST[models[i]].append(sst_change)

    return SST

def components(model,SST):
    FULL = SST[model][0]
    GW = FULL.mean(dim='lon').mean(dim='lat').values
    base = FULL/FULL
    ZONAL = base * FULL.mean(dim='lon')
    ASYM = FULL/GW - ZONAL/GW
    return FULL/GW,ZONAL/GW,ASYM

def sst_index(modelos,box,SST):
    index_out = np.array([])
    for model in modelos:
        print(model)
        full = SST[model][0]
        if model == 'IPSL-CM6A-LR':
            index = full.sel(lat=slice(box[0],box[1])).sel(lon=slice(box[2],box[3])).mean(dim='y').mean(dim='x').mean(dim='lat').mean(dim='lon')
        else:
            index = full.sel(lat=slice(box[0],box[1])).sel(lon=slice(box[2],box[3])).mean(dim='lat').mean(dim='lon')
        print(index)
        index_out = np.append(index_out,index)
    return index_out