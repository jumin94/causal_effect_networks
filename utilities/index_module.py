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

#Select season and compute seasonal mean
def seasonal_data(data,season):
    # select DJF
    DA_season = data.sel(time = data.time.dt.season==season)

    # calculate mean per year
    DA_season_mean = DA_season.groupby(DA_season.time.dt.year).mean(dim='time')
    return DA_season_mean

def cross_year_season(month,season):
    #Season is a list with two values, begining and endig season
    return (month >= season[0]) & (month <= season[1])

def ONDJFMA_season(month,season):
    #Season is a list with two values, begining and endig season
    return (month >= season[0]) | (month <= season[1])

def seasonal_data_month(data,season):
    # select DJF
    DA_season = data.sel(time = data.time.dt.month==slice(season[0],season[1]))

    # calculate mean per year
    DA_season_mean = DA_season.groupby(DA_season.time.dt.year).mean(dim='time')
    return DA_season_mean

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

def tropical_warming_time_series(dato,season,models,scenarios):
    TS_h = {}; TS_rcp = {}
    for model in models:
        ta_hist = xr.open_dataset(dato[scenarios[0]][model]['1940-1969'][0])
        ta_rcp5 = xr.open_dataset(dato[scenarios[1]][model]['2070-2099'][0])
        #Selecciono las variables y calculo los indices
        ta_h = ta_hist.ta
        ta_h.attrs = ta_hist.ta.attrs
        ta_rcp = ta_rcp5.ta
        ta_rcp.attrs = ta_rcp5.ta.attrs

        seasonalh = seasonal_data(ta_h,season)
        ta_h_lonMEAN = seasonalh.mean(dim='lon')
        ta_h_lonMEAN = ta_h_lonMEAN.isel(plev=0)
        ta_h_MEAN= ta_h_lonMEAN.sel(lat=slice(15,-15)).mean(dim='lat')

        seasonalrcp = seasonal_data(ta_rcp,season)
        ta_rcp_lonMEAN = seasonalrcp.mean(dim='lon')
        ta_rcp_lonMEAN = ta_rcp_lonMEAN.isel(plev=0)
        ta_rcp_MEAN = ta_rcp_lonMEAN.sel(lat=slice(15,-15)).mean(dim='lat') 
      
        TS_h[model] = ta_h_MEAN; TS_rcp[model] = ta_rcp_MEAN
    return TS_h, TS_rcp

def tropical_warming_time_series_flex_season(dato,season,models,scenarios):
    TS_h = {}; TS_rcp = {}
    for model in models:
        ta_hist = xr.open_dataset(dato[scenarios[0]][model]['1940-1969'][0])
        ta_rcp5 = xr.open_dataset(dato[scenarios[1]][model]['2070-2099'][0])
        #Selecciono las variables y calculo los indices
        ta_h = ta_hist.ta
        ta_h.attrs = ta_hist.ta.attrs
        ta_rcp = ta_rcp5.ta
        ta_rcp.attrs = ta_rcp5.ta.attrs

        seasonalh = ta_h.sel(time=cross_year_season(ta_h['time.month'],season))
    
        ta_h_lonMEAN = seasonalh.mean(dim='lon')
        ta_h_lonMEAN = ta_h_lonMEAN.isel(plev=0)
        ta_h_MEAN= ta_h_lonMEAN.sel(lat=slice(15,-15)).mean(dim='lat')

        seasonalrcp = ta_rcp.sel(time=cross_year_season(ta_rcp['time.month'],season))
        ta_rcp_lonMEAN = seasonalrcp.mean(dim='lon')
        ta_rcp_lonMEAN = ta_rcp_lonMEAN.isel(plev=0)
        ta_rcp_MEAN = ta_rcp_lonMEAN.sel(lat=slice(15,-15)).mean(dim='lat') 
      
        TS_h[model] = ta_h_MEAN; TS_rcp[model] = ta_rcp_MEAN
    return TS_h, TS_rcp

def tropical_warming_time_series_ONDJFMA_season(dato,season,models,scenarios):
    TS_h = {}; TS_rcp = {}
    for model in models:
        ta_hist = xr.open_dataset(dato[scenarios[0]][model]['1940-1969'][0])
        ta_rcp5 = xr.open_dataset(dato[scenarios[1]][model]['2070-2099'][0])
        #Selecciono las variables y calculo los indices
        ta_h = ta_hist.ta
        ta_h.attrs = ta_hist.ta.attrs
        ta_rcp = ta_rcp5.ta
        ta_rcp.attrs = ta_rcp5.ta.attrs

        seasonalh = ta_h.sel(time=ONDJFMA_season(ta_h['time.month'],season))
    
        ta_h_lonMEAN = seasonalh.mean(dim='lon')
        ta_h_lonMEAN = ta_h_lonMEAN.isel(plev=0)
        ta_h_MEAN= ta_h_lonMEAN.sel(lat=slice(15,-15)).mean(dim='lat')

        seasonalrcp = ta_rcp.sel(time=ONDJFMA_season(ta_rcp['time.month'],season))
        ta_rcp_lonMEAN = seasonalrcp.mean(dim='lon')
        ta_rcp_lonMEAN = ta_rcp_lonMEAN.isel(plev=0)
        ta_rcp_MEAN = ta_rcp_lonMEAN.sel(lat=slice(15,-15)).mean(dim='lat') 
      
        TS_h[model] = ta_h_MEAN; TS_rcp[model] = ta_rcp_MEAN
    return TS_h, TS_rcp

def tropical_warming(dato,season,models,scenarios):
    DT_SONi = np.array([])
    for model in models:
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

def tropical_warming_flex_season(dato,season,models,scenarios):
    DTi = np.array([])
    for model in models:
        ta_hist = xr.open_dataset(dato[scenarios[0]][model]['1940-1969'][0])
        ta_rcp5 = xr.open_dataset(dato[scenarios[1]][model]['2070-2099'][0])
        #Selecciono las variables y calculo los indices
        ta_h = ta_hist.ta
        ta_h.attrs = ta_hist.ta.attrs
        ta_rcp = ta_rcp5.ta
        ta_rcp.attrs = ta_rcp5.ta.attrs
        seasonalrcp = ta_rcp.groupby('time.month').mean(dim='time')
        seasonalh = ta_h.groupby('time.month').mean(dim='time')
        ta_h = seasonalh.sel(month=slice(season[0],season[1])).mean(dim='month')
        ta_hMEAN = ta_h.mean(dim='lon')
        ta_hMEAN = ta_hMEAN.isel(plev=0)
        ta_hMEAN = ta_hMEAN.sel(lat=slice(15,-15))
        ta_hMEAN = ta_hMEAN.mean(dim='lat')
        ta_rcp = seasonalrcp.sel(month=slice(season[0],season[1])).mean(dim='month')
        ta_rcpMEAN = ta_rcp.mean(dim='lon')
        ta_rcpMEAN = ta_rcpMEAN.isel(plev=0)
        ta_rcpMEAN = ta_rcpMEAN.sel(lat=slice(15,-15))
        ta_rcpMEAN = ta_rcpMEAN.mean(dim='lat')
        print(model)
        DT = ta_rcpMEAN - ta_hMEAN
        print(DT)
        DTi = np.append(DTi,DT)
        del ta_h
        del ta_rcp
    
    return DTi


#Global Warming--------------------------------------------------------------------------------------------

def global_warming_time_series(dato,season,models,scenarios):
    TS_h = {}; TS_rcp = {}
    for model in models:
        ta_hist = xr.open_dataset(dato[scenarios[0]][model]['1940-1969'][0])
        ta_rcp5 = xr.open_dataset(dato[scenarios[1]][model]['2070-2099'][0])
        #Selecciono las variables y calculo los indices
        ta_h = ta_hist.tas
        ta_h.attrs = ta_hist.tas.attrs
        ta_rcp = ta_rcp5.tas
        ta_rcp.attrs = ta_rcp5.tas.attrs
        
        seasonalh = seasonal_data(ta_h,season)
        ta_h_lonMEAN = seasonalh.mean(dim='lon')
        ta_h_lonMEAN = ta_h_lonMEAN.fillna(ta_h_lonMEAN[-1]-1)
        lats = np.cos(ta_h_lonMEAN.lat.values*np.pi/180)
        s = sum(lats)
        ta_h_MEAN = (ta_h_lonMEAN*lats).sum(dim='lat')/s

        seasonalrcp = seasonal_data(ta_rcp,season)
        ta_rcp_lonMEAN = seasonalrcp.mean(dim='lon')
        ta_rcp_lonMEAN = ta_rcp_lonMEAN.fillna(ta_rcp_lonMEAN[-1]-1)
        lats = np.cos(ta_rcp_lonMEAN.lat.values*np.pi/180)
        s = sum(lats)
        ta_rcp_MEAN = (ta_rcp_lonMEAN*lats).sum(dim='lat')/s
        TS_h[model] = ta_h_MEAN; TS_rcp[model] = ta_rcp_MEAN
    return TS_h, TS_rcp

def global_warming_time_series_flex_season(dato,season,models,scenarios):
    TS_h = {}; TS_rcp = {}
    for model in models:
        ta_hist = xr.open_dataset(dato[scenarios[0]][model]['1940-1969'][0])
        ta_rcp5 = xr.open_dataset(dato[scenarios[1]][model]['2070-2099'][0])
        #Selecciono las variables y calculo los indices
        ta_h = ta_hist.tas
        ta_h.attrs = ta_hist.tas.attrs
        ta_rcp = ta_rcp5.tas
        ta_rcp.attrs = ta_rcp5.tas.attrs
        
        seasonalh = ta_h.sel(time=cross_year_season(ta_h['time.month'],season))
        ta_h_lonMEAN = seasonalh.mean(dim='lon')
        ta_h_lonMEAN = ta_h_lonMEAN.fillna(ta_h_lonMEAN[-1]-1)
        lats = np.cos(ta_h_lonMEAN.lat.values*np.pi/180)
        s = sum(lats)
        ta_h_MEAN = (ta_h_lonMEAN*lats).sum(dim='lat')/s

        seasonalrcp = ta_rcp.sel(time=cross_year_season(ta_rcp['time.month'],season))
        ta_rcp_lonMEAN = seasonalrcp.mean(dim='lon')
        ta_rcp_lonMEAN = ta_rcp_lonMEAN.fillna(ta_rcp_lonMEAN[-1]-1)
        lats = np.cos(ta_rcp_lonMEAN.lat.values*np.pi/180)
        s = sum(lats)
        ta_rcp_MEAN = (ta_rcp_lonMEAN*lats).sum(dim='lat')/s
        TS_h[model] = ta_h_MEAN; TS_rcp[model] = ta_rcp_MEAN
    return TS_h, TS_rcp

def global_warming_time_series_ONDJFMA_season(dato,season,models,scenarios):
    TS_h = {}; TS_rcp = {}
    for model in models:
        ta_hist = xr.open_dataset(dato[scenarios[0]][model]['1940-1969'][0])
        ta_rcp5 = xr.open_dataset(dato[scenarios[1]][model]['2070-2099'][0])
        #Selecciono las variables y calculo los indices
        ta_h = ta_hist.tas
        ta_h.attrs = ta_hist.tas.attrs
        ta_rcp = ta_rcp5.tas
        ta_rcp.attrs = ta_rcp5.tas.attrs
        
        seasonalh = ta_h.sel(time=ONDJFMA_season(ta_h['time.month'],season))
        ta_h_lonMEAN = seasonalh.mean(dim='lon')
        ta_h_lonMEAN = ta_h_lonMEAN.fillna(ta_h_lonMEAN[-1]-1)
        lats = np.cos(ta_h_lonMEAN.lat.values*np.pi/180)
        s = sum(lats)
        ta_h_MEAN = (ta_h_lonMEAN*lats).sum(dim='lat')/s

        seasonalrcp = ta_rcp.sel(time=ONDJFMA_season(ta_rcp['time.month'],season))
        ta_rcp_lonMEAN = seasonalrcp.mean(dim='lon')
        ta_rcp_lonMEAN = ta_rcp_lonMEAN.fillna(ta_rcp_lonMEAN[-1]-1)
        lats = np.cos(ta_rcp_lonMEAN.lat.values*np.pi/180)
        s = sum(lats)
        ta_rcp_MEAN = (ta_rcp_lonMEAN*lats).sum(dim='lat')/s
        TS_h[model] = ta_h_MEAN; TS_rcp[model] = ta_rcp_MEAN
    return TS_h, TS_rcp

def global_warming(dato,season,models,scenarios):
    DTi = np.array([])
    for model in models:
        ta_hist = xr.open_dataset(dato[scenarios[0]][model]['1940-1969'][0])
        ta_rcp5 = xr.open_dataset(dato[scenarios[1]][model]['2070-2099'][0])
        #Selecciono las variables y calculo los indices
        ta_h = ta_hist.tas
        ta_h.attrs = ta_hist.tas.attrs
        ta_rcp = ta_rcp5.tas
        ta_rcp.attrs = ta_rcp5.tas.attrs
        
        seasonalrcp = ta_rcp.groupby('time.season').mean(dim='time')
        seasonalh = ta_h.groupby('time.season').mean(dim='time')
        ta_h = seasonalh.sel(season=season)
        ta_hMEAN = ta_h.mean(dim='lon')
        ta_hMEAN = ta_hMEAN.fillna(ta_hMEAN[-1]-1)
        lats = np.cos(ta_hMEAN.lat.values*np.pi/180)
        s = sum(lats)
        ta_hMEAN = sum(ta_hMEAN*lats)/s
        
        ta_rcp = seasonalrcp.sel(season=season)
        ta_rcpMEAN = ta_rcp.mean(dim='lon')
        ta_rcpMEAN = ta_rcpMEAN.fillna(ta_rcpMEAN[-1]-1)
        lats = np.cos(ta_rcpMEAN.lat.values*np.pi/180)
        s = sum(lats)
        ta_rcpMEAN = sum(ta_rcpMEAN*lats)/s
        DT = ta_rcpMEAN - ta_hMEAN
        print(DT)
        DTi = np.append(DTi,DT)
        del ta_h
        del ta_rcp
    return DTi


def global_warming_flex_season(dato,season,models,scenarios):
    DTi = np.array([])
    for model in models:
        ta_hist = xr.open_dataset(dato[scenarios[0]][model]['1940-1969'][0])
        ta_rcp5 = xr.open_dataset(dato[scenarios[1]][model]['2070-2099'][0])
        #Selecciono las variables y calculo los indices
        ta_h = ta_hist.tas
        ta_h.attrs = ta_hist.tas.attrs
        ta_rcp = ta_rcp5.tas
        ta_rcp.attrs = ta_rcp5.tas.attrs
        
        seasonalrcp = ta_rcp.groupby('time.month').mean(dim='time')
        seasonalh = ta_h.groupby('time.month').mean(dim='time')
        ta_h = seasonalh.sel(month=slice(season[0],season[1])).mean(dim='month')
        ta_hMEAN = ta_h.mean(dim='lon')
        ta_hMEAN = ta_hMEAN.fillna(ta_hMEAN[-1]-1)
        lats = np.cos(ta_hMEAN.lat.values*np.pi/180)
        s = sum(lats)
        ta_hMEAN = sum(ta_hMEAN*lats)/s
        
        ta_rcp = seasonalrcp.sel(month=slice(season[0],season[1])).mean(dim='month')
        ta_rcpMEAN = ta_rcp.mean(dim='lon')
        ta_rcpMEAN = ta_rcpMEAN.fillna(ta_rcpMEAN[-1]-1)
        lats = np.cos(ta_rcpMEAN.lat.values*np.pi/180)
        s = sum(lats)
        ta_rcpMEAN = sum(ta_rcpMEAN*lats)/s
        DT = ta_rcpMEAN - ta_hMEAN
        print(DT)
        DTi = np.append(DTi,DT)
        del ta_h
        del ta_rcp
    return DTi



#Stratospheric Polar Vortex---------------------------------------------------------------------------------

def stratospheric_vortex_time_series_flex_season(dato,season,models,scenarios):
    TS_h = {}; TS_rcp = {}
    for model in models:
        ua_hist = xr.open_dataset(dato[scenarios[0]][model]['1940-1969'][0])
        ua_rcp5 = xr.open_dataset(dato[scenarios[1]][model]['2070-2099'][0])
        #Selecciono las variables y calculo los indices
        ua_h = ua_hist.ua
        ua_h.attrs = ua_hist.ua.attrs
        ua_rcp = ua_rcp5.ua
        ua_rcp.attrs = ua_rcp5.ua.attrs
        
        seasonalh = ua_h.sel(time=cross_year_season(ua_h['time.month'],season))
        ua_hJJA = seasonalh
        ua_hu_JJA = ua_hJJA.isel(plev=0)
        ua_hu_JJA = ua_hu_JJA.mean(dim='lon')
        ua_hu_JJA = ua_hu_JJA.sel(lat=slice(-50,-60))
        #lats = np.cos(ua_hu_JJA.lat.values*np.pi/180)
        #s = sum(lats)
        #ua_hu_JJA = sum(ua_hu_JJA*lats)/s
        ua_hu_JJA = ua_hu_JJA.mean(dim='lat')

        seasonalrcp = ua_rcp.sel(time=cross_year_season(ua_rcp['time.month'],season))
        ua_rcpJJA = seasonalrcp
        ua_rcpu_JJA = ua_rcpJJA.isel(plev=0)
        ua_rcpu_JJA = ua_rcpu_JJA.mean(dim='lon')
        ua_rcpu_JJA = ua_rcpu_JJA.sel(lat=slice(-50,-60))
        #lats = np.cos(ua_rcpu_JJA.lat.values*np.pi/180)
        #s = sum(lats)
        #ua_rcpu_JJA = sum(ua_rcpu_JJA*lats)/s
        ua_rcpu_JJA = ua_rcpu_JJA.mean(dim='lat')
        
        TS_h[model] = ua_hu_JJA; TS_rcp[model] = ua_rcpu_JJA 
    return TS_h, TS_rcp


def stratospheric_vortex(dato,season,models,scenarios):
    DU_JJAi = np.array([])
    for model in models:
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

def stratospheric_vortex_flex_season(dato,season,models,scenarios):
    DUi = np.array([])
    for model in models:
        ua_hist = xr.open_dataset(dato[scenarios[0]][model]['1940-1969'][0])
        ua_rcp5 = xr.open_dataset(dato[scenarios[1]][model]['2070-2099'][0])
        #Selecciono las variables y calculo los indices
        ua_h = ua_hist.ua
        ua_h.attrs = ua_hist.ua.attrs
        ua_rcp = ua_rcp5.ua
        ua_rcp.attrs = ua_rcp5.ua.attrs
        
        seasonalrcp = ua_rcp.groupby('time.month').mean(dim='time')
        seasonalh = ua_h.groupby('time.month').mean(dim='time')

        ua_h = seasonalh.sel(month=slice(season[0],season[1])).mean(dim='month')
        ua_hu = ua_h.isel(plev=0)
        ua_hu = ua_hu.mean(dim='lon')
        ua_hu = ua_hu.sel(lat=slice(-50,-60))
        lats = np.cos(ua_hu.lat.values*np.pi/180)
        s = sum(lats)
        ua_hu = sum(ua_hu*lats)/s

        ua_rcp = seasonalrcp.sel(month=slice(season[0],season[1])).mean(dim='month')
        ua_rcpu = ua_rcp.isel(plev=0)
        ua_rcpu = ua_rcpu.mean(dim='lon')
        ua_rcpu = ua_rcpu.sel(lat=slice(-50,-60))
        lats = np.cos(ua_rcpu.lat.values*np.pi/180)
        s = sum(lats)
        ua_rcpu = sum(ua_rcpu*lats)/s

        DU = ua_rcpu - ua_hu
        DUi = np.append(DUi,DU)

        del ua_rcpu, ua_hu
        
    return DUi

# Sea surface temperature-----------------------------------------------------------------
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

def changes_list_asym(datos,scenarios,models,season):
    SST = {}
    for i in range(len(models)):
        print(models[i])
        tos_hist = xr.open_dataset(datos[scenarios[0]][models[i]]['1940-1969'][0])
        tos_rcp5 = xr.open_dataset(datos[scenarios[1]][models[i]]['2070-2099'][0])
        tos_h = tos_hist.tos - tos_hist.tos.mean(dim='lon')
        tos_h.attrs = tos_hist.tos.attrs
        tos_rcp = tos_rcp5.tos - tos_rcp5.tos.mean(dim='lon')
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

def changes_list_flex_season(datos,scenarios,models,season):
    SST = {}
    for i in range(len(models)):
        print(models[i])
        tos_hist = xr.open_dataset(datos[scenarios[0]][models[i]]['1940-1969'][0])
        tos_rcp5 = xr.open_dataset(datos[scenarios[1]][models[i]]['2070-2099'][0])
        tos_h = tos_hist.tos
        tos_h.attrs = tos_hist.tos.attrs
        tos_rcp = tos_rcp5.tos
        tos_rcp.attrs = tos_rcp5.tos.attrs
        seasonal = tos_h.groupby('time.month').mean(dim='time')
        tosDJF = seasonal.sel(month=slice(season[0],season[1])).mean(dim='month')
        tosDJF.attrs = tos_h.attrs
        seasonal_r = tos_rcp.groupby('time.month').mean(dim='time')
        tosDJF_r = seasonal_r.sel(month=slice(season[0],season[1])).mean(dim='month')
        tosDJF_r.attrs = tos_rcp.attrs
        sst_change = tosDJF_r - tosDJF
        SST[models[i]] = []
        SST[models[i]].append(sst_change)

    return SST

def changes_list_asym_flex_season(datos,scenarios,models,season):
    SST = {}
    for i in range(len(models)):
        print(models[i])
        tos_hist = xr.open_dataset(datos[scenarios[0]][models[i]]['1940-1969'][0])
        tos_rcp5 = xr.open_dataset(datos[scenarios[1]][models[i]]['2070-2099'][0])
        tos_h = tos_hist.tos - tos_hist.tos.mean(dim='lon')
        tos_h.attrs = tos_hist.tos.attrs
        tos_rcp = tos_rcp5.tos - tos_rcp5.tos.mean(dim='lon')
        tos_rcp.attrs = tos_rcp5.tos.attrs
        seasonal = tos_h.groupby('time.month').mean(dim='time')
        tos_season = seasonal.sel(month=slice(season[0],season[1])).mean(dim='month')
        tos_season.attrs = tos_h.attrs
        seasonal_r = tos_rcp.groupby('time.month').mean(dim='time')
        tos_season_r = seasonal_r.sel(month=slice(season[0],season[1])).mean(dim='month')
        tos_season_r.attrs = tos_rcp.attrs
        sst_change = tos_season_r - tos_season
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

def sst_index_full(modelos,box,SST):
    index_out = np.array([])
    for model in modelos:
        print(model)
        full = SST[model][0]
        if model == 'IPSL-CM6A-LR2':
            index = full.sel(lat=slice(box[0],box[1])).sel(lon=slice(box[2],box[3])).mean(dim='y').mean(dim='x').mean(dim='lat').mean(dim='lon')
        else:
            index = full.sel(lat=slice(box[0],box[1])).sel(lon=slice(box[2],box[3])).mean(dim='lat').mean(dim='lon')
        print(index)
        index_out = np.append(index_out,index)
    return index_out

def sst_index_asym(modelos,box,SST):
    index_out = np.array([])
    for model in modelos:
        print(model)
        asym = SST[model][0]
        if model == 'IPSL-CM6A-LR2':
            index = asym.sel(lat=slice(box[0],box[1])).sel(lon=slice(box[2],box[3])).mean(dim='y').mean(dim='x').mean(dim='lat').mean(dim='lon')
        else:
            index = asym.sel(lat=slice(box[0],box[1])).sel(lon=slice(box[2],box[3])).mean(dim='lat').mean(dim='lon')
        print(index)
        index_out = np.append(index_out,index)
    return index_out

def sst_index_asym_change(modelos,box,SST):
    index_out = np.array([])
    for model in modelos:
        print(model)
        asym = components(model,SST)[2]
        if model == 'IPSL-CM6A-LR2':
            index = asym.sel(lat=slice(box[0],box[1])).sel(lon=slice(box[2],box[3])).mean(dim='y').mean(dim='x').mean(dim='lat').mean(dim='lon')
        else:
            index = asym.sel(lat=slice(box[0],box[1])).sel(lon=slice(box[2],box[3])).mean(dim='lat').mean(dim='lon')
        print(index)
        index_out = np.append(index_out,index)
    return index_out


def full_sst_box_time_series(dato,season,models,scenarios,box):
    TS_h = {}; TS_rcp = {}
    for model in models:
        print(model)
        tos_hist = xr.open_dataset(dato[scenarios[0]][model]['1940-1969'][0])
        tos_rcp5 = xr.open_dataset(dato[scenarios[1]][model]['2070-2099'][0])
        #Selecciono las variables y calculo los indices
        tos_h = tos_hist.tos
        tos_h.attrs = tos_hist.tos.attrs
        tos_rcp = tos_rcp5.tos
        tos_rcp.attrs = tos_rcp5.tos.attrs

        seasonalh = seasonal_data(tos_h,season)
        tos_h_full_box = seasonalh.sel(lat=slice(box[0],box[1])).sel(lon=slice(box[2],box[3])).mean(dim='lat').mean(dim='lon')

        seasonalrcp = seasonal_data(ta_rcp,season)
        tos_rcp_full_box = seasonalrcp.sel(lat=slice(box[0],box[1])).sel(lon=slice(box[2],box[3])).mean(dim='lat').mean(dim='lon')

        TS_h[model] = tos_h_full_box; TS_rcp[model] = tos_rcp_full_box
    return TS_h, TS_rcp


def asym_sst_box_time_series(dato,season,models,scenarios,box):
    TS_h = {}; TS_rcp = {}
    for model in models:
        print(model)
        tos_hist = xr.open_dataset(dato[scenarios[0]][model]['1940-1969'][0])
        tos_rcp5 = xr.open_dataset(dato[scenarios[1]][model]['2070-2099'][0])
        #Selecciono las variables y calculo los indices
        tos_h = tos_hist.tos
        tos_h.attrs = tos_hist.tos.attrs
        tos_rcp = tos_rcp5.tos
        tos_rcp.attrs = tos_rcp5.tos.attrs

        seasonalh = seasonal_data(tos_h,season)
        tos_h_lonMEAN = seasonalh.mean(dim='lon')
        tos_h_asym = seasonalh - tos_h_lonMEAN 
        tos_h_asym_box = tos_h_asym.sel(lat=slice(box[0],box[1])).sel(lon=slice(box[2],box[3])).mean(dim='lat').mean(dim='lon')

        seasonalrcp = seasonal_data(tos_rcp,season)
        tos_rcp_lonMEAN = seasonalrcp.mean(dim='lon')
        tos_rcp_asym = seasonalrcp - tos_rcp_lonMEAN
        tos_rcp_asym_box = tos_rcp_asym.sel(lat=slice(box[0],box[1])).sel(lon=slice(box[2],box[3])).mean(dim='lat').mean(dim='lon')

        TS_h[model] = tos_h_asym_box; TS_rcp[model] = tos_rcp_asym_box
    return TS_h, TS_rcp

def asym_sst_box_time_series_flex_season(dato,season,models,scenarios,box):
    TS_h = {}; TS_rcp = {}
    for model in models:
        print(model)
        tos_hist = xr.open_dataset(dato[scenarios[0]][model]['1940-1969'][0])
        tos_rcp5 = xr.open_dataset(dato[scenarios[1]][model]['2070-2099'][0])
        #Selecciono las variables y calculo los indices
        tos_h = tos_hist.tos
        tos_h.attrs = tos_hist.tos.attrs
        tos_rcp = tos_rcp5.tos
        tos_rcp.attrs = tos_rcp5.tos.attrs

        seasonalh = tos_h.sel(time=cross_year_season(tos_h['time.month'],season))
        tos_h_lonMEAN = seasonalh.mean(dim='lon')
        tos_h_asym = seasonalh - tos_h_lonMEAN 
        tos_h_asym_box = tos_h_asym.sel(lat=slice(box[0],box[1])).sel(lon=slice(box[2],box[3])).mean(dim='lat').mean(dim='lon')

        seasonalrcp = tos_rcp.sel(time=cross_year_season(tos_rcp['time.month'],season))
        tos_rcp_lonMEAN = seasonalrcp.mean(dim='lon')
        tos_rcp_asym = seasonalrcp - tos_rcp_lonMEAN
        tos_rcp_asym_box = tos_rcp_asym.sel(lat=slice(box[0],box[1])).sel(lon=slice(box[2],box[3])).mean(dim='lat').mean(dim='lon')

        TS_h[model] = tos_h_asym_box; TS_rcp[model] = tos_rcp_asym_box
    return TS_h, TS_rcp

def asym_sst_box_time_series_flex_season(dato,season,models,scenarios,box):
    TS_h = {}; TS_rcp = {}
    for model in models:
        print(model)
        tos_hist = xr.open_dataset(dato[scenarios[0]][model]['1940-1969'][0])
        tos_rcp5 = xr.open_dataset(dato[scenarios[1]][model]['2070-2099'][0])
        #Selecciono las variables y calculo los indices
        tos_h = tos_hist.tos
        tos_h.attrs = tos_hist.tos.attrs
        tos_rcp = tos_rcp5.tos
        tos_rcp.attrs = tos_rcp5.tos.attrs

        seasonalh = tos_h.sel(time=ONDJFMA_season(tos_h['time.month'],season))
        tos_h_lonMEAN = seasonalh.mean(dim='lon')
        tos_h_asym = seasonalh - tos_h_lonMEAN 
        tos_h_asym_box = tos_h_asym.sel(lat=slice(box[0],box[1])).sel(lon=slice(box[2],box[3])).mean(dim='lat').mean(dim='lon')

        seasonalrcp = tos_rcp.sel(time=ONDJFMA_season(tos_rcp['time.month'],season))
        tos_rcp_lonMEAN = seasonalrcp.mean(dim='lon')
        tos_rcp_asym = seasonalrcp - tos_rcp_lonMEAN
        tos_rcp_asym_box = tos_rcp_asym.sel(lat=slice(box[0],box[1])).sel(lon=slice(box[2],box[3])).mean(dim='lat').mean(dim='lon')

        TS_h[model] = tos_h_asym_box; TS_rcp[model] = tos_rcp_asym_box
    return TS_h, TS_rcp
