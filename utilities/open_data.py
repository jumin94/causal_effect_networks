#funciones.py

import numpy as np
import os
import glob
import pandas as pd
import xarray as xr
import os, fnmatch
import matplotlib.pyplot as plt
# Subplot number three for mean changes and other figures
import cartopy.crs as ccrs
import cartopy.feature
from cartopy.util import add_cyclic_point
import matplotlib.path as mpath
import netCDF4
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
mpl.rcParams['hatch.linewidth'] = 0.5  # previous pdf hatch linewidth
import cartopy.util as cutil

def cargo_todo_zg(scenarios,models,ruta,var):
    os.chdir(ruta)
    os.getcwd()
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
                periods = ['1940-1969'] #1940-1969
            for period in periods:
                dic[scenario][model][period] = []
                pattern1 = "*"+model+"*"+scenario+"*"+"1950-1979"+"*T42*"
                pattern2 = "*"+model+"*"+scenario+"*"+"2070-2099"+"*T42*"
                for entry in listOfFiles:
                    if fnmatch.fnmatch(entry,pattern1):
                        dato = xr.open_dataset(ruta+'/'+scenario+'/'+var+'/'+entry)
                        dic[scenario][model][period].append(dato)
                    elif fnmatch.fnmatch(entry,pattern2):
                        dato = xr.open_dataset(ruta+'/'+scenario+'/'+var+'/'+entry)
                        dic[scenario][model][period].append(dato)
    return dic


def cargo_todo(scenarios,models,ruta,var):
    os.chdir(ruta)
    os.getcwd()
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
                periods = ['1940-1969'] #1940-1969
            for period in periods:
                dic[scenario][model][period] = []
                pattern = "*"+model+"*"+scenario+"*"+period+"*T42*"
                for entry in listOfFiles:
                    if fnmatch.fnmatch(entry,pattern):
                        dato = xr.open_dataset(ruta+'/'+scenario+'/'+var+'/'+entry)
                        dic[scenario][model][period].append(dato)
    return dic