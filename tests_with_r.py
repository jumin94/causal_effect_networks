import xarray as xr
import numpy as np
import statsmodels.api as sm
import pandas as pd
import json 
import os
from esmvaltool.diag_scripts.shared import run_diagnostic, get_cfg, group_metadata
from sklearn import linear_model
import glob
from scipy import signal
import netCDF4
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature
from cartopy.util import add_cyclic_point
import matplotlib.path as mpath
import matplotlib as mpl
# To use R packages:
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
import rpy2.robjects as robjects
# Convert pandas.DataFrames to R dataframes automatically.
pandas2ri.activate()
relaimpo = importr("relaimpo")
    
path = '/datos/julia.mindlin/output/Amon_Omon_recipe_filled_datasets_ssp585_20230328_174233/preproc/multiple_regression_indices'
dato = xr.open_dataset('/datos/julia.mindlin/output/Amon_Omon_recipe_filled_datasets_ssp585_20230328_174233/preproc/multiple_regression_indices/ua850/CMIP6_ACCESS-CM2_Amon_historical-ssp585_r1i1p1f1_ua_1900-2022.nc')

def stand(dato):
    anom = (dato - np.mean(dato))/np.std(dato)
    return anom


def filtro(dato):
    """Apply a rolling mean of 5 years and remov the NaNs resulting bigining and end"""
    signal = dato - dato.rolling(time=20, center=True).mean()
    signal_out = signal.dropna('time', how='all')
    return signal_out
        
def replace_nans_with_zero(x):
    return np.where(np.isnan(x), 0, x)


def iod(iod_e,iod_w):
    iod_index = iod_w - iod_e
    return iod_index


ts_dict = {v: filtro(xr.open_dataset(path+'/'+v+'/CMIP6_CanESM5_'+m+'mon_historical-ssp585_r5i1p2f1_'+s+'_1900-2022.nc')[s].sel(time=slice('1900','2099'))).values for s,v,m in zip(sn,var,mon) if (v != "tos_iod_e") & (v != "tos_iod_w")}
ts_dict["iod"] = iod([filtro(xr.open_dataset(path+'/tos_iod_e/CMIP6_CanESM5_Omon_historical-ssp585_r5i1p2f1_tos_1900-2022.nc')['tos'].sel(time=slice('1900','2099'))).values][0],[filtro(xr.open_dataset(path+'/tos_iod_w/CMIP6_CanESM5_Omon_historical-ssp585_r5i1p2f1_tos_1900-2022.nc')['tos'].sel(time=slice('1900','2099'))).values][0])
        
y = pd.DataFrame(ts_dict).values
x = filtro(dato.sel(time=slice('1900','2099')).ua).isel(lat=30).isel(lon=40).values
matrix = np.c_[x,y_ones]
print(matrix)

robjects.r('''
    rel_importante <- function(x,y) {
        relation <- lm(x~1+y)
        metrics <- calc.relimp(relation, type = "lmg")
        return(metrics)
    }
''')
res = robjects.globalenv['rel_importante'](x,y)