#I AM HUMAN
# I AM ROBOT
# I AM GAIA
import xarray as xr
import numpy as np
import statsmodels.api as sm
import pandas as pd
import json 
import os
from esmvaltool.diag_scripts.shared import run_diagnostic, get_cfg, group_metadata
from sklearn import linear_model
import glob
import netCDF4
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature
import matplotlib.path as mpath
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf
import seaborn as sns

def figure(target,predictors):
    fig = plt.figure()
    y = predictors.apply(stand_detr,axis=0).values
    for i in range(len(predictors.keys())):
        plt.plot(y[:,i])
    plt.plot(stand_detr(target))
    return fig

def autocorrelation(alias,predictors,path):
    figure = plt.figure(figsize=(20,10))
    dic = {}
    for i,key in enumerate(predictors.keys()):
        ax = plt.subplot(3,3,i+1)
        plot_acf(predictors[key], lags=20, ax=ax)
        ax.set_title(key)
        cor, conf = calculate(pd.DataFrame({key:predictors[key]}))
        sig_coefs = [c for i,c in enumerate(cor) if np.abs(c) > np.max(conf[i])]
        dic[key] = sig_coefs
    plt.savefig(path+'/autocorrelogram_'+alias)
    return dic
        
def calculate(df):
        """Calculate the ACF.
        Args:
            X (dataframe): input data
        Returns:
            autocors (array): array of autocorrelations
            conf_int (array): array of confidence intervals
        """
        autocors, conf_int = acf(
            x=df.values,
            nlags=20,
            alpha=0.05,
        )
        return autocors, conf_int 

def iod(iod_e,iod_w):
    iod_index = iod_w - iod_e
    return iod_index


def main(config):
    """Run the diagnostic."""
    cfg=get_cfg(os.path.join(config["run_dir"],"settings.yml"))
    print(cfg)
    meta = group_metadata(config["input_data"].values(), "alias")
    print(f"\n\n\n{meta}")
    cen_dict = {}
    dic_acorr = {}
    sig_acorrs = {}
    for alias, alias_list in meta.items():
        cen_dict[alias] = {}
        dic_acorr[alias] = {}
        print(f"Computing index regression for {alias}\n")
        ts_dict = {m["variable_group"]: xr.open_dataset(m["filename"])[m["short_name"]].values for m in alias_list if (m["variable_group"] != "ua850") & (m["variable_group"] != "tos_iod_e") & (m["variable_group"] != "tos_iod_w")}
        ts_dict["iod"] = iod([xr.open_dataset(m["filename"])[m["short_name"]].values for m in alias_list if m["variable_group"] == "tos_iod_e"][0],[xr.open_dataset(m["filename"])[m["short_name"]].values for m in alias_list if m["variable_group"] == "tos_iod_w"][0])
        dic_acorr[alias] = autocorrelation(alias,pd.DataFrame(ts_dict),config['plot_dir'])
        for key in pd.DataFrame(ts_dict).keys():
            if key in sig_acorrs.keys():
                if len(dic_acorr[alias][key]) != 0:
                    sig_acorrs[key].append(np.max(np.array(dic_acorr[alias][key])))
                else:
                    sig_acorrs[key].append(0.)
            else: 
                sig_acorrs[key] = []
                if len(dic_acorr[alias][key]) != 0:
                    sig_acorrs[key].append(np.max(np.array(dic_acorr[alias][key])))
                else: 
                    sig_acorrs[key].append(0.)
        
    df = pd.DataFrame([sig_acorrs[var] for var in sig_acorrs.keys()],  columns = alias_list, index = pd.DataFrame(ts_dict).keys())
    sns.heatmap(df, cmap ='RdYlGn', linewidths = 0.30, annot = True)
        
    with open(os.path.join(config["work_dir"], 'CEN_regressor_autocorrelations.json'), 'w') as f:
        json.dump(dic_acorr, f)   

if __name__ == "__main__":
    with run_diagnostic() as config:
        main(config)
                                    
