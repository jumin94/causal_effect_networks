# Genera la clase across_models que genera una regresion across models
#Imports
import numpy as np
import pandas as pd
import xarray as xr
import math
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import os, fnmatch
import glob
import csv2nc
import metpy


#Across models regression class

class across_models(object):
    def __init__(self):
        self.what_is_this = 'This performs a regression across models and plots everything'
    
    def regression_data(self,data_dic,scenarios,models,gw_index,var,levs='no'):

        #----------------------------------------
        psl = []

        for i in range(len(models)):
            print(models[i])
            hist = data_dic[scenarios[0]][models[i]]['1940-1969'][0]
            rcp585 = data_dic[scenarios[1]][models[i]]['2070-2099'][0]
            if var == 'pr':
                h = hist.pr
                h.attrs = hist.pr.attrs
                rcp = rcp585.pr
                rcp.attrs = rcp585.pr.attrs
            elif var == 'ua':
                h = hist.ua
                h.attrs = hist.ua.attrs
                rcp = rcp585.ua
                rcp.attrs = rcp585.ua.attrs
            elif var == 'psl':
                h = hist.psl
                h.attrs = hist.psl.attrs
                rcp = rcp585.psl
                rcp.attrs = rcp585.psl.attrs
            elif var == 'zg':
                h = hist.zg
                h.attrs = hist.zg.attrs
                rcp = rcp585.zg
                rcp.attrs = rcp585.zg.attrs
            
            seasonal_h = h.groupby('time.season').mean(dim='time')
            DJF_h = seasonal_h.sel(season='DJF')
            h1DJF = DJF_h.sel(lat=slice(90,-90))
            if levs == 'no':
                a = 0
            else:
                h1DJF = h1DJF.isel(plev=0)
            h1DJF.attrs = h.attrs
            seasonal_r = rcp.groupby('time.season').mean(dim='time')
            DJF_r = seasonal_r.sel(season='DJF')
            r1DJF = DJF_r.sel(lat=slice(90,-90))
            if levs == 'no':
                a = 0
            else:
                r1DJF = r1DJF.isel(plev=0)
            r1DJF.attrs = rcp.attrs
            psl_change = (r1DJF -h1DJF)/gw_index[i]
            psl.append(psl_change)
            
        self.psl_change = psl

    def regression_data_div(self,data_dic_ua,data_dic_va,scenarios,models,gw_index,var,levs='si'):

        #----------------------------------------
        psl = []

        for i in range(len(models)):
            print(models[i])
            hist_ua = data_dic_ua[scenarios[0]][models[i]]['1940-1969'][0]
            rcp585_ua = data_dic_ua[scenarios[1]][models[i]]['2070-2099'][0]
            hist_va = data_dic_va[scenarios[0]][models[i]]['1940-1969'][0]
            rcp585_va = data_dic_va[scenarios[1]][models[i]]['2070-2099'][0]
            h_ua = hist_ua.ua
            h_ua.attrs = hist_ua.ua.attrs
            rcp_ua = rcp585_ua.ua
            rcp_ua.attrs = rcp585_ua.ua.attrs
            h_va = hist_va.va
            h_va.attrs = hist_va.va.attrs
            rcp_va = rcp585_va.va
            rcp_va.attrs = rcp585_va.va.attrs

            #Calculo divergencia climatologica en pasado
            seasonal_h_ua = h_ua.groupby('time.season').mean(dim='time')
            DJF_h_ua = seasonal_h_ua.sel(season='DJF')
            h1DJF_ua = DJF_h_ua.sel(lat=slice(90,-90))
            if levs == 'no':
                a = 0
            else:
                h1DJF_ua = h1DJF_ua.isel(plev=0)
            h1DJF_ua.attrs = h_ua.attrs
            seasonal_h_va = h_va.groupby('time.season').mean(dim='time')
            DJF_h_va = seasonal_h_va.sel(season='DJF')
            h1DJF_va = DJF_h_va.sel(lat=slice(90,-90))
            if levs == 'no':
                a = 0
            else:
                h1DJF_va = h1DJF_va.isel(plev=0)
            h1DJF_va.attrs = h_va.attrs
            h1DJF_div = metpy.calc.divergence(h1DJF_ua,h1DJF_va)
            
            #Calculo divergencia climatologica en futuro 
            seasonal_r_ua = rcp_ua.groupby('time.season').mean(dim='time')
            DJF_r_ua = seasonal_r_ua.sel(season='DJF')
            r1DJF_ua = DJF_r_ua.sel(lat=slice(90,-90))
            if levs == 'no':
                a = 0
            else:
                r1DJF_ua = r1DJF_ua.isel(plev=0)
            r1DJF_ua.attrs = rcp_ua.attrs
            seasonal_r_va = rcp_va.groupby('time.season').mean(dim='time')
            DJF_r_va = seasonal_r_va.sel(season='DJF')
            r1DJF_va = DJF_r_va.sel(lat=slice(90,-90))
            if levs == 'no':
                a = 0
            else:
                r1DJF_va = r1DJF_va.isel(plev=0)
            r1DJF_va.attrs = rcp_va.attrs
            r1DJF_div = metpy.calc.divergence(r1DJF_ua,r1DJF_va)
            
            #Calculo el cambio en el campo de divergencia total
            div_change = (r1DJF_div -h1DJF_div)/gw_index[i]
            psl.append(div_change)
            
        self.psl_change = psl

    
    def perform_regression(self,indices,indices_names,gw_index,path): 
        
        Aij = pd.DataFrame(columns=['a','lat','lon'])
        Bij = pd.DataFrame(columns=['b','lat','lon'])
        Cij = pd.DataFrame(columns=['c','lat','lon'])
        Dij = pd.DataFrame(columns=['d','lat','lon'])
        Eij = pd.DataFrame(columns=['e','lat','lon'])
        Atij = pd.DataFrame(columns=['at','lat','lon'])
        Btij = pd.DataFrame(columns=['bt','lat','lon'])
        Ctij = pd.DataFrame(columns=['ct','lat','lon'])
        Dtij = pd.DataFrame(columns=['dt','lat','lon'])
        Etij = pd.DataFrame(columns=['et','lat','lon'])
        R2ij = pd.DataFrame(columns=['r2','lat','lon'])
        x = np.array([])
        
        #Generate indices and regressors diccionary 
        scaled_indices = []
        for i in range(len(indices)):
            index_1 = indices[i] #/gw_index
            a = np.ones(len(index_1))*np.mean(index_1)
            b = np.ones(len(index_1))*(2)*np.std(index_1)
            index_1 = (index_1-a)/b
            scaled_indices.append(index_1)
            
        regressors = pd.DataFrame({indices_names[0]:scaled_indices[0],
                                   indices_names[1]:scaled_indices[1],
                                   indices_names[2]:scaled_indices[2],
                                   indices_names[3]:scaled_indices[3]})
        
        #Regresion lineal
        y = sm.add_constant(regressors.values)
        lat = self.psl_change[0].lat
        lon = self.psl_change[0].lon
        reg = linear_model.LinearRegression()
        
        pr = self.psl_change
        for i in range(len(lat)):
            for j in range(len(lon)):
                if np.isnan(pr[0][i-1,j-1].values) or np.isnan(pr[1][i-1,j-1].values) or np.isnan(pr[2][i-1,j-1].values) or np.isnan(pr[3][i-1,j-1].values) or np.isnan(pr[4][i-1,j-1].values) or np.isnan(pr[5][i-1,j-1].values) or np.isnan(pr[6][i-1,j-1].values) or np.isnan(pr[7][i-1,j-1].values) or np.isnan(pr[8][i-1,j-1].values) or np.isnan(pr[9][i-1,j-1].values) or np.isnan(pr[10][i-1,j-1].values) or np.isnan(pr[11][i-1,j-1].values) or np.isnan(pr[12][i-1,j-1].values) or np.isnan(pr[13][i-1,j-1].values) or np.isnan(pr[14][i-1,j-1].values) or np.isnan(pr[15][i-1,j-1].values) or np.isnan(pr[16][i-1,j-1].values) or np.isnan(pr[17][i-1,j-1].values) or np.isnan(pr[18][i-1,j-1].values) or np.isnan(pr[19][i-1,j-1].values) or np.isnan(pr[20][i-1,j-1].values) or np.isnan(pr[21][i-1,j-1].values) or np.isnan(pr[22][i-1,j-1].values) or np.isnan(pr[23][i-1,j-1].values) or np.isnan(pr[24][i-1,j-1].values) or np.isnan(pr[25][i-1,j-1].values) or np.isnan(pr[26][i-1,j-1].values) or np.isnan(pr[27][i-1,j-1].values):
                    a = pd.DataFrame({'a':[np.nan],'lat':[lat[i-1].values.tolist()],'lon':[lon[j-1].values.tolist()]})
                    Aij = pd.concat([Aij,a],axis=0)
                    b = pd.DataFrame({'b':[np.nan],'lat':[lat[i-1].values.tolist()],'lon':[lon[j-1].values.tolist()]})
                    Bij = pd.concat([Bij,b],axis=0)
                    c = pd.DataFrame({'c':[np.nan],'lat':[lat[i-1].values.tolist()],'lon':[lon[j-1].values.tolist()]})
                    Cij = pd.concat([Cij,c],axis=0)
                    d = pd.DataFrame({'d':[np.nan],'lat':[lat[i-1].values.tolist()],'lon':[lon[j-1].values.tolist()]})
                    Dij = pd.concat([Dij,d],axis=0)
                    e = pd.DataFrame({'e':[np.nan],'lat':[lat[i-1].values.tolist()],'lon':[lon[j-1].values.tolist()]})
                    Eij = pd.concat([Eij,e],axis=0)
                    at = pd.DataFrame({'at':[np.nan],'lat':[lat[i-1].values.tolist()],'lon':[lon[j-1].values.tolist()]})
                    Atij = pd.concat([Atij,at],axis=0)
                    bt = pd.DataFrame({'bt':[np.nan],'lat':[lat[i-1].values.tolist()],'lon':[lon[j-1].values.tolist()]})
                    Btij = pd.concat([Btij,bt],axis=0)
                    ct = pd.DataFrame({'ct':[np.nan],'lat':[lat[i-1].values.tolist()],'lon':[lon[j-1].values.tolist()]})
                    Ctij = pd.concat([Ctij,ct],axis=0)
                    dt = pd.DataFrame({'dt':[np.nan],'lat':[lat[i-1].values.tolist()],'lon':[lon[j-1].values.tolist()]})
                    Dtij = pd.concat([Dtij,dt],axis=0)
                    et = pd.DataFrame({'et':[np.nan],'lat':[lat[i-1].values.tolist()],'lon':[lon[j-1].values.tolist()]})
                    Etij = pd.concat([Etij,et],axis=0)
                    del b, c, a, bt, ct, at, x, d, dt, e, et
                    x = np.array([])
                    continue
                x = self.create_x(i,j,pr)
                res = sm.OLS(x,y).fit()
                a = res.params[0]
                b = res.params[1]
                c = res.params[2]
                d = res.params[3]
                e = res.params[4]
                at = res.pvalues[0]
                bt = res.pvalues[1]
                ct = res.pvalues[2]
                dt = res.pvalues[3]
                et = res.pvalues[4]
                r2 = res.rsquared
                mse = res.conf_int(alpha=0.05, cols=None) #mse_model
                a = pd.DataFrame({'a':a,'lat':[lat[i-1].values.tolist()],'lon':[lon[j-1].values.tolist()]})
                Aij = pd.concat([Aij,a],axis=0)
                b = pd.DataFrame({'b':b,'lat':[lat[i-1].values.tolist()],'lon':[lon[j-1].values.tolist()]})
                Bij = pd.concat([Bij,b],axis=0)
                c = pd.DataFrame({'c':c,'lat':[lat[i-1].values.tolist()],'lon':[lon[j-1].values.tolist()]})
                Cij = pd.concat([Cij,c],axis=0)
                d = pd.DataFrame({'d':d,'lat':[lat[i-1].values.tolist()],'lon':[lon[j-1].values.tolist()]})
                Dij = pd.concat([Dij,d],axis=0)
                e = pd.DataFrame({'e':e,'lat':[lat[i-1].values.tolist()],'lon':[lon[j-1].values.tolist()]})
                Eij = pd.concat([Eij,e],axis=0)
                r2  = pd.DataFrame({'r2':r2,'lat':[lat[i-1].values.tolist()],'lon':[lon[j-1].values.tolist()]})
                R2ij = pd.concat([R2ij,r2],axis=0)
                at = pd.DataFrame({'at':at,'lat':[lat[i-1].values.tolist()],'lon':[lon[j-1].values.tolist()]})
                Atij = pd.concat([Atij,at],axis=0)
                bt = pd.DataFrame({'bt':bt,'lat':[lat[i-1].values.tolist()],'lon':[lon[j-1].values.tolist()]})
                Btij = pd.concat([Btij,bt],axis=0)
                ct = pd.DataFrame({'ct':ct,'lat':[lat[i-1].values.tolist()],'lon':[lon[j-1].values.tolist()]})
                Ctij = pd.concat([Ctij,ct],axis=0)
                dt = pd.DataFrame({'dt':dt,'lat':[lat[i-1].values.tolist()],'lon':[lon[j-1].values.tolist()]})
                Dtij = pd.concat([Dtij,dt],axis=0)
                et = pd.DataFrame({'et':et,'lat':[lat[i-1].values.tolist()],'lon':[lon[j-1].values.tolist()]})
                Etij = pd.concat([Etij,et],axis=0)
                del r2, res, ct, bt, at, b, c, a, x, d, dt, e, et
                x = np.array([])
                
        #Guardo resultados en DataFrames
        TA = {'coef':Bij.iloc[:,0],'lat':Bij.iloc[:,1],'lon':Bij.iloc[:,2]}
        TAij = pd.DataFrame(TA).fillna(0)

        VB = {'coef':Cij.iloc[:,0],'lat':Cij.iloc[:,1],'lon':Cij.iloc[:,2]}
        VBij = pd.DataFrame(VB).fillna(0)
        
        SST_1 = {'coef':Dij.iloc[:,0],'lat':Dij.iloc[:,1],'lon':Dij.iloc[:,2]}
        SST_1ij = pd.DataFrame(SST_1).fillna(0)
        
        SST_2 = {'coef':Eij.iloc[:,0],'lat':Eij.iloc[:,1],'lon':Eij.iloc[:,2]}
        SST_2ij = pd.DataFrame(SST_2).fillna(0)
        
        A = {'coef':Aij.iloc[:,0],'lat':Aij.iloc[:,1],'lon':Aij.iloc[:,2]}
        Aaij = pd.DataFrame(A).fillna(0)
        
        TAp = {'coef':Btij.iloc[:,0],'lat':Btij.iloc[:,1],'lon':Btij.iloc[:,2]}
        TApij = pd.DataFrame(TAp).fillna(10)
        
        VBp = {'coef':Ctij.iloc[:,0],'lat':Ctij.iloc[:,1],'lon':Ctij.iloc[:,2]}
        VBpij = pd.DataFrame(VBp).fillna(10)
        
        SST_1p = {'coef':Dtij.iloc[:,0],'lat':Dtij.iloc[:,1],'lon':Dtij.iloc[:,2]}
        SST_1pij = pd.DataFrame(SST_1p).fillna(10)
        
        SST_2p = {'coef':Etij.iloc[:,0],'lat':Etij.iloc[:,1],'lon':Etij.iloc[:,2]}
        SST_2pij = pd.DataFrame(SST_2p).fillna(10)
        
        Ap = {'coef':Atij.iloc[:,0],'lat':Atij.iloc[:,1],'lon':Atij.iloc[:,2]}
        Aapij = pd.DataFrame(Ap).fillna(10)
        
        R2 = {'coef':R2ij.iloc[:,0],'lat':R2ij.iloc[:,1],'lon':R2ij.iloc[:,2]}
        R2ij = pd.DataFrame(R2).fillna(0)

        TAij.to_csv(path+'/TAij.csv', float_format='%g')
        VBij.to_csv(path+'/VBij.csv', float_format='%g')
        SST_2ij.to_csv(path+'/SST_2ij.csv', float_format='%g')
        SST_1ij.to_csv(path+'/SST_1ij.csv', float_format='%g')
        Aaij.to_csv(path+'/Aij.csv', float_format='%g')
        TApij.to_csv(path+'/TApij.csv', float_format='%g')
        VBpij.to_csv(path+'/VBpij.csv', float_format='%g')
        SST_2pij.to_csv(path+'/SST_2pij.csv', float_format='%g')
        SST_1pij.to_csv(path+'/SST_1pij.csv', float_format='%g')
        Aapij.to_csv(path+'/Apij.csv', float_format='%g')
        R2ij.to_csv(path+'/R2ij.csv', float_format='%g')
                
        
    def convert_csv_files(self,path):
        file_names_nc = csv2nc.csv_to_nc(path)
        return file_names_nc
    
    
    def create_x(self,i,j,pr):
        x = np.array([])
        for k in range(len(pr)):
            aux = pr[k]
            x = np.append(x,aux[i-1,j-1].values)
            x_mean = np.mean(x); x_anom = (x - x_mean)
            x_new = x
        return x_new
    