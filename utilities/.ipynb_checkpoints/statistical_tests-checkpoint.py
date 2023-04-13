#Hypothesis testing: This code recieves a .nc file with independent p-values from independent hypothesis tests and evaluates field significance implementing the False Discovery Rate test (Benjamini and Hochberg 1995, Ventura et al. 2004, Wilks 2016). 

import xarray as xr
import numpy as np 


def fdr(pvals,alpha):
    p = pvals.stack(aux=('lat','lon'))
    p_sorted = p.sortby(p)
    p_fdr = np.arange(1,len(p_sorted.aux)+1,1)/len(p_sorted.aux)*alpha
    auxiliar = p_sorted.copy()
    auxiliar.values= p_fdr
    final = (p_sorted.coef / auxiliar.coef)
    final.unstack().where(final.unstack() < 1).plot()
    return final 

def s2n