# Subplot number three for mean changes and other figures
import cartopy.crs as ccrs
import numpy as np
import cartopy.feature
from cartopy.util import add_cyclic_point
import matplotlib.path as mpath
import os
import glob
import pandas as pd
import xarray as xr
import netCDF4
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
mpl.rcParams['hatch.linewidth'] = 0.5  # previous pdf hatch linewidth
import cartopy.util as cutil


def plot_sensitivity_pr_viejo(indices,indices_pval,frac_var,path_fig):
    
    GlobalWarming = indices[0]
    TropicalWarming = indices[1]
    VorBreak_GW = indices[2]
    SeaSurfaceTemperature = indices[3]
    SeaSurfaceTemperature2 = indices[4]
    SeaSurfaceTemperature_IOD = indices[5]
    FracVar = frac_var
    GlobalWarming_pval = indices_pval[0]
    TropicalWarming_pval = indices_pval[1]
    VorBreak_GW_pval = indices_pval[2]
    SeaSurfaceTemperature_pval = indices_pval[3]
    SeaSurfaceTemperature2_pval = indices_pval[4]
    SeaSurfaceTemperature_IOD_pval = indices_pval[5]
    
    cmapPr = mpl.colors.ListedColormap(['sienna','darkgoldenrod','burlywood','wheat',
                                        'moccasin','white','white','paleturquoise','mediumaquamarine',
                                        'mediumseagreen','seagreen','darkgreen'])
                                                                                                            
    cmapPr.set_over('darkslategrey')
    cmapPr.set_under('saddlebrown')

    latr = FracVar.lat 
    lat = GlobalWarming.lat
    lon = np.arange(0,357.188,2.81)
    gw, lon_c = add_cyclic_point(GlobalWarming.coef*86400,lon)
    ta, lon_c = add_cyclic_point(TropicalWarming.coef*86400,lon)
    vb_gw, lon_c = add_cyclic_point(VorBreak_GW.coef*86400,lon)
    sst1, lon_c = add_cyclic_point(SeaSurfaceTemperature.coef*86400,lon)
    sst2, lon_c = add_cyclic_point(SeaSurfaceTemperature2.coef*86400,lon)
    sst_iod, lon_c = add_cyclic_point(SeaSurfaceTemperature_IOD.coef*86400,lon)
    fv, lon_c  = add_cyclic_point(FracVar.coef,lon)
    gwp, lon_c = add_cyclic_point(GlobalWarming_pval.coef,lon)
    tap,lon_c = add_cyclic_point(TropicalWarming_pval.coef,lon)
    gwp, lon_c = add_cyclic_point(GlobalWarming_pval.coef,lon)
    vb_gwp,lon_c = add_cyclic_point(VorBreak_GW_pval.coef,lon)
    sst1p,lon_c = add_cyclic_point(SeaSurfaceTemperature_pval.coef,lon)
    sst2p,lon_c = add_cyclic_point(SeaSurfaceTemperature2_pval.coef,lon)
    sst_iodp,lon_c = add_cyclic_point(SeaSurfaceTemperature_IOD_pval.coef,lon)
    
    #SoutherHemisphere Stereographic
    fig = plt.figure(figsize=(20, 16),dpi=300,constrained_layout=True)
    projection = ccrs.SouthPolarStereo(central_longitude=300)
    data_crs = ccrs.PlateCarree()

    ax1 = plt.subplot(3,3,1,projection=projection)
    ax1.set_extent([0,359.9, -90, 0], crs=data_crs)
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax1.set_boundary(circle, transform=ax1.transAxes)
    clevels = np.arange(-.2,.24,0.04)
    im1=ax1.contourf(lon_c, lat, ta,clevels,transform=data_crs,cmap=cmapPr,extend='both')
    #cnt=ax1.contour(lonh,lath, climU850,levels=[8],transform=data_crs,linewidths=1.2, colors='black', linestyles='-')
    #plt.clabel(cnt,inline=True,fmt='%1.0f',fontsize=8)
    levels = [tap.min(),0.05,tap.max()]
    ax1.contourf(lon_c, lat, tap,levels, transform=data_crs,levels=levels, hatches=["...", ""], alpha=0)
    plt.title('a) Tropical warming',fontsize=18)
    ax1.add_feature(cartopy.feature.COASTLINE,alpha=.5)
    ax1.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
    ax1.gridlines(crs=data_crs, linewidth=0.3, linestyle='-')
    ax1.set_extent([-180, 180, -90, 0], ccrs.PlateCarree())
    plt1_ax = plt.gca()
    left_1, bottom_1, width_1, height_1 = plt1_ax.get_position().bounds

    ax2 = plt.subplot(3,3,2,projection=projection)
    ax2.set_extent([0,359.9, -90, 0], crs=data_crs)
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax2.set_boundary(circle, transform=ax2.transAxes)
    #clevels = np.arange(-.05,.06, 0.01)
    im2=ax2.contourf(lon_c, lat, vb_gw,clevels,transform=data_crs,cmap=cmapPr,extend='both')
    #cnt=ax2.contour(lonh, lath, climU850,levels=[8],transform=data_crs,linewidths=1.2, colors='black', linestyles='-')
    #plt.clabel(cnt,inline=True,fmt='%1.0f',fontsize=8)
    levels = [vb_gwp.min(),0.05,vb_gwp.max()]
    ax2.contourf(lon_c, lat,vb_gwp,levels, transform=data_crs,levels=levels, hatches=["...", ""], alpha=0)
    plt.title('b) Vortex breakdown delay (GW)',fontsize=18)
    ax2.add_feature(cartopy.feature.COASTLINE,alpha=.5)
    ax2.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
    ax2.gridlines(crs=data_crs, linewidth=0.3, linestyle='-')
    ax2.set_extent([-180, 180, -90, 0], ccrs.PlateCarree())
    plt2_ax = plt.gca()
    left_2, bottom_2, width_2, height_2 = plt2_ax.get_position().bounds

    ax3 = plt.subplot(3,3,3,projection=projection)
    ax3.set_extent([0,359.9, -90, 0], crs=data_crs)
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax3.set_boundary(circle, transform=ax3.transAxes)
    #clevels = np.arange(-.05,.06, 0.01)
    im3=ax3.contourf(lon_c, lat, sst1,clevels,transform=data_crs,cmap=cmapPr,extend='both')
    #cnt=ax2.contour(lonh, lath, climU850,levels=[8],transform=data_crs,linewidths=1.2, colors='black', linestyles='-')
    #plt.clabel(cnt,inline=True,fmt='%1.0f',fontsize=8)
    levels = [sst1p.min(),0.05,sst1p.max()]
    ax3.contourf(lon_c, lat,sst1p,levels, transform=data_crs,levels=levels, hatches=["...", ""], alpha=0)
    plt.title('c) Sea Surface Temperature (Central)',fontsize=18)
    ax3.add_feature(cartopy.feature.COASTLINE,alpha=.5)
    ax3.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
    ax3.gridlines(crs=data_crs, linewidth=0.3, linestyle='-')
    ax3.set_extent([-180, 180, -90, 0], ccrs.PlateCarree())
    plt3_ax = plt.gca()
    left_3, bottom_3, width_3, height_3 = plt3_ax.get_position().bounds

    ax4 = plt.subplot(3,3,4,projection=projection)
    ax4.set_extent([0,359.9, -90, 0], crs=data_crs)
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax4.set_boundary(circle, transform=ax4.transAxes)
    #clevels = np.arange(-.05,.06, 0.01)
    im4=ax4.contourf(lon_c, lat, sst2,clevels,transform=data_crs,cmap=cmapPr,extend='both')
    #cnt=ax2.contour(lonh, lath, climU850,levels=[8],transform=data_crs,linewidths=1.2, colors='black', linestyles='-')
    #plt.clabel(cnt,inline=True,fmt='%1.0f',fontsize=8)
    levels = [sst2p.min(),0.05,sst2p.max()]
    ax4.contourf(lon_c, lat,sst2p,levels, transform=data_crs,levels=levels, hatches=["...", ""], alpha=0)
    plt.title('d) Sea Surface Temperature (Eastern)',fontsize=18)
    ax4.add_feature(cartopy.feature.COASTLINE,alpha=.5)
    ax4.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
    ax4.gridlines(crs=data_crs, linewidth=0.3, linestyle='-')
    ax4.set_extent([-180, 180, -90, 0], ccrs.PlateCarree())
    plt4_ax = plt.gca()
    left_4, bottom_4, width_4, height_4 = plt4_ax.get_position().bounds
    
    ax7 = plt.subplot(3,3,5,projection=projection)
    ax7.set_extent([0,359.9, -90, 0], crs=data_crs)
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax7.set_boundary(circle, transform=ax7.transAxes)
    #clevels = np.arange(-.05,.06, 0.01)
    im7=ax7.contourf(lon_c, lat, sst_iod,clevels,transform=data_crs,cmap=cmapPr,extend='both')
    #cnt=ax2.contour(lonh, lath, climU850,levels=[8],transform=data_crs,linewidths=1.2, colors='black', linestyles='-')
    #plt.clabel(cnt,inline=True,fmt='%1.0f',fontsize=8)
    levels = [sst_iodp.min(),0.05,sst_iodp.max()]
    ax7.contourf(lon_c, lat,sst_iodp,levels, transform=data_crs,levels=levels, hatches=["...", ""], alpha=0)
    plt.title('e) Sea Surface Temperature (IOD)',fontsize=18)
    ax7.add_feature(cartopy.feature.COASTLINE,alpha=.5)
    ax7.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
    ax7.gridlines(crs=data_crs, linewidth=0.3, linestyle='-')
    ax7.set_extent([-180, 180, -90, 0], ccrs.PlateCarree())
    plt7_ax = plt.gca()
    left_7, bottom_7, width_7, height_7 = plt7_ax.get_position().bounds
    
    ax5 = plt.subplot(3,3,6,projection=projection)
    ax5.set_extent([0,359.9, -90, 0], crs=data_crs)
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax5.set_boundary(circle, transform=ax5.transAxes)
    clevels = np.arange(-.3,.36,.06)
    im5=ax5.contourf(lon_c, lat, gw,clevels,transform=data_crs,cmap=cmapPr,extend='both')
    #cnt=ax1.contour(lonh,lath, climU850,levels=[8],transform=data_crs,linewidths=1.2, colors='black', linestyles='-')
    #plt.clabel(cnt,inline=True,fmt='%1.0f',fontsize=8)
    levels = [gwp.min(),0.05,gwp.max()]
    ax5.contourf(lon_c, lat, gwp,levels, transform=data_crs,levels=levels, hatches=["...", ""], alpha=0)
    plt.title('f) Global warming',fontsize=18)
    ax5.add_feature(cartopy.feature.COASTLINE,alpha=.5)
    ax5.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
    ax5.gridlines(crs=data_crs, linewidth=0.3, linestyle='-')
    ax5.set_extent([-180, 180, -90, 0], ccrs.PlateCarree())
    plt5_ax = plt.gca()
    left_5, bottom_5, width_5, height_5 = plt5_ax.get_position().bounds

    ax6 = plt.subplot(3,3,7,projection=projection)
    ax6.set_extent([0,359.9, -90, 0], crs=data_crs)
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    clevels = np.arange(0,1,0.1)
    ax6.set_boundary(circle, transform=ax6.transAxes)
    im6=ax6.contourf(lon_c, latr, fv,clevels,transform=data_crs,cmap='OrRd',extend='both')
    plt.title('g) Fraction of variance explained',fontsize=18)
    ax6.add_feature(cartopy.feature.COASTLINE,alpha=.5)
    ax6.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
    ax6.gridlines(crs=data_crs, linewidth=0.3, linestyle='-')
    ax6.set_extent([-180, 180, -90, 0], ccrs.PlateCarree())
    plt6_ax = plt.gca()
    left_6, bottom_6, width_6, height_6 = plt6_ax.get_position().bounds

    plt.subplots_adjust(bottom=0.2, right=1.2, top=0.8)

    fourth_plot_left = plt4_ax.get_position().bounds[0]
    colorbar_axes4 = fig.add_axes([fourth_plot_left +0.25, bottom_4, 0.01, height_4])
    cbar = fig.colorbar(im4, colorbar_axes4, orientation='vertical')
    cbar.set_label('mmday$^{-1}$K$^{-1}$',fontsize=14) #rotation = radianes
    cbar.ax.tick_params(axis='both',labelsize=14)
    
    fifth_plot_left = plt5_ax.get_position().bounds[0]
    colorbar_axes5 = fig.add_axes([fifth_plot_left +0.25, bottom_5 , 0.01, height_5])
    cbar = fig.colorbar(im5, colorbar_axes5, orientation='vertical')
    cbar.set_label(' ',fontsize=14) #rotation = radianes
    cbar.ax.tick_params(axis='both',labelsize=14)

    first_plot_left = plt1_ax.get_position().bounds[0]
    colorbar_axes1 = fig.add_axes([first_plot_left +0.25, bottom_1, 0.01, height_1])
    cbar = fig.colorbar(im1, colorbar_axes1, orientation='vertical')
    cbar.set_label('mmday$^{-1}$K$^{-1}$',fontsize=14) #rotation = radianes
    cbar.ax.tick_params(axis='both',labelsize=14)

    second_plot_left = plt2_ax.get_position().bounds[0]
    colorbar_axes2 = fig.add_axes([second_plot_left +0.25, bottom_2, 0.01, height_2])
    cbar = fig.colorbar(im2, colorbar_axes2, orientation='vertical')
    cbar.set_label('mmday$^{-1}$K$^{-1}$',fontsize=14) #rotation = radianes
    cbar.ax.tick_params(axis='both',labelsize=14)

    third_plot_left = plt3_ax.get_position().bounds[0]
    colorbar_axes3 = fig.add_axes([third_plot_left +0.25, bottom_3, 0.01, height_3])
    cbar = fig.colorbar(im3, colorbar_axes3, orientation='vertical')
    cbar.set_label(' ',fontsize=14) #rotation = radianes
    cbar.set_label('mmday$^{-1}$K$^{-1}$',fontsize=14) #rotation = radianes
    cbar.ax.tick_params(axis='both',labelsize=14)
    
    seventh_plot_left = plt7_ax.get_position().bounds[0]
    colorbar_axes7 = fig.add_axes([seventh_plot_left +0.25, bottom_7, 0.01, height_7])
    cbar = fig.colorbar(im7, colorbar_axes7, orientation='vertical')
    cbar.set_label(' ',fontsize=14) #rotation = radianes
    cbar.set_label('mmday$^{-1}$K$^{-1}$',fontsize=14) #rotation = radianes
    cbar.ax.tick_params(axis='both',labelsize=14)

    sixth_plot_left = plt6_ax.get_position().bounds[0]
    colorbar_axes6 = fig.add_axes([sixth_plot_left +0.25, bottom_6, 0.01, height_6])
    cbar = fig.colorbar(im6, colorbar_axes6, orientation='vertical')
    cbar.set_label(' ',fontsize=14) #rotation = radianes
    cbar.ax.tick_params(axis='both',labelsize=14)

    plt.savefig(path_fig+'/prDJFsensitivities_TW_VB_SSTs.png',bbox_inches='tight')
    plt.clf

    return fig

def plot_sensitivity_pr(indices,indices_pval,frac_var,path_fig):
    
    GlobalWarming = indices[0]
    TropicalWarming = indices[1]
    VorBreak_GW = indices[2]
    SeaSurfaceTemperature = indices[3]
    SeaSurfaceTemperature2 = indices[4]
    SeaSurfaceTemperature_IOD = indices[5]
    FracVar = frac_var
    GlobalWarming_pval = indices_pval[0]
    TropicalWarming_pval = indices_pval[1]
    VorBreak_GW_pval = indices_pval[2]
    SeaSurfaceTemperature_pval = indices_pval[3]
    SeaSurfaceTemperature2_pval = indices_pval[4]
    SeaSurfaceTemperature_IOD_pval = indices_pval[5]
    
    cmapPr = mpl.colors.ListedColormap(['sienna','darkgoldenrod','burlywood','wheat',
                                        'moccasin','white','white','paleturquoise','mediumaquamarine',
                                        'mediumseagreen','seagreen','darkgreen'])
                                                                                                            
    cmapPr.set_over('darkslategrey')
    cmapPr.set_under('saddlebrown')

    latr = FracVar.lat 
    lat = GlobalWarming.lat
    lon = np.arange(0,357.188,2.81)
    gw, lon_c = add_cyclic_point(GlobalWarming.coef*86400,lon)
    ta, lon_c = add_cyclic_point(TropicalWarming.coef*86400,lon)
    vb_gw, lon_c = add_cyclic_point(VorBreak_GW.coef*86400,lon)
    sst1, lon_c = add_cyclic_point(SeaSurfaceTemperature.coef*86400,lon)
    sst2, lon_c = add_cyclic_point(SeaSurfaceTemperature2.coef*86400,lon)
    sst_iod, lon_c = add_cyclic_point(SeaSurfaceTemperature_IOD.coef*86400,lon)
    fv, lon_c  = add_cyclic_point(FracVar.coef,lon)
    gwp, lon_c = add_cyclic_point(GlobalWarming_pval.coef,lon)
    tap,lon_c = add_cyclic_point(TropicalWarming_pval.coef,lon)
    gwp, lon_c = add_cyclic_point(GlobalWarming_pval.coef,lon)
    vb_gwp,lon_c = add_cyclic_point(VorBreak_GW_pval.coef,lon)
    sst1p,lon_c = add_cyclic_point(SeaSurfaceTemperature_pval.coef,lon)
    sst2p,lon_c = add_cyclic_point(SeaSurfaceTemperature2_pval.coef,lon)
    sst_iodp,lon_c = add_cyclic_point(SeaSurfaceTemperature_IOD_pval.coef,lon)
    
    #SoutherHemisphere Stereographic
    fig = plt.figure(figsize=(20, 16),dpi=300,constrained_layout=True)
    projection = ccrs.SouthPolarStereo(central_longitude=300)
    data_crs = ccrs.PlateCarree()

    ax1 = plt.subplot(3,3,1,projection=projection)
    ax1.set_extent([0,359.9, -90, 0], crs=data_crs)
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax1.set_boundary(circle, transform=ax1.transAxes)
    clevels = np.arange(-.2,.24,0.04)
    im1=ax1.contourf(lon_c, lat, ta,clevels,transform=data_crs,cmap=cmapPr,extend='both')
    levels = [tap.min(),0.05,tap.max()]
    ax1.contourf(lon_c, lat, tap,levels, transform=data_crs,levels=levels, hatches=["...", ""], alpha=0)
    plt.title('a) Tropical warming',fontsize=18)
    ax1.add_feature(cartopy.feature.COASTLINE,alpha=.5)
    ax1.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
    ax1.gridlines(crs=data_crs, linewidth=0.3, linestyle='-')
    ax1.set_extent([-180, 180, -90, 0], ccrs.PlateCarree())
    plt1_ax = plt.gca()
    left_1, bottom_1, width_1, height_1 = plt1_ax.get_position().bounds

    ax2 = plt.subplot(3,3,2,projection=projection)
    ax2.set_extent([0,359.9, -90, 0], crs=data_crs)
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax2.set_boundary(circle, transform=ax2.transAxes)
    #clevels = np.arange(-.08,.1, 0.02)
    im2=ax2.contourf(lon_c, lat, vb_gw,clevels,transform=data_crs,cmap=cmapPr,extend='both')
    levels = [vb_gwp.min(),0.05,vb_gwp.max()]
    ax2.contourf(lon_c, lat,vb_gwp,levels, transform=data_crs,levels=levels, hatches=["...", ""], alpha=0)
    plt.title('b) Stratospheric polar vortex',fontsize=18)
    ax2.add_feature(cartopy.feature.COASTLINE,alpha=.5)
    ax2.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
    ax2.gridlines(crs=data_crs, linewidth=0.3, linestyle='-')
    ax2.set_extent([-180, 180, -90, 0], ccrs.PlateCarree())
    plt2_ax = plt.gca()
    left_2, bottom_2, width_2, height_2 = plt2_ax.get_position().bounds

    ax3 = plt.subplot(3,3,3,projection=projection)
    ax3.set_extent([0,359.9, -90, 0], crs=data_crs)
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax3.set_boundary(circle, transform=ax3.transAxes)
    #clevels = np.arange(-.08,.1, 0.02)
    im3=ax3.contourf(lon_c, lat, sst1,clevels,transform=data_crs,cmap=cmapPr,extend='both')
    levels = [sst1p.min(),0.05,sst1p.max()]
    ax3.contourf(lon_c, lat,sst1p,levels, transform=data_crs,levels=levels, hatches=["...", ""], alpha=0)
    plt.title('c) Sea Surface Temperature (Central)',fontsize=18)
    ax3.add_feature(cartopy.feature.COASTLINE,alpha=.5)
    ax3.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
    ax3.gridlines(crs=data_crs, linewidth=0.3, linestyle='-')
    ax3.set_extent([-180, 180, -90, 0], ccrs.PlateCarree())
    plt3_ax = plt.gca()
    left_3, bottom_3, width_3, height_3 = plt3_ax.get_position().bounds

    ax4 = plt.subplot(3,3,4,projection=projection)
    ax4.set_extent([0,359.9, -90, 0], crs=data_crs)
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax4.set_boundary(circle, transform=ax4.transAxes)
    #clevels = np.arange(-.08,.1, 0.02)
    im4=ax4.contourf(lon_c, lat, sst2,clevels,transform=data_crs,cmap=cmapPr,extend='both')
    levels = [sst2p.min(),0.05,sst2p.max()]
    ax4.contourf(lon_c, lat,sst2p,levels, transform=data_crs,levels=levels, hatches=["...", ""], alpha=0)
    plt.title('d) Sea Surface Temperature (Eastern)',fontsize=18)
    ax4.add_feature(cartopy.feature.COASTLINE,alpha=.5)
    ax4.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
    ax4.gridlines(crs=data_crs, linewidth=0.3, linestyle='-')
    ax4.set_extent([-180, 180, -90, 0], ccrs.PlateCarree())
    plt4_ax = plt.gca()
    left_4, bottom_4, width_4, height_4 = plt4_ax.get_position().bounds
    
    ax7 = plt.subplot(3,3,5,projection=projection)
    ax7.set_extent([0,359.9, -90, 0], crs=data_crs)
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax7.set_boundary(circle, transform=ax7.transAxes)
    #clevels = np.arange(-.08,.1, 0.02)
    im7=ax7.contourf(lon_c, lat, sst_iod,clevels,transform=data_crs,cmap=cmapPr,extend='both')
    levels = [sst_iodp.min(),0.05,sst_iodp.max()]
    ax7.contourf(lon_c, lat,sst_iodp,levels, transform=data_crs,levels=levels, hatches=["...", ""], alpha=0)
    plt.title('e) Sea Surface Temperature (IOD)',fontsize=18)
    ax7.add_feature(cartopy.feature.COASTLINE,alpha=.5)
    ax7.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
    ax7.gridlines(crs=data_crs, linewidth=0.3, linestyle='-')
    ax7.set_extent([-180, 180, -90, 0], ccrs.PlateCarree())
    plt7_ax = plt.gca()
    left_7, bottom_7, width_7, height_7 = plt7_ax.get_position().bounds
    
    ax5 = plt.subplot(3,3,6,projection=projection)
    ax5.set_extent([0,359.9, -90, 0], crs=data_crs)
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax5.set_boundary(circle, transform=ax5.transAxes)
    clevels = np.arange(-.3,.36,.06)
    im5=ax5.contourf(lon_c, lat, gw,clevels,transform=data_crs,cmap=cmapPr,extend='both')
    levels = [gwp.min(),0.05,gwp.max()]
    ax5.contourf(lon_c, lat, gwp,levels, transform=data_crs,levels=levels, hatches=["...", ""], alpha=0)
    plt.title('f) Global warming',fontsize=18)
    ax5.add_feature(cartopy.feature.COASTLINE,alpha=.5)
    ax5.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
    ax5.gridlines(crs=data_crs, linewidth=0.3, linestyle='-')
    ax5.set_extent([-180, 180, -90, 0], ccrs.PlateCarree())
    plt5_ax = plt.gca()
    left_5, bottom_5, width_5, height_5 = plt5_ax.get_position().bounds

    ax6 = plt.subplot(3,3,7,projection=projection)
    ax6.set_extent([0,359.9, -90, 0], crs=data_crs)
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    clevels = np.arange(0,1,0.1)
    ax6.set_boundary(circle, transform=ax6.transAxes)
    im6=ax6.contourf(lon_c, latr, fv,clevels,transform=data_crs,cmap='OrRd',extend='both')
    plt.title('g) Fraction of variance explained',fontsize=18)
    ax6.add_feature(cartopy.feature.COASTLINE,alpha=.5)
    ax6.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
    ax6.gridlines(crs=data_crs, linewidth=0.3, linestyle='-')
    ax6.set_extent([-180, 180, -90, 0], ccrs.PlateCarree())
    plt6_ax = plt.gca()
    left_6, bottom_6, width_6, height_6 = plt6_ax.get_position().bounds

    plt.subplots_adjust(bottom=0.2, right=0.8, top=0.8)

    fourth_plot_left = plt4_ax.get_position().bounds[0]
    colorbar_axes4 = fig.add_axes([fourth_plot_left +0.15, bottom_4+0.05, 0.01, height_4*0.6])
    cbar = fig.colorbar(im4, colorbar_axes4, orientation='vertical')
    cbar.set_label('mm day$^{-1}$K$^{-1}$',fontsize=14) #rotation = radianes
    cbar.ax.tick_params(axis='both',labelsize=14)
    
    fifth_plot_left = plt5_ax.get_position().bounds[0]
    colorbar_axes5 = fig.add_axes([fifth_plot_left +0.15, bottom_5+0.05, 0.01, height_5*0.6])
    cbar = fig.colorbar(im5, colorbar_axes5, orientation='vertical')
    cbar.set_label(' ',fontsize=14) #rotation = radianes
    cbar.ax.tick_params(axis='both',labelsize=14)

    first_plot_left = plt1_ax.get_position().bounds[0]
    colorbar_axes1 = fig.add_axes([first_plot_left +0.15, bottom_1, 0.01, height_1*0.6])
    cbar = fig.colorbar(im1, colorbar_axes1, orientation='vertical')
    cbar.set_label('mm day$^{-1}$K$^{-1}$',fontsize=14) #rotation = radianes
    cbar.ax.tick_params(axis='both',labelsize=14)

    second_plot_left = plt2_ax.get_position().bounds[0]
    colorbar_axes2 = fig.add_axes([second_plot_left +0.15, bottom_2, 0.01, height_2*0.6])
    cbar = fig.colorbar(im2, colorbar_axes2, orientation='vertical')
    cbar.set_label('mm day$^{-1}$K$^{-1}$',fontsize=14) #rotation = radianes
    cbar.ax.tick_params(axis='both',labelsize=14)

    third_plot_left = plt3_ax.get_position().bounds[0]
    colorbar_axes3 = fig.add_axes([third_plot_left +0.15, bottom_3, 0.01, height_3*0.6])
    cbar = fig.colorbar(im3, colorbar_axes3, orientation='vertical')
    cbar.set_label(' ',fontsize=14) #rotation = radianes
    cbar.set_label('mm day$^{-1}$K$^{-1}$',fontsize=14) #rotation = radianes
    cbar.ax.tick_params(axis='both',labelsize=14)
    
    seventh_plot_left = plt7_ax.get_position().bounds[0]
    colorbar_axes7 = fig.add_axes([seventh_plot_left +0.15, bottom_7+0.05, 0.01, height_7*0.6])
    cbar = fig.colorbar(im7, colorbar_axes7, orientation='vertical')
    cbar.set_label('mm day$^{-1}$K$^{-1}$',fontsize=14) #rotation = radianes
    cbar.ax.tick_params(axis='both',labelsize=14)

    sixth_plot_left = plt6_ax.get_position().bounds[0]
    colorbar_axes6 = fig.add_axes([sixth_plot_left +0.15, bottom_6+0.1, 0.01, height_6*0.6])
    cbar = fig.colorbar(im6, colorbar_axes6, orientation='vertical')
    cbar.set_label(' ',fontsize=14) #rotation = radianes
    cbar.ax.tick_params(axis='both',labelsize=14)

    plt.savefig(path_fig+'/prSONsensitivities_TW_VB_SSTs.png',bbox_inches='tight')
    plt.clf

    return fig

def plot_sensitivity_ua(indices,indices_pval,frac_var,path_fig):
    
    GlobalWarming = indices[0]
    TropicalWarming = indices[1]
    VorBreak_GW = indices[2]
    SeaSurfaceTemperature = indices[3]
    SeaSurfaceTemperature2 = indices[4]
    SeaSurfaceTemperature_IOD = indices[5]
    FracVar = frac_var
    GlobalWarming_pval = indices_pval[0]
    TropicalWarming_pval = indices_pval[1]
    VorBreak_GW_pval = indices_pval[2]
    SeaSurfaceTemperature_pval = indices_pval[3]
    SeaSurfaceTemperature2_pval = indices_pval[4]
    SeaSurfaceTemperature_IOD_pval = indices_pval[5]
    
    cmapU850 = mpl.colors.ListedColormap(['darkblue','navy','steelblue','lightblue',
                                          'lightsteelblue','white','white','mistyrose',
                                          'lightcoral','indianred','brown','firebrick'])
    cmapU850.set_over('maroon')
    cmapU850.set_under('midnightblue')

    path_era = '/datos/ERA5/mon'
    u_ERA = xr.open_dataset(path_era+'/era5.mon.mean.nc')
    u_ERA = u_ERA.u.sel(lev=850).mean(dim='time')

    latr = FracVar.lat 
    lat = GlobalWarming.lat
    lon = np.arange(0,357.188,2.81)
    gw, lon_c = add_cyclic_point(GlobalWarming.coef,lon)
    ta, lon_c = add_cyclic_point(TropicalWarming.coef,lon)
    vb_gw, lon_c = add_cyclic_point(VorBreak_GW.coef,lon)
    sst1, lon_c = add_cyclic_point(SeaSurfaceTemperature.coef,lon)
    sst2, lon_c = add_cyclic_point(SeaSurfaceTemperature2.coef,lon)
    sst_iod, lon_c = add_cyclic_point(SeaSurfaceTemperature_IOD.coef,lon)
    fv, lon_c  = add_cyclic_point(FracVar.coef,lon)
    gwp, lon_c = add_cyclic_point(GlobalWarming_pval.coef,lon)
    tap,lon_c = add_cyclic_point(TropicalWarming_pval.coef,lon)
    gwp, lon_c = add_cyclic_point(GlobalWarming_pval.coef,lon)
    vb_gwp,lon_c = add_cyclic_point(VorBreak_GW_pval.coef,lon)
    sst1p,lon_c = add_cyclic_point(SeaSurfaceTemperature_pval.coef,lon)
    sst2p,lon_c = add_cyclic_point(SeaSurfaceTemperature2_pval.coef,lon)
    sst_iodp,lon_c = add_cyclic_point(SeaSurfaceTemperature_IOD_pval.coef,lon)
    
    #SoutherHemisphere Stereographic
    fig = plt.figure(figsize=(20, 16),dpi=300,constrained_layout=True)
    projection = ccrs.SouthPolarStereo(central_longitude=300)
    data_crs = ccrs.PlateCarree()

    ax1 = plt.subplot(3,3,1,projection=projection)
    ax1.set_extent([0,359.9, -90, 0], crs=data_crs)
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax1.set_boundary(circle, transform=ax1.transAxes)
    clevels = np.arange(-.2,.24,0.04)
    im1=ax1.contourf(lon_c, lat, ta,clevels,transform=data_crs,cmap=cmapU850,extend='both')
    cnt=ax1.contour(u_ERA.lon,u_ERA.lat, u_ERA.values,levels=[8],transform=data_crs,linewidths=1.2, colors='black', linestyles='-')
    plt.clabel(cnt,inline=True,fmt='%1.0f',fontsize=8)
    levels = [tap.min(),0.05,tap.max()]
    ax1.contourf(lon_c, lat, tap,levels, transform=data_crs,levels=levels, hatches=["...", ""], alpha=0)
    plt.title('a) Tropical warming',fontsize=18)
    ax1.add_feature(cartopy.feature.COASTLINE,alpha=.5)
    ax1.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
    ax1.gridlines(crs=data_crs, linewidth=0.3, linestyle='-')
    ax1.set_extent([-180, 180, -90, 0], ccrs.PlateCarree())
    plt1_ax = plt.gca()
    left_1, bottom_1, width_1, height_1 = plt1_ax.get_position().bounds

    ax2 = plt.subplot(3,3,2,projection=projection)
    ax2.set_extent([0,359.9, -90, 0], crs=data_crs)
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax2.set_boundary(circle, transform=ax2.transAxes)
    #clevels = np.arange(-.08,.1, 0.02)
    im2=ax2.contourf(lon_c, lat, vb_gw,clevels,transform=data_crs,cmap=cmapU850,extend='both')
    cnt=ax2.contour(u_ERA.lon,u_ERA.lat, u_ERA.values,levels=[8],transform=data_crs,linewidths=1.2, colors='black', linestyles='-')
    plt.clabel(cnt,inline=True,fmt='%1.0f',fontsize=8)
    levels = [vb_gwp.min(),0.05,vb_gwp.max()]
    ax2.contourf(lon_c, lat,vb_gwp,levels, transform=data_crs,levels=levels, hatches=["...", ""], alpha=0)
    plt.title('b) Stratospheric polar vortex',fontsize=18)
    ax2.add_feature(cartopy.feature.COASTLINE,alpha=.5)
    ax2.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
    ax2.gridlines(crs=data_crs, linewidth=0.3, linestyle='-')
    ax2.set_extent([-180, 180, -90, 0], ccrs.PlateCarree())
    plt2_ax = plt.gca()
    left_2, bottom_2, width_2, height_2 = plt2_ax.get_position().bounds

    ax3 = plt.subplot(3,3,3,projection=projection)
    ax3.set_extent([0,359.9, -90, 0], crs=data_crs)
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax3.set_boundary(circle, transform=ax3.transAxes)
    #clevels = np.arange(-.08,.1, 0.02)
    im3=ax3.contourf(lon_c, lat, sst1,clevels,transform=data_crs,cmap=cmapU850,extend='both')
    cnt=ax3.contour(u_ERA.lon,u_ERA.lat, u_ERA.values,levels=[8],transform=data_crs,linewidths=1.2, colors='black', linestyles='-')
    plt.clabel(cnt,inline=True,fmt='%1.0f',fontsize=8)
    levels = [sst1p.min(),0.05,sst1p.max()]
    ax3.contourf(lon_c, lat,sst1p,levels, transform=data_crs,levels=levels, hatches=["...", ""], alpha=0)
    plt.title('c) Sea Surface Temperature (Central)',fontsize=18)
    ax3.add_feature(cartopy.feature.COASTLINE,alpha=.5)
    ax3.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
    ax3.gridlines(crs=data_crs, linewidth=0.3, linestyle='-')
    ax3.set_extent([-180, 180, -90, 0], ccrs.PlateCarree())
    plt3_ax = plt.gca()
    left_3, bottom_3, width_3, height_3 = plt3_ax.get_position().bounds

    ax4 = plt.subplot(3,3,4,projection=projection)
    ax4.set_extent([0,359.9, -90, 0], crs=data_crs)
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax4.set_boundary(circle, transform=ax4.transAxes)
    #clevels = np.arange(-.08,.1, 0.02)
    im4=ax4.contourf(lon_c, lat, sst2,clevels,transform=data_crs,cmap=cmapU850,extend='both')
    cnt=ax4.contour(u_ERA.lon,u_ERA.lat, u_ERA.values,levels=[8],transform=data_crs,linewidths=1.2, colors='black', linestyles='-')
    plt.clabel(cnt,inline=True,fmt='%1.0f',fontsize=8)
    levels = [sst2p.min(),0.05,sst2p.max()]
    ax4.contourf(lon_c, lat,sst2p,levels, transform=data_crs,levels=levels, hatches=["...", ""], alpha=0)
    plt.title('d) Sea Surface Temperature (Eastern)',fontsize=18)
    ax4.add_feature(cartopy.feature.COASTLINE,alpha=.5)
    ax4.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
    ax4.gridlines(crs=data_crs, linewidth=0.3, linestyle='-')
    ax4.set_extent([-180, 180, -90, 0], ccrs.PlateCarree())
    plt4_ax = plt.gca()
    left_4, bottom_4, width_4, height_4 = plt4_ax.get_position().bounds
    
    ax7 = plt.subplot(3,3,5,projection=projection)
    ax7.set_extent([0,359.9, -90, 0], crs=data_crs)
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax7.set_boundary(circle, transform=ax7.transAxes)
    #clevels = np.arange(-.08,.1, 0.02)
    im7=ax7.contourf(lon_c, lat, sst_iod,clevels,transform=data_crs,cmap=cmapU850,extend='both')
    cnt=ax7.contour(u_ERA.lon,u_ERA.lat, u_ERA.values,levels=[8],transform=data_crs,linewidths=1.2, colors='black', linestyles='-')
    plt.clabel(cnt,inline=True,fmt='%1.0f',fontsize=8)
    levels = [sst_iodp.min(),0.05,sst_iodp.max()]
    ax7.contourf(lon_c, lat,sst_iodp,levels, transform=data_crs,levels=levels, hatches=["...", ""], alpha=0)
    plt.title('e) Sea Surface Temperature (IOD)',fontsize=18)
    ax7.add_feature(cartopy.feature.COASTLINE,alpha=.5)
    ax7.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
    ax7.gridlines(crs=data_crs, linewidth=0.3, linestyle='-')
    ax7.set_extent([-180, 180, -90, 0], ccrs.PlateCarree())
    plt7_ax = plt.gca()
    left_7, bottom_7, width_7, height_7 = plt7_ax.get_position().bounds
    
    ax5 = plt.subplot(3,3,6,projection=projection)
    ax5.set_extent([0,359.9, -90, 0], crs=data_crs)
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax5.set_boundary(circle, transform=ax5.transAxes)
    clevels = np.arange(-.3,.36,.06)
    im5=ax5.contourf(lon_c, lat, gw,clevels,transform=data_crs,cmap=cmapU850,extend='both')
    cnt=ax5.contour(u_ERA.lon,u_ERA.lat, u_ERA.values,levels=[8],transform=data_crs,linewidths=1.2, colors='black', linestyles='-')
    plt.clabel(cnt,inline=True,fmt='%1.0f',fontsize=8)
    levels = [gwp.min(),0.05,gwp.max()]
    ax5.contourf(lon_c, lat, gwp,levels, transform=data_crs,levels=levels, hatches=["...", ""], alpha=0)
    plt.title('f) Global warming',fontsize=18)
    ax5.add_feature(cartopy.feature.COASTLINE,alpha=.5)
    ax5.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
    ax5.gridlines(crs=data_crs, linewidth=0.3, linestyle='-')
    ax5.set_extent([-180, 180, -90, 0], ccrs.PlateCarree())
    plt5_ax = plt.gca()
    left_5, bottom_5, width_5, height_5 = plt5_ax.get_position().bounds

    ax6 = plt.subplot(3,3,7,projection=projection)
    ax6.set_extent([0,359.9, -90, 0], crs=data_crs)
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    clevels = np.arange(0,1,0.1)
    ax6.set_boundary(circle, transform=ax6.transAxes)
    im6=ax6.contourf(lon_c, latr, fv,clevels,transform=data_crs,cmap='OrRd',extend='both')
    plt.title('g) Fraction of variance explained',fontsize=18)
    ax6.add_feature(cartopy.feature.COASTLINE,alpha=.5)
    ax6.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
    ax6.gridlines(crs=data_crs, linewidth=0.3, linestyle='-')
    ax6.set_extent([-180, 180, -90, 0], ccrs.PlateCarree())
    plt6_ax = plt.gca()
    left_6, bottom_6, width_6, height_6 = plt6_ax.get_position().bounds

    plt.subplots_adjust(bottom=0.2, right=0.8, top=0.8)

    fourth_plot_left = plt4_ax.get_position().bounds[0]
    colorbar_axes4 = fig.add_axes([fourth_plot_left +0.15, bottom_4+0.05, 0.01, height_4*0.6])
    cbar = fig.colorbar(im4, colorbar_axes4, orientation='vertical')
    cbar.set_label('ms$^{-1}$K$^{-1}$',fontsize=14) #rotation = radianes
    cbar.ax.tick_params(axis='both',labelsize=14)
    
    fifth_plot_left = plt5_ax.get_position().bounds[0]
    colorbar_axes5 = fig.add_axes([fifth_plot_left +0.15, bottom_5+0.05, 0.01, height_5*0.6])
    cbar = fig.colorbar(im5, colorbar_axes5, orientation='vertical')
    cbar.set_label(' ',fontsize=14) #rotation = radianes
    cbar.ax.tick_params(axis='both',labelsize=14)

    first_plot_left = plt1_ax.get_position().bounds[0]
    colorbar_axes1 = fig.add_axes([first_plot_left +0.15, bottom_1, 0.01, height_1*0.6])
    cbar = fig.colorbar(im1, colorbar_axes1, orientation='vertical')
    cbar.set_label('ms$^{-1}$K$^{-1}$',fontsize=14) #rotation = radianes
    cbar.ax.tick_params(axis='both',labelsize=14)

    second_plot_left = plt2_ax.get_position().bounds[0]
    colorbar_axes2 = fig.add_axes([second_plot_left +0.15, bottom_2, 0.01, height_2*0.6])
    cbar = fig.colorbar(im2, colorbar_axes2, orientation='vertical')
    cbar.set_label('ms$^{-1}$K$^{-1}$',fontsize=14) #rotation = radianes
    cbar.ax.tick_params(axis='both',labelsize=14)

    third_plot_left = plt3_ax.get_position().bounds[0]
    colorbar_axes3 = fig.add_axes([third_plot_left +0.15, bottom_3, 0.01, height_3*0.6])
    cbar = fig.colorbar(im3, colorbar_axes3, orientation='vertical')
    cbar.set_label(' ',fontsize=14) #rotation = radianes
    cbar.set_label('ms$^{-1}$K$^{-1}$',fontsize=14) #rotation = radianes
    cbar.ax.tick_params(axis='both',labelsize=14)
    
    seventh_plot_left = plt7_ax.get_position().bounds[0]
    colorbar_axes7 = fig.add_axes([seventh_plot_left +0.15, bottom_7+0.05, 0.01, height_7*0.6])
    cbar = fig.colorbar(im7, colorbar_axes7, orientation='vertical')
    cbar.set_label('ms$^{-1}$K$^{-1}$',fontsize=14) #rotation = radianes
    cbar.ax.tick_params(axis='both',labelsize=14)

    sixth_plot_left = plt6_ax.get_position().bounds[0]
    colorbar_axes6 = fig.add_axes([sixth_plot_left +0.15, bottom_6+0.1, 0.01, height_6*0.6])
    cbar = fig.colorbar(im6, colorbar_axes6, orientation='vertical')
    cbar.set_label(' ',fontsize=14) #rotation = radianes
    cbar.ax.tick_params(axis='both',labelsize=14)

    plt.savefig(path_fig+'/u850SONsensitivities_TW_VB_SSTs.png',bbox_inches='tight')
    plt.clf

    return fig

def plot_sensitivity_psl(indices,indices_pval,frac_var,path_fig):
    
    GlobalWarming = indices[0]
    TropicalWarming = indices[1]
    VorBreak_GW = indices[2]
    SeaSurfaceTemperature = indices[3]
    SeaSurfaceTemperature2 = indices[4]
    SeaSurfaceTemperature_IOD = indices[5]
    FracVar = frac_var
    GlobalWarming_pval = indices_pval[0]
    TropicalWarming_pval = indices_pval[1]
    VorBreak_GW_pval = indices_pval[2]
    SeaSurfaceTemperature_pval = indices_pval[3]
    SeaSurfaceTemperature2_pval = indices_pval[4]
    SeaSurfaceTemperature_IOD_pval = indices_pval[5]

    cmapPSL = mpl.colors.ListedColormap(['darkblue','navy','steelblue','lightblue',
                                          'lightsteelblue','white','white','mistyrose',
                                          'pink','lightcoral','indianred','brown'])
    cmapPSL.set_over('maroon')
    cmapPSL.set_under('midnightblue')


    latr = FracVar.lat 
    lat = GlobalWarming.lat
    lon = np.arange(0,357.188,2.81)
    gw, lon_c = add_cyclic_point(GlobalWarming.coef/100,lon)
    ta, lon_c = add_cyclic_point(TropicalWarming.coef/100,lon)
    vb_gw, lon_c = add_cyclic_point(VorBreak_GW.coef/100,lon)
    sst1, lon_c = add_cyclic_point(SeaSurfaceTemperature.coef/100,lon)
    sst2, lon_c = add_cyclic_point(SeaSurfaceTemperature2.coef/100,lon)
    sst_iod, lon_c = add_cyclic_point(SeaSurfaceTemperature_IOD.coef/100,lon)
    fv, lon_c  = add_cyclic_point(FracVar.coef,lon)
    gwp, lon_c = add_cyclic_point(GlobalWarming_pval.coef,lon)
    tap,lon_c = add_cyclic_point(TropicalWarming_pval.coef,lon)
    gwp, lon_c = add_cyclic_point(GlobalWarming_pval.coef,lon)
    vb_gwp,lon_c = add_cyclic_point(VorBreak_GW_pval.coef,lon)
    sst1p,lon_c = add_cyclic_point(SeaSurfaceTemperature_pval.coef,lon)
    sst2p,lon_c = add_cyclic_point(SeaSurfaceTemperature2_pval.coef,lon)
    sst_iodp,lon_c = add_cyclic_point(SeaSurfaceTemperature_IOD_pval.coef,lon)
    
    #SoutherHemisphere Stereographic
    fig = plt.figure(figsize=(20, 16),dpi=300,constrained_layout=True)
    projection = ccrs.SouthPolarStereo(central_longitude=300)
    data_crs = ccrs.PlateCarree()

    ax1 = plt.subplot(3,3,1,projection=projection)
    ax1.set_extent([0,359.9, -90, 0], crs=data_crs)
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax1.set_boundary(circle, transform=ax1.transAxes)
    clevels = np.arange(-.24,.28,0.04)
    im1=ax1.contourf(lon_c, lat, ta,clevels,transform=data_crs,cmap=cmapPSL,extend='both')
    levels = [tap.min(),0.05,tap.max()]
    ax1.contourf(lon_c, lat, tap,levels, transform=data_crs,levels=levels, hatches=["...", ""], alpha=0)
    plt.title('a) Tropical warming',fontsize=18)
    ax1.add_feature(cartopy.feature.COASTLINE,alpha=.5)
    ax1.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
    ax1.gridlines(crs=data_crs, linewidth=0.3, linestyle='-')
    ax1.set_extent([-180, 180, -90, 0], ccrs.PlateCarree())
    plt1_ax = plt.gca()
    left_1, bottom_1, width_1, height_1 = plt1_ax.get_position().bounds

    ax2 = plt.subplot(3,3,2,projection=projection)
    ax2.set_extent([0,359.9, -90, 0], crs=data_crs)
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax2.set_boundary(circle, transform=ax2.transAxes)
    #clevels = np.arange(-.08,.1, 0.02)
    im2=ax2.contourf(lon_c, lat, vb_gw,clevels,transform=data_crs,cmap=cmapPSL,extend='both')
    levels = [vb_gwp.min(),0.05,vb_gwp.max()]
    ax2.contourf(lon_c, lat,vb_gwp,levels, transform=data_crs,levels=levels, hatches=["...", ""], alpha=0)
    plt.title('b) Stratospheric polar vortex',fontsize=18)
    ax2.add_feature(cartopy.feature.COASTLINE,alpha=.5)
    ax2.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
    ax2.gridlines(crs=data_crs, linewidth=0.3, linestyle='-')
    ax2.set_extent([-180, 180, -90, 0], ccrs.PlateCarree())
    plt2_ax = plt.gca()
    left_2, bottom_2, width_2, height_2 = plt2_ax.get_position().bounds

    ax3 = plt.subplot(3,3,3,projection=projection)
    ax3.set_extent([0,359.9, -90, 0], crs=data_crs)
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax3.set_boundary(circle, transform=ax3.transAxes)
    #clevels = np.arange(-.08,.1, 0.02)
    im3=ax3.contourf(lon_c, lat, sst1,clevels,transform=data_crs,cmap=cmapPSL,extend='both')
    levels = [sst1p.min(),0.05,sst1p.max()]
    ax3.contourf(lon_c, lat,sst1p,levels, transform=data_crs,levels=levels, hatches=["...", ""], alpha=0)
    plt.title('c) Sea Surface Temperature (Central)',fontsize=18)
    ax3.add_feature(cartopy.feature.COASTLINE,alpha=.5)
    ax3.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
    ax3.gridlines(crs=data_crs, linewidth=0.3, linestyle='-')
    ax3.set_extent([-180, 180, -90, 0], ccrs.PlateCarree())
    plt3_ax = plt.gca()
    left_3, bottom_3, width_3, height_3 = plt3_ax.get_position().bounds

    ax4 = plt.subplot(3,3,4,projection=projection)
    ax4.set_extent([0,359.9, -90, 0], crs=data_crs)
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax4.set_boundary(circle, transform=ax4.transAxes)
    #clevels = np.arange(-.08,.1, 0.02)
    im4=ax4.contourf(lon_c, lat, sst2,clevels,transform=data_crs,cmap=cmapPSL,extend='both')
    levels = [sst2p.min(),0.05,sst2p.max()]
    ax4.contourf(lon_c, lat,sst2p,levels, transform=data_crs,levels=levels, hatches=["...", ""], alpha=0)
    plt.title('d) Sea Surface Temperature (Eastern)',fontsize=18)
    ax4.add_feature(cartopy.feature.COASTLINE,alpha=.5)
    ax4.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
    ax4.gridlines(crs=data_crs, linewidth=0.3, linestyle='-')
    ax4.set_extent([-180, 180, -90, 0], ccrs.PlateCarree())
    plt4_ax = plt.gca()
    left_4, bottom_4, width_4, height_4 = plt4_ax.get_position().bounds
    
    ax7 = plt.subplot(3,3,5,projection=projection)
    ax7.set_extent([0,359.9, -90, 0], crs=data_crs)
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax7.set_boundary(circle, transform=ax7.transAxes)
    #clevels = np.arange(-.08,.1, 0.02)
    im7=ax7.contourf(lon_c, lat, sst_iod,clevels,transform=data_crs,cmap=cmapPSL,extend='both')
    levels = [sst_iodp.min(),0.05,sst_iodp.max()]
    ax7.contourf(lon_c, lat,sst_iodp,levels, transform=data_crs,levels=levels, hatches=["...", ""], alpha=0)
    plt.title('e) Sea Surface Temperature (IOD)',fontsize=18)
    ax7.add_feature(cartopy.feature.COASTLINE,alpha=.5)
    ax7.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
    ax7.gridlines(crs=data_crs, linewidth=0.3, linestyle='-')
    ax7.set_extent([-180, 180, -90, 0], ccrs.PlateCarree())
    plt7_ax = plt.gca()
    left_7, bottom_7, width_7, height_7 = plt7_ax.get_position().bounds
    
    ax5 = plt.subplot(3,3,6,projection=projection)
    ax5.set_extent([0,359.9, -90, 0], crs=data_crs)
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax5.set_boundary(circle, transform=ax5.transAxes)
    clevels = np.arange(-.4,.45,.15)
    im5=ax5.contourf(lon_c, lat, gw,clevels,transform=data_crs,cmap=cmapPSL,extend='both')
    levels = [gwp.min(),0.05,gwp.max()]
    ax5.contourf(lon_c, lat, gwp,levels, transform=data_crs,levels=levels, hatches=["...", ""], alpha=0)
    plt.title('f) Global warming',fontsize=18)
    ax5.add_feature(cartopy.feature.COASTLINE,alpha=.5)
    ax5.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
    ax5.gridlines(crs=data_crs, linewidth=0.3, linestyle='-')
    ax5.set_extent([-180, 180, -90, 0], ccrs.PlateCarree())
    plt5_ax = plt.gca()
    left_5, bottom_5, width_5, height_5 = plt5_ax.get_position().bounds

    ax6 = plt.subplot(3,3,7,projection=projection)
    ax6.set_extent([0,359.9, -90, 0], crs=data_crs)
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    clevels = np.arange(0,1,0.1)
    ax6.set_boundary(circle, transform=ax6.transAxes)
    im6=ax6.contourf(lon_c, latr, fv,clevels,transform=data_crs,cmap='OrRd',extend='both')
    plt.title('g) Fraction of variance explained',fontsize=18)
    ax6.add_feature(cartopy.feature.COASTLINE,alpha=.5)
    ax6.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
    ax6.gridlines(crs=data_crs, linewidth=0.3, linestyle='-')
    ax6.set_extent([-180, 180, -90, 0], ccrs.PlateCarree())
    plt6_ax = plt.gca()
    left_6, bottom_6, width_6, height_6 = plt6_ax.get_position().bounds

    plt.subplots_adjust(bottom=0.2, right=0.8, top=0.8)

    fourth_plot_left = plt4_ax.get_position().bounds[0]
    colorbar_axes4 = fig.add_axes([fourth_plot_left +0.15, bottom_4+0.05, 0.01, height_4*0.6])
    cbar = fig.colorbar(im4, colorbar_axes4, orientation='vertical')
    cbar.set_label('hPa$^{-1}$K$^{-1}$',fontsize=14) #rotation = radianes
    cbar.ax.tick_params(axis='both',labelsize=14)
    
    fifth_plot_left = plt5_ax.get_position().bounds[0]
    colorbar_axes5 = fig.add_axes([fifth_plot_left +0.15, bottom_5+0.05, 0.01, height_5*0.6])
    cbar = fig.colorbar(im5, colorbar_axes5, orientation='vertical')
    cbar.set_label('hPa$^{-1}$K$^{-1}$',fontsize=14) #rotation = radianes
    cbar.ax.tick_params(axis='both',labelsize=14)

    first_plot_left = plt1_ax.get_position().bounds[0]
    colorbar_axes1 = fig.add_axes([first_plot_left +0.15, bottom_1, 0.01, height_1*0.6])
    cbar = fig.colorbar(im1, colorbar_axes1, orientation='vertical')
    cbar.set_label('hPa$^{-1}$K$^{-1}$',fontsize=14) #rotation = radianes
    cbar.ax.tick_params(axis='both',labelsize=14)

    second_plot_left = plt2_ax.get_position().bounds[0]
    colorbar_axes2 = fig.add_axes([second_plot_left +0.15, bottom_2, 0.01, height_2*0.6])
    cbar = fig.colorbar(im2, colorbar_axes2, orientation='vertical')
    cbar.set_label('hPa$^{-1}$K$^{-1}$',fontsize=14) #rotation = radianes
    cbar.ax.tick_params(axis='both',labelsize=14)

    third_plot_left = plt3_ax.get_position().bounds[0]
    colorbar_axes3 = fig.add_axes([third_plot_left +0.15, bottom_3, 0.01, height_3*0.6])
    cbar = fig.colorbar(im3, colorbar_axes3, orientation='vertical')
    cbar.set_label(' ',fontsize=14) #rotation = radianes
    cbar.set_label('hPa$^{-1}$K$^{-1}$',fontsize=14) #rotation = radianes
    cbar.ax.tick_params(axis='both',labelsize=14)
    
    seventh_plot_left = plt7_ax.get_position().bounds[0]
    colorbar_axes7 = fig.add_axes([seventh_plot_left +0.15, bottom_7+0.05, 0.01, height_7*0.6])
    cbar = fig.colorbar(im7, colorbar_axes7, orientation='vertical')
    cbar.set_label('hPa$^{-1}$K$^{-1}$',fontsize=14) #rotation = radianes
    cbar.ax.tick_params(axis='both',labelsize=14)

    sixth_plot_left = plt6_ax.get_position().bounds[0]
    colorbar_axes6 = fig.add_axes([sixth_plot_left +0.15, bottom_6+0.1, 0.01, height_6*0.6])
    cbar = fig.colorbar(im6, colorbar_axes6, orientation='vertical')
    cbar.set_label(' ',fontsize=14) #rotation = radianes
    cbar.ax.tick_params(axis='both',labelsize=14)

    plt.savefig(path_fig+'/pslSONsensitivities_TW_VB_SSTs.png',bbox_inches='tight')
    plt.clf

    return fig 

def plot_sensitivity_zg(indices,indices_pval,frac_var,path_fig):
    
    GlobalWarming = indices[0]
    TropicalWarming = indices[1]
    VorBreak_GW = indices[2]
    SeaSurfaceTemperature = indices[3]
    SeaSurfaceTemperature2 = indices[4]
    SeaSurfaceTemperature_IOD = indices[5]
    FracVar = frac_var
    GlobalWarming_pval = indices_pval[0]
    TropicalWarming_pval = indices_pval[1]
    VorBreak_GW_pval = indices_pval[2]
    SeaSurfaceTemperature_pval = indices_pval[3]
    SeaSurfaceTemperature2_pval = indices_pval[4]
    SeaSurfaceTemperature_IOD_pval = indices_pval[5]

    cmapPSL = mpl.colors.ListedColormap(['darkblue','navy','steelblue','lightblue',
                                          'lightsteelblue','white','white','mistyrose',
                                          'pink','lightcoral','indianred','brown'])
    cmapPSL.set_over('maroon')
    cmapPSL.set_under('midnightblue')


    latr = FracVar.lat 
    lat = GlobalWarming.lat
    lon = np.arange(0,357.188,2.81)
    gw, lon_c = add_cyclic_point(GlobalWarming.coef,lon)
    ta, lon_c = add_cyclic_point(TropicalWarming.coef,lon)
    vb_gw, lon_c = add_cyclic_point(VorBreak_GW.coef,lon)
    sst1, lon_c = add_cyclic_point(SeaSurfaceTemperature.coef,lon)
    sst2, lon_c = add_cyclic_point(SeaSurfaceTemperature2.coef,lon)
    sst_iod, lon_c = add_cyclic_point(SeaSurfaceTemperature_IOD.coef,lon)
    fv, lon_c  = add_cyclic_point(FracVar.coef,lon)
    gwp, lon_c = add_cyclic_point(GlobalWarming_pval.coef,lon)
    tap,lon_c = add_cyclic_point(TropicalWarming_pval.coef,lon)
    gwp, lon_c = add_cyclic_point(GlobalWarming_pval.coef,lon)
    vb_gwp,lon_c = add_cyclic_point(VorBreak_GW_pval.coef,lon)
    sst1p,lon_c = add_cyclic_point(SeaSurfaceTemperature_pval.coef,lon)
    sst2p,lon_c = add_cyclic_point(SeaSurfaceTemperature2_pval.coef,lon)
    sst_iodp,lon_c = add_cyclic_point(SeaSurfaceTemperature_IOD_pval.coef,lon)
    
    #SoutherHemisphere Stereographic
    fig = plt.figure(figsize=(20, 16),dpi=300,constrained_layout=True)
    projection = ccrs.SouthPolarStereo(central_longitude=300)
    data_crs = ccrs.PlateCarree()

    ax1 = plt.subplot(3,3,1,projection=projection)
    ax1.set_extent([0,359.9, -90, 0], crs=data_crs)
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax1.set_boundary(circle, transform=ax1.transAxes)
    clevels = np.arange(-2,2.4,0.4)
    im1=ax1.contourf(lon_c, lat, ta,clevels,transform=data_crs,cmap='PuOr',extend='both')
    levels = [tap.min(),0.1,tap.max()]
    #ax1.contourf(lon_c, lat, tap,levels, transform=data_crs,levels=levels, hatches=["...", ""], alpha=0)
    plt.title('a) Tropical warming',fontsize=18)
    ax1.add_feature(cartopy.feature.COASTLINE,alpha=.5)
    ax1.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
    ax1.gridlines(crs=data_crs, linewidth=0.3, linestyle='-')
    ax1.set_extent([-180, 180, -90, 0], ccrs.PlateCarree())
    plt1_ax = plt.gca()
    left_1, bottom_1, width_1, height_1 = plt1_ax.get_position().bounds

    ax2 = plt.subplot(3,3,2,projection=projection)
    ax2.set_extent([0,359.9, -90, 0], crs=data_crs)
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax2.set_boundary(circle, transform=ax2.transAxes)
    #clevels = np.arange(-.08,.1, 0.02)
    im2=ax2.contourf(lon_c, lat, vb_gw,clevels,transform=data_crs,cmap='PuOr',extend='both')
    levels = [vb_gwp.min(),0.1,vb_gwp.max()]
    #ax2.contourf(lon_c, lat,vb_gwp,levels, transform=data_crs,levels=levels, hatches=["...", ""], alpha=0)
    plt.title('b) Stratospheric polar vortex',fontsize=18)
    ax2.add_feature(cartopy.feature.COASTLINE,alpha=.5)
    ax2.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
    ax2.gridlines(crs=data_crs, linewidth=0.3, linestyle='-')
    ax2.set_extent([-180, 180, -90, 0], ccrs.PlateCarree())
    plt2_ax = plt.gca()
    left_2, bottom_2, width_2, height_2 = plt2_ax.get_position().bounds

    ax3 = plt.subplot(3,3,3,projection=projection)
    ax3.set_extent([0,359.9, -90, 0], crs=data_crs)
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax3.set_boundary(circle, transform=ax3.transAxes)
    #clevels = np.arange(-.08,.1, 0.02)
    im3=ax3.contourf(lon_c, lat, sst1,clevels,transform=data_crs,cmap='PuOr',extend='both')
    levels = [sst1p.min(),0.1,sst1p.max()]
    #ax3.contourf(lon_c, lat,sst1p,levels, transform=data_crs,levels=levels, hatches=["...", ""], alpha=0)
    plt.title('c) Sea Surface Temperature (Central)',fontsize=18)
    ax3.add_feature(cartopy.feature.COASTLINE,alpha=.5)
    ax3.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
    ax3.gridlines(crs=data_crs, linewidth=0.3, linestyle='-')
    ax3.set_extent([-180, 180, -90, 0], ccrs.PlateCarree())
    plt3_ax = plt.gca()
    left_3, bottom_3, width_3, height_3 = plt3_ax.get_position().bounds

    ax4 = plt.subplot(3,3,4,projection=projection)
    ax4.set_extent([0,359.9, -90, 0], crs=data_crs)
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax4.set_boundary(circle, transform=ax4.transAxes)
    #clevels = np.arange(-.08,.1, 0.02)
    im4=ax4.contourf(lon_c, lat, sst2,clevels,transform=data_crs,cmap='PuOr',extend='both')
    levels = [sst2p.min(),0.1,sst2p.max()]
    #ax4.contourf(lon_c, lat,sst2p,levels, transform=data_crs,levels=levels, hatches=["...", ""], alpha=0)
    plt.title('d) Sea Surface Temperature (Eastern)',fontsize=18)
    ax4.add_feature(cartopy.feature.COASTLINE,alpha=.5)
    ax4.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
    ax4.gridlines(crs=data_crs, linewidth=0.3, linestyle='-')
    ax4.set_extent([-180, 180, -90, 0], ccrs.PlateCarree())
    plt4_ax = plt.gca()
    left_4, bottom_4, width_4, height_4 = plt4_ax.get_position().bounds
    
    ax7 = plt.subplot(3,3,5,projection=projection)
    ax7.set_extent([0,359.9, -90, 0], crs=data_crs)
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax7.set_boundary(circle, transform=ax7.transAxes)
    #clevels = np.arange(-.08,.1, 0.02)
    im7=ax7.contourf(lon_c, lat, sst_iod,clevels,transform=data_crs,cmap='PuOr',extend='both')
    levels = [sst_iodp.min(),0.1,sst_iodp.max()]
    ax7.contourf(lon_c, lat,sst_iodp,levels, transform=data_crs,levels=levels, hatches=["...", ""], alpha=0)
    plt.title('e) Sea Surface Temperature (IOD)',fontsize=18)
    ax7.add_feature(cartopy.feature.COASTLINE,alpha=.5)
    ax7.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
    ax7.gridlines(crs=data_crs, linewidth=0.3, linestyle='-')
    ax7.set_extent([-180, 180, -90, 0], ccrs.PlateCarree())
    plt7_ax = plt.gca()
    left_7, bottom_7, width_7, height_7 = plt7_ax.get_position().bounds
    
    ax5 = plt.subplot(3,3,6,projection=projection)
    ax5.set_extent([0,359.9, -90, 0], crs=data_crs)
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax5.set_boundary(circle, transform=ax5.transAxes)
    clevels = np.arange(-40,48,8)
    im5=ax5.contourf(lon_c, lat, gw,clevels,transform=data_crs,cmap='PuOr',extend='both')
    levels = [gwp.min(),0.05,gwp.max()]
    #ax5.contourf(lon_c, lat, gwp,levels, transform=data_crs,levels=levels, hatches=["...", ""], alpha=0)
    plt.title('f) Global warming',fontsize=18)
    ax5.add_feature(cartopy.feature.COASTLINE,alpha=.5)
    ax5.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
    ax5.gridlines(crs=data_crs, linewidth=0.3, linestyle='-')
    ax5.set_extent([-180, 180, -90, 0], ccrs.PlateCarree())
    plt5_ax = plt.gca()
    left_5, bottom_5, width_5, height_5 = plt5_ax.get_position().bounds

    ax6 = plt.subplot(3,3,7,projection=projection)
    ax6.set_extent([0,359.9, -90, 0], crs=data_crs)
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    clevels = np.arange(0,1,0.1)
    ax6.set_boundary(circle, transform=ax6.transAxes)
    im6=ax6.contourf(lon_c, latr, fv,clevels,transform=data_crs,cmap='OrRd',extend='both')
    plt.title('g) Fraction of variance explained',fontsize=18)
    ax6.add_feature(cartopy.feature.COASTLINE,alpha=.5)
    ax6.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
    ax6.gridlines(crs=data_crs, linewidth=0.3, linestyle='-')
    ax6.set_extent([-180, 180, -90, 0], ccrs.PlateCarree())
    plt6_ax = plt.gca()
    left_6, bottom_6, width_6, height_6 = plt6_ax.get_position().bounds

    plt.subplots_adjust(bottom=0.2, right=0.8, top=0.8)

    fourth_plot_left = plt4_ax.get_position().bounds[0]
    colorbar_axes4 = fig.add_axes([fourth_plot_left +0.15, bottom_4+0.05, 0.01, height_4*0.6])
    cbar = fig.colorbar(im4, colorbar_axes4, orientation='vertical')
    cbar.set_label('m$^{-1}$K$^{-1}$',fontsize=14) #rotation = radianes
    cbar.ax.tick_params(axis='both',labelsize=14)
    
    fifth_plot_left = plt5_ax.get_position().bounds[0]
    colorbar_axes5 = fig.add_axes([fifth_plot_left +0.15, bottom_5+0.05, 0.01, height_5*0.6])
    cbar = fig.colorbar(im5, colorbar_axes5, orientation='vertical')
    cbar.set_label('m$^{-1}$K$^{-1}$',fontsize=14) #rotation = radianes
    cbar.ax.tick_params(axis='both',labelsize=14)

    first_plot_left = plt1_ax.get_position().bounds[0]
    colorbar_axes1 = fig.add_axes([first_plot_left +0.15, bottom_1, 0.01, height_1*0.6])
    cbar = fig.colorbar(im1, colorbar_axes1, orientation='vertical')
    cbar.set_label('m$^{-1}$K$^{-1}$',fontsize=14) #rotation = radianes
    cbar.ax.tick_params(axis='both',labelsize=14)

    second_plot_left = plt2_ax.get_position().bounds[0]
    colorbar_axes2 = fig.add_axes([second_plot_left +0.15, bottom_2, 0.01, height_2*0.6])
    cbar = fig.colorbar(im2, colorbar_axes2, orientation='vertical')
    cbar.set_label('m$^{-1}$K$^{-1}$',fontsize=14) #rotation = radianes
    cbar.ax.tick_params(axis='both',labelsize=14)

    third_plot_left = plt3_ax.get_position().bounds[0]
    colorbar_axes3 = fig.add_axes([third_plot_left +0.15, bottom_3, 0.01, height_3*0.6])
    cbar = fig.colorbar(im3, colorbar_axes3, orientation='vertical')
    cbar.set_label(' ',fontsize=14) #rotation = radianes
    cbar.set_label('m$^{-1}$K$^{-1}$',fontsize=14) #rotation = radianes
    cbar.ax.tick_params(axis='both',labelsize=14)
    
    seventh_plot_left = plt7_ax.get_position().bounds[0]
    colorbar_axes7 = fig.add_axes([seventh_plot_left +0.15, bottom_7+0.05, 0.01, height_7*0.6])
    cbar = fig.colorbar(im7, colorbar_axes7, orientation='vertical')
    cbar.set_label('m$^{-1}$K$^{-1}$',fontsize=14) #rotation = radianes
    cbar.ax.tick_params(axis='both',labelsize=14)

    sixth_plot_left = plt6_ax.get_position().bounds[0]
    colorbar_axes6 = fig.add_axes([sixth_plot_left +0.15, bottom_6+0.1, 0.01, height_6*0.6])
    cbar = fig.colorbar(im6, colorbar_axes6, orientation='vertical')
    cbar.set_label(' ',fontsize=14) #rotation = radianes
    cbar.ax.tick_params(axis='both',labelsize=14)

    plt.savefig(path_fig+'/zgSONsensitivities_TW_VB_SSTs.png',bbox_inches='tight')
    plt.clf

    return fig 



def indices_box(indices,indices_names,path):
    fig = plt.figure(figsize=(4, 6), sharey=True)
    plot = pd.DataFrame({indices_names[0]:indices[0],indices_names[2]:indices[2],
                         indices_names[3]:indices[3],indices_names[4]:indices[4]})
    plot.plot.box(patch_artist=True)
    #ax1.set_aspect(1)
    plt.ylabel('Temperature change (K)')
    plt.savefig(path+'/IndicesT_box.pdf', bbox_inches='tight')
    return fig

def plot_sst_indices(indices,indices_names,path):
    index = pd.Index(np.arange(len(indices[0])))
    frame = pd.DataFrame({indices_names[2]:indices[2],indices_names[3]:indices[3],indices_names[4]:indices[4]},index=index)
    indice = pd.Series(['SST change']*len(indices[0]), name='index', dtype='category')
    indice.index = frame.index
    frame['Index'] = indice
    periodos = [indices_names[2:]]
    df = pd.melt(frame, id_vars=['Index'])
    df2 = df.rename(columns = {'variable': 'Period', 'value': 'K'}, inplace = False)
    fig = sns.factorplot(data=df2, x='Index', y='K',
                         col='Period',
                         kind='box', legend=True)
    fig.savefig(path+'/SST_indices.png')
    return fig
                # alternativa ------------------
    #g = sns.FacetGrid(df2, col="Period", size=4, aspect=.7)
    #fig = g.map(sns.boxplot, "Index", "Julian days")
    #fig.savefig("/home/julia.mindlin/Trabajo/storylines_CMIP6/plots/VBdelay_index.png")




#----------------------------------------------------------------------------------

def plot_sensitivity_ua_carree(indices,indices_pval,frac_var,path_fig):
 

    GlobalWarming = indices[0]
    TropicalWarming = indices[1]
    VorBreak_GW = indices[2]
    SeaSurfaceTemperature = indices[3]
    SeaSurfaceTemperature2 = indices[4]
    SeaSurfaceTemperature_IOD = indices[5]
    FracVar = frac_var
    GlobalWarming_pval = indices_pval[0]
    TropicalWarming_pval = indices_pval[1]
    VorBreak_GW_pval = indices_pval[2]
    SeaSurfaceTemperature_pval = indices_pval[3]
    SeaSurfaceTemperature2_pval = indices_pval[4]
    SeaSurfaceTemperature_IOD_pval = indices_pval[5]
    
    cmapU850 = mpl.colors.ListedColormap(['darkblue','navy','steelblue','lightblue',
                                          'lightsteelblue','white','white','mistyrose',
                                          'lightcoral','indianred','brown','firebrick'])
    cmapU850.set_over('maroon')
    cmapU850.set_under('midnightblue')

    path_era = '/datos/ERA5/mon'
    u_ERA = xr.open_dataset(path_era+'/era5.mon.mean.nc')
    u_ERA = u_ERA.u.sel(lev=850).mean(dim='time')

    latr = FracVar.lat 
    lat = GlobalWarming.lat
    lon = np.arange(0,357.188,2.81)
    gw, lon_c = add_cyclic_point(GlobalWarming.coef,lon)
    ta, lon_c = add_cyclic_point(TropicalWarming.coef,lon)
    vb_gw, lon_c = add_cyclic_point(VorBreak_GW.coef,lon)
    sst1, lon_c = add_cyclic_point(SeaSurfaceTemperature.coef,lon)
    sst2, lon_c = add_cyclic_point(SeaSurfaceTemperature2.coef,lon)
    sst_iod, lon_c = add_cyclic_point(SeaSurfaceTemperature_IOD.coef,lon)
    fv, lon_c  = add_cyclic_point(FracVar.coef,lon)
    gwp, lon_c = add_cyclic_point(GlobalWarming_pval.coef,lon)
    tap,lon_c = add_cyclic_point(TropicalWarming_pval.coef,lon)
    gwp, lon_c = add_cyclic_point(GlobalWarming_pval.coef,lon)
    vb_gwp,lon_c = add_cyclic_point(VorBreak_GW_pval.coef,lon)
    sst1p,lon_c = add_cyclic_point(SeaSurfaceTemperature_pval.coef,lon)
    sst2p,lon_c = add_cyclic_point(SeaSurfaceTemperature2_pval.coef,lon)
    sst_iodp,lon_c = add_cyclic_point(SeaSurfaceTemperature_IOD_pval.coef,lon)
    
    #SoutherHemisphere Stereographic
    fig = plt.figure(figsize=(20, 16),dpi=300,constrained_layout=True)
    projection = ccrs.PlateCarree(central_longitude=300)
    data_crs = ccrs.PlateCarree()

    ax1 = plt.subplot(3,3,1,projection=projection)
    ax1.set_extent([0,359.9, -90, 0], crs=data_crs)
    clevels = np.arange(-.24,.28,0.04)
    im1=ax1.contourf(lon_c, lat, ta,clevels,transform=data_crs,cmap=cmapU850,extend='both')
    cnt=ax1.contour(u_ERA.lon,u_ERA.lat, u_ERA.values,levels=[8],transform=data_crs,linewidths=1.2, colors='black', linestyles='-')
    plt.clabel(cnt,inline=True,fmt='%1.0f',fontsize=8)
    levels = [tap.min(),0.05,tap.max()]
    ax1.contourf(lon_c, lat, tap,levels, transform=data_crs,levels=levels, hatches=["...", ""], alpha=0)
    plt.title('a) Tropical warming',fontsize=18)
    ax1.add_feature(cartopy.feature.COASTLINE,alpha=.5)
    ax1.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
    ax1.gridlines(crs=data_crs, linewidth=0.3, linestyle='-')
    ax1.set_extent([-180, 180, -90, 0], ccrs.PlateCarree())
    plt1_ax = plt.gca()
    left_1, bottom_1, width_1, height_1 = plt1_ax.get_position().bounds

    ax2 = plt.subplot(3,3,2,projection=projection)
    ax2.set_extent([0,359.9, -90, 0], crs=data_crs)
    #clevels = np.arange(-.08,.1, 0.02)
    im2=ax2.contourf(lon_c, lat, vb_gw,clevels,transform=data_crs,cmap=cmapU850,extend='both')
    cnt=ax2.contour(u_ERA.lon,u_ERA.lat, u_ERA.values,levels=[8],transform=data_crs,linewidths=1.2, colors='black', linestyles='-')
    plt.clabel(cnt,inline=True,fmt='%1.0f',fontsize=8)
    levels = [vb_gwp.min(),0.05,vb_gwp.max()]
    ax2.contourf(lon_c, lat,vb_gwp,levels, transform=data_crs,levels=levels, hatches=["...", ""], alpha=0)
    plt.title('b) Stratospheric Polar Vortex',fontsize=18)
    ax2.add_feature(cartopy.feature.COASTLINE,alpha=.5)
    ax2.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
    ax2.gridlines(crs=data_crs, linewidth=0.3, linestyle='-')
    ax2.set_extent([-180, 180, -90, 0], ccrs.PlateCarree())
    plt2_ax = plt.gca()
    left_2, bottom_2, width_2, height_2 = plt2_ax.get_position().bounds

    ax3 = plt.subplot(3,3,3,projection=projection)
    ax3.set_extent([0,359.9, -90, 0], crs=data_crs)
    #clevels = np.arange(-.08,.1, 0.02)
    im3=ax3.contourf(lon_c, lat, sst1,clevels,transform=data_crs,cmap=cmapU850,extend='both')
    cnt=ax3.contour(u_ERA.lon,u_ERA.lat, u_ERA.values,levels=[8],transform=data_crs,linewidths=1.2, colors='black', linestyles='-')
    plt.clabel(cnt,inline=True,fmt='%1.0f',fontsize=8)
    levels = [sst1p.min(),0.05,sst1p.max()]
    ax3.contourf(lon_c, lat,sst1p,levels, transform=data_crs,levels=levels, hatches=["...", ""], alpha=0)
    plt.title('c) Sea Surface Temperature (Central)',fontsize=18)
    ax3.add_feature(cartopy.feature.COASTLINE,alpha=.5)
    ax3.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
    ax3.gridlines(crs=data_crs, linewidth=0.3, linestyle='-')
    ax3.set_extent([-180, 180, -90, 0], ccrs.PlateCarree())
    plt3_ax = plt.gca()
    left_3, bottom_3, width_3, height_3 = plt3_ax.get_position().bounds

    ax4 = plt.subplot(3,3,4,projection=projection)
    ax4.set_extent([0,359.9, -90, 0], crs=data_crs)
    #clevels = np.arange(-.08,.1, 0.02)
    im4=ax4.contourf(lon_c, lat, sst2,clevels,transform=data_crs,cmap=cmapU850,extend='both')
    cnt=ax4.contour(u_ERA.lon,u_ERA.lat, u_ERA.values,levels=[8],transform=data_crs,linewidths=1.2, colors='black', linestyles='-')
    plt.clabel(cnt,inline=True,fmt='%1.0f',fontsize=8)
    levels = [sst2p.min(),0.05,sst2p.max()]
    ax4.contourf(lon_c, lat,sst2p,levels, transform=data_crs,levels=levels, hatches=["...", ""], alpha=0)
    plt.title('d) Sea Surface Temperature (Eastern)',fontsize=18)
    ax4.add_feature(cartopy.feature.COASTLINE,alpha=.5)
    ax4.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
    ax4.gridlines(crs=data_crs, linewidth=0.3, linestyle='-')
    ax4.set_extent([-180, 180, -90, 0], ccrs.PlateCarree())
    plt4_ax = plt.gca()
    left_4, bottom_4, width_4, height_4 = plt4_ax.get_position().bounds
    
    ax5 = plt.subplot(3,3,5,projection=projection)
    ax5.set_extent([0,359.9, -90, 0], crs=data_crs)
    #clevels = np.arange(-.08,.1, 0.02)
    im5=ax5.contourf(lon_c, lat, sst_iod,clevels,transform=data_crs,cmap=cmapU850,extend='both')
    cnt=ax5.contour(u_ERA.lon,u_ERA.lat, u_ERA.values,levels=[8],transform=data_crs,linewidths=1.2, colors='black', linestyles='-')
    plt.clabel(cnt,inline=True,fmt='%1.0f',fontsize=8)
    levels = [sst_iodp.min(),0.05,sst_iodp.max()]
    ax5.contourf(lon_c, lat,sst_iodp,levels, transform=data_crs,levels=levels, hatches=["...", ""], alpha=0)
    plt.title('e) Sea Surface Temperature (IOD)',fontsize=18)
    ax5.add_feature(cartopy.feature.COASTLINE,alpha=.5)
    ax5.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
    ax5.gridlines(crs=data_crs, linewidth=0.3, linestyle='-')
    ax5.set_extent([-180, 180, -90, 0], ccrs.PlateCarree())
    plt5_ax = plt.gca()
    left_5, bottom_5, width_5, height_5 = plt5_ax.get_position().bounds
    
    ax6 = plt.subplot(3,3,6,projection=projection)
    ax6.set_extent([0,359.9, -90, 0], crs=data_crs)
    clevels = np.arange(-.4,.45,.15)
    im6=ax6.contourf(lon_c, lat, gw,clevels,transform=data_crs,cmap=cmapU850,extend='both')
    cnt=ax6.contour(u_ERA.lon,u_ERA.lat, u_ERA.values,levels=[8],transform=data_crs,linewidths=1.2, colors='black', linestyles='-')
    plt.clabel(cnt,inline=True,fmt='%1.0f',fontsize=8)
    levels = [gwp.min(),0.05,gwp.max()]
    ax6.contourf(lon_c, lat, gwp,levels, transform=data_crs,levels=levels, hatches=["...", ""], alpha=0)
    plt.title('f) Global warming',fontsize=18)
    ax6.add_feature(cartopy.feature.COASTLINE,alpha=.5)
    ax6.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
    ax6.gridlines(crs=data_crs, linewidth=0.3, linestyle='-')
    ax6.set_extent([-180, 180, -90, 0], ccrs.PlateCarree())
    plt6_ax = plt.gca()
    left_6, bottom_6, width_6, height_6 = plt6_ax.get_position().bounds

    ax7 = plt.subplot(3,3,7,projection=projection)
    ax7.set_extent([0,359.9, -90, 0], crs=data_crs)
    clevels = np.arange(0,1,.1)
    im7=ax7.contourf(lon_c, latr, fv,clevels,transform=data_crs,cmap='OrRd',extend='both')
    plt.title('g) Fraction of variance explained',fontsize=18)
    ax7.add_feature(cartopy.feature.COASTLINE,alpha=.5)
    ax7.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
    ax7.gridlines(crs=data_crs, linewidth=0.3, linestyle='-')
    ax7.set_extent([-180, 180, -90, 0], ccrs.PlateCarree())
    plt7_ax = plt.gca()
    left_7, bottom_7, width_7, height_7 = plt7_ax.get_position().bounds

    plt.subplots_adjust(bottom=0.1, right=0.8, top=0.4)

    sixth_plot_left = plt6_ax.get_position().bounds[0]
    colorbar_axes6 = fig.add_axes([left_6 +0.15, bottom_6-0.30 , 0.01, height_6*1.4])
    cbar = fig.colorbar(im6, colorbar_axes6, orientation='vertical')
    cbar.set_label('ms$^{-1}$K$^{-1}$',fontsize=14) #rotation = radianes
    cbar.ax.tick_params(axis='both',labelsize=14)

    third_plot_left = plt3_ax.get_position().bounds[0]
    colorbar_axes3 = fig.add_axes([left_3 +0.15, bottom_3-0.42, 0.01, height_3*1.4])
    cbar = fig.colorbar(im3, colorbar_axes3, orientation='vertical')
    cbar.set_label(' ',fontsize=14) #rotation = radianes
    cbar.set_label('ms$^{-1}$K$^{-1}$',fontsize=14) #rotation = radianes
    cbar.ax.tick_params(axis='both',labelsize=14)

    seventh_plot_left = plt7_ax.get_position().bounds[0]
    colorbar_axes7 = fig.add_axes([left_7 +0.22, bottom_7-0.1, 0.01, height_7*1.4])
    cbar = fig.colorbar(im7, colorbar_axes7, orientation='vertical')
    cbar.set_label(' ',fontsize=14) #rotation = radianes
    cbar.ax.tick_params(axis='both',labelsize=14)

    plt.savefig(path_fig+'/uaSONsensitivities_TW_VB_SSTs_carree.png',bbox_inches='tight')
    plt.clf

    return fig


def plot_sensitivity_zg_carree(indices,indices_pval,frac_var,path_fig):
 

    GlobalWarming = indices[0]
    TropicalWarming = indices[1]
    VorBreak_GW = indices[2]
    SeaSurfaceTemperature = indices[3]
    SeaSurfaceTemperature2 = indices[4]
    SeaSurfaceTemperature_IOD = indices[5]
    FracVar = frac_var
    GlobalWarming_pval = indices_pval[0]
    TropicalWarming_pval = indices_pval[1]
    VorBreak_GW_pval = indices_pval[2]
    SeaSurfaceTemperature_pval = indices_pval[3]
    SeaSurfaceTemperature2_pval = indices_pval[4]
    SeaSurfaceTemperature_IOD_pval = indices_pval[5]


    latr = FracVar.lat 
    lat = GlobalWarming.lat
    lon = np.arange(0,357.188,2.81)
    gw, lon_c = add_cyclic_point(GlobalWarming.coef,lon)
    ta, lon_c = add_cyclic_point(TropicalWarming.coef,lon)
    vb_gw, lon_c = add_cyclic_point(VorBreak_GW.coef,lon)
    sst1, lon_c = add_cyclic_point(SeaSurfaceTemperature.coef,lon)
    sst2, lon_c = add_cyclic_point(SeaSurfaceTemperature2.coef,lon)
    sst_iod, lon_c = add_cyclic_point(SeaSurfaceTemperature_IOD.coef,lon)
    fv, lon_c  = add_cyclic_point(FracVar.coef,lon)
    gwp, lon_c = add_cyclic_point(GlobalWarming_pval.coef,lon)
    tap,lon_c = add_cyclic_point(TropicalWarming_pval.coef,lon)
    gwp, lon_c = add_cyclic_point(GlobalWarming_pval.coef,lon)
    vb_gwp,lon_c = add_cyclic_point(VorBreak_GW_pval.coef,lon)
    sst1p,lon_c = add_cyclic_point(SeaSurfaceTemperature_pval.coef,lon)
    sst2p,lon_c = add_cyclic_point(SeaSurfaceTemperature2_pval.coef,lon)
    sst_iodp,lon_c = add_cyclic_point(SeaSurfaceTemperature_IOD_pval.coef,lon)
    
    #SoutherHemisphere Stereographic
    fig = plt.figure(figsize=(20, 16),dpi=300,constrained_layout=True)
    projection = ccrs.PlateCarree(central_longitude=300)
    data_crs = ccrs.PlateCarree()

    ax1 = plt.subplot(3,3,1,projection=projection)
    ax1.set_extent([0,359.9, -90, 0], crs=data_crs)
    clevels = np.arange(-2,2.4,0.4)
    im1=ax1.contourf(lon_c, lat, ta,clevels,transform=data_crs,cmap='PuOr',extend='both')
    levels = [tap.min(),0.1,tap.max()]
    ax1.contourf(lon_c, lat, tap,levels, transform=data_crs,levels=levels, hatches=["...", ""], alpha=0)
    plt.title('a) Tropical warming',fontsize=18)
    ax1.add_feature(cartopy.feature.COASTLINE,alpha=.5)
    ax1.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
    ax1.gridlines(crs=data_crs, linewidth=0.3, linestyle='-')
    ax1.set_extent([-180, 180, -90, 0], ccrs.PlateCarree())
    plt1_ax = plt.gca()
    left_1, bottom_1, width_1, height_1 = plt1_ax.get_position().bounds

    ax2 = plt.subplot(3,3,2,projection=projection)
    ax2.set_extent([0,359.9, -90, 0], crs=data_crs)
    #clevels = np.arange(-.08,.1, 0.02)
    im2=ax2.contourf(lon_c, lat, vb_gw,clevels,transform=data_crs,cmap='PuOr',extend='both')
    levels = [vb_gwp.min(),0.1,vb_gwp.max()]
    ax2.contourf(lon_c, lat,vb_gwp,levels, transform=data_crs,levels=levels, hatches=["...", ""], alpha=0)
    plt.title('b) Stratospheric Polar Vortex',fontsize=18)
    ax2.add_feature(cartopy.feature.COASTLINE,alpha=.5)
    ax2.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
    ax2.gridlines(crs=data_crs, linewidth=0.3, linestyle='-')
    ax2.set_extent([-180, 180, -90, 0], ccrs.PlateCarree())
    plt2_ax = plt.gca()
    left_2, bottom_2, width_2, height_2 = plt2_ax.get_position().bounds

    ax3 = plt.subplot(3,3,3,projection=projection)
    ax3.set_extent([0,359.9, -90, 0], crs=data_crs)
    #clevels = np.arange(-.08,.1, 0.02)
    im3=ax3.contourf(lon_c, lat, sst1,clevels,transform=data_crs,cmap='PuOr',extend='both')
    levels = [sst1p.min(),0.1,sst1p.max()]
    ax3.contourf(lon_c, lat,sst1p,levels, transform=data_crs,levels=levels, hatches=["...", ""], alpha=0)
    plt.title('c) Sea Surface Temperature (Central)',fontsize=18)
    ax3.add_feature(cartopy.feature.COASTLINE,alpha=.5)
    ax3.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
    ax3.gridlines(crs=data_crs, linewidth=0.3, linestyle='-')
    ax3.set_extent([-180, 180, -90, 0], ccrs.PlateCarree())
    plt3_ax = plt.gca()
    left_3, bottom_3, width_3, height_3 = plt3_ax.get_position().bounds

    ax4 = plt.subplot(3,3,4,projection=projection)
    ax4.set_extent([0,359.9, -90, 0], crs=data_crs)
    #clevels = np.arange(-.08,.1, 0.02)
    im4=ax4.contourf(lon_c, lat, sst2,clevels,transform=data_crs,cmap='PuOr',extend='both')
    levels = [sst2p.min(),0.1,sst2p.max()]
    ax4.contourf(lon_c, lat,sst2p,levels, transform=data_crs,levels=levels, hatches=["...", ""], alpha=0)
    plt.title('d) Sea Surface Temperature (Eastern)',fontsize=18)
    ax4.add_feature(cartopy.feature.COASTLINE,alpha=.5)
    ax4.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
    ax4.gridlines(crs=data_crs, linewidth=0.3, linestyle='-')
    ax4.set_extent([-180, 180, -90, 0], ccrs.PlateCarree())
    plt4_ax = plt.gca()
    left_4, bottom_4, width_4, height_4 = plt4_ax.get_position().bounds
    
    ax5 = plt.subplot(3,3,5,projection=projection)
    ax5.set_extent([0,359.9, -90, 0], crs=data_crs)
    #clevels = np.arange(-.08,.1, 0.02)
    im5=ax5.contourf(lon_c, lat, sst_iod,clevels,transform=data_crs,cmap='PuOr',extend='both')
    levels = [sst_iodp.min(),0.1,sst_iodp.max()]
    ax5.contourf(lon_c, lat,sst_iodp,levels, transform=data_crs,levels=levels, hatches=["...", ""], alpha=0)
    plt.title('e) Sea Surface Temperature (IOD)',fontsize=18)
    ax5.add_feature(cartopy.feature.COASTLINE,alpha=.5)
    ax5.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
    ax5.gridlines(crs=data_crs, linewidth=0.3, linestyle='-')
    ax5.set_extent([-180, 180, -90, 0], ccrs.PlateCarree())
    plt5_ax = plt.gca()
    left_5, bottom_5, width_5, height_5 = plt5_ax.get_position().bounds
    
    ax6 = plt.subplot(3,3,6,projection=projection)
    ax6.set_extent([0,359.9, -90, 0], crs=data_crs)
    clevels = np.arange(-40,48,8)
    im6=ax6.contourf(lon_c, lat, gw,clevels,transform=data_crs,cmap='PuOr',extend='both')
    levels = [gwp.min(),0.05,gwp.max()]
    #ax6.contourf(lon_c, lat, gwp,levels, transform=data_crs,levels=levels, hatches=["...", ""], alpha=0)
    plt.title('f) Global warming',fontsize=18)
    ax6.add_feature(cartopy.feature.COASTLINE,alpha=.5)
    ax6.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
    ax6.gridlines(crs=data_crs, linewidth=0.3, linestyle='-')
    ax6.set_extent([-180, 180, -90, 0], ccrs.PlateCarree())
    plt6_ax = plt.gca()
    left_6, bottom_6, width_6, height_6 = plt6_ax.get_position().bounds

    ax7 = plt.subplot(3,3,7,projection=projection)
    ax7.set_extent([0,359.9, -90, 0], crs=data_crs)
    clevels = np.arange(0,1,.1)
    im7=ax7.contourf(lon_c, latr, fv,clevels,transform=data_crs,cmap='OrRd',extend='both')
    plt.title('g) Fraction of variance explained',fontsize=18)
    ax7.add_feature(cartopy.feature.COASTLINE,alpha=.5)
    ax7.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
    ax7.gridlines(crs=data_crs, linewidth=0.3, linestyle='-')
    ax7.set_extent([-180, 180, -90, 0], ccrs.PlateCarree())
    plt7_ax = plt.gca()
    left_7, bottom_7, width_7, height_7 = plt7_ax.get_position().bounds

    plt.subplots_adjust(bottom=0.1, right=0.8, top=0.4)

    sixth_plot_left = plt6_ax.get_position().bounds[0]
    colorbar_axes6 = fig.add_axes([left_6 +0.15, bottom_6-0.30 , 0.01, height_6*1.4])
    cbar = fig.colorbar(im6, colorbar_axes6, orientation='vertical')
    cbar.set_label('m$^{-1}$K$^{-1}$',fontsize=14) #rotation = radianes
    cbar.ax.tick_params(axis='both',labelsize=14)

    third_plot_left = plt3_ax.get_position().bounds[0]
    colorbar_axes3 = fig.add_axes([left_3 +0.15, bottom_3-0.42, 0.01, height_3*1.4])
    cbar = fig.colorbar(im3, colorbar_axes3, orientation='vertical')
    cbar.set_label(' ',fontsize=14) #rotation = radianes
    cbar.set_label('m$^{-1}$K$^{-1}$',fontsize=14) #rotation = radianes
    cbar.ax.tick_params(axis='both',labelsize=14)

    seventh_plot_left = plt7_ax.get_position().bounds[0]
    colorbar_axes7 = fig.add_axes([left_7 +0.22, bottom_7-0.1, 0.01, height_7*1.4])
    cbar = fig.colorbar(im7, colorbar_axes7, orientation='vertical')
    cbar.set_label(' ',fontsize=14) #rotation = radianes
    cbar.ax.tick_params(axis='both',labelsize=14)

    plt.savefig(path_fig+'/zgMAMsensitivities_TW_VB_SSTs_carree.png',bbox_inches='tight')
    plt.clf

    return fig
