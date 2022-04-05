######
# plot Cloud Top Temperature (CTT) classification
# author: Leonie von Terzi
######

import numpy as np
import xarray as xr
import glob as gb
import pandas as pd
import os
import math
from scipy.optimize import curve_fit
import scipy.interpolate as intp

import string

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LogNorm
from matplotlib.colors import ListedColormap

from turbo import turbo
turbo = turbo()

def estPercentil(medFreq, cumFreq, centDwrBins, perc):
    
    func = intp.interp1d(cumFreq, centDwrBins)
    intMedian = func(medFreq)
    
    return intMedian

def getPercentil(cumFreq, centDwrBins, perc):
    
    medFreq = cumFreq[:,-1]*perc # this is the number of profiles at median
    medDwr = np.ones_like(medFreq)*np.nan
    
    for i, tmpFreq in enumerate(medFreq):
        try:
            medDwr[i] = estPercentil(tmpFreq, cumFreq[i], centDwrBins, perc)
            
        except:
            medDwr[i] = np.nan #print(medDwr[i])
        
    return medDwr,medFreq

def plotMedianPaper(fig,ax,bins,varProf,varBins,varName,units,lim,pos,label,legend=False,ylabel=False):
  colors = ['C0','C1','C2']
  
  if varName == 'sEdges':
    for i, dwrLim in enumerate(bins):
    
      trpolName = 'classification_output/histograms_trpol_masked_mean_KDP_from_phidp_cont_profile_CT_minus10_CTT_class_{0}_{1}_contProfThres2.nc'.format(dwrLim[0],dwrLim[1])
      trpolData = xr.open_dataset(trpolName)
      data = trpolData 
      for ii,varProf,varBins in zip(range(2),['minVelHistProf','maxVelHistProf'],['minVel','maxVel']):
        ## getting the median profile from DWR-KaW 
        cumFreq = np.nancumsum(data[varProf].values.T, axis=1)
        profMedFast,medFreq = getPercentil(cumFreq, data[varBins].values, 0.5)
        profMedFast25,medFreq = getPercentil(cumFreq, data[varBins].values, 0.25)
        profMedFast75,medFreq = getPercentil(cumFreq, data[varBins].values, 0.75)

        number = data[varProf].sum(dim=('ta',varBins)).values
        if ii == 0:
            ax.plot(profMedFast, data.ta, 
                 label = 'class '+str(i+1),#+'total number:'+str(number), #r'DWR$_{KaW}$ '+str(dwrLim[0])+':'+str(dwrLim[1]),
                 color='C{0}'.format(i),linewidth=3)
        else:
            ax.plot(profMedFast, data.ta, 
                 color='C{0}'.format(i),linewidth=3)
        ax.fill_betweenx(data.ta,profMedFast25,profMedFast75,
                         color='C{0}'.format(i),LineStyle='--',hatch='+',LineWidth=2,alpha=0.2)
        
  elif varName == 'W_icon':
    data = xr.open_dataset('/work/lvonterz/tripex_pol/ICON/wind_ICON_hist.nc')
    cumFreq = np.nancumsum(data[varProf].values.T, axis=1)
    profMedFast,medFreq = getPercentil(cumFreq, data[varBins].values, 0.5)
    profMedFast25,medFreq = getPercentil(cumFreq, data[varBins].values, 0.25)
    profMedFast75,medFreq = getPercentil(cumFreq, data[varBins].values, 0.75)
    #i = 4
    number = data[varProf].sum(dim=('ta',varBins)).values
    ax.plot(profMedFast, data.ta, 
            color='k',linewidth=2)
        
    ax.fill_betweenx(data.ta,profMedFast25,profMedFast75,
                     color='k',LineStyle='--',LineWidth=2,alpha=0.2)
  
  else:  
    for i, dwrLim in enumerate(bins):
    
      trpolName = 'classification_output/histograms_trpol_masked_mean_KDP_from_phidp_cont_profile_CT_minus10_CTT_class_{0}_{1}_contProfThres2.nc'.format(dwrLim[0],dwrLim[1])
      trpolData = xr.open_dataset(trpolName)
      data = trpolData 
      ## getting the median profile from DWR-KaW 
      cumFreq = np.nancumsum(data[varProf].values.T, axis=1)
      profMedFast,medFreq = getPercentil(cumFreq, data[varBins].values, 0.5)
      profMedFast25,medFreq = getPercentil(cumFreq, data[varBins].values, 0.25)
      profMedFast75,medFreq = getPercentil(cumFreq, data[varBins].values, 0.75)
      profMedFast80,medFreq = getPercentil(cumFreq, data[varBins].values, 0.8)
      profMedFast100,medFreq = getPercentil(cumFreq, data[varBins].values, 0.99)
      
      number = data[varProf].sum(dim=('ta',varBins)).values
      ax.plot(profMedFast, data.ta, 
              label = 'CTT: '+str(dwrLim[0])+'...'+str(dwrLim[1]),#+'total number:'+str(number), #r'DWR$_{KaW}$ '+str(dwrLim[0])+':'+str(dwrLim[1]),
              color=colors[i],linewidth=3)#'C{0}'.format(i*2),linewidth=3)
        
      #ax.fill_betweenx(data.ta,profMedFast25,profMedFast75,
      #                 color='C{0}'.format(i),LineStyle='--',LineWidth=2,alpha=0.2)
      
      ax.fill_betweenx(data.ta,profMedFast25,profMedFast75,
                       facecolor=colors[i],ls='--',lw=2,alpha=0.2)  
    
  ax.set_xlabel(varName+' '+units,fontsize=22)
  ax.grid(True,linestyle='-.')
  #ax.set_ylabel('Temp [°C]',fontsize=18)
  if varName == 'CTT':
    ax.set_ylim(0, -60)
    ax.set_xlabel('# of CTT/# profiles',fontsize=22) 
  if legend == True:
    if varName != 'CTT':
      ax.legend(loc='upper right',fontsize=20) 
    else:
      ax.legend(loc='lower right',fontsize=20) 
  if ylabel==True:
    if varName != 'CTT':
      ax.set_ylim(0, -30)
    ax.set_ylabel('T [°C]',fontsize=22)
    
    #ax.axhline(y=-20,ls='--',color='r',linewidth=2)
    #ax.axhline(y=-10,ls='--',color='r',linewidth=2)
  else:
    if varName != 'CTT':
      ax.set_ylim(0, -30)
    ax.set_yticklabels('')
  ax.axhline(y=-20,ls='--',color='r',linewidth=2)
  ax.axhline(y=-10,ls='--',color='r',linewidth=2)
  ax.set_xlim(lim)
  ax.tick_params(labelsize=20)
  ax.text(pos[0],pos[1],label,fontsize=26)
  plt.tight_layout()
    
  return ax
    

