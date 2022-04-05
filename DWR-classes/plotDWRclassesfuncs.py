###########
# function to plot the medians and quantiles of DWR-classes
#author: Leonie von Terzi
##########

import numpy as np
import xarray as xr
import glob as gb
import pandas as pd
import os

from scipy.optimize import curve_fit
import scipy.interpolate as intp

import string

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LogNorm
from matplotlib.colors import ListedColormap
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

def estPercentil(medFreq, cumFreq, centDwrBins, perc):
    
    func = intp.interp1d(cumFreq, centDwrBins)
    intMedian = func(medFreq)
    
    return intMedian

def getPercentil(cumFreq, centDwrBins, perc):
    
    medFreq = cumFreq[:,-1]*perc # this is the number of profiles at median
    medDwr = np.ones_like(medFreq)*np.nan
    
    for i, tmpFreq in enumerate(medFreq):
        try:
            #plt.plot(centDwrBins,cumFreq[i])
            #print(tmpFreq)
            medDwr[i] = estPercentil(tmpFreq, cumFreq[i], centDwrBins, perc)
            #plt.axhline(y=tmpFreq)
            #plt.axvline(x=medDwr[i],c='r')
            #plt.savefig('prof_'+str(i)+'.png')
            #plt.close()
        except:
            medDwr[i] = np.nan #print(medDwr[i])
        
    return medDwr,medFreq
 

def plotMedianPaper(fig,ax,classVar,bins,varProf,varBins,varName,units,lim,pos,label,ylim=(0,-30),legend=False,ylabel=False,legendLoc='upper right',zoom=False):
  lns = 0
  
  if varName == 'spectral edges':
    for i, dwrLim in enumerate(bins):
    
      if classVar == 'KDP':
        trpolName = 'classification_output/KDP_classes/histograms_trpol_masked_mean_total_cont_profile_2010_2classes_classKDP_{0}_{1}.nc'.format(dwrLim[0],dwrLim[1]) 
                     #histograms_trpol_masked_mean_KDP_from_phidp_cont_profile_2010_3classes_classDWRKaW_0_1.5.nc
      else:
        trpolName = 'classification_output/DWR_classes/histograms_trpol_masked_mean_newsEdges_total_cont_profile_2010_3classes_classDWR_{0}_{1}.nc'.format(dwrLim[0],dwrLim[1]) #histograms_trpol_masked_mean_KDP_from_phidp_cont_profile_2010_3classes_classDWR
      trpolData = xr.open_dataset(trpolName)
      data = trpolData 
      cumFreq = np.nancumsum(data['minVelHistProf'].values.T, axis=1)
      profMedFastmin,medFreq = getPercentil(cumFreq, data['minVel'].values, 0.5)
      cumFreq = np.nancumsum(data['maxVelHistProf'].values.T, axis=1)
      profMedFastmax,medFreq = getPercentil(cumFreq, data['maxVel'].values, 0.5)
      #if i == 2:
      #  ax.plot(profMedFastmin-profMedFastmax, data.ta, 
      #          color='C{0}'.format(i),linewidth=3)
      
      for varProf,varBins in zip(['minVelHistProf','maxVelHistProf'],['minVel','maxVel']):
        ## getting the median profile from DWR-KaW 
        cumFreq = np.nancumsum(data[varProf].values.T, axis=1)
        profMedFast,medFreq = getPercentil(cumFreq, data[varBins].values, 0.5)
        profMedFast25,medFreq = getPercentil(cumFreq, data[varBins].values, 0.25)
        profMedFast75,medFreq = getPercentil(cumFreq, data[varBins].values, 0.75)

        number = data[varProf].sum(dim=('ta',varBins)).values
        
        if zoom == True:
          if (i == 2) and (varProf == 'minVelHistProf'):# only plot 3rd DWR class, add this one to legend
            lns = ax.plot(profMedFast, data.ta, 
                          label = 'spectral edges class '+str(i+1),#+'total number:'+str(number), #r'DWR$_{KaW}$ '+str(dwrLim[0])+':'+str(dwrLim[1]),
                          color='C{0}'.format(i),linewidth=3)
            ax.fill_betweenx(data.ta,profMedFast25,profMedFast75,
                             color='C{0}'.format(i),ls='--',lw=2,alpha=0.2)
            cumFreq = np.nancumsum(data['mdvKaHistProf'].values.T, axis=1)
            profMedFast,medFreq = getPercentil(cumFreq, data['mdv'].values, 0.5)
            profMedFast25,medFreq = getPercentil(cumFreq, data['mdv'].values, 0.25)
            profMedFast75,medFreq = getPercentil(cumFreq, data['mdv'].values, 0.75)
            ax.fill_betweenx(data.ta,profMedFast25,profMedFast75,
                             color='k',ls='--',lw=2,alpha=0.2)
            lns1 = ax.plot(profMedFast, data.ta, 
                   label = 'MDV class '+str(i+1),
                   color='k',linewidth=3)  
            lns = lns + lns1    
            
          elif (i==2) and  (varProf == 'maxVelHistProf'): # TODO: remove this again and uncomment part above
            ax.plot(profMedFast, data.ta, 
                    color='C{0}'.format(i),linewidth=3)
            ax.fill_betweenx(data.ta,profMedFast25,profMedFast75,
                             color='C{0}'.format(i),ls='--',lw=2,alpha=0.2)              
          
            
        
        else: # now this is the normal plot
          ax.fill_betweenx(data.ta,profMedFast25,profMedFast75,
                         color='C{0}'.format(i),ls='--',lw=2,alpha=0.2)
          if (i == 0) and (varProf == 'minVelHistProf'):# and (legend == True):
            lns = ax.plot(profMedFast, data.ta, 
                          label = 'class '+str(i+1),#+'total number:'+str(number), #r'DWR$_{KaW}$ '+str(dwrLim[0])+':'+str(dwrLim[1]),
                          color='C{0}'.format(i),linewidth=3)
          #ax.fill_betweenx(data.ta,profMedFast25,profMedFast75,
          #               color='C{0}'.format(i),LineStyle='--',LineWidth=2,alpha=0.2) 
          elif (varProf == 'minVelHistProf'): 
            lns1 = ax.plot(profMedFast, data.ta, 
                          label = 'class '+str(i+1),#+'total number:'+str(number), #r'DWR$_{KaW}$ '+str(dwrLim[0])+':'+str(dwrLim[1]),
                          color='C{0}'.format(i),linewidth=3)
            lns = lns + lns1
          else:
            ax.plot(profMedFast, data.ta, 
                    color='C{0}'.format(i),linewidth=3)
      
    ax.set_xlabel('DV '+units,fontsize=22)#(varName+' '+units,fontsize=22) 
  elif varName == r'W$_{\rm icon}$':
    data = xr.open_dataset('/work/lvonterz/tripex_pol/ICON/wind_ICON_hist_bins_11_exclude_spinup.nc')
    cumFreq = np.nancumsum(data[varProf].values.T, axis=1)
    profMedFast,medFreq = getPercentil(cumFreq, data[varBins].values, 0.5)
    profMedFast25,medFreq = getPercentil(cumFreq, data[varBins].values, 0.25)
    profMedFast75,medFreq = getPercentil(cumFreq, data[varBins].values, 0.75)
    #i = 4
    number = data[varProf].sum(dim=('ta',varBins)).values
    ax.plot(profMedFast, data.ta, 
            color='k',linewidth=3)
        
    ax.fill_betweenx(data.ta,profMedFast25,profMedFast75,
                     color='k',LineStyle='--',LineWidth=2,alpha=0.2)
    ax.set_xlabel(varName+' '+units,fontsize=22)                     
  elif varName == 'CTT':
    for i, dwrLim in enumerate(bins):
      if classVar == 'KDP':
        data = xr.open_dataset('/work/lvonterz/tripex_pol/CTT/classification_output/CTT_classified_KDP_classes_largeDWR.nc')
      else:
        data = xr.open_dataset('/work/lvonterz/tripex_pol/CTT/classification_output/CTT_classified_DWR_classes_2deg_step.nc')
      ax.plot(data['CTTHistProf{0}_{1}'.format(dwrLim[0],dwrLim[1])]/data['CTTHistProf{0}_{1}'.format(dwrLim[0],dwrLim[1])].sum(),data.CTT,
                                                          label = 'class '+str(i+1),#+'total number:'+str(number), #r'DWR$_{KaW}$ '+str(dwrLim[0])+':'+str(dwrLim[1]),
                                                          color='C{0}'.format(i),linewidth=2)        
    ax.set_xlabel('# of CTT',fontsize=22) 
    print(lim)                                                                   
  elif varName == 'number of peaks':
    for i, dwrLim in enumerate(bins):
      trpolName = 'classification_output/DWR_classes/histograms_trpol_masked_mean_total_cont_profile_2010_3classes_classDWRKaW_{0}_{1}.nc'.format(dwrLim[0],dwrLim[1])
      trpolData = xr.open_dataset(trpolName)
      data = trpolData 
      print(data)
      #quit()
      ## getting the median profile from DWR-KaW 
      cumFreq = np.nancumsum(data[varProf].values.T, axis=1)
      profMedFast,medFreq = getPercentil(cumFreq, data[varBins].values, 0.5)
      profMedFast25,medFreq = getPercentil(cumFreq, data[varBins].values, 0.25)
      profMedFast75,medFreq = getPercentil(cumFreq, data[varBins].values, 0.75)
      profMedFast90,medFreq = getPercentil(cumFreq, data[varBins].values, 0.9)
      
      #if i == 2: # TODO: remove that again after you plotted T zoom in
      ax.plot(profMedFast, data.ta, 
              label = 'class '+str(i+1),#r'DWR$_{KaW}$ '+str(dwrLim[0])+':'+str(dwrLim[1]),#'class '+str(i+1),#+'total number:'+str(number), #r'DWR$_{KaW}$ '+str(dwrLim[0])+':'+str(dwrLim[1]),
              color='C{0}'.format(i),linewidth=3)
        
      ax.fill_betweenx(data.ta,profMedFast25,profMedFast75,
                     color='C{0}'.format(i),ls='--',lw=2,alpha=0.2)
        
      #ax.plot(profMedFast90, data.ta, 
              #label = 'class '+str(i+1),#r'DWR$_{KaW}$ '+str(dwrLim[0])+':'+str(dwrLim[1]),#'class '+str(i+1),#+'total number:'+str(number), #r'DWR$_{KaW}$ '+str(dwrLim[0])+':'+str(dwrLim[1]),
       #       color='C{0}'.format(i),linewidth=3,linestyle='--')
              
      ax.set_xlabel(varName+' '+units,fontsize=22)
  
  elif varName == 'peak velocity':
    for i, dwrLim in enumerate(bins):
      trpolName = 'classification_output/DWR_classes/histograms_trpol_masked_mean_peakNumber_cont_profile_2010_3classes_classDWRKaW_{0}_{1}.nc'.format(dwrLim[0],dwrLim[1])
      trpolData = xr.open_dataset(trpolName)
      data = trpolData 
      print(data)
      #quit()
      ## getting the median profile from DWR-KaW 
      for varProf,varBins in zip(['peak2VelProf'],['peak2vel']):
        ## getting the median profile from DWR-KaW 
        cumFreq = np.nancumsum(data[varProf].values.T, axis=1)
        profMedFast,medFreq = getPercentil(cumFreq, data[varBins].values, 0.5)
        profMedFast25,medFreq = getPercentil(cumFreq, data[varBins].values, 0.25)
        profMedFast75,medFreq = getPercentil(cumFreq, data[varBins].values, 0.75)

        number = data[varProf].sum(dim=('ta',varBins)).values
        
        
        ax.fill_betweenx(data.ta,profMedFast25,profMedFast75,
                         color='C{0}'.format(i),LineStyle='--',LineWidth=2,alpha=0.2)
        if (i == 0) and (varProf == 'peak1VelProf'):# and (legend == True):
          lns = ax.plot(profMedFast, data.ta, 
                 label = 'class '+str(i+1),#+'total number:'+str(number), #r'DWR$_{KaW}$ '+str(dwrLim[0])+':'+str(dwrLim[1]),
                 color='C{0}'.format(i),linewidth=3)
          ax.fill_betweenx(data.ta,profMedFast25,profMedFast75,
                         color='C{0}'.format(i),LineStyle='--',LineWidth=2,alpha=0.2) # TODO remove and uncomment ax.fill_betweenx above
        elif (varProf == 'peak1VelProf'): 
          lns1 = ax.plot(profMedFast, data.ta, 
                 label = 'class '+str(i+1),#+'total number:'+str(number), #r'DWR$_{KaW}$ '+str(dwrLim[0])+':'+str(dwrLim[1]),
                 color='C{0}'.format(i),linewidth=3)
          lns = lns + lns1
        else:
          ax.plot(profMedFast, data.ta, 
                 color='C{0}'.format(i),linewidth=3)
              
      ax.set_xlabel(varName+' '+units,fontsize=22)
  else:  
    for i, dwrLim in enumerate(bins):
    
      if classVar == 'KDP':
        trpolName = 'classification_output/KDP_classes/histograms_trpol_masked_mean_total_cont_profile_2010_2classes_classKDP_{0}_{1}.nc'.format(dwrLim[0],dwrLim[1])
      else:
        trpolName = 'classification_output/DWR_classes/histograms_trpol_masked_mean_total_cont_profile_2010_3classes_classDWR_{0}_{1}.nc'.format(dwrLim[0],dwrLim[1])
      trpolData = xr.open_dataset(trpolName)
      data = trpolData 
      print(data)
      
      ## getting the median profile from DWR-KaW 
      cumFreq = np.nancumsum(data[varProf].values.T, axis=1)
      profMedFast,medFreq = getPercentil(cumFreq, data[varBins].values, 0.5)
      profMedFast25,medFreq = getPercentil(cumFreq, data[varBins].values, 0.25)
      profMedFast75,medFreq = getPercentil(cumFreq, data[varBins].values, 0.75)
      profMedFast80,medFreq = getPercentil(cumFreq, data[varBins].values, 0.8)
      #with open('sZDRmax_median_DWRclass_{0}.txt'.format(i),'a+') as f:
      #  np.savetxt(f, np.vstack(('Temp','med')).T, fmt="%s")
      #  np.savetxt(f, np.vstack((data.ta.values,profMedFast)).T, fmt="%.6e")
      #print(profMedFast90)
      #quit()
      
      number = data[varProf].sum(dim=('ta',varBins)).values
      #if i == 2: # TODO: remove that again after you plotted T zoom in
      ax.plot(profMedFast, data.ta, 
              label = 'class '+str(i+1),#r'DWR$_{KaW}$ '+str(dwrLim[0])+':'+str(dwrLim[1]),#'class '+str(i+1),#+'total number:'+str(number), #r'DWR$_{KaW}$ '+str(dwrLim[0])+':'+str(dwrLim[1]),
              color='C{0}'.format(i),linewidth=3)
        
      ax.fill_betweenx(data.ta,profMedFast25,profMedFast75,
                       color='C{0}'.format(i),ls='--',lw=2,alpha=0.2)
        
      #ax.plot(profMedFast90, data.ta, 
              #label = 'class '+str(i+1),#r'DWR$_{KaW}$ '+str(dwrLim[0])+':'+str(dwrLim[1]),#'class '+str(i+1),#+'total number:'+str(number), #r'DWR$_{KaW}$ '+str(dwrLim[0])+':'+str(dwrLim[1]),
       #       color='C{0}'.format(i),linewidth=3,linestyle='--')
              
      ax.set_xlabel(varName+' '+units,fontsize=22)
  #ax.set_ylabel('Temp [°C]',fontsize=18)
  ax.grid(True,linestyle='-.')
  
  if zoom == True:
    ax.vlines(x=-1.58,ymax = -10, ymin=-16,color='r',ls='--',linewidth=2)
    ax.vlines(x=-1.47,ymax = -10, ymin=-14.2,color='r',ls='--',linewidth=2)
    ax.vlines(x=-0.33,ymax = -10, ymin=-18.3,color='r',ls='--',linewidth=2)
    ax.vlines(x=-0.045,ymax = -10, ymin=-14.5,color='r',ls='--',linewidth=2)
    ax.vlines(x=-0.985,ymax = -10, ymin=-15.9,color='r',ls='--',linewidth=2)
    ax.vlines(x=-0.8,ymax = -10, ymin=-13.4,color='r',ls='--',linewidth=2)
  else:
    ax.axhline(y=-20,ls='--',color='r',linewidth=2)
    ax.axhline(y=-10,ls='--',color='r',linewidth=2)
  if legend == True:
    if zoom == True:
      print(lns)
      ax.legend(bbox_to_anchor=(0.1, 1.17),loc=legendLoc,fontsize=16) # TODO: change back to 20 when plotting for paper 
    else:
      ax.legend(loc=legendLoc,fontsize=16)
  if varName == 'CTT':
    ax.set_ylim(0, -60)
    ax.set_ylabel('CTT [°C]',fontsize=22)
  if ylabel==True:
    ax.set_ylim(ylim)
    ax.set_ylabel('T [°C]',fontsize=22)
  else:
    ax.set_ylim(ylim)
    ax.set_yticklabels('')
  
  
  ax.set_xlim(lim)
  ax.tick_params(labelsize=18)
  #ax.hlines(-18, ls='--', color='r',linewidth=1.5)
  
  #ax.xaxis.set_minor_locator(MultipleLocator(.5))
  ax.text(pos[0],pos[1],label,fontsize=26)
  plt.tight_layout()
   
  return ax#,lns

