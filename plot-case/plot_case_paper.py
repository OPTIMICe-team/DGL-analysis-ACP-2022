'''
Plot case studie for ACP paper. 
Author: Leonie von Terzi
'''

import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
import numpy as np
import glob
import os
import matplotlib as mpl
import matplotlib.dates as mdates
from matplotlib import cm
from matplotlib.colors import ListedColormap
import string
import postProcessSpectra as post
import math

def round_down_to_even(f):
    return math.floor(f / 2.) * 2

def getNewNipySpectral_r():

    numEnt = 15

    viridis = cm.get_cmap('nipy_spectral_r', 256)
    newcolors = viridis(np.linspace(0, 1, 256))

    colorSpace = np.linspace(144, 198, numEnt)/256
    colorTest=np.zeros((numEnt,4))
    colorTest[:,3] = 1
    colorTest[:,0]=colorSpace

    newcolors[0:15, :] = colorTest
    newcmp = ListedColormap(newcolors)

    return newcmp
def getNewNipySpectral(r=False):

    numEnt = 15
    if r == True:
      viridis = cm.get_cmap('nipy_spectral_r', 256)
      newcolors = viridis(np.linspace(0, 1, 256))
      colorSpace = np.linspace(144, 198, numEnt)/256
      colorTest=np.zeros((numEnt,4))
      colorTest[:,3] = 1
      colorTest[:,0]=colorSpace
      newcolors[0:15, :] = colorTest
    else:
      viridis = cm.get_cmap('nipy_spectral', 256)
      colorSpace = np.linspace(198, 144, numEnt)/(256)
      newcolors = viridis(np.linspace(0, 1, 256))
      colorTest=np.zeros((numEnt,4))
      colorTest[:,3] = 1
      colorTest[:,0]=colorSpace
      newcolors[- numEnt:, :] = colorTest
    newcmp = ListedColormap(newcolors)

    return newcmp

date2start = '20190130'
date2end = '20190130'
dateStart = pd.to_datetime(date2start); dateEnd = pd.to_datetime(date2end)
dateList = pd.date_range(dateStart, dateEnd,freq='d')
pathProcessed = '/data/obs/campaigns/tripex-pol/processed/'
level2_ID = 'tripex_pol_3fr_L2_mom.nc'
level0_ID = 'tripex_pol_3fr_spec_filtered_regridded.nc'
pol_level0_ID = 'tripex_pol_poldata_L0_spec_regridded_dealized.nc'
dataPol_ID = 'tripex_pol_poldata_L1_mom_wind.nc'
dataPath_W = '/data/obs/campaigns/tripex-pol/wband_gra/l0/'
dataPath_Ka = '/data/obs/site/jue/joyrad35/'
dataPath_X = '/data/obs/site/jue/joyrady10/'
dataPathMean = '/data/obs/campaigns/tripex-pol/5minmean/'
for date2proc in dateList:
    # open radar data   
    dataLV2 = xr.open_dataset(pathProcessed+'tripex_pol_level_2/{date}_{ID}'.format(date=date2proc.strftime('%Y%m%d'),ID = level2_ID))
    data = xr.open_dataset(dataPathMean+'{date}_tripex_pol_data_mean.nc'.format(date=date2proc.strftime('%Y%m%d')))
    timeRef = pd.date_range(date2proc,date2proc+pd.offsets.Day(1),freq='Min')
    
    
    pos0 = date2proc + pd.offsets.Minute(10)#position of the enumeration at time
    timesel = pd.to_datetime('20190130 13:30:20') # time at which to plot spectra and moments
    
    '''
    plot moments
    '''
    data['range'] = data['range']*1e-3
    dataLV2['range'] = dataLV2['range']*1e-3
    fig,ax = plt.subplots(nrows=5,figsize=(12,10),sharex=True) #12,10
    radData = {'Ze':{'data':data.Ka_DBZ.interp_like(dataLV2),'axis':ax[0], 'lim':(-30,20),
                     'cbLabel':'Ze [dB]','cmap':getNewNipySpectral(),'ticks':[-30,-5,20]},
               'MDV':{'data':data.Ka_VEL.interp_like(dataLV2),'axis':ax[1],'lim':(0,-1.5),
                      'cbLabel':r'MDV [ms$^{-1}$]','cmap':getNewNipySpectral_r(),'ticks':[0,-0.5,-1.0,-1.5]},
               'DWR_KaW':{'data':data.DWR_KaW.interp_like(dataLV2),'axis':ax[2], 'lim':(-1,15),
                          'cbLabel':r'DWR$_{\rm KaW}$ [dB]','cmap':getNewNipySpectral(),'ticks':[0,7,15]},
               'sZDRmax':{'data':data.sZDRmax.interp_like(dataLV2),'axis':ax[3],'lim':(0,4),
                          'cbLabel':r'sZDR$_{\rm max}$ [dB]','cmap':getNewNipySpectral(),'ticks':[0,2,4]},
               'KDP':{'data':data.KDP.where(~np.isnan(data.sZDRmax)).interp_like(dataLV2),'axis':ax[4],'lim':(0,3),
                      'cbLabel':r'KDP [°km$^{-1}$]','cmap':getNewNipySpectral(),'ticks':[0,1,2,3]},
               }
               
    
    for i,rad in enumerate(radData.keys()):
      print(rad)
      plot = radData[rad]['data'].plot(ax=radData[rad]['axis'],
                                       x='time',
                                       vmax=radData[rad]['lim'][1],
                                       vmin=radData[rad]['lim'][0],
                                       cmap=radData[rad]['cmap'],add_colorbar=False)#,shading='gouraud')#plot_rout.getNewNipySpectral(),add_colorbar=False)
        
      #v1 = np.linspace(radData[rad]['lim'][0], radData[rad]['lim'][1], 5, endpoint=True)
      cb = plt.colorbar(plot,ax=radData[rad]['axis'],ticks=radData[rad]['ticks'],pad=0.01,aspect=10)
      cb.set_label(radData[rad]['cbLabel'],fontsize=18)
      cb.ax.tick_params(labelsize=16)
      CS = data.ta.plot.contour(ax=radData[rad]['axis'],y='range',levels=[-30,-15,0],colors='r',linewidths=[2,2,2],linestyles=['--','--','--'])
      plt.clabel(CS, CS.levels, fontsize=18, fmt='%1.f '+'°C')
      if i == 0:
        radData[rad]['axis'].set_title(date2proc.strftime('%Y%m%d'),fontsize=20)
      if rad == 'KDP':
        plt.setp(plot.axes.xaxis.get_majorticklabels(), rotation=0,horizontalalignment='left')
        radData[rad]['axis'].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
      else:
        radData[rad]['axis'].tick_params(labelbottom=False)      
      radData[rad]['axis'].set_ylabel('')      
      plt.setp(plot.axes.xaxis.get_majorticklabels(), rotation=0,horizontalalignment='left')
      radData[rad]['axis'].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
      radData[rad]['axis'].grid()
      radData[rad]['axis'].set_xlabel('')      
      radData[rad]['axis'].tick_params(axis='y',labelsize=16)
      radData[rad]['axis'].tick_params(axis='x',labelsize=16)      
      radData[rad]['axis'].set_ylim([0,10])
      radData[rad]['axis'].text(pos0,8.4,'('+string.ascii_lowercase[i]+')',fontsize=20)
    fig.supylabel('height above ground [km]',fontsize=18)
    plt.tight_layout()
    filePathName = '{date}_moments'.format(date=date2proc.strftime('%Y%m%d'))
    print(filePathName)
    plt.savefig(filePathName+'.png',bbox_inches='tight',dpi=300)
    #plt.savefig(filePathName+'.pdf',bbox_inches='tight')
    plt.close()
    quit()
    '''
    #now we plot the spectra and profiles
    '''
    
    fileName = '{date}_{hour}_{ID}'.format(date=date2proc.strftime('%Y%m%d'),hour=timesel.strftime('%H'),ID=level0_ID)
    dataLV0 = xr.open_dataset(pathProcessed+'/tripex_pol_level_0/{year}/{month}/{day}/{fileName}'.format(year=date2proc.strftime('%Y'),
                                                                                                     month=date2proc.strftime('%m'),
                                                                                                     day=date2proc.strftime('%d'),
                                                                                                     fileName = fileName))
    # for some reason, the time is not decoded automatically (the units are not in the exact wording that python expects it to be..)
    dataLV0.time.attrs['units'] = 'seconds since 1970-01-01 00:00:00 UTC'
    dataLV0 = xr.decode_cf(dataLV0)
    
    fileNamePol = '{date}_{hour}_{ID}'.format(date=date2proc.strftime('%Y%m%d'),hour=timesel.strftime('%H'),ID=pol_level0_ID)
    datasetPol = xr.open_dataset(pathProcessed+'/tripex_pol_level_0/{year}/{month}/{day}/{fileName}'.format(year=date2proc.strftime('%Y'),
                                                                                                     month=date2proc.strftime('%m'),
                                                                                                     day=date2proc.strftime('%d'),
                                                                                                     fileName = fileNamePol))
    
    
    
    
    #fileEdges = '/data/obs/campaigns/tripex-pol/spectralEdges/spectMaxMinVelTKa_{datestr}.nc'.format(datestr=date2proc.strftime('%Y%m%d'))
    fileEdges = '/data/optimice/tripex-pol/joyrad35/resampled/{year}/{month}/{day}/{datestr}_{hour}_specEdges.nc'.format(year = date2proc.strftime('%Y'),
                                                                                                                     month = date2proc.strftime('%m'),
                                                                                                                     day = date2proc.strftime('%d'),
                                                                                                                     hour=round_down_to_even(float(timesel.strftime('%H'))),
                                                                                                                     datestr=date2proc.strftime('%Y%m%d'))
    print(fileEdges)
    dataEdges = xr.open_dataset(fileEdges)
    
    filePeaks = '/data/obs/campaigns/tripex-pol/spectralPeaks/{datestr}_peaks_joyrad35.nc'.format(datestr=date2proc.strftime('%Y%m%d'))
    dataPeaks = xr.open_dataset(filePeaks)
    peak1 = dataPeaks.sel(peakIndex=0)
    peak1 = peak1.sel(time=timesel)
    
    #quit()    
    datasel = dataLV0.sel(time=timesel)
    #for r in datasel.range:
    #  data = datasel.sel(range=r)
    #  peak1sel = peak1.sel(range=r)
      
    #  data.KaSpecH.plot()
    #  plt.axvline(x=peak1sel.peakVelClass)
      #plt.plot(peak1sel.peakVelClass,peak1sel.peakPowClass)
    #  plt.show()
    #quit()
    data2 = dataLV2.sel(time=timesel)
    data5minmean = data.sel(time=timesel,method='nearest')
    dataEdgessel = dataEdges.sel(time=timesel)
    print(dataEdgessel)
    dataPol = datasetPol.sel(time=timesel)
    #dataPol = dataPol[['sZDR','HSpec','Vel2ZeroH']].where(dataPol['sSNR_H'] > 10.0)
    dataPol = dataPol[['sZDR','HSpec']].where(dataPol['sSNR_H'] > 10.0)
    # move sZDR velocity to 0 for visualisation
    maxVelH,minVelH = post.calcOffset(dataPol,'HSpec')
    dataPol['maxVelH'] = maxVelH
    dataPol['minVelH'] = minVelH
    dataPol = post.removeOffset(dataPol) 
    #print(dataPol)
    #quit()
    # we need to interpolate the spectra to the same velocity grid
    data_interp = post.regridSpec(datasel,windowWidth=10)
    
    # add offsets from LV2 file:
    dataDWR = post.addOffsets(data_interp,data2)
    
    fig,ax = plt.subplots(ncols=3,nrows=2,figsize=(15,10))#,sharey=True)
    ax[0,0].plot(data5minmean.Ka_DBZ,data5minmean.ta,color='k',lw=2)
    ax[0,0].set_xlabel('Ze [dB]',fontsize=22)
    ax[0,0].text(-34,-27,'(a)',fontsize=26)
    ax[0,0].set_ylabel(r'T [$^{\circ}$C]',fontsize=22)
    ax[0,0].set_xticks([-30,-10,10])
    ax[0,0].set_xlim([-35,17])
    
    ax2 = ax[0,0].twiny()
    ax2.plot(data5minmean.Ka_VEL,data5minmean.ta,color='grey',lw=2)
    ax2.set_xlabel(r'MDV [ms$^{-1}$]',color='grey',fontsize=22)
    ax2.tick_params(axis='x', labelcolor='grey')
    ax2.spines['top'].set_edgecolor('grey')
    #ax2.set_xlim([-19,10])
    ax[0,1].plot(data5minmean.Ka_SK,data5minmean.ta,lw=2,color='k')
    ax[0,1].set_xlabel('Skewness',fontsize=22)
    ax[0,1].text(-0.45,-27,'(b)',fontsize=26)
    ax[0,1].set_xlim([-0.5,1.5])
    ax[0,1].tick_params(labelleft=False)
    ax[0,1].set_xticks([-0.5,0.,0.5,1.,1.5])
    
    KDPnew = data5minmean.KDP.where(data5minmean.ta>-21)
    KDPnew = KDPnew.where(data5minmean.ta<-3.3)
    ax[0,2].plot(KDPnew,data5minmean.ta,lw=2,color='k')
    ax[0,2].set_xlabel(r'KDP [°km$^{-1}$]',fontsize=22)
    ax[0,2].text(0,-27,'(c)',fontsize=26)
    ax[0,2].set_xlim([-0.1,2.5])
    ax[0,2].tick_params(labelleft=False)
    ax3 = ax[0,2].twiny()
    ax3.plot(data5minmean.ZDR,data5minmean.ta,lw=2,color='grey')
    ax3.set_xlabel('ZDR [dB]',color='grey',fontsize=22)
    ax3.tick_params(axis='x', labelcolor='grey')
    ax3.spines['top'].set_edgecolor('grey')
    
    for i,a in enumerate([ax[0,0],ax2,ax[0,1],ax[0,2],ax3]):
      if (i != 1) and (i!=4):
        a.grid()
        a.set_ylim([0,-30])
      a.tick_params(axis='both', which='major', labelsize=20)
      
    radData = {'sZe(Ka)':{'data':10*np.log10(datasel['KaSpecH']),'velD':datasel['dopplerKa'],'title':'sZe','axis':ax[1,0],'lim':(-30,10),'xlim':[-2.1,0.3],
                          'cblabel':'sZe [dB]','pos0':-2.0,'label':'(d)'},
               'DWR_KaW':{'data':dataDWR.DWR_Ka_W,'velD':dataDWR.doppler,'title':r'sDWR$_{\rm KaW}$','axis':ax[1,1],'lim':(0,15),'xlim':[-2.1,.3],
                          'cblabel':r'sDWR$_{\rm KaW}$ [dB]','pos0':-2.0,'label':'(e)'},
               'sZDR':{'data':dataPol.sZDR,'velD':dataPol.Vel2ZeroH.fillna(0),'title':r'sZDR','axis':ax[1,2],'lim':(0,4),'xlim':[-2,0.01],
                        'cblabel':'sZDR [dB]','pos0':-1.95,'label':'(f)'}}
    
    for rad in radData.keys():
        print(rad)
        plot = radData[rad]['axis'].pcolormesh(radData[rad]['velD'].values.T,
                              data2['ta'].values,
                              radData[rad]['data'].values,
                              vmin=radData[rad]['lim'][0],vmax=radData[rad]['lim'][1],
                              cmap=getNewNipySpectral())
                              
        
        cb = plt.colorbar(plot,ax=radData[rad]['axis'],pad=0.02,aspect=20)#,ticks=v1)
        cb.set_label(radData[rad]['cblabel'],fontsize=18)
        cb.ax.tick_params(labelsize=16)
        time = pd.to_datetime(str(timesel)).strftime('%Y%m%d %H:%M:%S')
        radData[rad]['axis'].grid()
        radData[rad]['axis'].set_xlabel(r'Doppler velocity [ms$^{-1}]$',fontsize=22)
        if rad == 'sZe(Ka)':
          radData[rad]['axis'].set_ylabel(r'T [$^{\circ}$C]',fontsize=22)
          #peak1.peakVelClass.plot(ax=radData[rad]['axis'],y='range',c='k',lw=2)
          #radData[rad]['axis'].plot(peak1.peakVelClass,data2.ta,c='k',lw=2)
          radData[rad]['axis'].plot(dataEdgessel.maxVel40dB,data2.ta,c='r',lw=2)
          radData[rad]['axis'].plot(dataEdgessel.minVel40dB,data2.ta,c='r',lw=2)
        else:
          radData[rad]['axis'].set_ylabel('')
          radData[rad]['axis'].tick_params(labelleft=False)
        if (rad == 'sZe(Ka)') or (rad == 'DWR_KaW'):
          radData[rad]['axis'].vlines(0.2,ymin=-17.7,ymax=-12.5,lw=3,color='magenta')
          radData[rad]['axis'].vlines(-0.5,ymin=-17.7,ymax=-12.5,lw=3,color='magenta')
          radData[rad]['axis'].hlines(-17.7,xmin=-0.5,xmax=0.2,lw=3,color='magenta')
          radData[rad]['axis'].hlines(-12.5,xmin=-0.5,xmax=0.2,lw=3,color='magenta')
        else:
          radData[rad]['axis'].vlines(0.0,ymin=-17.5,ymax=-12.5,lw=3,color='magenta')
          radData[rad]['axis'].vlines(-0.3,ymin=-17.5,ymax=-12.5,lw=3,color='magenta')
          radData[rad]['axis'].hlines(-17.5,xmin=-0.3,xmax=0.,lw=3,color='magenta')
          radData[rad]['axis'].hlines(-12.5,xmin=-0.3,xmax=0.,lw=3,color='magenta')
        radData[rad]['axis'].tick_params(axis='both', which='major', labelsize=20)
        radData[rad]['axis'].set_xticks([-2,-1,0])
        radData[rad]['axis'].set_ylim([0,-30])
        radData[rad]['axis'].set_xlim(radData[rad]['xlim'])
        radData[rad]['axis'].text(radData[rad]['pos0'],-27,radData[rad]['label'],fontsize=26)
        
    filePathName = '{date}_case_spectra'.format(date=date2proc.strftime('%Y%m%d'))
    print(filePathName)
    plt.tight_layout()    
    plt.savefig(filePathName+'.png',bbox_inches='tight',dpi=300)
    #plt.savefig(filePathName+'.pdf',bbox_inches='tight',dpi=300)
    plt.show()
    quit()
    
    
    
    fig,axes = plt.subplots(ncols=2,figsize=(11,5))
    plot = axes[0].pcolormesh(datasel['dopplerKa'],
                         data2['ta'].values,
                         10*np.log10(datasel['KaSpecH']).values,
                         vmin=-30,vmax=10,
                         cmap=getNewNipySpectral())
    cb = plt.colorbar(plot,ax=axes[0],pad=0.02,aspect=20)#,ticks=v1)
    cb.set_label('sZe [dB]',fontsize=20)
    cb.ax.tick_params(labelsize=16)       
    axes[0].plot(peak1.peakVelClass,data2.ta,c='magenta',lw=3)
    axes[0].set_ylim([-10,-16])
    axes[0].set_xlim([-1.5,0.1])     
    axes[0].set_xticks([-1.5,-1,-0.5,0])
    axes[0].axhline(y=data2.ta.sel(range=2556),c='b',lw=3)
    axes[0].plot(dataEdgessel.maxVel40dB,data2.ta,c='r',lw=3)
    axes[0].plot(dataEdgessel.minVel40dB,data2.ta,c='r',lw=3)
    axes[0].tick_params(axis='both', which='major', labelsize=16)
    axes[0].grid()
    axes[0].set_xlabel(r'Doppler velocity [ms$^{-1}]$',fontsize=20)
    axes[0].set_ylabel(r'T [$^{\circ}$C]',fontsize=22)
    axes[0].text(-1.475,-15.25,'(a)',fontsize=20)
    (10*np.log10(datasel.sel(range=2556).KaSpecH)).plot(ax=axes[1],c='k',lw=3)
    axes[1].set_xlim([-1.5,0.1])
    axes[1].set_ylim([-30,2])
    axes[1].set_xticks([-1.5,-1,-0.5,0])
    axes[1].set_ylabel(r'Ze [dB]',fontsize=22)
    axes[1].set_xlabel(r'Doppler velocity [ms$^{-1}]$',fontsize=20)
    axes[1].set_title('')
    axes[1].tick_params(axis='both', which='major', labelsize=16)
    axes[1].text(-1.475,-2.0,'(b)',fontsize=20)
    axes[1].grid()
    plt.tight_layout()
    plt.savefig('zoom_sZe.png',bbox_inches='tight',dpi=300)
    plt.savefig('zoom_sZe.pdf',bbox_inches='tight',dpi=300)
    plt.show()         
    
               
