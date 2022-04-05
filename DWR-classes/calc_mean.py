'''
This skript is meant to calculate the mean over each period where the elevation of the polarimetric radar is constant at an elevation of 30°. This allows us to ensure that the polarimetric and non-pol radars are viewing similar volumns. 
author: Leonie von Terzi
date: 09.03.2021
last modified: 30.03.2022
'''
import os
import time
import numpy as np
import pandas as pd
import xarray as xr
import glob as glob
from sys import argv
from netCDF4 import Dataset
from matplotlib import pyplot as plt
from datetime import date
import scipy.stats as stats
#own routines:
import processData as pro
#import plotting_routines as plot

def globalAttr(data):
    data.attrs['Experiment']= 'TRIPEX-POL, Forschungszentrum Juelich'
    data.attrs['Data']= 'Produced by Leonie von Terzi, lterzi@uni-koeln.de'
    data.attrs['Institution']= 'Data processed within the Emmy-Noether Group OPTIMIce, Institute for Geophysics and Meteorology, University of Cologne, Germany'
    data.attrs['Latitude']= '50.908547 N'
    data.attrs['Longitude']= '6.413536 E'
    data.attrs['Altitude']= 'Altitude of the JOYCE (www.joyce.cloud) platform: 111m asl'
    data.attrs['process_date'] = str(date.today())
    data.attrs['explanation'] = 'data in this file is the temporal mean over periods where the pol. W-Band radar was at 30° elv.'
    return data


def maskData(variable,flag):
# apply masks from LV2 dataset:
    maskP = int('1000000000000000',2) #n points
    maskC = int('0100000000000000',2) # correl
    maskV = int('0010000000000000',2) # Variance
    
    variable = variable.where((flag.values & maskP) != maskP)
    variable = variable.where((flag.values & maskC) != maskC)
    variable = variable.where((flag.values & maskV) != maskV)
    return variable

dateStart = pd.to_datetime('20181101'); dateEnd = pd.to_datetime('20190130')
dateList = pd.date_range(dateStart, dateEnd,freq='D')
dataOutPath = '/work/lvonterz/tripex_pol/output/pol_moments_wind/'

for dayIndex,date2proc in enumerate(dateList):
    print('now reprocessing '+date2proc.strftime('%Y%m%d'))

    # define inputFiles
    fileLV2 = '/data/obs/campaigns/tripex-pol/processed/tripex_pol_level_2/{datestr}_tripex_pol_3fr_L2_mom.nc'.format(datestr=date2proc.strftime('%Y%m%d'))
    fileEdges = glob.glob('/data/optimice/tripex-pol/joyrad35/resampled/{year}/{month}/{day}/{datestr}_*_specEdges.nc'.format(year = date2proc.strftime('%Y'),
                                                                                                                     month = date2proc.strftime('%m'),
                                                                                                                     day = date2proc.strftime('%d'),
                                                                                                                     datestr=date2proc.strftime('%Y%m%d')))
    filePolSpec = '/data/obs/campaigns/tripex-pol/polSpectralVariables/{datestr}_tripex_pol_poldata_maxsZDR_ZDP.nc'.format(datestr=date2proc.strftime('%Y%m%d'))
    filePolLV1 = '/data/obs/campaigns/tripex-pol/processed/tripex_pol_level_1/{datestr}_tripex_pol_poldata_L1_mom.nc'.format(datestr=date2proc.strftime('%Y%m%d'))
    
    if glob.glob(fileLV2):
        #read in all files, since we sometimes have LV2 missing (because no CN file was available to process) I am doing that with if
        dataLV2 = xr.open_dataset(fileLV2)
        #edges
        dataEdges = xr.open_mfdataset(fileEdges)#xr.open_dataset(fileEdges)
        _, index_time = np.unique(dataEdges['time'], return_index=True)
        dataEdges = dataEdges.isel(time=index_time)        
        dataEdges = dataEdges.reindex({'time':dataLV2.time},method='nearest',tolerance='2S')
        # pol spectra
        dataPolSpec = xr.open_dataset(filePolSpec)
        dataPolSpec = dataPolSpec.rename({'height':'range'})
        SNR_H = (10*np.log10(dataPolSpec.SNR_H))       
        
        # pol moments
        dataLV1 = xr.open_dataset(filePolLV1)      
        dataLV1 = dataLV1.rename({'height':'range'}) # all other datasets have range as "height coordinate"
       
        # now mask all data
        ZDRdrop = dataLV1.ZDR.dropna(dim='time',how='all')
        dbz_ka = dataLV2.Ka_DBZ_H
        dbz_x = dataLV2.X_DBZ_H
        dbz_w = dataLV2.W_DBZ_H
        # apply qualitx flags defined in level2 processing step
        qFlagW = dataLV2.quality_flag_offset_w
        qFlagX = dataLV2.quality_flag_offset_x
        dbz_w = maskData(dbz_w, qFlagW)
        dbz_x = maskData(dbz_x, qFlagX)
        KaVel = dataLV2.Ka_VEL_H
        DWR_KaW = dbz_ka - dbz_w
        DWR_XKa = dbz_x - dbz_ka
        # we need to mask the polarimetric spectral data when SNR<10, because otherwise signal to noise ratio is not large enough
        sZDRmax = dataPolSpec.sZDR_max.where(dataPolSpec.SNR_H>10) # only where SNR > 10 
        sZDRmax = sZDRmax.astype(dtype='float32')
        
        newday=True
        # first idea of how to only take means of periods where the pol radar measured at 30°. For now not pretty but it works. I am basically looking if the next time step of the pol data is less or equal to 8s after the current time step. If so, then the data is concatenated into ZDRcon (or other varcon). If the next time is further away, that means that we have left the 30° period and the mean is calculated over that period
        n=0
        for i,t in enumerate(ZDRdrop.time[0:-1]):
            #print(dataLV1)
            tasel = dataLV2.ta.sel(time=t)         
            ZDRsel = ZDRdrop.sel(time=t)
            KDPsel = dataLV1.KDP.sel(time=t)
            Phisel = dataLV1.PhiDP.sel(time=t)
            KaDBZsel = dbz_ka.sel(time=t)
            XDBZsel = dbz_x.sel(time=t)
            WDBZsel = dbz_w.sel(time=t)
            SKsel = dataLV2.Ka_SK_H.sel(time=t)
            DWR_KaWsel = DWR_KaW.sel(time=t)
            DWR_XKasel = DWR_XKa.sel(time=t)
            KaVelsel = KaVel.sel(time=t)
            maxVelsel = dataEdges.maxVel40dB.sel(time=t)
            minVelsel = dataEdges.minVel40dB.sel(time=t)
            sZDRmsel = sZDRmax.sel(time=t)
            
            if pd.to_datetime(str(ZDRdrop.time[i+1].values))<=pd.to_datetime(str(t.values))+pd.offsets.Second(8):
                if n==0:
                    tacon = tasel
                    ZDRcon = ZDRsel
                    KDPcon = KDPsel
                    Phicon = Phisel
                    Kacon = KaDBZsel
                    SKcon = SKsel
                    Xcon = XDBZsel
                    Wcon = WDBZsel
                    KaVelcon = KaVelsel
                    DWR_KaWcon = DWR_KaWsel
                    DWR_XKacon = DWR_XKasel
                    maxVelcon = maxVelsel
                    minVelcon = minVelsel
                    sZDRmcon = sZDRmsel
                    
                else:
                    tacon = xr.concat([tacon,tasel],dim='time')
                    ZDRcon = xr.concat([ZDRcon,ZDRsel],dim='time')
                    KDPcon = xr.concat([KDPcon,KDPsel],dim='time')
                    Phicon = xr.concat([Phicon,Phisel],dim='time')
                    Xcon = xr.concat([Xcon,XDBZsel],dim='time')
                    Kacon = xr.concat([Kacon,KaDBZsel],dim='time')
                    SKcon = xr.concat([SKcon,SKsel],dim='time')
                    Wcon = xr.concat([Wcon,WDBZsel],dim='time')
                    KaVelcon = xr.concat([KaVelcon,KaVelsel],dim='time')
                    DWR_KaWcon = xr.concat([DWR_KaWcon,DWR_KaWsel],dim='time')
                    DWR_XKacon = xr.concat([DWR_XKacon,DWR_XKasel],dim='time')
                    maxVelcon = xr.concat([maxVelcon,maxVelsel],dim='time')
                    minVelcon = xr.concat([minVelcon,minVelsel],dim='time')
                    sZDRmcon = xr.concat([sZDRmcon,sZDRmsel],dim='time')
                n+=1
            else:
                if  n>1 and newday==False:
                    tamean = xr.concat([tamean,tacon.mean(dim='time',keep_attrs=True)],dim='time')
                    tmean = xr.concat([tmean,DWR_KaWcon.time.mean(keep_attrs=True)],dim='time')
                    ZDRmean = xr.concat([ZDRmean,ZDRcon.mean(dim='time',keep_attrs=True)],dim='time')
                    KDPmean = xr.concat([KDPmean,KDPcon.mean(dim='time',keep_attrs=True)],dim='time')
                    Phimean = xr.concat([Phimean,Phicon.mean(dim='time',keep_attrs=True)],dim='time')
                    RHVmean = xr.concat([RHVmean,RHVcon.mean(dim='time',keep_attrs=True)],dim='time')
                    Xmean = xr.concat([Xmean,Xcon.mean(dim='time',keep_attrs=True)],dim='time')
                    Kamean = xr.concat([Kamean,Kacon.mean(dim='time',keep_attrs=True)],dim='time')
                    SKmean = xr.concat([SKmean,SKcon.mean(dim='time',keep_attrs=True)],dim='time')
                    KaVelmean = xr.concat([KaVelmean,KaVelcon.mean(dim='time',keep_attrs=True)],dim='time')
                    Wmean = xr.concat([Wmean,Wcon.mean(dim='time',keep_attrs=True)],dim='time')
                    DWR_KaWmean = xr.concat([DWR_KaWmean,DWR_KaWcon.mean(dim='time',keep_attrs=True)],dim='time')
                    DWR_XKamean = xr.concat([DWR_XKamean,DWR_XKacon.mean(dim='time',keep_attrs=True)],dim='time')
                    maxVelmean = xr.concat([maxVelmean,maxVelcon.mean(dim='time',keep_attrs=True)],dim='time')
                    minVelmean = xr.concat([minVelmean,minVelcon.mean(dim='time',keep_attrs=True)],dim='time')
                    sZDRmmean = xr.concat([sZDRmmean,sZDRmcon.mean(dim='time',keep_attrs=True)],dim='time')
                    
                elif newday==True and n>1:
                    tmean = DWR_KaWcon.time.mean(keep_attrs=True)
                    ZDRmean = ZDRcon.mean(dim='time',keep_attrs=True)
                    KDPmean = KDPcon.mean(dim='time',keep_attrs=True)
                    Phimean = Phicon.mean(dim='time',keep_attrs=True)
                    RHVmean = RHVcon.mean(dim='time',keep_attrs=True)
                    Xmean = Xcon.mean(dim='time',keep_attrs=True)
                    Kamean = Kacon.mean(dim='time',keep_attrs=True)
                    SKmean = SKcon.mean(dim='time',keep_attrs=True)
                    KaVelmean = KaVelcon.mean(dim='time',keep_attrs=True)
                    Wmean = Wcon.mean(dim='time',keep_attrs=True)
                    DWR_KaWmean = DWR_KaWcon.mean(dim='time',keep_attrs=True)
                    DWR_XKamean = DWR_XKacon.mean(dim='time',keep_attrs=True)
                    maxVelmean = maxVelcon.mean(dim='time',keep_attrs=True)
                    minVelmean = minVelcon.mean(dim='time',keep_attrs=True)
                    sZDRmmean = sZDRmcon.mean(dim='time',keep_attrs=True)
                    tamean = tacon.mean(dim='time',keep_attrs=True)
                    newday=False
                n=0
        # now save everything into a nc file
        
        meanDS = xr.merge([tamean.rename('ta'),ZDRmean.rename('ZDR'),KDPmean.rename('KDP'),Phimean.rename('PhiDP'),sZDRmmean.rename('sZDRmax'),
                           Xmean.rename('X_DBZ'),Kamean.rename('Ka_DBZ'),Wmean.rename('W_DBZ'),SKmean.rename('Ka_SK'),
                           KaVelmean.rename('Ka_VEL'),maxVelmean.rename('max_vel'),minVelmean.rename('min_vel'),
                           DWR_XKamean.rename('DWR_XKa'),DWR_KaWmean.rename('DWR_KaW'),
                           ])
        meanDS['time'] = tmean        
        meanDS.to_netcdf(dataOutPath+'{datestr}_tripex_pol_data_new_sEdges.nc'.format(datestr=date2proc.strftime('%Y%m%d')))
        
        
