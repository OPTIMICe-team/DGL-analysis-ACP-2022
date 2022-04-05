######################################################
#post-processing of the spectral data
#this skript is used to interpolate the spectra to the same (fine) doppler grid, 
#smooth the data and add all necessary offsets to have the same reflectivity as the LV2 dataset
######################################################

import os
import time
import xarray as xr
import numpy as np
import netCDF4 as nc
import glob as glob
import pandas as pd

def regridSpec(data,newVelRes=0.01,windowWidth=10):
#-- regrid along the doppler dimension in order to calculate spectral DWR... since we want to smooth the spectra, I am going to interpolate to a finer grid, which will then be smoothed (so we dont loose much information)
# now this is really intricate, but xarray doesnt let me reindex and then rename the doppler coordinate inside of the original dataset... so now we do it like that...
    dataDWR = xr.Dataset()
    newVel = np.arange(-10, 10, newVelRes) #new resolution and doppler velocity vector
    dvX = np.abs(np.diff(data['dopplerX'].values)[0])# normalize spectrum
    X = data['XSpecH']/dvX
    XSpecInt = X.interp(dopplerX=newVel) # interpolate to new vel
    XSpecInt = XSpecInt*newVelRes
    XSpecInt = XSpecInt.rename({'dopplerX':'doppler'})
    #print('interp X done')
    dvKa = np.abs(np.diff(data['dopplerKa'].values)[0])
    Ka = data['KaSpecH']/dvKa
    KaSpecInt = Ka.interp(dopplerKa=newVel)
    KaSpecInt = KaSpecInt*newVelRes
    KaSpecInt = KaSpecInt.rename({'dopplerKa':'doppler'})
    #print('interp Ka done')
    dvW = np.abs(np.diff(data['dopplerW'].values)[0])
    W = data['WSpecH']/dvW
    WSpecInt = W.interp(dopplerW=newVel)
    WSpecInt = WSpecInt*newVelRes
    WSpecInt = WSpecInt.rename({'dopplerW':'doppler'})
    #print('interp W done')
    dataDWR = xr.merge([dataDWR,WSpecInt,KaSpecInt,XSpecInt])
    #print('merging datasets done')
    dataDWR = dataDWR.rolling(doppler=windowWidth,min_periods=1,center=True).mean() #smooth dataset
    return dataDWR

def addOffsets(data,data2,test_interp=False):
 #- add offset found and the offset calculated during the LV2 processing and the atmospheric attenuation. If you want to test against the LV2 Ze, just set test_interp=True
    data['XSpecH'] = 10*np.log10(data['XSpecH']) + data2.rain_offset_X + data2.offset_x + data2.pia_x
    #print('X offsets added')
    data['KaSpecH'] = 10*np.log10(data['KaSpecH']) + data2.rain_offset_Ka +  data2.pia_ka
    data['WSpecH'] = 10*np.log10(data['WSpecH']) + data2.rain_offset_W + data2.offset_w.values[0] + data2.pia_w
    if test_interp==True:
        data['linKaSpec'] = 10**(data['KaSpecH']/10)
        data['ZeKa'] = data['linKaSpec'].sum(dim='doppler')
        data['Ka_DBZ'] = 10*np.log10(data['ZeKa'])

        data['linWSpec'] = 10**(data['WSpecH']/10)
        data['ZeW'] = data['linWSpec'].sum(dim='doppler')
        data['W_DBZ'] = 10*np.log10(data['ZeW'])
        #dataDWR['W_DBZ_LV0'] = 10*np.log10(data['W_Z_H'])+offsetW+data2.pia_w
        #data1['W_DBZ_H'] = data1['W_DBZ_H']+offsetW+data2.pia_w
        data['linXSpec'] = 10**(data['XSpecH']/10)
        data['ZeX'] = data['linXSpec'].sum(dim='doppler')
        data['X_DBZ'] = 10*np.log10(data['ZeX'])

    data['DWR_X_Ka'] = data['XSpecH'] - data['KaSpecH']
    data['DWR_Ka_W'] = data['KaSpecH'] - data['WSpecH']
    return data

def calcOffset(data,var):
# this function calculates the right and left edge of the polarimetric dataset (so meaning that the vector we get here can be used to move the fast edge of the psectra to 0). This is needed, because since we are looking slanted, the Doppler velocities are influenced by the horizontal wind component as well ( so not just the vertical as it would be if we look zenith). Thespectra therefore looks more snake like and we don't have the actual information of the fall velocities of the particles. For now moving it to 0 is the easiest and cleanest way to deal with it. 
    vel = data.Vel.values
    spectra = data[var].values
    velMatrix = spectra/spectra*vel
    
    maxVel = np.nanmax(velMatrix,axis=1)
    minVel = np.nanmin(velMatrix,axis=1)
    minVelXR = xr.DataArray(minVel,
                            dims=('height'),
                            coords={'height':data.height})
    maxVelXR = xr.DataArray(maxVel,
                            dims=('height'),
                            coords={'height':data.height})
    
    return maxVelXR,minVelXR
  
def removeOffset(data):
# here the offset we calculated in calcOffset is used to retrieve a matrix with which we can move the fast edge to 0 m/s. The dataset needs to hae the variable maxVel from th calcOffset function.
    newVelH = data.Vel - data['maxVelH']#.fillna(0)
    
    #newVelH = newVelH.transpose('time','height','Vel')
    data['Vel2ZeroH'] = newVelH.fillna(0)
#    newVelV = data.Vel - data['maxVelV'].fillna(0)
#    newVelV = newVelV.transpose('Time','range','Vel')
#    data['Vel2ZeroV'] = newVelV
#    newVelZDR = data.Vel - data['maxVelZDR'].fillna(0)
#    newVelZDR = newVelZDR.transpose('Time','range','Vel')
#    data['Vel2ZeroZDR'] = newVelZDR
    return data  
#def removeOffset(data):
# here the offset we calculated in calcOffset is used to retrieve a matrix with which we can move the fast edge to 0 m/s. The dataset needs to hae the variable maxVel from th calcOffset function.
#    newVelH = data.Vel - data['maxVelH'].fillna(0)
#    newVelH = newVelH.transpose('time','range','Vel')
#    data['Vel2ZeroH'] = newVelH.fillna(0)
#    return data
