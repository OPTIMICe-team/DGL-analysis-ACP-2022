'''
This is classifying the TRIPEx-pol variables into cloudtop-temperature (cct) classes. sofar: only one cloudtop, if we have multiple clouds this does not work!
Author: Leonie von Terzi
'''
import numpy as np
import xarray as xr
import glob as glob
import pandas as pd

import scipy.interpolate as intp

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LogNorm

# own functions:

import classifyCTTfuncs as classify

outputPath = '/work/lvonterz/tripex_pol/CTT/classification_output/'
histograms = xr.Dataset()

# class cct
#cctMin, cctMax = -30, -20
cctMin, cctMax = -40, -30
#cctMin, cctMax = -50, -40
#cctMin, cctMax = -60, -50
# these are all the variables we want to sort into classes
dwrHistDic = {'dwrKaWHistProf':{'hist':0},
              'dwrXKaHistProf':{'hist':0},
              'dbzKaHistProf':{'hist':0},
              'mdvKaHistProf':{'hist':0},
              'zdrHistProf':{'hist':0},
              'rhvHistProf':{'hist':0},
              'kdpHistProf':{'hist':0},
              'kdpAlexHistProf':{'hist':0},
              'sZDRmaxHistProf':{'hist':0},
              'sZDRwidthHistProf':{'hist':0},
              'sZDRmaxwidthHistProf':{'hist':0},
              'zdpHistProf':{'hist':0},
              'upwindHistProf':{'hist':0},
              'maxVelHistProf':{'hist':0},
              'minVelHistProf':{'hist':0},
            }

# read in files to classify (in this case the files are the 5min means of tripex-pol)
filePath = '/work/lvonterz/tripex_pol/output/pol_moments_wind/'
files = sorted(glob.glob(filePath+'*_tripex_pol_data_mean_KDP_from_phidp.nc'))
#with open('times_CTT_{0}_{1}_Ze.txt'.format(cctMin,cctMax),'a+') as file2save:
for f in files:
    print(f)
    data = xr.open_dataset(f)
    # calculate CCT
    range_in_cloud = data.range.where(~np.isnan(data.Ka_DBZ)) # calculate all ranges that have a cloud
    cloudtop = range_in_cloud.max(dim='range') # now find the maximum range of the cloud, alas the cloudtop range
    #print(cloudtop)
    cct = data.ta.sel(range=cloudtop.fillna(0))
    
    # now mask dataset according to Ka_DBZ profile, to make sure that we are continuous from cloud top to the bottom of DGZ
    ta_masked = data.ta.where((data.ta > cct) & (data.ta < -10))        
    mask,cont,tacont = classify.get_cont_prof(data.Ka_DBZ,ta_masked,tol=2)
    
    mask = mask.where(cct < -10)
    
    #--classification of each profile
    
    xNBins = 110
    xRange = [-2.5, 1.5]
    dwrHistDic, rvBins, taBins =  classify.variableProfHist2dSimple(dwrHistDic, data.ta, cct, data.Ka_VEL, 
                                                      cctMin, cctMax, 'mdvKaHistProf', xRange, xNBins,mask)#,file2save) #
    # dwr_kaw
    xNBins = 110
    xRange = [-5,15]
    dwrHistDic, dwrBins, taBins =  classify.variableProfHist2dSimple(dwrHistDic, data.ta, cct, data.DWR_KaW, 
                                                      cctMin, cctMax,'dwrKaWHistProf', xRange, xNBins,mask)
    # dwr_xka
    xNBins = 110
    xRange = [-5,15]
    dwrHistDic, dwrBins, taBins =  classify.variableProfHist2dSimple(dwrHistDic, data.ta, cct, data.DWR_XKa, 
                                                      cctMin, cctMax, 'dwrXKaHistProf', xRange, xNBins,mask)
    
    # dbz_ka
    xNBins = 110
    xRange = [-30,20]
    dwrHistDic, dbzBins, taBins =  classify.variableProfHist2dSimple(dwrHistDic, data.ta, cct, data.Ka_DBZ, 
                                                      cctMin, cctMax, 'dbzKaHistProf', xRange, xNBins,mask)#,write_time=True,file2save=file2save)
    
    # zdr
    xNBins = 110
    xRange = [-1,3]
    dwrHistDic, zdrBins, taBins =  classify.variableProfHist2dSimple(dwrHistDic, data.ta, cct, data.ZDR, 
                                                      cctMin, cctMax, 'zdrHistProf', xRange, xNBins,mask)
    # rhv
    xNBins = 110
    xRange = [0.9,1]
    dwrHistDic, rhvBins, taBins =  classify.variableProfHist2dSimple(dwrHistDic, data.ta, cct, data.RHV, 
                                                      cctMin, cctMax, 'rhvHistProf', xRange, xNBins,mask)
    
    # kdp
    xNBins = 110
    xRange = [-1,4]
    dwrHistDic, kdpBins, taBins =  classify.variableProfHist2dSimple(dwrHistDic, data.ta, cct, data.KDP, 
                                                      cctMin, cctMax, 'kdpHistProf', xRange, xNBins,mask)
    
    # kdp_alexander
    xNBins = 110
    xRange = [-1,4]
    dwrHistDic, kdpBins, taBins =  classify.variableProfHist2dSimple(dwrHistDic, data.ta, cct, data.KDP_Alexander, 
                                                      cctMin, cctMax, 'kdpAlexHistProf', xRange, xNBins,mask)
    
    # zdp
    xNBins = 110
    xRange = [-25,5]
    dwrHistDic, zdpBins, taBins =  classify.variableProfHist2dSimple(dwrHistDic, data.ta, cct, data.ZDP, 
                                                      cctMin, cctMax, 'zdpHistProf', xRange, xNBins,mask)
 
    # sZDRmax
    xNBins = 110
    xRange = [-1,4]
    dwrHistDic, sZDRmaxBins, taBins =  classify.variableProfHist2dSimple(dwrHistDic, data.ta, cct, data.sZDRmax, 
                                                      cctMin, cctMax, 'sZDRmaxHistProf', xRange, xNBins,mask)
    # sZDRwidth
    xNBins = 110
    xRange = [0,2]
    dwrHistDic, sZDRwidthBins, taBins =  classify.variableProfHist2dSimple(dwrHistDic, data.ta, cct, data.sZDRwidth, 
                                                      cctMin, cctMax, 'sZDRwidthHistProf', xRange, xNBins,mask)
    # sZDRmaxwidth
    xNBins = 110
    xRange = [0,0.5]
    dwrHistDic, sZDRmaxwidthBins, taBins =  classify.variableProfHist2dSimple(dwrHistDic, data.ta, cct, data.sZDRmaxwidth, 
                                                      cctMin, cctMax, 'sZDRmaxwidthHistProf', xRange, xNBins,mask)
       
    # maxVel spectrum
    xNBins = 110
    xRange = [-2,2]
    dwrHistDic, maxVelBins, taBins =  classify.variableProfHist2dSimple(dwrHistDic, data.ta, cct, data.max_vel, 
                                                      cctMin, cctMax, 'maxVelHistProf', xRange, xNBins,mask)
    
    # minVel spectrum
    xNBins = 110
    xRange = [-3,1]
    dwrHistDic, minVelBins, taBins =  classify.variableProfHist2dSimple(dwrHistDic, data.ta, cct, data.min_vel, 
                                                      cctMin, cctMax, 'minVelHistProf', xRange, xNBins,mask)
    
#--defining the coordenate bins 
#print(dwrBins)
dwrCenterBin = dwrBins[0:-1] + np.diff(dwrBins)/2.
dbzCenterBin = dbzBins[0:-1] + np.diff(dbzBins)/2.
taCenterBin = taBins[0:-1] + np.diff(taBins)/2.
rvCenterBin = rvBins[0:-1] + np.diff(rvBins)/2.
kdpCenterBin = kdpBins[0:-1] + np.diff(kdpBins)/2.
zdrCenterBin = zdrBins[0:-1] + np.diff(zdrBins)/2.
rhvCenterBin = rhvBins[0:-1] + np.diff(rhvBins)/2.
zdpCenterBin = zdpBins[0:-1] + np.diff(zdpBins)/2.
sZDRmaxCenterBin = sZDRmaxBins[0:-1] + np.diff(sZDRmaxBins)/2.
sZDRwidthCenterBin = sZDRwidthBins[0:-1] + np.diff(sZDRwidthBins)/2.
sZDRmaxwidthCenterBin = sZDRmaxwidthBins[0:-1] + np.diff(sZDRmaxwidthBins)/2.
maxVelCenterBin = maxVelBins[0:-1] + np.diff(maxVelBins)/2.
minVelCenterBin = minVelBins[0:-1] + np.diff(minVelBins)/2.

#--histogram variables
rvKaProfs = xr.DataArray(dwrHistDic['mdvKaHistProf']['hist'], 
                           coords={'ta':taCenterBin,'mdv':rvCenterBin}, 
                           dims=('mdv','ta'),
                           name='mdvKaHistProf')
dwrKaWProfs = xr.DataArray(dwrHistDic['dwrKaWHistProf']['hist'], 
                           coords={'ta':taCenterBin,'dwr':dwrCenterBin}, 
                           dims=('dwr','ta'),
                           name='dwrKaWHistProf')
dwrXKaProfs = xr.DataArray(dwrHistDic['dwrXKaHistProf']['hist'], 
                           coords={'ta':taCenterBin,'dwr':dwrCenterBin}, 
                           dims=('dwr','ta'),
                           name='dwrXKaHistProf')
dbzKaProfs = xr.DataArray(dwrHistDic['dbzKaHistProf']['hist'],
                           coords={'ta':taCenterBin,'dbz':dbzCenterBin},
                           dims=('dbz','ta'),
                           name='dbzKaHistProf')
kdpProfs = xr.DataArray(dwrHistDic['kdpHistProf']['hist'], 
                           coords={'ta':taCenterBin,'kdp':kdpCenterBin}, 
                           dims=('kdp','ta'),
                           name='kdpHistProf')
kdpAlexProfs = xr.DataArray(dwrHistDic['kdpAlexHistProf']['hist'], 
                           coords={'ta':taCenterBin,'kdp':kdpCenterBin}, 
                           dims=('kdp','ta'),
                           name='kdpAlexHistProf')                           
zdrProfs = xr.DataArray(dwrHistDic['zdrHistProf']['hist'], 
                           coords={'ta':taCenterBin,'zdr':zdrCenterBin}, 
                           dims=('zdr','ta'),
                           name='zdrHistProf')
rhvProfs = xr.DataArray(dwrHistDic['rhvHistProf']['hist'], 
                           coords={'ta':taCenterBin,'rhv':rhvCenterBin}, 
                           dims=('rhv','ta'),
                           name='rhvHistProf')
zdpProfs = xr.DataArray(dwrHistDic['zdpHistProf']['hist'],
                           coords={'ta':taCenterBin,'zdp':zdpCenterBin},
                           dims=('zdp','ta'),
                           name='zdpHistProf')
sZDRmaxProfs = xr.DataArray(dwrHistDic['sZDRmaxHistProf']['hist'],
                           coords={'ta':taCenterBin,'sZDRmax':sZDRmaxCenterBin},
                           dims=('sZDRmax','ta'),
                           name='sZDRmaxHistProf')
sZDRwidthProfs = xr.DataArray(dwrHistDic['sZDRwidthHistProf']['hist'],
                           coords={'ta':taCenterBin,'sZDRwidth':sZDRwidthCenterBin},
                           dims=('sZDRwidth','ta'),
                           name='sZDRwidthHistProf')
sZDRmaxwidthProfs = xr.DataArray(dwrHistDic['sZDRmaxwidthHistProf']['hist'],
                           coords={'ta':taCenterBin,'sZDRmaxwidth':sZDRmaxwidthCenterBin},
                           dims=('sZDRmaxwidth','ta'),
                           name='sZDRmaxwidthHistProf')
maxVelProfs = xr.DataArray(dwrHistDic['maxVelHistProf']['hist'],
                           coords={'ta':taCenterBin,'maxVel':maxVelCenterBin},
                           dims=('maxVel','ta'),
                           name='maxVelHistProf')
minVelProfs = xr.DataArray(dwrHistDic['minVelHistProf']['hist'],
                           coords={'ta':taCenterBin,'minVel':minVelCenterBin},
                           dims=('minVel','ta'),
                           name='minVelHistProf')

## # saving as netCDF
dwrHist = xr.Dataset()
dwrHist = xr.merge([dwrHist, dwrKaWProfs, dwrXKaProfs, dbzKaProfs, rvKaProfs, zdrProfs, rhvProfs,zdpProfs, kdpProfs,kdpAlexProfs, sZDRmaxProfs, sZDRwidthProfs, sZDRmaxwidthProfs, minVelProfs,maxVelProfs])
fileName = 'histograms_trpol_masked_mean_KDP_from_phidp_cont_profile_CT_minus10_CTT_class_{0}_{1}_contProfThres2.nc'.format(cctMin, cctMax)
dwrHist.to_netcdf(outputPath+fileName)
    
