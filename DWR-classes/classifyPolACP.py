###############################################################
# this is classifying the polarimetric dataset into the same DWR-classes
# author: Leonie von Terzi
###############################################################
import numpy as np
import xarray as xr
import glob as glob
import pandas as pd
# own functions
import classifyPolFunc as classify


##################################################################
outputPath = '/work/lvonterz/tripex_pol/classification/classification_output/DWR_classes/'
histograms = xr.Dataset()

# class dwr
dwrMin, dwrMax = 0, 1.5
dwrMin, dwrMax = 1.5, 4.0
dwrMin, dwrMax = 4.0, 9.5


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
              'WHistProf':{'hist':0},
              'RHHistProf':{'hist':0},
              'maxVelHistProf':{'hist':0},
              'minVelHistProf':{'hist':0},
              'CTTHistProf':{'hist':0},
              'peakNumberProf':{'hist':0},
              'skewHistProf':{'hist':0},
              }

dwrlarge = False
#filePathList = classify.getPeakPathList()
filePathList = sorted(glob.glob('/data/obs/campaigns/tripex-pol/5minmean/*_tripex_pol_data_mean.nc'))
for i,filePath in enumerate(filePathList): #Testing file
# for filePath in filePathList:
    print(filePath)
    date = filePath.split('/')[-1].split('_')[0]
    # read in file:
    
    data = xr.open_dataset(filePath)
    
    #selecting tempearture region   
    ta = data.ta.where((data.ta < -10) & (data.ta > -20)).copy() 
    dwr_kaw = data.DWR_KaW.where(np.isfinite(ta)).copy()
    #check where dwr_kaw has continuous profile
    dwr_kaw = classify.get_cont_prof(dwr_kaw,ta)
    # get maximum of DWR within temperature region
    maxDWR_kaw = dwr_kaw.max(dim='range')
    #--classification of each profile    
    xNBins = 120
    xRange = [-2.5, 1.5]
    dwrHistDic, rvBins, taBins =  classify.variableProfHist2dSimple(dwrHistDic, data.ta, ta, data.Ka_VEL, maxDWR_kaw,
                                                      dwrMin, dwrMax, 'mdvKaHistProf', xRange, xNBins)
    
    # dwr_kaw
    xNBins = 110
    xRange = [-5,15]
    dwrHistDic, dwrBins, taBins =  classify.variableProfHist2dSimple(dwrHistDic, data.ta, ta, data.DWR_KaW, maxDWR_kaw,
                                                      dwrMin, dwrMax,'dwrKaWHistProf', xRange, xNBins)
    
    # dwr_xka
    xNBins = 110
    xRange = [-5,15]
    dwrHistDic, dwrBins, taBins =  classify.variableProfHist2dSimple(dwrHistDic, data.ta, ta, data.DWR_XKa, maxDWR_kaw,
                                                      dwrMin, dwrMax, 'dwrXKaHistProf', xRange, xNBins)
    # dbz_ka
    xNBins = 110
    xRange = [-30,20]
    dwrHistDic, dbzBins, taBins =  classify.variableProfHist2dSimple(dwrHistDic, data.ta, ta, data.Ka_DBZ, maxDWR_kaw,
                                                      dwrMin, dwrMax, 'dbzKaHistProf', xRange, xNBins)
    # dbz_ka
    xNBins = 110
    xRange = [-1,1]
    dwrHistDic, skewBins, taBins =  classify.variableProfHist2dSimple(dwrHistDic, data.ta, ta, data.Ka_SK, maxDWR_kaw,
                                                      dwrMin, dwrMax, 'skewHistProf', xRange, xNBins)                                                  
    # zdr
    xNBins = 110
    xRange = [-1,3]
    dwrHistDic, zdrBins, taBins =  classify.variableProfHist2dSimple(dwrHistDic, data.ta, ta, data.ZDR, maxDWR_kaw,
                                                      dwrMin, dwrMax, 'zdrHistProf', xRange, xNBins)
    # rhv
    xNBins = 110
    xRange = [0.9,1]
    dwrHistDic, rhvBins, taBins =  classify.variableProfHist2dSimple(dwrHistDic, data.ta, ta, data.RHV, maxDWR_kaw,
                                                      dwrMin, dwrMax, 'rhvHistProf', xRange, xNBins)
    
    # kdp
    xNBins = 110
    xRange = [-1,4]
    dwrHistDic, kdpBins, taBins =  classify.variableProfHist2dSimple(dwrHistDic, data.ta, ta, KDP, maxDWR_kaw,
                                                      dwrMin, dwrMax, 'kdpHistProf', xRange, xNBins)
    # sZDRmax
    xNBins = 110
    xRange = [-1,4]
    dwrHistDic, sZDRmaxBins, taBins =  classify.variableProfHist2dSimple(dwrHistDic, data.ta, ta, data.sZDRmax, maxDWR_kaw,
                                                      dwrMin, dwrMax, 'sZDRmaxHistProf', xRange, xNBins)
    
    # maxVel spectrum
    xNBins = 110
    xRange = [-2,2]
    dwrHistDic, maxVelBins, taBins =  classify.variableProfHist2dSimple(dwrHistDic, data.ta, ta, data.max_vel, maxDWR_kaw,
                                                      dwrMin, dwrMax, 'maxVelHistProf', xRange, xNBins)
    
    # minVel spectrum
    xNBins = 110
    xRange = [-3,1]
    dwrHistDic, minVelBins, taBins =  classify.variableProfHist2dSimple(dwrHistDic, data.ta, ta, data.min_vel, maxDWR_kaw,
                                                      dwrMin, dwrMax, 'minVelHistProf', xRange, xNBins)
    


#--defining the coordenate bins 
#print(dwrBins)
dwrCenterBin = dwrBins[0:-1] + np.diff(dwrBins)/2.
taCenterBin = taBins[0:-1] + np.diff(taBins)/2.
skewCenterBin = skewBins[0:-1] + np.diff(skewBins)/2.
dbzCenterBin = dbzBins[0:-1] + np.diff(dbzBins)/2.
rvCenterBin = rvBins[0:-1] + np.diff(rvBins)/2.
kdpCenterBin = kdpBins[0:-1] + np.diff(kdpBins)/2.
zdrCenterBin = zdrBins[0:-1] + np.diff(zdrBins)/2.
rhvCenterBin = rhvBins[0:-1] + np.diff(rhvBins)/2.
sZDRmaxCenterBin = sZDRmaxBins[0:-1] + np.diff(sZDRmaxBins)/2.
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
skewProfs = xr.DataArray(dwrHistDic['skewHistProf']['hist'],
                           coords={'ta':taCenterBin,'skewness':skewCenterBin},
                           dims=('skewness','ta'),
                           name='skewHistProf')                           
kdpProfs = xr.DataArray(dwrHistDic['kdpHistProf']['hist'], 
                           coords={'ta':taCenterBin,'kdp':kdpCenterBin}, 
                           dims=('kdp','ta'),
                           name='kdpHistProf')
                   
zdrProfs = xr.DataArray(dwrHistDic['zdrHistProf']['hist'], 
                           coords={'ta':taCenterBin,'zdr':zdrCenterBin}, 
                           dims=('zdr','ta'),
                           name='zdrHistProf')
rhvProfs = xr.DataArray(dwrHistDic['rhvHistProf']['hist'], 
                           coords={'ta':taCenterBin,'rhv':rhvCenterBin}, 
                           dims=('rhv','ta'),
                           name='rhvHistProf')
#zdpProfs = xr.DataArray(dwrHistDic['zdpHistProf']['hist'],
#                           coords={'ta':taCenterBin,'zdp':zdpCenterBin},
#                           dims=('zdp','ta'),
#                           name='zdpHistProf')
sZDRmaxProfs = xr.DataArray(dwrHistDic['sZDRmaxHistProf']['hist'],
                           coords={'ta':taCenterBin,'sZDRmax':sZDRmaxCenterBin},
                           dims=('sZDRmax','ta'),
                           name='sZDRmaxHistProf')

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
dwrHist = xr.merge([dwrHist, dwrKaWProfs, dwrXKaProfs, dbzKaProfs, rvKaProfs, zdrProfs, rhvProfs, kdpProfs, sZDRmaxProfs,  minVelProfs,maxVelProfs,skewProfs])#,peakProfs])
#dwrHist = xr.merge([dwrHist,dwrKaWProfs,peakProfs])
fileName = 'histograms_trpol_masked_mean_newsEdges_total_cont_profile_2010_3classes_classDWR_{0}_{1}.nc'.format(dwrMin, dwrMax)
dwrHist.to_netcdf(outputPath+fileName)

