###############################################################
# this is classifying the polarimetric dataset into CTT classes
# author: leonie von Terzi
###############################################################
import numpy as np
import xarray as xr
import glob as gb
import pandas as pd
import matplotlib.pyplot as plt
import scipy.interpolate as intp


def maskData(variable, flag, mask):
    
    variable = variable.where((flag.values & mask) != mask)
    
    return variable

def variableProfHist2dSimple(dwrHistDic, ta, cct, variable, 
                            cctMin, cctMax, varTarget, xRange, xBins, mask,contProf=False,upwind=False,KDP=False,write_time=False,file2save=None):
# classify into cct-classes, input: histogram dictionary, temperature, cct, the variable to classify, cct intervals, name of variable in histogram dic
# the bins to make histogram, if contProf: only use continuous profiles from cloudtop to -10    
    if contProf and upwind==False:# and KDP==False:
        ta_masked = ta.where((ta > cct) & (ta < -10))        
        variable,cont,tacont = get_cont_prof(variable,ta_masked,tol=2)
        
    variable = variable.where(~np.isnan(mask))           
    variable = variable.where((cct > cctMin) & (cct <= cctMax))
    if write_time == True:
      allna_range = variable.isnull().all(dim='range')
      time = variable.time.where(~allna_range,drop=True)
      np.savetxt(file2save,time.dt.strftime('%Y%m%d_%H%M%S').values,fmt="%s")
    
    counts, varBins, taBinsProf = np.histogram2d(variable.values.flatten(),
                                                 ta.values.copy().flatten(),
                                                 range=[xRange,[-60, 0]], 
                                                 bins=[xBins, 100])
    
    dwrHistDic[varTarget]['hist'] = dwrHistDic[varTarget]['hist'] + counts
    
    return dwrHistDic, varBins, taBinsProf

def get_cont_prof(variable,ta,tol=0):
# this checks wether the variable is continously available in a certain temperature interval (e.g. -20 to -10°C), tolerance of range gates when missing data, default=0
    ta_count = ta.count(dim='range')
    variable_count = variable.where(np.isfinite(ta)).count(dim='range')
    variable = variable.where(ta_count<=variable_count+tol)
    return variable,variable_count,ta_count
