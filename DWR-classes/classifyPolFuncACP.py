###############################################################
# this is classifying the polarimetric dataset into the same DWR-classes
# author: Leonie von Terzi
###############################################################
import numpy as np
import xarray as xr
import glob as gb
import pandas as pd
import scipy.interpolate as intp


def variableProfHist2dSimple(dwrHistDic, ta,ta_2010, variable, maxDWR_kaw, 
                            dwrMin, dwrMax, varTarget, xRange, xBins,contProf=True,upwind=False,KDP=False,file2save=None,write_time=False):
    
    if contProf==True and upwind==False:
        
        ta_2010 = ta.where((ta < -10) & (ta > -20))
        variable = get_cont_prof(variable,ta_2010)
        
               
    variable = variable.where((maxDWR_kaw > dwrMin) & (maxDWR_kaw <= dwrMax))
    
    if write_time == True:
      allna_range = variable.isnull().all(dim='range')
      time = variable.time.where(~allna_range,drop=True)
      np.savetxt(file2save,time.dt.strftime('%Y%m%d_%H%M%S').values,fmt="%s")
    
    counts, varBins, taBinsProf = np.histogram2d(variable.values.flatten(),
                                                 ta.values.copy().flatten(),
                                                 range=[xRange,[-50, 0]], 
                                                 bins=[xBins, 100])
    
    
    dwrHistDic[varTarget]['hist'] = dwrHistDic[varTarget]['hist'] + counts
    
    return dwrHistDic, varBins, taBinsProf
def get_cont_prof(variable,ta,KDP=False):
  
  for i,t in enumerate(variable.time):
    ta_sel = ta.sel(time=t)  
    variable_sel = variable.sel(time=t)
    if KDP == True:
      variable_sel = variable_sel.where((variable_sel < 10.0) & (variable_sel > -10.0))
    ta_count = ta_sel.count(dim='range')
    variable_count = variable_sel.where(np.isfinite(ta_sel)).count(dim='range')
    variable_sel = variable_sel.where(ta_count==variable_count)
    #print(variable_sel)
    if i == 0:
      variable_tot = variable_sel
    else:
      variable_tot = xr.concat([variable_tot,variable_sel],dim='time')
    
  
  return variable_tot
