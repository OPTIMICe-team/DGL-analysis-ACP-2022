######
# plot Cloud Top Temperature (CTT) classification
# author: Leonie von Terzi
######
import numpy as np
import xarray as xr
import glob as gb
import pandas as pd
import plotCTTclassesfuncs as plot
import matplotlib.pyplot as plt 
bins = [(-60,-50),(-50,-40),(-40,-30)]

fig,axes = plt.subplots(nrows=2,ncols=3,figsize=(12,10))#,sharey=True) #figsize=(cols,rows)
axes[0,0] = plot.plotMedianPaper(fig,axes[0,0],bins,'dbzKaHistProf','dbz',r'Ze$_{\rm Ka}$','[dB]',(-25,15),(-24.5,-27),'(a)',ylabel=True)
plt.tight_layout()
axes[0,1] = plot.plotMedianPaper(fig,axes[0,1],bins,'dwrKaWHistProf','dwr',r'DWR$_{\rm KaW}$','[dB]',(-1.5,6.5),(-1.4,-27),'(b)')#(-1,5)
plt.tight_layout()
axes[0,2] = plot.plotMedianPaper(fig,axes[0,2],bins,'dwrXKaHistProf','dwr',r'DWR$_{\rm XKa}$','[dB]',(-1.5,6.5),(-1.4,-27),'(c)',legend=True)#(-1,5)
plt.tight_layout()
axes[1,0] = plot.plotMedianPaper(fig,axes[1,0],bins,'zdrHistProf','zdr','ZDR','[dB]',(0,0.85),(0.03,-27),'(d)',ylabel=True)#(0,0.8)
plt.tight_layout()
axes[1,1] = plot.plotMedianPaper(fig,axes[1,1],bins,'kdpHistProf','kdp','KDP',r'[°km$^{-1}$]',(-0.3,1.4),(-0.25,-27),'(e)')#(-0.25,1.5)
plt.tight_layout()
axes[1,2] = plot.plotMedianPaper(fig,axes[1,2],bins,'sZDRmaxHistProf','sZDRmax',r'sZDR$_{\rm max}$','[dB]',(0.4,2.4),(0.45,-27),'(f)')#(0.5,2.5)
plt.tight_layout()
plt.savefig('plot_medians_paper_CTT_classes_DBZ_DWR_KaW_XKa_KDP_sZDRmax_ZDR.png',bbox_inches='tight')#plot_medians_paper_KDP_2classes_largeDWR_4x2
plt.savefig('plot_medians_paper_CTT_classes_DBZ_DWR_KaW_XKa_KDP_sZDRmax_ZDR.pdf',bbox_inches='tight')#plot_medians_paper_KDP_2classes_largeDWR_4x2
plt.close()

