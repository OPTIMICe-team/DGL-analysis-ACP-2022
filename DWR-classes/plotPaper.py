#########
# plot DWR classes
#########

import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import plotDWRclassesfuncs as plot

bins = [(0,1.5),(1.5,4.0),(4.0,9.5)]

# dbz and dwr, MDV
fig,axes = plt.subplots(nrows=1,ncols=4,figsize=(16,5)) #(20,5)
axes[0] = plot.plotMedianPaper(fig,axes[0],'KaW',bins,'dbzKaHistProf','dbz',r'Ze$_{\rm Ka}$','[dB]',(-25,15),(-24.5,-27),'(a)',ylabel=True)
axes[1] = plot.plotMedianPaper(fig,axes[1],'KaW',bins,'dwrKaWHistProf','dwr',r'DWR$_{\rm KaW}$','[dB]',(-1.5,7),(-1.4,-27),'(b)')
axes[2] = plot.plotMedianPaper(fig,axes[2],'KaW',bins,'dwrXKaHistProf','dwr',r'DWR$_{\rm XKa}$','[dB]',(-1.5,7),(-1.4,-27),'(c)',legend=True)
axes[3] = plot.plotMedianPaper(fig,axes[3],'KaW',bins,'mdvKaHistProf','mdv','MDV',r'[ms$^{-1}$]',(-1.3,-0.4),(-1.28,-27),'(d)')
plt.tight_layout()
plt.savefig('plot_medians_paper_DBZ_DWR_MDV_new_DWRclasses1.png',bbox_inches='tight',dpi=300)
plt.savefig('plot_medians_paper_DBZ_DWR_MDV_new_DWRclasses1.pdf',bbox_inches='tight',dpi=300)
plt.close()


# pol and skewness
fig,axes = plt.subplots(nrows=1,ncols=4,figsize=(16,5))#12,5
axes[0] = plot.plotMedianPaper(fig,axes[0],'KaW',bins,'zdrHistProf','zdr','ZDR','[dB]',(0,0.9),(0.01,-27),'(a)',ylabel=True)
axes[1] = plot.plotMedianPaper(fig,axes[1],'KaW',bins,'sZDRmaxHistProf','sZDRmax',r'sZDR$_{\rm max}$','[dB]',(0.5,2.5),(0.55,-27),'(b)')
axes[2] = plot.plotMedianPaper(fig,axes[2],'KaW',bins,'kdpHistProf','kdp','KDP',r'[°km$^{-1}$]',(-0.25,1.6),(-0.2,-27),'(c)')
axes[3] = plot.plotMedianPaper(fig,axes[3],'KaW',bins,'skewHistProf','skewness','skewness','',(-0.2,0.5),(-0.19,-27),'(d)',legend=True)
plt.tight_layout()
plt.savefig('plot_medians_paper_ZDR_sZDRmax_KDP_SK.png',bbox_inches='tight',dpi=300)
plt.savefig('plot_medians_paper_ZDR_sZDRmax_KDP_SK.pdf',bbox_inches='tight',dpi=300)
plt.close()

# spectral edges and zoom
fig,axes = plt.subplots(nrows=1,ncols=1,figsize=(5,5))#,sharey=True)
axes = plot.plotMedianPaper(fig,axes,'KaW',bins,'','','spectral edges',r'[ms$^{-1}$]',(-2.5,0.5),(0.1,-27),'(a)',ylabel=True,legend=True,legendLoc='upper left')#(-30,-32)
plt.tight_layout()
plt.savefig('plot_medians_paper_sEdges_newDWRclasses.png',bbox_inches='tight')
plt.savefig('plot_medians_paper_sEdges_newDWRclasses.pdf',bbox_inches='tight')
plt.close()
fig,axes = plt.subplots(nrows=1,ncols=1,figsize=(5,5),sharey=True)
axes = plot.plotMedianPaper(fig,axes,'KaW',bins,'','','spectral edges',r'[ms$^{-1}$]',(-2,0.3),(-2,-20.5),'b)',ylim=(-9.9,-20.1),legend=True,ylabel=True,zoom=True)#(-30,-32)
plt.tight_layout()
plt.savefig('plot_medians_paper_sEdges_zoom_only3_MDV.png',bbox_inches='tight',dpi=300)
plt.close()
