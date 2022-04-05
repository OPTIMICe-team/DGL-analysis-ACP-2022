import numpy as np
import matplotlib.pyplot as plt

from snowScatt import refractiveIndex as ref

from readARMdb import openScatteringRed
from readARMdb import wlDict

from readARMdb import InterpScattering

def expPSD(N0,lam,D):
  return N0*np.exp(-lam*D)
def dB(x):
  return 10.0*np.log10(x)


#####################################################################################
# Load scattering tables         ####################################################
#####################################################################################

subtype_agg = 'HD-P1d'
thick_ratio_den = 1.0
slanted_elev = 30

# 3-freq vertically pointing
aggX = openScatteringRed('aggregates', subtype=subtype_agg, band='X', elevation=90)
aggKa = openScatteringRed('aggregates', subtype=subtype_agg, band='Ka', elevation=90)
aggW = openScatteringRed('aggregates', subtype=subtype_agg, band='W', elevation=90)

denX = openScatteringRed('dendrites', thick_ratio=thick_ratio_den, band='X', elevation=90)
denKa = openScatteringRed('dendrites', thick_ratio=thick_ratio_den, band='Ka', elevation=90)
denW = openScatteringRed('dendrites', thick_ratio=thick_ratio_den, band='W', elevation=90)
#plates = openScatteringRed('plates', thick_ratio=1.0)

# polarimetric W-band 30deg elevation
agg_pol = openScatteringRed('aggregates', subtype='HD-P1d', band='W', elevation=slanted_elev)
den_pol = openScatteringRed('dendrites', thick_ratio=1.0, band='W', elevation=slanted_elev)
#columns = openScatteringRed('columns', thick_ratio=1.0)
#graupel = openScatteringRed('graupel')


#####################################################################################
# Derive interpolating functions ####################################################
#####################################################################################

aggXFunc = InterpScattering(aggX)
aggKaFunc = InterpScattering(aggKa)
aggWFunc = InterpScattering(aggW)

denXFunc = InterpScattering(denX)
denKaFunc = InterpScattering(denKa)
denWFunc = InterpScattering(denW)

aggPFunc = InterpScattering(agg_pol)
denPFunc = InterpScattering(den_pol)

#print(lambda x:1.0e-6*aggPFunc('mass', x*1.0e3))
#quit()


D = np.logspace(-6, np.log10(0.1), 1000)
#convFac = 3.67
lam = [1/(2.225e-3)]#[1/(4.1e-3)] #[1/(2.7e-3), 1/(4.1e-3), 1/(5e-3)]#[500.,  1000., 1500., 2000., 3000., 4000., 5000.,  6000., 1/(2e-3)] #600., 700., 800., 900.,
N0 = 5.6e4#1.2e4 #4e4
#-- now compute the scattering: at zenith
#vRadar = np.vectorize(radarSnow2parts)
bands = ['Ka','W']
Ze = np.empty((len(bands),len(lam)))
wl =  wlDict['Ka']
K2 = ref.utilities.K2(ref.water.eps(273.15, 299792458000./wl))
preFac = wl**4/(np.pi**5*K2)
for j,l in enumerate(lam):
  Ze[0,j] = dB(preFac * (expPSD(N0,l,D)*np.gradient(D)*aggKaFunc('sigma_backward_hh', D*1e3)).sum())

wl =  wlDict['W']
K2 = ref.utilities.K2(ref.water.eps(273.15, 299792458000./wl))
preFac = wl**4/(np.pi**5*K2)
for j,l in enumerate(lam):
  Ze[1,j] = dB(preFac * (expPSD(N0,l,D)*np.gradient(D)*aggWFunc('sigma_backward_hh', D*1e3)).sum())  

#print(Ze[0]-Ze[1])
#print(Ze[0])
#quit()
# now for that PSD setting at 30° for pol observations and for aggregates
kdpFac = 1e-3*(180.0/np.pi)*wl
kdp = kdpFac*(expPSD(N0,l,D)*np.gradient(D)*(aggPFunc('Shh_forward_real', D*1e3) - aggPFunc('Svv_forward_real',D*1e3))).sum()
ZeHH = dB(preFac * (expPSD(N0,l,D)*np.gradient(D)*aggPFunc('sigma_backward_hh', D*1e3)).sum())  
ZeVV = dB(preFac * (expPSD(N0,l,D)*np.gradient(D)*aggPFunc('sigma_backward_vv', D*1e3)).sum())  

#print(kdp)
#quit()

# kdp dendrites at different D and conc:
D = 1e-3; conc = [2000.,2500.,3000., 4000., 4150.,]
kdp1 = np.empty(len(conc))
for i,c in enumerate(conc):
  kdp1[i] = c*kdpFac*(aggPFunc('Shh_forward_real', D*1e3) - aggPFunc('Svv_forward_real',D*1e3))

D = 2e-3; conc = [500., 600., 700., 1000., 1030.]
kdp2 = np.empty(len(conc))
for i,c in enumerate(conc):
  kdp2[i] = c*kdpFac*(aggPFunc('Shh_forward_real', D*1e3) - aggPFunc('Svv_forward_real',D*1e3))


D = 3e-3; conc = [300., 400.,  485., 500.]
kdp3 = np.empty(len(conc))
for i,c in enumerate(conc):
  kdp3[i] = c*kdpFac*(aggPFunc('Shh_forward_real', D*1e3) - aggPFunc('Svv_forward_real',D*1e3))


D = 4e-3; conc = [170.,180., 200., 300., 305.]
kdp4 = np.empty(len(conc))
for i,c in enumerate(conc):
  kdp4[i] = c*kdpFac*(aggPFunc('Shh_forward_real', D*1e3) - aggPFunc('Svv_forward_real',D*1e3))


D = 5e-3; conc = [120., 150., 200.]
kdp5 = np.empty(len(conc))
for i,c in enumerate(conc):
  kdp5[i] = c*kdpFac*(aggPFunc('Shh_forward_real', D*1e3) - aggPFunc('Svv_forward_real',D*1e3))


print('Ze',Ze[0])
print('DWR',Ze[0,:]-Ze[1,:])
print('ZDR',ZeHH - ZeVV)
print('KDP',kdp)
#print('KDP 1mm',kdp1)
print('KDP agg + 1mm', kdp1+kdp)
#print('KDP 2mm',kdp2)
print('KDP agg + 2mm', kdp2+kdp)
#print('KDP 3mm',kdp3)
print('KDP agg + 3mm', kdp3+kdp)
#print('KDP 4mm',kdp4)
print('KDP agg + 4mm', kdp4+kdp)
#print('KDP 5mm',kdp5)
print('KDP agg + 5mm', kdp5+kdp)




'''
fig, (axm, axa, axv) = plt.subplots(1, 3, figsize=(16,3))
for p, particle in enumerate([den_pol, agg_pol]):
    axm.scatter(particle.maximum_dimension, particle.mass, label=p)
    axa.scatter(particle.maximum_dimension, particle.projected_area, label=p)
    axv.scatter(particle.maximum_dimension, KC05(particle.maximum_dimension*1.0e-3, 
                                                 particle.mass*1.0e-6, 
                                                 particle.projected_area*1.0e-6), label=p)
D = np.logspace(-6, np.log10(0.1), 1000)
axm.plot(D*1.0e3, 1.0e6*mD_den(D), label='dendrites')
axm.plot(D*1.0e3, 1.0e6*mD_agg(D), label='aggregates')
axm.plot(D*1.0e3, denPFunc('mass', D*1.0e3), label='spline den')
axm.plot(D*1.0e3, aggPFunc('mass', D*1.0e3), label='spline agg')
axa.plot(D*1.0e3, 1.0e6*aD_den(D))
axa.plot(D*1.0e3, 1.0e6*aD_agg(D))
axa.plot(D*1.0e3, denPFunc('projected_area', D*1.0e3), label='spline den')
axa.plot(D*1.0e3, aggPFunc('projected_area', D*1.0e3), label='spline agg')
axv.plot(D*1.0e3, vD_den(D))
axv.plot(D*1.0e3, vD_agg(D))
axv.plot(D*1.0e3, KC05(D, 1.0e-6*denPFunc('mass', D*1.0e3), 1.0e-6*denPFunc('projected_area', D*1.0e3)), label='spline den')
axv.plot(D*1.0e3, KC05(D, 1.0e-6*aggPFunc('mass', D*1.0e3), 1.0e-6*aggPFunc('projected_area', D*1.0e3)), label='spline agg')

axm.set_xscale('log')
axm.set_yscale('log')
axm.set_xlim(1.0e-1, 1e2)
axm.set_ylim(1.0e-3, 1e2)
axm.legend()
axa.set_xscale('log')
axa.set_yscale('log')
axa.set_xlim(1.0e-1, 1e2)
axa.set_ylim(1.0e-2, 1.0e2)
axv.set_xscale('log')
axv.set_yscale('log')
axv.set_xlim(1.0e-1, 1e2)
axv.set_ylim(2.0e-1, 3e0)
plt.show()

fig, (ax, axm) = plt.subplots(1, 2, figsize=(15,5))
ax.plot(D, PSDden(D)/np.max(PSDden(D)), label='dendrites')
ax.plot(D, PSDagg(D)/np.max(PSDagg(D)), label='aggregates')
ax.plot(D,expPSD(1000,1/(2e-3),D)/np.max(expPSD(1000,1/(2e-3),D)),label='expPSD')
ax.set_xscale('log')
ax.set_xlim(1.0e-7, 0.1)
ax.set_xlabel('Dmax')
axm.plot(D, np.cumsum(PSDden(D)*mD_den(D)))
axm.plot(D, np.cumsum(PSDagg(D)*mD_agg(D)))
#axm.plot(D,np.cumsum(expPSD(1000,1/(2e-3),D)*mD_agg(D)))
#axm.set_yscale('log')
axm.set_xscale('log')
axm.set_xlim(1.0e-4, 0.1)
ax.legend(loc=4)
plt.tight_layout()
plt.savefig('testPSD.png')
plt.show()
'''
'''
# First let's interpolate mD, aD relations to the aggregate particles selected in the database.
bm_agg, am_agg, mD_agg = _loglog_interp(1.0e-3*agg_pol.maximum_dimension, agg_pol.mass*1.0e-6)
# = lambda x: am_agg*x**bm_agg
ba_agg, aa_agg, aD_agg = _loglog_interp(1.0e-3*agg_pol.maximum_dimension, agg_pol.projected_area*1.0e-6)
#aD_agg = lambda x: aa_agg*x**ba_agg
# Then, create a wrapper of KC05 for aggregates that becomes a function of diameters only
vD_agg = lambda x: KC05(x, mD_agg(x), aD_agg(x))

# Repeate the same exercise for the dendrites
bm_den, am_den, mD_den = _loglog_interp(1.0e-3*den_pol.maximum_dimension, den_pol.mass*1.0e-6)
#mD_den = lambda x: am_den*x**bm_den
ba_den, aa_den, aD_den = _loglog_interp(1.0e-3*den_pol.maximum_dimension, den_pol.projected_area*1.0e-6)
#aD_den = lambda x: aa_den*x**ba_den
vD_den = lambda x: KC05(x, mD_den(x), aD_den(x))

# Now build PSDs for aggregates and dendrites for testing
N = 1.0e4
q = 1.0e-3
#PSDagg = PSDGamma3(q, N, mD_agg, mu=1.0, minD=0.3e-3)
PSDagg = PSDGamma3(q, N, lambda x:1.0e-6*aggPFunc('mass', x*1.0e3), mu=1.0, minD=0.3e-4)

N = 1.0e5
q = 1.0e-4
#PSDden = PSDGamma3(q, N, mD_agg, mu=1.5, minD=0.3e-4)
PSDden = PSDGamma3(q, N, lambda x:1.0e-6*denPFunc('mass', x*1.0e3), mu=1.5, minD=0.3e-4)
fig, ax = plt.subplots(figsize=(15,5))
#ax.plot(D, PSDden(D)/np.max(PSDden(D)), label='dendrites')
#ax.plot(D, PSDagg(D)/np.max(PSDagg(D)), label='aggregates')
for l in lam:
  ax.plot(D,expPSD(1000,l,D)/np.max(expPSD(1000,1/(2e-3),D)),label='expPSD,lam={0}'.format(l))
#ax.plot(D,expPSD(1000,1/(5e-3),D)/np.max(expPSD(1000,1/(5e-3),D)),label='expPSD,Dm=5mm')
#ax.plot(D,expPSD(1000,1/(10e-3),D)/np.max(expPSD(1000,1/(10e-3),D)),label='expPSD,Dm=10mm')
ax.set_xscale('log')
ax.set_xlim(1.0e-7, 0.1)
ax.set_xlabel('Dmax')

ax.legend(loc='lower left')
plt.tight_layout()
plt.savefig('testPSD.png')
#plt.show()
#quit()
'''
