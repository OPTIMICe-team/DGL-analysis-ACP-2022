import numpy as np
from snowScatt import refractiveIndex as ref
from readARMdb import wlDict

def ZeARM(psd, scattFunc, band, pol='hh'):
    """ Radar reflectivity with polarization
    
    Parameters
    ----------
    psd : a PSDqN object. PSDGamma3
        a callable PSDGamma3 object or equivalent.
        It must have the integrate method implemented
    scattFunc : InterpScattering object from readARMdb
        a callable interpolating scattering set of function.
        It must have the relevant sigma_backscattering_{pol} interpolated
    band : str
        one of the four ['W', 'Ka', 'Ku', 'X'] bands available in readARMdb
    pol : str
        polarization configuration. 
        Options:
            'hh' - classical co-polar horizontally polarized
            'vv' - co-polar vertically polarized
            'hv' - cross-polar horizontal incident vertical return
            'vh' - cross-polar vertical incident horizontal return
    
    Returns
    -------
    Ze - float
        radar reflectivity with polarization mm**6/m*3
    spec ----- ???? 
    """
    wl = wlDict[band]
    K2 = ref.utilities.K2(ref.water.eps(273.15, 299792458000./wl))
    Ze = wl**4 * psd.integrate(lambda x : scattFunc('sigma_backward_'+pol, x*1.0e3) / (np.pi**5 * K2))
    return Ze

#def Ze(Z, wl, K2, pol=None):
    """ Radar reflectivity with polarization mm**6/m**3
    This is intended to be a private local function called by Ze
    Parameters
    ----------
    Z : 2D array-like minimum dimensions 2x2
        upper left portion of the 4x4 real scattering matrix [mm**2]
    wl : float
        wavelength [mm]
    K2 : dielectric factor
        if computed with the Clausius-Mossotti formula it would be |K|**2
        where K=(m**2-1)/(n**2+2)
        where n is the (wavelength dependent) refractive index of water
    pol : str or None
        polarization configuration. 
        If None only unpolarized radiation is considered (both incident and scattered)
        Options:
            'hh' - classical co-polar horizontally polarized
            'vv' - co-polar vertically polarized
            'hv' - cross-polar horizontal incident vertical return
            'vh' - cross-polar vertical incident horizontal return
            None - unpolarized radiation
    
    Returns
    -------
    rcs - float
        radar backscattering cross-section with the requested polarization
    """
#    return wl**4/(np.pi**5*K2) * _radar_xsect(Z, pol)

def kdpARM(psd, scattFunc, band):
    """ Radar specific differential phase shift [°/km]
    
    Parameters
    ----------
    psd : a PSDqN object. PSDGamma3
        a callable PSDGamma3 object or equivalent.
        It must have the integrate method implemented
    scattFunc : InterpScattering object from readARMdb
        a callable interpolating scattering set of function.
        It must have the relevant sigma_backscattering_{pol} interpolated
    band : str
        one of the four ['W', 'Ka', 'Ku', 'X'] bands available in readARMdb
    
    Returns
    -------
    kdp - float
        radar specific differential phase shift [°/km]
    spec ----- ???? 
    """
    wl = wlDict[band]
    kdp = psd.integrate(lambda x : scattFunc('Shh_forward_real', x*1.0e3)-scattFunc('Svv_forward_real', x*1.0e3))
    return kdp * 1e-3 * (180.0/np.pi) * wl


def rhohv():
    raise NotImplementedError


def Ai(S1i, S2i, wl, pol):
    raise NotImplementedError


#def _radar_xsect(Z, pol=None):
    """ Radar backscattering cross section with polarization
    This is intended to be a private local function called by Ze
    Parameters
    ----------
    Z : 2D array-like minimum dimensions 2x2
        upper left portion of the 4x4 real scattering matrix [mm**2]
    pol : str or None
        polarization configuration. If None only unpolarized radiation is considered (both incident and scattered)
        Options:
            'hh' - classical co-polar horizontally polarized
            'vv' - co-polar vertically polarized
            'hv' - cross-polar horizontal incident vertical return
            'vh' - cross-polar vertical incident horizontal return
            None - unpolarized radiation
    
    Returns
    -------
    rcs - float
        radar backscattering cross-section with the requested polarization
    """
    
#    if pol == 'hh':
#        return 2*np.pi*(Z[0,0]-Z[0,1]-Z[1,0]+Z[1,1])
#    elif pol == 'vv':
#        return 2*np.pi*(Z[0,0]+Z[0,1]+Z[1,0]+Z[1,1])
#    elif pol == 'hv':
#        return 2*np.pi*(Z[0,0]-Z[0,1]+Z[1,0]-Z[1,1])
#    elif pol == 'vh':
#        return 2*np.pi*(Z[0,0]+Z[0,1]-Z[1,0]-Z[1,1])
#    elif pol is None:
#        return 2*np.pi*Z[0,0]
#    else:
#        raise AttributeError(f'I cannot interprete the polarization {pol}')