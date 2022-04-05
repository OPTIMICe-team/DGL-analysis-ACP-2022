import xarray as xr
import numpy as np
from glob import glob
from scipy.interpolate import interp1d


#####################################################################################
# ARM database path (adjust if needed) ##############################################
#####################################################################################
ARMpath = '/data/optimice/scattering_databases/ARM_aydin_GMM/arm-iop/0pi-data/aydin/'

# Some useful path-resolving dictionaries
types = ['aggregates', 'dendrites', 'columns', 'graupel', 'plates']
typeFolders = ['Aggregates/', 'BranchedPlanarCrystals/', 'Columns/',  'ConicalGraupel/', 'Plates/']
typeFldDict = dict(zip(types, typeFolders))
geomNames = ['aggregate', 'branchedplanar', 'column', 'graupel', 'plate']
reducedNames = [i+('' if i == 'graupel' else 's') for i in geomNames] 
geomNamesDict = dict(zip(types, geomNames))
reducedNamesDict = dict(zip(types, reducedNames))

bands = ['W', 'Ka', 'Ku', 'X']
wl_mm = [3.19 ,  8.4  , 22.4  , 31.859]
wlDict = dict(zip(bands, wl_mm))

# Aggregates subtypes
aggTypLabels = ['HD-P1d', 'HD-N1e', 'LD-P1d', 'LD-N1e', 'LDt-P1d']
aggTypDict = {l:i+1 for i,l in enumerate(aggTypLabels)} # translate label to code for data extraction

# Comment vars to keep
drop = [#'thickness_ratio',
        #'mass',
        #'maximum_dimension',
        'thickness',
        'projected_maximum_dimension',
        'projected_thickness',
        #'projected_area',
        'single_scattering_albedo',
        'asymmetry_parameter',
        #'sigma_backward_hh',
        #'sigma_backward_vv',
        #'sigma_backward_hv',
        'sigma_ext',
        'sigma_ext_h',
        'sigma_ext_v',
        'sigma_abs',
        'sigma_abs_h',
        'sigma_abs_v',
        'sigma_sca',
        'sigma_sca_h',
        'sigma_sca_v',
        'Shh_backward_real',
        'Shh_backward_imag',
        'Shv_backward_real',
        'Shv_backward_imag',
        'Svh_backward_real',
        'Svh_backward_imag',
        'Svv_backward_real',
        'Svv_backward_imag',
        #'Shh_forward_real',
        #'Shh_forward_imag',
        #'Shv_forward_real',
        #'Shv_forward_imag',
        #'Svh_forward_real',
        #'Svh_forward_imag',
        #'Svv_forward_real',
        #'Svv_forward_imag',
        #'particle_index',
        #'wavelength',
        #'incident_polar_angle'
       ]

#####################################################################################
# I/O functions for ARM database ####################################################
#####################################################################################

def _selData(data, elevation=None, band=None):
    """Data reduction through elevation and wavelength selection and azimuthal averaging
    This is a private function of the libray to reduce code repetition.
    data is optionally reduced by selecting elevation angle and wavelength
    azimuthal averaging is always performed, hence only horizonatally aligned snowflakes
    are considered.
    
    Parameters
    ----------
    data : xarray.Dataset
        the extracted scattering dataset
    elevation : float
        the radar elevation in degrees
    band : str
        the radar frequency band. It is internally translated to wavelength in millimenters
    
    Returns
    -------
    reduced : xarray.Dataset
        the reduced dataset with squeezed dimensions
    """
    reduced = data.mean(dim='incident_azimuth_angle')
    if elevation is not None:
        if elevation not in 90-data.incident_polar_angle:
            raise AttributeError(f'Unavailable elevation angle, use one of {90-data.incident_polar_angle.values}')
        reduced = reduced.sel(incident_polar_angle=90-elevation)
    if band is not None:
        if band not in bands:
            raise AttributeError(f'Do not know {band} band, use one of {bands}')
        reduced = reduced.sel(wavelength=wlDict[band])
    return reduced.squeeze()


def openScatteringRed(snowtype, scatt='GMM',
                      elevation=None, band=None,
                      subtype=None, thick_ratio=None):
    """ Function that loads the content of a Reduced-Size .nc file in the ARM database
    This function loads only one snow type dataset in the form of a xarray.Dataset
    Multiple Datasets can be stacked together afterwards along a new dimension "snowtype",
    but beware that some shapes have symmetries or might be more complex, hence their range
    of incident angles might vary.
    The dimension incident_azimuth_angle is dropped through averaging.
    Performs also data selection of wavelength and elevation angle through optional parameters
    
    Parameters
    ----------
    snowtype : str
        The string identifying the snowflake type. The function checks for a valid string
        Available strings are defined in the list "types"
    scatt : str default 'GMM'
        For most shapes both DDA and GMM are available. For aggregates only GMM. If DDA is
        requested for aggregates the function prints a warning and returns the GMM data
    elevation : float
        Radar elevation angle in degrees
    band : str
        the radar frequency band. It is internally translated to wavelength in millimenters
    subtype : str
        Valid only for aggregates. String that specifies the type of aggregate
        Could be one of aggTypeLabels. If left None the entire data type is returned
    thick_ratio : float
        Valid only for plates, columns and dendrites. Specifies the subset of thickness ratio.
        The valid values depend on the particle type. Plates(0.5, 1.0, 2.0), Dendrites(0.5, 1.0)
        and columns(1.0, 2.0). If left None the entire data type is returned
    
    Returns
    -------
    xarray.Dataset
        dataset of most important scattering properties for radar applications. Cross sections,
        scattering matrices and amplitude materices are given for the forward and backward
        scattering direction at multiple elevation angles and azimuths.
    
    """
    if snowtype not in types:
        raise AttributeError(f'Unrecognized snow type {snowtype}, use one of {types}')
    pathname = f'{ARMpath}{typeFldDict[snowtype]}ReducedSizeFiles/psuaydinetal_{reducedNamesDict[snowtype]}*.nc'
    files = glob(pathname)
    filename = None
    for f in files:
        if scatt in f:
            filename = f
    if filename is None:
        filename = files[0]
        print(f'Requested scattering not found using {filename}')
        
    # Selecting subtype of aggregates
    if subtype is not None:
        if snowtype != 'aggregates':
            print('WARNING I cannot apply subtype selection to a non-aggregate type, returning full data')
        else:
            if thick_ratio is not None:
                print('WARNING I cannot apply thick_ratio selection to a non-crystal type (plates, dendrites, columns) ignoring this argument')
            if subtype not in aggTypDict.keys():
                print(f'WARNING {subtype} is not in the list of available subtypes {aggTypDict.keys()}. Returning full data')
            else:
                data = xr.open_dataset(filename, drop_variables=drop)
                return _selData(data.where(data.type==aggTypDict[subtype], drop=True), elevation, band)
    
    # Selecting thickness ratio of crystals
    if thick_ratio is not None:
        if snowtype not in ['plates', 'columns', 'dendrites']:
            print('WARNING I cannot apply thick_ratio selection to a non-crystal type (plates, dendrites, columns), returning full data')
        else:
            data = xr.open_dataset(filename, drop_variables=drop)
            if thick_ratio not in np.unique(data.thickness_ratio):
                print(f'WARNING the {snowtype} does not have the requested {thick_ratio}, returning full data, consider using {np.unique(data.thickness_ratio)}')
                return _selData(data, elevation, band)
            else:
                return _selData(data.where(data.thickness_ratio==thick_ratio, drop=True), elevation, band)
    return _selData(xr.open_dataset(filename, drop_variables=drop), elevation, band)


def openShapefiles(snowtype, scatt='GMM', subtype=None, thick_ratio=None):
    """ Function that loads the content of all the shapefiles.nc available for a specific snowtype
    This function loads every snow shapefile in the form of a xarray.Dataset and stacks them into
    a unique dataset along a new "shape identifier" dimension
    Multiple Datasets can be further stacked together along a new dimension "snowtype".
    
    Parameters
    ----------
    snowtype : str
        The string identifying the snowflake type. The function checks for a valid string
        Available strings are defined in the list "types"
    scatt : str default 'GMM'
        For most shapes both DDA and GMM are available. For aggregates only GMM. If DDA is
        requested for aggregates the function prints a warning and returns the GMM data.
        The shapes used for GMM and DDA are necesseraly slightly different.
    subtype : str
        Valid only for aggregates. String that specifies the type of aggregate
        Could be one of aggTypeLabels. If left None the entire data type is returned
    thick_ratio : float
        Valid only for plates, columns and dendrites. Specifies the subset of thickness ratio.
        The valid values depend on the particle type. Plates(0.5, 1.0, 2.0), Dendrites(0.5, 1.0)
        and columns(1.0, 2.0). If left None the entire data type is returned
    
    Returns
    -------
    xarray.Dataset
        combined dataset of shapefiles used to compute scattering properties with GMM and DDA
        
    
    """
    if snowtype not in types:
        raise AttributeError('Unrecognized snow type ', snowtype)
    pathname = f'{ARMpath}{typeFldDict[snowtype]}GeometryFiles/psuaydinetal_geometry_{geomNamesDict[snowtype]}*.nc'
    filesglob = glob(pathname)
    if not filesglob:
        raise OSError('Files not found, check paths')
    files = sorted([f for f in filesglob if scatt in f.split('/')[-1]])
    if scatt not in ['DDA', 'GMM']:
        raise AttributeError(f'Unknown scatt attribute {scatt}, use DDA or GMM')
    if not files:
        print(f'Requested scattering {scatt} not found, trying using what I got')
        files = sorted(filesglob)
    
    # Subtype selection for aggregates
    if subtype is not None:
        if snowtype != 'aggregates':
            print('WARNING I cannot apply subtype selection to a non-aggregate type, returning full data')
        else:
            if thick_ratio is not None:
                print('WARNING I cannot apply thick_ratio selection to a non-crystal type (plates, dendrites, columns) ignoring this argument')
            if subtype not in aggTypDict.keys():
                print(f'WARNING {subtype} is not in the list of available subtypes {aggTypDict.keys()}. Returning full data')
            else:
                data = xr.open_mfdataset(files, combine='by_coords')
                return data.where(data.type==aggTypDict[subtype], drop=True)
    
    # Now thick_ratio selection for crystals
    if thick_ratio is not None:
        if snowtype not in ['plates', 'columns', 'dendrites']:
            print('WARNING I cannot apply thick_ratio selection to a non-crystal type (plates, dendrites, columns), returning full data')
        else:
            data = xr.open_mfdataset(files, combine='by_coords')
            if thick_ratio not in np.unique(data.thickness_ratio):
                print(f'WARNING the {snowtype} does not have the requested {thick_ratio}, returning full data, consider using {np.unique(data.thickness_ratio)}')
                return data
            else:
                return data.where(data.thickness_ratio==thick_ratio, drop=True)
    return xr.open_mfdataset(files, combine='by_coords')



#####################################################################################
# Fitting functions for the database ################################################
#####################################################################################

def _get_Dmax_bins(ds, th=0.025):
    """ Bin edges according to Dmax
    This function compute the bin edges according to the variable maximum_dimension
    Binnin is performed with simple clustering.
    This algorithm is based on the assumption that particles ar sort of distributed
    in log space of maximum_dimension
    
    Parameters
    ----------
    ds : xarray.Dataset
        an xarray dataset containing the variable maximum_dimension
        (could also be a coordinate)
    th : float
        the spacing in logaritmic scale of the maximum dimension to be considered for
        the clustering
    
    Returns
    -------
    aggBins : array-like n+1
        given n unique Dmax it computes the n+1 bin edges
    """
    a = np.sort(np.log10(ds.maximum_dimension))
    b = a[1:] - a[:-1]
    c = [0.5*(a[i+1]+a[i]) for i, ib in enumerate(b) if ib > th]
    aggBins = 10.0**np.array([2.0*c[0]-c[1]] + c + [2*c[-1]-c[-2]])
    
    return aggBins


def _get_mean_grouped_ds(ds, **kwargs):
    """ get averaged scattering properties for particles with grouped dimensions
    Parameters
    ----------
    ds : xarray.Dataset
        an xarray dataset containing the scattering properties and
        a variable maximum_dimension
    kwargs : optional keyword arguments
        th : log10 threshold for 1D clustering of maximum dimension
            see _get_Dmax_bins(ds, th)
            
    Returns
    -------
    dsAVG : xarray.Dataset
        same dataset but reduced by groupby.mean() over the binned dimension
    """
    bin_edges = _get_Dmax_bins(ds, **kwargs)
    grp = ds.groupby_bins('maximum_dimension', bin_edges)
    dsAVG = grp.mean()
    
    return dsAVG


_interp_var_list = [ 'mass',
                     'projected_area',
                     'sigma_backward_hh',
                     'sigma_backward_vv',
                     'sigma_backward_hv',
                     'Shh_forward_real',
                     'Shh_forward_imag',
#                     'Shv_forward_real', # cross-pol S elements need special treatment
#                     'Shv_forward_imag', # they do not combine well with log-log fit
#                     'Svh_forward_real', # since they can be negative
#                     'Svh_forward_imag',
                     'Svv_forward_real',
                     'Svv_forward_imag'
                   ]


class InterpScattering(object):
    """ Callable object that holds a dictionary of interpolating functions
    
    Generate array of interpolating functions
    using the log-log space and interp1d with extrapolation    
    """
    
    
    def __init__(self, ds, var_list=None):
        """ Initialize with ds and a list of vars
        Maybe I can put this in a separate _add_var function and call iteratively that
        
        Sets the self.interp_dict attribute, which is a dictionary of interpolating functions
        for each of the needed scattering variables
        
        Parameters
        ----------
        ds : xarray.Dataset
            a Dataset of scattering properties of particles that can be identified by
            their maximum dimension. Better if it is the result of _get_mean_grouped_ds
            to avoid unwanted internal noise
        var_list : array-like(str)
            list of string identifying the variables to be interpolated
            default is None which corresponds to any scattering and geometric variable
            see _interp_var_list for details
        """
        
        dsAVG = _get_mean_grouped_ds(ds)
        if var_list is None:
            var_list = _interp_var_list
        self.interp_dict = {variable:interp1d(np.log10(dsAVG.maximum_dimension), 
                                              np.log10(dsAVG[variable]), 
                                              fill_value='extrapolate') for variable in var_list}
            
            
    def __call__(self, variable, D):
        """ Call the appropriate fitting function and perform transformations
        
        Parameters
        ----------
            variable : str
                name of the variable for which an interpolating function has been derived
            D : array-like(float)
                works also with scalars, defines the sizes D for which the interpolating function
                is to be computed
            
        Returns
        -------
            array-like(float)
                or scalar if D is scalar. Values of the interpolating functions at D points
                Conversion back to linear space is also performed since interp_dict holds the log-log
                interpolation
        """
        
        return 10**self.interp_dict[variable](np.log10(D))