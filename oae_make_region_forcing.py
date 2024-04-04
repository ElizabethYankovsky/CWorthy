import os
from glob import glob

import numpy as np
import xarray as xr
import pop_tools

import cesm_tools
import config


def forcing_template():
    
    '''Generate a OAE forcing template'''
    
    # Get POP grid
    global grid, time # keep grid and time global and unique
    
    grid = pop_tools.get_grid('POP_gx1v7')[['TAREA', 'KMT', 'TLAT', 'TLONG', 'REGION_MASK']]

    # Set up time axis for the forcing data set
    time_bnds = xr.cftime_range('1999-01-01', '2020-01-01', freq='1D', calendar='noleap')
    #originally: time_bnds = xr.cftime_range('1999-01-01', '2020-01-01', freq='1D', calendar='noleap')

    time_bnds = xr.DataArray(np.vstack((time_bnds[:-1], time_bnds[1:])).T, dims=('time', 'd2'))
    time = time_bnds.mean('d2')
    
    # Construct forcing dataset template.
    alk_forcing = xr.full_like(grid.TLAT.expand_dims({'time': time}), fill_value=0.0)
    alk_forcing.name = 'alk_forcing'
    alk_forcing.attrs['long_name'] = 'Alkalinity forcing'
    alk_forcing.attrs['units'] = 'mol/m^2/s'

    dso = xr.Dataset(dict(
        time=time,
        time_bnds=time_bnds,
        alk_forcing=alk_forcing,
        TLAT=grid.TLAT,
        TLONG=grid.TLONG,
        KMT=grid.KMT,
        TAREA=grid.TAREA,
    )).set_coords(['time', 'TLAT', 'TLONG']).rename({'nlat': 'Y', 'nlon': 'X'})

    dso['Y'] = xr.DataArray(np.arange(1, grid.sizes['nlat']+1), dims=('Y'))
    dso['X'] = xr.DataArray(np.arange(1, grid.sizes['nlon']+1), dims=('X'))
    dso.time.encoding['dtype'] = np.float64
    dso.time.encoding['_FillValue'] = None
    dso.time.encoding['units'] = 'days since 1999-01-01 00:00:00'
    dso.time_bnds.encoding['dtype'] = np.float64
    dso.time_bnds.encoding['_FillValue'] = None
    dso.time.attrs['bounds'] = 'time_bnds'
    
    return dso

def oae_loading_rate():
    
    '''Prescibe OAE loading rate'''
    
    # OAE loading rate
    alk_mass_loading_rate_molm2yr = 10.0 # mol/m^2/yr
    peryr_to_pers = (1.0 / 365.0 / 86400.0) 
    alk_mass_loading_rate = alk_mass_loading_rate_molm2yr * peryr_to_pers # mol/m^2/s
    
    return alk_mass_loading_rate

def oae_loading_time(months_to_perturb):
    
    '''
    Generate time function for OAE loading
    
    months_to_perturb: ['1999-01'] or ['1999-01', '1999-04'].
                        generate ONE forcing file for all months.
    '''
    
    time_function = xr.DataArray(
        np.array([
            f'{d.year:04d}-{d.month:02d}' in months_to_perturb for d in time.values]).astype(np.float64),
        dims=('time'),
        coords={'time': time},
    )

    return time_function

def get_region_mask(region_mask_name):
    
    '''generate region masks
    
    region_mask: 3D, (number of regions, nlat, nlon)'''
    
    if region_mask_name == 'cal':
        region_mask = np.zeros((1, grid.TLONG.shape[0], grid.TLONG.shape[1]))
        region_mask[0,:,:] = xr.where(
            (grid.TLONG>234) & (grid.TLONG<239) & 
            (grid.TLAT>35) & (grid.TLAT<40), 1.0, 0.0
        )
        region_mask = xr.DataArray(region_mask, dims=('region', 'nlat', 'nlon'))
        #region_mask['region'] = region_mask_name
        
    elif region_mask_name == 'lat-range-basin':
        region_mask = pop_tools.region_mask_3d('POP_gx1v7', region_mask_name)

        # only look at -160 - 66
        keep = np.ones(region_mask.sizes['region']).astype(bool)
        for i in range(region_mask.sizes["region"]):
            region_mask.data[i, :, :] = xr.where(
                (region_mask[i, :, :] == 1.0) & (-60 < grid.TLAT) & (grid.TLAT < 66), 1.0, 0.0
            )
            if not region_mask[i, :, :].any():
                keep[i] = False

        region_mask = region_mask[keep, :, :]
        
    elif region_mask_name == 'Atlantic_polygon53':
        
        # read polygon masks
        Atlantic_polygon_mask53 = np.load('/glade/work/mengyangz/GVP/Atlantic_polygon_mask54.npy')
        
        region_mask = np.zeros((1, grid.TLONG.shape[0], grid.TLONG.shape[1]))
        region_mask[0,:,:] = Atlantic_polygon_mask53
        region_mask = xr.DataArray(region_mask, dims=('region', 'nlat', 'nlon'))
        
        
    elif region_mask_name == 'North_Atlantic_basin':
        
        # read polygon masks
        region_mask = np.load('/glade/work/mengyangz/GVP/Atlantic_final_polygon_mask.npy')
        region_mask = xr.DataArray(region_mask, dims=('region', 'nlat', 'nlon'), coords={'region': np.arange(0,150)}) #region_mask[63], coords=...(63,64)
        region_mask.attrs['mask_name'] = 'North_Atlantic_basin'
        
    elif region_mask_name == 'North_Pacific_basin':
        
        # read polygon masks
        region_mask = np.load('/glade/work/mengyangz/GVP/Pacific_final_polygon_mask.npy')
        region_mask = xr.DataArray(region_mask, dims=('region', 'nlat', 'nlon'), coords={'region': np.arange(0,200)})
        region_mask.attrs['mask_name'] = 'North_Pacific_basin'
    
    elif region_mask_name == 'South':
        
        # read polygon masks
        region_mask = np.load('/glade/work/mengyangz/GVP/South_final_polygon_mask_120EEZ_180openocean.npy')
        region_mask = xr.DataArray(region_mask, dims=('region', 'nlat', 'nlon'), coords={'region': np.arange(0,300)})
        region_mask.attrs['mask_name'] = 'South'
    
    elif region_mask_name == 'Southern_Ocean':
        
        # read polygon masks
        region_mask = np.load('/glade/work/mengyangz/GVP/Southern_Ocean_final_polygon_mask.npy')
        region_mask = xr.DataArray(region_mask, dims=('region', 'nlat', 'nlon'), coords={'region': np.arange(0,40)})
        region_mask.attrs['mask_name'] = 'Southern_Ocean'


    return region_mask
    
def write_one_forcing(region_mask_name, months_to_perturb):
    
    '''
    Write out 1 forcing file for a time list, but many files for different regions.
    
    region_mask_name: 'lat-range-basin'
    months_to_perturb: a list of months: ['1999-01'], or ['1999-01', '1999-04'].
    '''

    # set up forcing template, loading rate and region mask
    dso = forcing_template()
    alk_mass_loading_rate = oae_loading_rate()
    region_mask = get_region_mask(region_mask_name)
    time_function = oae_loading_time(months_to_perturb)
        
    clobber = False
    forcing_files = []
    #for i in range(0, region_mask.shape[0]):
    for i in [22, 6]: # rewrite 060-1999-04

        # /glade/work is full. 1TB limit
        #default: file_out = f'{config.dir_scratch}/{config.project_name}/data/alk-forcing-{region_mask_name}.{i:03d}-{months_to_perturb}.nc'
        file_out = f'{config.dir_scratch}/{config.project_name}/data/alk-forcing-{region_mask_name}.{i:03d}-5year.nc' #change name based on length of forcing

        if os.path.exists(file_out) and not clobber:
            continue
        dso.alk_forcing.data[:] = 0.0
        alk_forcing = time_function * xr.where(region_mask[i, :, :], alk_mass_loading_rate, 0.0)   # broadcast thru time, where to perturb alk
        alk_forcing = alk_forcing.reset_coords('region', drop=True) # drop region
        dso.alk_forcing.data = alk_forcing

        cesm_tools.to_netcdf_clean(dso, file_out)
        forcing_files.append(file_out)
        
    print(forcing_files)

def write_multi_forcings(region_mask_name, months_to_perturb, seperate_months=True):
    
    '''
    Write out forcing files.
    
    reperate_months: if True, will generate seperate forcing files for each month
                     if False, will generate 1 forcing file for all the months
    '''
        
    if not seperate_months:
        write_one_forcing(region_mask_name, months_to_perturb)
    else:
        for month in months_to_perturb:
            print('-------Generating forcing files for: ', month)
            write_one_forcing(region_mask_name, month)
         
        
if __name__ == '__main__':
    
    #region_mask_name = 'lat-range-basin'
    #region_mask_name = 'cal'
    #region_mask_name = 'Atlantic_polygon53'
    region_mask_name = 'North_Pacific_basin'
    #region_mask_name = 'North_Pacific_basin'
    #region_mask_name = 'South'
    #region_mask_name = 'Southern_Ocean' #originally specified
    
#    months_to_perturb = ['2014-01']
    months_to_perturb = ['1999-01','1999-02','1999-03','1999-04','1999-05','1999-06','1999-07','1999-08','1999-09','1999-10','1999-11','1999-12',
                         '2000-01','2000-02','2000-03','2000-04','2000-05','2000-06','2000-07','2000-08','2000-09','2000-10','2000-11','2000-12',
                         '2001-01','2001-02','2001-03','2001-04','2001-05','2001-06','2001-07','2001-08','2001-09','2001-10','2001-11','2001-12',
                         '2002-01','2002-02','2002-03','2002-04','2002-05','2002-06','2002-07','2002-08','2002-09','2002-10','2002-11','2002-12',
                         '2003-01','2003-02','2003-03','2003-04','2003-05','2003-06','2003-07','2003-08','2003-09','2003-10','2003-11','2003-12']

    #months_to_perturb = ['1999-04','1999-07','1999-10'] 
    #months_to_perturb = ['1999-10','1999-07'] 
    
    write_multi_forcings(region_mask_name, months_to_perturb, seperate_months=False) #default True, generates separate forcing files for each month for seasonal runs
