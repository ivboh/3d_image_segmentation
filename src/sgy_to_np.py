import numpy as np
import matplotlib.pyplot as plt
from obspy.io.segy.segy import _read_segy
import bruges
import csv
from scipy import misc


def get_mask(point_velocity):
    """Function to return a binary mask for salt rock"""
    if point_velocity>4400:
        return 1
    else:
        return 0

def get_density(point_velocity):
    """Function to compute density from input accoustic velocity"""
    """ input : int or float
        output: float"""
    val = 0.31*point_velocity**0.25
    return val

def get_impedance(point_velocity):
    """Function to compute impedance from input accoustic velocity"""
    """input: int or float
       output: float"""
     return get_density(point_velocity)*point_velocity



def get_reflectivity(trace_velocity):
    """Function to generate reflectivity for a wavelet with input accoustic velocity"""
    """ input: Nx1 array 
        output: Nx1 array"""
    nsamples = len(trace_velocity)
    vfunct_imp = np.vectorize(get_impedance)
    imp = vfunct_imp(trace_velocity.reshape(nsamples,))
    rc = (imp[1:] - imp[:-1])/(imp[1:] + imp[:-1])
    rc = np.append(rc,rc[-1])
    return rc


xl = 676
il = 676
n_samples = 201

stream = _read_segy('../data/SEG_C3NA_Velocity.sgy', headonly=True)
full_data = np.stack(t.data for t in stream.traces).reshape(xl*il*n_samples,)

rc = np.apply_along_axis(get_reflectivity, 1, full_data.reshape(676*676,201))
w= bruges.filters.ricker(duration = 0.100, dt=0.001, f=120)
rc_f = np.apply_along_axis(lambda t:np.convolve(t, w, mode='same'),
                            axis=1,
                            arr=rc)

for i in range(676):    
    np.savetxt(f"../data/img_{i:03}.csv", rc_f.reshape(676,676,201)[i,:,:], delimiter = ",", fmt='%.2f')





