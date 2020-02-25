import numpy as np
import matplotlib.pyplot as plt
from obspy.io.segy.segy import _read_segy



stream = _read_segy('../data/SEG_C3NA_Velocity.sgy', headonly=True)
# one_trace = stream.traces[0]

# print(one_trace.data.shape)
#plt.figure(figsize=(16,2))
#plt.plot(one_trace.data)
#plt.show()

data = np.stack(t.data for t in stream.traces)
#print(data.shape)

#vm = np.percentile(data, 99)
#print("The 99th percentile is {:.0f}; the max amplitude is {:.0f}".format(vm, data.max()))


# plt.imshow(data[200000:203000,:].T, cmap="viridis", vmin=-vm, vmax=vm, aspect='auto')
# plt.show()

#print(type(data))



shape_3d = (676, 676, 201)
new_data = data.reshape(676,676,201)

new_data = new_data[300:330,:,:]

a,b,c, = new_data.shape

def get_mask(point_velocity):
    if point_velocity>4400:
        return 1
    else:
        return 0

vfunc = np.vectorize(get_mask)
mask_data = vfunc(new_data.reshape(a*b*c,)).reshape(a,b,c)

print(new_data.shape)
plt.imshow(mask_data[0,:].T, cmap="viridis", vmin=0, vmax=1, aspect='auto')
#plt.show()


def get_density(point_velocity):
    val = 0.31*point_velocity**0.25
    return val


def get_impedance(point_velocity):
     return get_density(point_velocity)*point_velocity

vfunc = np.vectorize(get_impedance)
impedance_data = vfunc(new_data.reshape(a*b*c,)).reshape(a,b,c)

def get_reflectivity(trace_velocity):
    nsamples = len(trace_velocity)
    #print(nsamples)
    vfunct_imp = np.vectorize(get_impedance)
    imp = vfunct_imp(trace_velocity.reshape(nsamples,))
    rc = (imp[1:] - imp[:-1])/(imp[1:] + imp[:-1])
    return rc



# rc =  get_reflectivity(new_data[0,300,:])
# plt.figure(figsize=(16,2))
# plt.plot(rc)
# plt.show()
# print(len(rc))


rc = np.apply_along_axis(get_reflectivity, 1, new_data.reshape(a*b,-1)).reshape(a,b,-1)
vm = np.percentile(rc, 99)
plt.imshow(rc[0,:,:].T, cmap="gray", vmin=-vm, vmax=vm, aspect='auto')
plt.show()




