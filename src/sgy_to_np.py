import numpy as np
import matplotlib.pyplot as plt
from obspy.io.segy.segy import _read_segy
import bruges
import csv

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
full_data = data.reshape(676,676,201)
#training_data = full_data[:,0:350 :,:]


#np.savetxt("vel.csv", full_data, delimiter=",", fmt='%d')
# for i in range(676):    
#     np.savetxt(f"../data/vel_{i:03}.csv", full_data[i,:,:], delimiter = ",", fmt='%d')


def get_mask(point_velocity):
    if point_velocity>4400:
        return 1
    else:
        return 0

vfunc = np.vectorize(get_mask)
#mask_data = vfunc(training_chunck.reshape(a*b*c,)).reshape(a,b,c)
mask_data = vfunc(full_data.reshape(676*676*201,))
for i in range(676):    
    np.savetxt(f"../data/msk_{i:03}.csv", mask_data.reshape(676,676,201)[i,:,:], delimiter = ",", fmt='%d')

#plt.imshow(mask_data[0,:].T, cmap="viridis", vmin=0, vmax=1, aspect='auto')
#plt.show()


def get_density(point_velocity):
    val = 0.31*point_velocity**0.25
    return val


def get_impedance(point_velocity):
     return get_density(point_velocity)*point_velocity

#vfunc = np.vectorize(get_impedance)
#impedance_data = vfunc(training_chunck.reshape(a*b*c,)).reshape(a,b,c)

def get_reflectivity(trace_velocity):
    nsamples = len(trace_velocity)
    #print(nsamples)
    vfunct_imp = np.vectorize(get_impedance)
    imp = vfunct_imp(trace_velocity.reshape(nsamples,))
    rc = (imp[1:] - imp[:-1])/(imp[1:] + imp[:-1])
    rc = np.append(rc,rc[-1])
    #print(len(rc))
    return rc


# rc =  get_reflectivity(new_data[0,300,:])
# plt.figure(figsize=(16,2))
# plt.plot(rc)
# plt.show()
# print(len(rc))


rc = np.apply_along_axis(get_reflectivity, 1, full_data.reshape(676*676,201))
# vm = np.percentile(rc, 99)
# plt.imshow(rc[0,:,:].T, cmap="gray", vmin=-vm, vmax=vm, aspect='auto')
# plt.show()


w= bruges.filters.ricker(duration = 0.100, dt=0.001, f=40)

rc_f = np.apply_along_axis(lambda t:np.convolve(t, w, mode='same'),
                            axis=1,
                            arr=rc)


# for i in range(676):    
#     np.savetxt(f"../data/img_{i:03}.csv", rc_f.reshape(676,676,201)[i,:,:], delimiter = ",", fmt='%d')



vm = np.percentile(rc_f, 99)
# plt.imshow(rc_f[0,:,:].T, cmap='gray', aspect='auto')
# plt.show()

#np.savetxt('image.csv', rc_f, delimiter=',', fmt='%.2f')

# training_data = full_data[:,:,:]
# testing_data = full_data[:,350:,:]

fg, ax = plt.subplots()
ax.imshow(rc_f.reshape(676,676,201)[300, :,:].T.squeeze(), cmap = "gray", vmin=-vm, vmax=vm,aspect='auto')
plt.show()





# w= bruges.filters.ricker(duration = 0.100, dt=0.001, f=40)
# for i in range(676):
#     outfile = open("images.csv", "r+")
#     training_chunck = full_data[i,:,:]
#     print("processing line  {} to {}".format(i,i))
#     a,b,c, = training_chunck.shape
#     rc = np.apply_along_axis(get_reflectivity, 1, training_chunck.reshape(a*b,-1)).reshape(a,b,-1)
#     rc_f = np.apply_along_axis(lambda t:np.convolve(t, w, mode='same'),
#                             axis=0,
#                             arr=rc)
#     np.savetxt(outfile, [rc_f.reshape(a*b*c,)], delimiter=',', fmt='%f')
#     outfile.close()




