# test python wrapper for fortran fmm 2d code

import numpy as np
import matplotlib.pyplot as plt
from fmm import fm2d, fm2d_ray


angle = np.arange(0., 2.*np.pi, np.pi/4)

src = np.array([[4*np.sin(x), 4*np.cos(x)] for x in angle])
rec = src

srcx = np.ascontiguousarray(src[:,0])
srcy = np.ascontiguousarray(src[:,1])
recx = np.ascontiguousarray(rec[:,0])
recy = np.ascontiguousarray(rec[:,1])

print(src)

mask = np.zeros((2,64),dtype=np.int32)
for i in range(8):
    for j in range(8):
        if(j>i):
             mask[0,i*8+j] = 1
             mask[1,i*8+j] = i*8 + j + 1
nx = 21
ny = 21
xmin = -5
ymin = -5
dx = 0.5
dy = 0.5
gdx = 2
gdy = 2
sdx = 4
sext = 4
sigma = 0.02

vel = np.ones(441) * 2
data, dtdv, ray = fm2d_ray(vel,srcx,srcy,recx,recy,mask,nx,ny,xmin,ymin,dx,dy,gdx,gdy,sdx,sext)
print(ray.shape)
print(mask.shape)

vel_new = vel.copy()
# vel_new[225] = 0.55
vel_new[204] = 1
# vel_new[0:9*21] = 0.55
data_new, dtdv_new, ray_new = fm2d_ray(vel_new,srcx,srcy,recx,recy,mask,nx,ny,xmin,ymin,dx,dy,gdx,gdy,sdx,sext)

ray_path = np.zeros(dtdv.shape)
for idata in mask[0,:].nonzero()[0]:
    ix = ((ray[idata,:,1] - xmin + dx/2) // dx).astype('int') # actually iy
    iy = ((ray[idata,:,0] - ymin + dy/2) // dy).astype('int') # actually ix
    ray_path[idata,iy * nx + ix] = 1
    ray_path[idata,iy * nx + ix -nx-1] = 1
    ray_path[idata,iy * nx + ix-nx] = 1
    ray_path[idata,iy * nx + ix-nx+1] = 1
    ray_path[idata,iy * nx + ix-1] = 1
    ray_path[idata,iy * nx + ix+1] = 1
    ray_path[idata,iy * nx + ix+nx-1] = 1
    ray_path[idata,iy * nx + ix+nx] = 1
    ray_path[idata,iy * nx + ix+nx+1] = 1

ray_path_new = np.zeros(dtdv.shape)
for idata in mask[0,:].nonzero()[0]:
    ix = ((ray_new[idata,:,1] - xmin + dx/2) // dx).astype('int')
    iy = ((ray_new[idata,:,0] - ymin + dy/2) // dy).astype('int')
    ray_path_new[idata,iy * nx + ix] = 1
    ray_path_new[idata,iy * nx + ix -nx-1] = 1
    ray_path_new[idata,iy * nx + ix-nx] = 1
    ray_path_new[idata,iy * nx + ix-nx+1] = 1
    ray_path_new[idata,iy * nx + ix-1] = 1
    ray_path_new[idata,iy * nx + ix+1] = 1
    ray_path_new[idata,iy * nx + ix+nx-1] = 1
    ray_path_new[idata,iy * nx + ix+nx] = 1
    ray_path_new[idata,iy * nx + ix+nx+1] = 1



idata = 3

print(ray_path[idata,:].nonzero())
print(ray_path[idata,:].reshape(ny, -1))

# print(ray_new[idata,:25,:])


print(np.all(data[idata] == data_new[idata]))
print(np.all(dtdv[idata] == dtdv_new[idata]))
print(np.all(ray[idata] == ray_new[idata]))
print(np.all(vel == vel_new))

print((data[idata]))
print((data_new[idata]))

index_dtdv = (dtdv[idata]).nonzero()
# print((dtdv[idata][index_dtdv]))
# print((dtdv_new[idata][index_dtdv]))
# print((dtdv[idata][index_dtdv] - dtdv_new[idata][index_dtdv]) / dtdv[idata][index_dtdv])
print(np.abs((dtdv[idata][index_dtdv] - dtdv_new[idata][index_dtdv]) / dtdv[idata][index_dtdv]).min())
print(np.abs((dtdv[idata][index_dtdv] - dtdv_new[idata][index_dtdv]) / dtdv[idata][index_dtdv]).max())

# index_ray = (ray[idata]).nonzero()
# # print(index_ray)
# print((ray[idata][index_ray] - ray_new[idata][index_ray]) / ray[idata][index_ray])

plt.figure()
plt.subplot(121)
plt.imshow(ray_path[idata,:].reshape(ny, -1), extent = (-5-dx/2, 5.+dx/2, 5.+dy/2, -5.-dx/2))
plt.scatter(ray[idata,:100,1], ray[idata,:100,0])
plt.xticks(np.linspace(-5, 5, nx))
plt.yticks(np.linspace(-5, 5, ny))
plt.grid()
plt.subplot(122)
plt.imshow(ray_path_new[idata,:].reshape(ny, -1), extent = (-5-dx/2, 5.+dx/2, 5.+dy/2, -5.-dx/2))
plt.scatter(ray_new[idata,:100,1], ray_new[idata,:100,0])
plt.xticks(np.linspace(-5, 5, nx))
plt.yticks(np.linspace(-5, 5, ny))
plt.grid()
plt.show()


M = dtdv[idata].max()
m = dtdv[idata].min()
plt.figure()
plt.subplot(131)
plt.imshow(dtdv[idata].reshape(ny, -1), clim = (m, M))
plt.subplot(132)
plt.imshow(dtdv_new[idata].reshape(ny, -1), clim = (m, M))
plt.subplot(133)
plt.imshow((dtdv - dtdv_new)[idata].reshape(ny, -1), clim = (m, M))
# plt.imshow((dtdv - dtdv_new)[idata].reshape(ny, -1), clim = (0.1* m, 0.1*M))
plt.show()