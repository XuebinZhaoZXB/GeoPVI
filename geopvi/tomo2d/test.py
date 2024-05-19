# test python wrapper for fortran fmm 2d code

import numpy as np
import matplotlib.pyplot as plt
from fmm import fm2d, fm2d_ray
import torch

def get_raypath_grid(ray_path):
    ray = np.zeros((mask.shape[1], 441))

    for idata in mask[0,:].nonzero()[0]:
        iy = ((ray_path[idata,:,1] - ymin + dy/2) // dy).astype('int') # actually iy
        ix = ((ray_path[idata,:,0] - xmin + dx/2) // dx).astype('int') # actually ix
        ray[idata,ix * ny + iy] = 1

        ## set the grids surrounding the true ray path grids as 1
        ray[idata,ix * ny + iy -ny-1] = 1
        ray[idata,ix * ny + iy-ny] = 1
        ray[idata,ix * ny + iy-ny+1] = 1
        ray[idata,ix * ny + iy-1] = 1
        ray[idata,ix * ny + iy+1] = 1
        ray[idata,ix * ny + iy+ny-1] = 1
        ray[idata,ix * ny + iy+ny] = 1
        ray[idata,ix * ny + iy+ny+1] = 1
        ## set 2 grids outside the true ray path grids as 1
        # ray[idata,(ix-2)*ny+iy-2] = 1
        # ray[idata,(ix-2)*ny+iy-1] = 1
        # ray[idata,(ix-2)*ny+iy-0] = 1
        # ray[idata,(ix-2)*ny+iy+1] = 1
        # ray[idata,(ix-2)*ny+iy+2] = 1
        # ray[idata,(ix-1)*ny+iy-2] = 1
        # ray[idata,(ix-1)*ny+iy+2] = 1
        # ray[idata,(ix-0)*ny+iy-2] = 1
        # ray[idata,(ix-0)*ny+iy+2] = 1
        # ray[idata,(ix+1)*ny+iy-2] = 1
        # ray[idata,(ix+1)*ny+iy+2] = 1
        # ray[idata,(ix+2)*ny+iy-2] = 1
        # ray[idata,(ix+2)*ny+iy-1] = 1
        # ray[idata,(ix+2)*ny+iy-0] = 1
        # ray[idata,(ix+2)*ny+iy+1] = 1
        # ray[idata,(ix+2)*ny+iy+2] = 1

    return ray

angle = np.arange(0., 2.*np.pi, np.pi/8)

src = np.array([[4*np.sin(x), 4*np.cos(x)] for x in angle])
rec = src

srcx = np.ascontiguousarray(src[:,0])
srcy = np.ascontiguousarray(src[:,1])
recx = np.ascontiguousarray(rec[:,0])
recy = np.ascontiguousarray(rec[:,1])


mask = np.zeros((2,256),dtype=np.int32)
for i in range(16):
    for j in range(16):
        if(j>i):
             mask[0,i*16+j] = 1
             mask[1,i*16+j] = i*16 + j + 1
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
sigma = 0.05

vel = np.ones(441,) * 2.
# vel = np.random.uniform(0.5, 3.0, (441,))
data, dtdv, ray_path = fm2d_ray(vel,srcx,srcy,recx,recy,mask,nx,ny,xmin,ymin,dx,dy,gdx,gdy,sdx,sext)
# print(ray_path.shape)
# print(ray_path[0,:100,:])
# print(dtdv[0,:])
# print(vel.shape)
# print(data.shape)


# extenison
z = torch.zeros((1, 441))
grad = torch.zeros((1, 256, 441))
ray = torch.zeros((1, 256, 441))
z[0] = torch.from_numpy(vel)
grad[0] = torch.from_numpy(dtdv)
ray[0] = torch.from_numpy(get_raypath_grid(ray_path))
# print(ray[0,1,:])

# z = torch.from_numpy(vel[None,...]).clone()
# grad = torch.from_numpy(dtdv[None,...]).clone()
# ray = torch.from_numpy(get_raypath_grid(ray_path)[None,...])
# print(z.shape)
# print(grad.shape)
# print(ray.shape)

max_grad = 0
ig = 0
ida = 0

number = 5
z_hat = z.expand(mask.shape[-1], number+1, z.shape[-1]).clone()
grad_hat = grad.squeeze(0)[:,None,:].expand(mask.shape[-1], number+1, z.shape[-1]).clone()
for i in range(mask.shape[-1]):
    if mask[0,i] != 0:
        # do extension along pixels without raypath
        for j in range(number):
            # 1st: determine the number of grids (grid_num) to be decreased
            grid_num = np.random.randint(1, (ray[:,i,:].shape[-1] - ray[:,i,:].sum()), (1,))
            # 2nd: get the index of grid_num grids
            extension_index = np.random.choice((ray[:,i,:]==0).nonzero(as_tuple = True)[-1], grid_num)
            # 3rd: extend the velocity value at the selected grids
            z_hat[i,j,extension_index] = torch.from_numpy(
                                np.random.uniform(0.5, z[:,extension_index], grid_num).astype(np.float32))
            # 4th: get the corresponding travel time and gradient
            grad_hat[i,j,extension_index] = grad[:,i,extension_index] * \
                        (z[:,extension_index] / z_hat[i,j,extension_index]) **2

            if (np.abs(grad[:,i,extension_index].min()) > max_grad):
                max_grad = np.abs(grad[:,i,extension_index].min())
                ig = j
                ida = i


print(max_grad, ig, ida)
print(z_hat.shape)
print(grad_hat.shape)

idata = ida
ivel = ig
data_new, dtdv_new, ray_path_new = fm2d_ray(z_hat[idata,ivel,:].numpy().astype(np.float64),srcx,srcy,recx,recy,mask,nx,ny,xmin,ymin,dx,dy,gdx,gdy,sdx,sext)
ray_new = torch.from_numpy(get_raypath_grid(ray_path_new)[None,...])


plt.figure()
plt.subplot(241)
plt.imshow(ray[0,idata,:].reshape(ny, -1), extent = (-5-dx/2, 5.+dx/2, 5.+dy/2, -5.-dx/2))
plt.scatter(ray_path[idata,:100,1], ray_path[idata,:100,0])
plt.xticks(np.linspace(-5, 5, nx))
plt.yticks(np.linspace(-5, 5, ny))
plt.grid()
plt.title('Original ray')
plt.subplot(245)
plt.imshow(ray_new[0,idata,:].reshape(ny, -1), extent = (-5-dx/2, 5.+dx/2, 5.+dy/2, -5.-dx/2))
plt.scatter(ray_path_new[idata,:100,1], ray_path_new[idata,:100,0])
plt.xticks(np.linspace(-5, 5, nx))
plt.yticks(np.linspace(-5, 5, ny))
plt.grid()
plt.title('Extension ray')

M = 0.
m = -0.08

# plt.figure()
plt.subplot(242)
# plt.imshow(grad[0, idata, :].reshape(ny, -1), clim = (m, M))
plt.imshow(grad[:,idata,:].reshape(ny, -1), clim = (m, M))
plt.title('Original gradient')
plt.subplot(246)
plt.imshow(dtdv_new[idata,:].reshape(ny, -1), clim = (m, M))
plt.title('Gradient using new vel')
plt.subplot(243)
plt.imshow(grad_hat[idata, ivel, :].reshape(ny, -1), clim = (m, M))
plt.title('Extension gradient')
plt.subplot(247)
plt.imshow((grad_hat[idata, ivel, :] - dtdv_new[idata, :]).reshape(ny, -1), clim = (m, M))
plt.title('grad_hat - dtdv_new')
# plt.imshow((dtdv - dtdv_new)[idata].reshape(ny, -1), clim = (0.1* m, 0.1*M))
# plt.show()

# plt.show()


# plt.figure()
plt.subplot(248)
plt.imshow(z.reshape(ny, -1), extent = (-5-dx/2, 5.+dx/2, 5.+dy/2, -5.-dx/2))
# plt.scatter(ray1[1,:100,0], ray[idata,:100,1])
plt.xticks(np.linspace(-5, 5, nx))
plt.yticks(np.linspace(-5, 5, ny))
plt.grid()
plt.subplot(244)
plt.imshow(z_hat[idata,ivel,:].reshape(ny, -1), extent = (-5-dx/2, 5.+dx/2, 5.+dy/2, -5.-dx/2))
# plt.scatter(ray_new[idata,:100,0], ray_new[idata,:100,1])
plt.xticks(np.linspace(-5, 5, nx))
plt.yticks(np.linspace(-5, 5, ny))
plt.grid()
plt.show()

print(data[idata])
print(data_new[idata])
# print(np.min(dtdv_new[1] - dtdv[1]))
# print(np.max(dtdv_new[1] - dtdv[1]))


vel_new = vel.copy()
# vel_new[225] = 0.55
vel_new[222] = 1
# vel_new[0:9*21] = 0.55
data_new, dtdv_new, ray_new = fm2d_ray(vel_new,srcx,srcy,recx,recy,mask,nx,ny,xmin,ymin,dx,dy,gdx,gdy,sdx,sext)

ray_path = np.zeros(dtdv.shape)
for idata in mask[0,:].nonzero()[0]:
    ix = ((ray[idata,:,0] - xmin + dx/2) // dx).astype('int')
    iy = ((ray[idata,:,1] - ymin + dy/2) // dy).astype('int')
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
    ix = ((ray_new[idata,:,0] - xmin + dx/2) // dx).astype('int')
    iy = ((ray_new[idata,:,1] - ymin + dy/2) // dy).astype('int')
    ray_path_new[idata,iy * nx + ix] = 1
    ray_path_new[idata,iy * nx + ix -nx-1] = 1
    ray_path_new[idata,iy * nx + ix-nx] = 1
    ray_path_new[idata,iy * nx + ix-nx+1] = 1
    ray_path_new[idata,iy * nx + ix-1] = 1
    ray_path_new[idata,iy * nx + ix+1] = 1
    ray_path_new[idata,iy * nx + ix+nx-1] = 1
    ray_path_new[idata,iy * nx + ix+nx] = 1
    ray_path_new[idata,iy * nx + ix+nx+1] = 1



idata = 2

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
plt.scatter(ray[idata,:100,0], ray[idata,:100,1])
plt.xticks(np.linspace(-5, 5, nx))
plt.yticks(np.linspace(-5, 5, ny))
plt.grid()
plt.subplot(122)
plt.imshow(ray_path_new[idata,:].reshape(ny, -1), extent = (-5-dx/2, 5.+dx/2, 5.+dy/2, -5.-dx/2))
plt.scatter(ray_new[idata,:100,0], ray_new[idata,:100,1])
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