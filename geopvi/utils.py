import numpy as np
import scipy.sparse as sparse


def delta(n):
    diag0 = np.full((n,),fill_value=-2); diag0[0]=-1; diag0[-1]=-1
    diag1 = np.full((n-1,),fill_value=1)
    diagonals = [diag0,diag1,diag1]
    D = sparse.diags(diagonals,[0,-1,1]).tocsc()
    return D

def smooth_matrix_2D(nx, nz, smoothx, smoothz):
    smoothx = np.full((nz,),fill_value=smoothx)
    smoothz = np.full((nz,),fill_value=smoothz)
    deltax = delta(nx)
    deltaz = delta(nz)/smoothz[:,None]
    Ix = sparse.eye(nx)
    Iz = sparse.eye(nz)
    Sx = sparse.kron(Iz/smoothx,deltax)
    Sz = sparse.kron(deltaz,Ix)
    L = sparse.vstack([Sx,Sz])
    return L

def smooth_matrix_3D(nx, ny, nz, smoothx, smoothy, smoothz):
    smoothx = np.full((nz,),fill_value=smoothx)
    smoothy = np.full((nz,),fill_value=smoothy)
    smoothz = np.full((nz,),fill_value=smoothz)
    deltax = delta(nx)
    deltay = delta(ny)
    deltaz = delta(nz)/smoothz[:,None]
    Iy = sparse.eye(ny); Ix = sparse.eye(nx); Iz = sparse.eye(nz)
    Sz = sparse.kron(Iy,sparse.kron(Ix,deltaz))
    Sx = sparse.kron(Iy,sparse.kron(deltax,Iz/smoothx))
    Sy = sparse.kron(deltay,sparse.kron(Ix,Iz/smoothy))
    L = sparse.vstack([Sx,Sy,Sz])
    return L

def psvi_mask_2D(correlation, ndim, nx = 1, nz = 1):
    z, x = correlation.shape
    rank = (correlation != 0).sum() // 2
    # cz, cx: coordinate for central point of the mask
    cz, cx = (correlation.size)//2 // x, (correlation.size)//2 % x
    offset = np.zeros(rank, dtype = int)
    mask = np.ones((rank, ndim), dtype = bool)
    i = 0
    for iz in range(z):
        for ix in range(x):
            if correlation[iz, ix] == 0 or iz * x + ix >= (correlation.size)//2:
                continue
            offset[i] = (cz - iz) * nx + (cx - ix)
            # self.non_diag[i, -offset[i]:] = torch.zeros(offset[i])
            mask[i, -offset[i]:] = False
            i += 1
    return mask

def psvi_mask_3D(correlation, ndim, nx = 1, ny = 1, nz = 1):
    y, x, z = correlation.shape
    rank = (correlation != 0).sum() // 2
    cy = correlation.size // 2 // (x*z)
    cx = correlation.size // 2 % (x*z) // z
    cz = correlation.size // 2 % (x*z) % z
    offset = np.zeros(rank, dtype = int)
    mask = np.ones((rank, ndim), dtype = bool)
    i = 0
    for iy in range(y):
        for ix in range(x):
            for iz in range(z):
                if correlation[iy, ix, iz] == 0 or iy*x*z + ix*z + iz >= (correlation.size)//2:
                    continue
                offset[i] = (cy - iy)*nx*nz + (cx - ix)*nz + (cz - iz)
                mask[i, -offset[i]:] = False
                i += 1
    return mask