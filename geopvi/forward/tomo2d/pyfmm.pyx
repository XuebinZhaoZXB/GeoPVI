import numpy as np
cimport numpy as np
from sys import exit

cdef extern from "pyfm2d.h":
    void c_fm2d(int *nsrc, double *srcx, double *srcy, int *nrec, double *recx, double *recy,
               int *nx, int *ny, int *mask, double *xmin, double *ymin, double *dx, double *dy,
               int *gdx, int *gdy, int *sdx, int *sext, double *vel, double *time, double *dtdv)
    void c_fm2d_ray(int *nsrc, double *srcx, double *srcy, int *nrec, double *recx, double *recy,
               int *nx, int *ny, int *mask, double *xmin, double *ymin, double *dx, double *dy,
               int *gdx, int *gdy, int *sdx, int *sext, double *vel, double *time, double *dtdv, 
               double *rayx, double *rayy)
    void c_fm2d_lglike(int *nsrc, double *srcx, double *srcy, int *nrec, double *recx, double *recy,
               int *nx, int *ny, int *mask, double *xmin, double *ymin, double *dx, double *dy,
               int *gdx, int *gdy, int *sdx, int *sext, int *nv, double *vel, double *tobs, 
               double *res, double *grads)

def fm2d(np.ndarray[double, ndim=1, mode="c"] vel not None, np.ndarray[double, ndim=1, mode="c"] srcx not None, 
        np.ndarray[double, ndim=1, mode="c"] srcy not None,
        np.ndarray[double, ndim=1, mode="c"] recx not None, np.ndarray[double, ndim=1, mode="c"] recy not None,
        np.ndarray[int, ndim=2, mode="c"] mask not None, int nx, int ny, double xmin, double ymin, double dx, 
        double dy, int gdx, int gdy, int sdx, int sext):
    cdef int nsrc, nrec
    nsrc = srcx.shape[0]
    nrec = recx.shape[0]

    if(np.isnan(vel).any()):
        print('NaN occured in python')
        exit()
    if(np.isinf(vel).any()):
        print('Inf occured in python')
        exit()
    
    cdef np.ndarray[double,ndim=1,mode="c"] time = np.empty(nsrc*nrec, dtype=np.float64)
    cdef np.ndarray[double,ndim=2, mode="c"] dtdv = np.empty((nsrc*nrec,nx*ny),dtype=np.float64)
    c_fm2d(&nsrc, &srcx[0], &srcy[0], &nrec, &recx[0], &recy[0], &nx, &ny, &mask[0,0],
          &xmin, &ymin, &dx, &dy, &gdx, &gdy, &sdx, &sext, &vel[0], &time[0], &dtdv[0,0])
    
    return time, dtdv

def fm2d_ray(np.ndarray[double, ndim=1, mode="c"] vel not None, np.ndarray[double, ndim=1, mode="c"] srcx not None, 
        np.ndarray[double, ndim=1, mode="c"] srcy not None,
        np.ndarray[double, ndim=1, mode="c"] recx not None, np.ndarray[double, ndim=1, mode="c"] recy not None,
        np.ndarray[int, ndim=2, mode="c"] mask not None, int nx, int ny, double xmin, double ymin, double dx, 
        double dy, int gdx, int gdy, int sdx, int sext):
    cdef int nsrc, nrec
    nsrc = srcx.shape[0]
    nrec = recx.shape[0]

    if(np.isnan(vel).any()):
        print('NaN occured in python')
        exit()
    if(np.isinf(vel).any()):
        print('Inf occured in python')
        exit()
    
    cdef np.ndarray[double,ndim=1,mode="c"] time = np.empty(nsrc*nrec, dtype=np.float64)
    cdef np.ndarray[double,ndim=2, mode="c"] dtdv = np.empty((nsrc*nrec,nx*ny),dtype=np.float64)
    cdef np.ndarray[double,ndim=2, mode="c"] rayx = np.empty((nsrc*nrec,((ny-1)*gdy+1)*((nx-1)*gdx+1)),dtype=np.float64)
    cdef np.ndarray[double,ndim=2, mode="c"] rayy = np.empty((nsrc*nrec,((ny-1)*gdy+1)*((nx-1)*gdx+1)),dtype=np.float64)

    for i in range(nsrc):
        for j in range(nrec):
            rayx[i*nrec+j, :] =  srcx[i]
            rayy[i*nrec+j, :] =  srcy[i]

    c_fm2d_ray(&nsrc, &srcx[0], &srcy[0], &nrec, &recx[0], &recy[0], &nx, &ny, &mask[0,0],
          &xmin, &ymin, &dx, &dy, &gdx, &gdy, &sdx, &sext, &vel[0], &time[0], &dtdv[0,0], 
          &rayx[0,0], &rayy[0,0])
    
    return time, dtdv, np.concatenate((rayx[:,:,None],rayy[:,:,None]), axis = 2)

def fm2d_lglike(np.ndarray[double, ndim=2, mode="c"] vel not None, np.ndarray[double, ndim=1, mode="c"] srcx not None, 
        np.ndarray[double, ndim=1, mode="c"] srcy not None,
        np.ndarray[double, ndim=1, mode="c"] recx not None, np.ndarray[double, ndim=1, mode="c"] recy not None,
        np.ndarray[int, ndim=2, mode="c"] mask not None, int nx, int ny, double xmin, double ymin, double dx, 
        double dy, int gdx, int gdy, int sdx, int sext, 
        np.ndarray[double, ndim=1, mode="c"] tobs not None):
    cdef int nsrc, nrec, nv
    nsrc = srcx.shape[0]
    nrec = recx.shape[0]
    nv = vel.shape[0]

    if(np.isnan(vel).any()):
        print('NaN occured in python')
        exit()
    
    cdef np.ndarray[double,ndim=1,mode="c"] res = np.empty(nv, dtype=np.float64)
    cdef np.ndarray[double,ndim=2, mode="c"] grads = np.empty((nv,nx*ny),dtype=np.float64)
    c_fm2d_lglike(&nsrc, &srcx[0], &srcy[0], &nrec, &recx[0], &recy[0], &nx, &ny, &mask[0,0],
          &xmin, &ymin, &dx, &dy, &gdx, &gdy, &sdx, &sext, &nv, 
          &vel[0,0], &tobs[0], &res[0], &grads[0,0])
    
    return res, grads
