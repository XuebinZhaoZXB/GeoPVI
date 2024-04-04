import numpy as np
import time
import torch
from torch.autograd import Function

import subprocess
import PyDDS.dds_io as io
import os.path
import re
from pathlib import Path
from scipy.sparse.linalg import spsolve
import scipy.sparse as sparse
import collections
# from protector import SGEJobArrayProtector


def argmap(key, settings):
    return key+"="+settings[key]

def prepare_samples(theta, ny, nx, nz, dx, dy, dz, datapath='./', fname='particles', sep='.'):
    nsamples = theta.shape[0]
    for i in range(nsamples):
        filename = os.path.join(datapath,fname+sep+str(i))
        model = theta[i,:].reshape((ny,nx,nz))
        io.tofile(filename,model,axes=['z','x','y'],units=['m','m','m'],
                    origins=[0,0,0],deltas=[dz,dx,dy],bases=[1,1,1],steps=[1,1,1])

    return 0

def prepare_batch(batch, datapath='input', geomfile='input/Hgeom',datafile='input/data'):
    geom = io.fromfile(geomfile)['Samples']
    geom_dict = io.DDSInputDict('',geomfile)
    shotsize = geom_dict.axis3.size
    slices = np.sort(np.random.choice(shotsize,size=batch,replace=False))
    geom = geom[slices,:,:]
    batch_geom = os.path.join(datapath,'Hgeom.batch')
    io.tofile(batch_geom,geom)

    data = io.fromfile(datafile)['Samples']
    data = data[slices,:,:]
    data_dict = io.DDSInputDict('',datafile)
    batch_data = os.path.join(datapath,'data.batch')
    io.tofile(batch_data, data,origins=[data_dict.axis1.origin,0,0],deltas=[data_dict.axis1.delta,1,1])
    geom_dict.fclose()
    data_dict.fclose()

    return shotsize*1./batch

def read_loss(filename):
    # read loss from output of external FWI code (revise as needed)
    with open(filename) as f:
        text = f.read()
    numeric_const_pattern = '( [-+]? (?: (?: \d* \. \d+ ) | (?: \d+ \.? ) )(?: [Ee] [+-]? \d+ ) ? )'
    pattern = 'iter,f,gnorm,tevalFG:\s*\d+\s*'+numeric_const_pattern
    rx = re.compile(pattern,re.VERBOSE)
    results = rx.findall(text)

    loss = -1.
    if(results):
        loss = float(results[-1])

    return loss

def cal_loss(filename):
    data_dict = io.DDSInputDict('',filename)
    rd = io.DDSFileReader(data_dict)
    res = rd[0,:]['Samples']
    loss = np.sum(res**2)

    return loss


def run_fwi_i(i, argument, config, mask):
    """
    warpper function for parallel FWI simulation using dask features
    i: sample index - looks like they have been run in series, but actually in parallel
    argument: a python parser.parse_args() object that defines hyperparameters for VI
    config: configure file for hyper-parameters
    Return
        loss: l2_norm loss function value: 0.5 * |d - d_syn|^2
        grad: -tdwi gradient output (masked only those parameters that need to be updated)
        Noth this returned grad is not dloss / dm. To get this, you need: grad / vel^3
    """
    options =['np','nproc','ph','imag','usedatmsk','tmp_data_path','bindata','gather_resamp',
            'nxaper','nyaper','Tmax_source','gathertype','operator','isogradient',
            'invtype','bctaper','Tmin_rcvr','Tmax_rcvr','nxtaper','nztaper',
            'nytaper','filterUpCrap','optype','illumOption','tdfwimod',
            'niter','kout','betascale','perturb_ls','vclipmin','vclipmax',
            'abs_surf','srcwpw','lessio','offmin','offmax',
            'aper_max_x','aper_max_y']
    args = []
    settings = config['FWI']
    args.append("/tstapps/asi/bin/tdwi")
    inpath = os.path.join(argument.basepath + config.get('path', 'inpath'))
    outpath = os.path.join(argument.basepath + config.get('path', 'outpath'))
    print_path = os.path.join(argument.basepath + config.get('path', 'print_path'))

    for op in options:
        args.append(argmap(op,settings))
    args.append("datmsk="+config.get('path', 'datmskfile'))
    args.append("source="+config.get('path', 'srcfile'))
    args.append("geom="+os.path.join(inpath,'Hgeom.batch'))
    args.append("in="+os.path.join(inpath,'data.batch'))
    args.append("vel="+os.path.join(inpath,'particles.'+str(i)))
    args.append("print_path="+os.path.join(print_path,'printout.'+str(i)))
    args.append("data_path="+outpath)
    args.append("velout="+os.path.join(outpath,'velout.'+str(i)))
    args.append("grad="+os.path.join(outpath,'gradout.'+str(i)))
    args.append("plotfile="+os.path.join(outpath,'plot.'+str(i)))
    args.append("ures="+os.path.join(outpath,'ures.'+str(i)))
    args.append("urestmp="+os.path.join(outpath,'urestmp.'+str(i)))
    args.append("uvsrc="+os.path.join(outpath,'uvsrc.'+str(i)))
    args.append("ucalc="+os.path.join(outpath,'ucalc.'+str(i)))

    #print(*args, sep=" ")
    try:
        subprocess.check_output(args, stderr=subprocess.STDOUT, shell=False, timeout=3600)
    except subprocess.CalledProcessError as e:
        print(e.output)
    except subprocess.TimeoutExpired:
        subprocess.check_output(args, stderr=subprocess.STDOUT, shell=False, timeout=3600)

    loss = read_loss(os.path.join(print_path,'printout.'+str(i)))
    if(loss<0):
        loss = cal_loss(os.path.join(outpath,'ures.'+str(i)))

    # tdfwimod: Set to 1 to just do synthetic modeling, and not  inversion.   Default=0.
    # if synthetic modeling, then set grad as None to save memory
    # output_grad: whether to output gradient of velocity or not
    output_grad = bool(1 - config.getint('FWI','tdfwimod'))
    grad = None
    if output_grad:
        grad = io.fromfile(os.path.join(outpath,'gradout.'+str(i)))['Samples']
        grad = grad.flatten()[mask]
    return loss, grad


class ForwardModel(Function):
    @staticmethod
    def forward(ctx, input, func):
        output, grad = func(input)
        ctx.save_for_backward(input, torch.tensor(grad))
        return torch.tensor(output)

    @staticmethod
    def backward(ctx, grad_output):
        '''
        this function returns the gradient of loss w.r.t the input tensor in the forward function
        therefore, the return shape should be the same as the shape of input tensor
        '''
        input, grad = ctx.saved_tensors
        # grad_input = (grad_output[...,None] * grad).sum(axis = -2)
        grad_input = (grad_output[...,None] * grad)
        return grad_input, None


class Posterior():
    def __init__(self, args, config, vel_water = None, mask = None, log_prior = None, client=None):
        '''
        args: a python argparser.parse_args() object that defines hyperparameters for VI
        config: a python configparser.ConfigParser() that defines hyperparameters for FWI forward simulation
        vel_water: velocity of water layer (in 3D synthetic test the true velocity value is used 
                                            thus obtianed from true model)
        mask: a mask array where the parameters with mask = 0 will be fixed 
        log_prior: a function that takes samples as input and calculates their log-prior values (using PyTorch)
                    Return: y = log_prior(x)
        client: a dask client to submit FWI running, must be specified (used to perform inter-node parallelisation)
        '''
        self.args = args
        self.config = config
        self.sigma = args.sigma
        self.client = client
        self.log_prior = log_prior
        self.vel_water = vel_water

        # create mask matrix for model parameters that are fixed during inversion (e.g., water layer)
        if mask is None:
            nx = config.getint('FWI','nx')
            ny = config.getint('FWI','ny')
            nz = config.getint('FWI','nz')
            mask = np.full((ny*nx*nz),True)
        self.mask = mask

    def fwi3d(self, x):
        '''
        Call external FWI code to get misfit value and gradient
        Note that this needs to be implemented for specific FWI code
        '''
        # get model info from self.config
        nx = self.config.getint('FWI','nx')
        ny = self.config.getint('FWI','ny')
        nz = self.config.getint('FWI','nz')
        dx = self.config.getfloat('FWI','dx')
        dy = self.config.getfloat('FWI','dy')
        dz = self.config.getfloat('FWI','dz')
        batch = self.config.getint('FWI','shot_batch')
        datapath = os.path.join(self.args.basepath + self.config.get('path','inpath'))

        # convert x from torch.tensor to np.ndarray and get parameters needed for optimisation
        m, _ = x.shape
        if not isinstance(x, np.ndarray):
            x = x.detach().numpy().astype(np.float64)
        vel = np.broadcast_to(self.vel_water[None], (m, ny, nx, nz)).reshape(m, -1)
        vel[:, self.mask] = x

        # prepare velocity models for FWI code (revise for sepcific code)
        prepare_samples(vel, ny, nx, nz, dx, dy, dz, datapath = datapath)
        scale = prepare_batch(batch, datapath = datapath, geomfile = self.config.get('path','geomfile'), 
                                datafile = self.config.get('path','datafile'))

        for filename in Path(self.args.basepath + self.config.get('path','outpath')).glob("*_restart.state*"):
            filename.unlink()

        # submit external FWI code to dask cluster for each sample in vel
        futures = []
        for i in range(m):
            # futures.append(self.client.submit(self.solver, i, pure=False))
            futures.append( self.client.submit(run_fwi_i, i, self.args, self.config, self.mask, pure=False))
        results = self.client.gather(futures)

        loss = np.zeros(m)            
        for i in range(m):
            loss[i] = results[i][0] * scale
            
        # tdfwimod: Set to 1 to just do synthetic modeling, and not  inversion.   Default=0.
        # if synthetic modeling, then set grad as None to save memory
        # output_grad: whether to output gradient of velocity or not
        output_grad = bool(1 - self.config.getint('FWI','tdfwimod'))
        grad = None
        if output_grad:
            grad = np.zeros(x.shape)
            for i in range(m):
                grad[i] = results[i][1]
            grad = 2 * grad / (x**3) * scale
    
            # # clip the grad to avoid numerical instability
            # clip = self.sigma**2 * self.config.getfloat('FWI','gclipmax')
            # #clip = clip * np.quantile(np.abs(grad),0.999)
            # grad[grad>=clip] = clip
            # grad[grad<=-clip] = -clip

        return 0.5 * loss, grad

    def log_prob(self, x):
        """
        calculate log likelihood and its gradient directly from model x
        Input
            x: 2D array with dimension of nsamples * ndim
        Return
            logp: a vector of (unnormalised) log-posterior value for each sample
        """    
        if(torch.isnan(x).any()):
            raise ValueError('NaN occured in sample!')

        t = time.time()
        loss = ForwardModel.apply(x, self.fwi3d)
        print('Simulation takes '+str(time.time()-t))

        logp = -loss/self.sigma**2 + self.log_prior(x)
        # log_prior =  self.prior(x)
        return logp