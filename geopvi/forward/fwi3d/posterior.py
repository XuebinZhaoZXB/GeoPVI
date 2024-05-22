import numpy as np
import torch
from torch.autograd import Function

import subprocess
import os.path
from pathlib import Path


def argmap(key, settings):
    return key+"="+settings[key]

def prepare_samples(theta, nx, ny, nz, datapath='./', fname='particles', sep='_'):
    nparticles = theta.shape[0]
    for i in range(nparticles):
        filename = os.path.join(datapath,fname+sep+str(i)+'.npy')
        model = theta[i,:].reshape((ny,nx,nz))
        np.save(filename,model)
    return 0

def prepare_batch(batch, datapath='input', srcfile='input/source.npy', datafile='input/data.npy'):
    shots = np.load(srcfile)
    shotsize = shots.shape[0]
    slices = np.sort(np.random.choice(shotsize,size=batch,replace=False))
    batch_shots = shots[slices,:,]
    batch_src = os.path.join(datapath,'batch_src.npy')
    np.save(batch_src,batch_shots)

    data = np.load(datafile)
    batch_data = data[slices,:,:]
    batch_file = os.path.join(datapath,'batch_data.npy')
    np.save(batch_file, batch_data)

    return shotsize*1./batch

def cal_loss(pred_file, data_file='input/batch_data.npy'):
    pred_data = np.load(pred_file)
    data = np.load(data_file)
    res = pred_data - data
    loss = np.sum(res**2)

    return loss

def run_fwi_i(i, argument, config, mask):
    """
    warpper function for parallel FWI simulation using dask features
    i: sample index - looks like they have been run in series, but actually in parallel
    argument: a python parser.parse_args() object that defines hyperparameters for VI
    config: configure file for hyper-parameters
    Return
        loss: l2_norm loss function value: |d - d_syn|^2
        grad: gradient output (masked only those parameters that need to be updated)
    """
    options =['waveletfile', 'recfile']
    args = []
    settings = config['FWI']
    args.append("/your/external/3DFWI/executable/file")
    inpath = os.path.join(config.get('path', 'inpath'))
    outpath = os.path.join(config.get('path', 'outpath'))

    for op in options:
        args.append(argmap(op,settings))
    args.append("data="+os.path.join(inpath,'batch_data.npy'))
    args.append("src="+os.path.join(outpath,'batch_src.npy'))

    try:
        subprocess.check_output(args, stderr=subprocess.STDOUT, shell=False, timeout=1200)
    except subprocess.CalledProcessError as e:
        print(e.output)
    except subprocess.TimeoutExpired:
        subprocess.check_output(args, stderr=subprocess.STDOUT, shell=False, timeout=1200)

    grad = np.load(os.path.join(outpath,'gradout_'+str(i)+'.npy'))
    loss = cal_loss(os.path.join(outpath,'pred_data_'+str(i)+'.npy'), os.path.join(inpath,'batch_data.npy'))

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
    def __init__(self, args, config, mask = None, log_prior = None, client=None):
        '''
        args: a python argparser.parse_args() object that defines hyperparameters for VI
        config: a python configparser.ConfigParser() that defines hyperparameters for FWI forward simulation
        mask: a mask array where the parameters with mask = 0 will be fixed 
        log_prior: a function that takes samples as input and calculates their log-prior values (using PyTorch)
                    Return: y = log_prior(x)
        client: a dask client to submit FWI running, must be specified (used to perform inter-node parallelisation)
        '''
        self.args = args
        self.config = config
        self.sigma = args.sigma
        self.client = client
        self.vel_water = config.getfloat('FWI','vel_fixed')
        self.log_prior = log_prior

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
        datapath = os.path.join(self.config.get('path','inpath'))

        # convert x from torch.tensor to np.ndarray and get parameters needed for optimisation
        m, _ = x.shape
        loss = np.zeros(m)
        grad = np.zeros(x.shape)
        if not isinstance(x, np.ndarray):
            x = x.detach().numpy().astype(np.float64)
        vel = np.broadcast_to(self.vel_water, (m, ny, nx, nz)).reshape(m, -1)
        vel[:, self.mask] = x

        # prepare velocity models for FWI code (revise for sepcific code)
        prepare_samples(vel, ny, nx, nz, datapath = datapath)
        scale = prepare_batch(batch, datapath = datapath, srcfile=self.config.get('path','srcfile'),
                                        datafile = self.config.get('path','datafile'))

        # submit external FWI code to dask cluster for each sample in vel
        futures = []
        for i in range(m):
            futures.append( self.client.submit(run_fwi_i, i, self.args, self.config, self.mask, pure=False))
        results = self.client.gather(futures)

        for i in range(m):
            loss[i] = results[i][0]
            grad[i] = results[i][1]

        return 0.5 * loss * scale, grad * scale

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
        loss = ForwardModel.apply(x, self.fwi3d)

        logp = -loss/self.sigma**2 + self.log_prior(x)
        return logp