import numpy as np
import torch

import os
from pathlib import Path
from argparse import ArgumentParser
import configparser
import time
from datetime import datetime

from geopvi.nfvi.models import FlowsBasedDistribution, VariationalInversion
from geopvi.nfvi.flows import Real2Constr, Constr2Real, Linear
from geopvi.forward.swi.posterior import Posterior3D_dc, Posterior3D_tt
from geopvi.utils import smooth_matrix_3D as smooth_matrix


class Uniform():
    '''
    A class that defines Uniform prior distribution used for inversion
    In this version, we consider additional prior information for dispersion inversion
    where the top layer should have the smallest velocity value
    '''
    def __init__(self, lower = np.array([0.]), upper = np.array([1.]), smooth_matrix = None, top_layer_constraint = 0, 
                        nx = 10, ny = 10, nz = 10):
        '''
        lower (numpy array): lower bound of the Uniform distribution 
        upper (numpy array): upper bound of the Uniform distribution
        smooth_matrix (scipy sparse): smooth matrix applied on model samples, used to define smooth prior pdf
        top_layer_constraint: scalar value defining prior information that first layer has the smallest velocity values
        '''
        if np.isscalar(upper):
            upper = np.array([upper])
        if np.isscalar(lower):
            lower = np.array([lower])
        self.lower = torch.from_numpy(lower)
        self.upper = torch.from_numpy(upper)
        # self.dim = lower.size
        self.smooth_matrix = smooth_matrix
        self.top_layer_constraint = top_layer_constraint
        self.nx = nx
        self.ny = ny
        self.nz = nz

    def prior_top_layer(self, vel):
        '''
        vel: torch array with a shape of (nsampls, nx, ny, nz) defining nsamples 3d vs model
        '''
        d_vel = torch.min(vel[:,:,:,1:], dim = -1)[0] - vel[:,:,:,0]
        index = d_vel < 0.
        log_prior = torch.sum(-0.5 * d_vel[index] ** 2 / self.top_layer_constraint ** 2)
        return log_prior

    def log_prob(self, x):
        '''
        Compute log probability using PyTorch, such that the result can be back propagated
        '''
        logp = - torch.log(self.upper - self.lower).sum(axis = -1)

        if self.top_layer_constraint != 0:
            vel_3d = x.reshape(x.shape[0], self.nx, self.ny, self.nz)
            logp += self.prior_top_layer(vel_3d)

        if self.smooth_matrix is not None:
            logp_smooth = Smoothing.apply(x, self.smooth_matrix)
            return logp + logp_smooth
        return logp


def get_offdiag_mask(correlation, ndim, nx = 1, ny = 1, nz = 1):
    x, y, z = correlation.shape
    rank = (correlation != 0).sum() // 2
    cx = correlation.size // 2 // (y*z)
    cy = correlation.size // 2 % (y*z) // z
    cz = correlation.size // 2 % (y*z) % z
    offset = np.zeros(rank, dtype = int)
    mask = np.ones((rank, ndim), dtype = bool)
    i = 0
    for iy in range(x):
        for ix in range(y):
            for iz in range(z):
                if correlation[ix, iy, iz] == 0 or ix*y*z + iy*z + iz >= (correlation.size)//2:
                    continue
                offset[i] = (cx - ix)*ny*nz + (cy - iy)*nz + (cz - iz)
                mask[i, -offset[i]:] = False
                i += 1
    return mask

def get_flow_param(flow):
    mus = flow.u.detach().numpy()
    sigmas = np.exp(flow.diag.detach().numpy())
    if flow.non_diag is None:
        return np.hstack([mus, sigmas])
    else:
        non_diag = flow.non_diag.detach().numpy().flatten()
        return np.hstack([mus, sigmas, non_diag])


if __name__ == "__main__":
    argparser = ArgumentParser(description='3D Bayesian surface wave dispersion inversion using GeoPVI')
    argparser.add_argument("--basepath", metavar='basepath', type=str, help='Project path', default='./')

    argparser.add_argument("--flow", default='Linear', type=str, help='Flows used to perform inversion')
    argparser.add_argument("--kernel", default='structured', type=str, help='Covariance kernel type if Linear flow is used')
    argparser.add_argument("--kernel_size", default=5, type=int, help='Local covariance kernel size if PSVI is performed')
    argparser.add_argument("--nflow", default=1, type=int, help='number of flows')
    argparser.add_argument("--nsample", default=8, type=int, help='number of samples for MC integration during each iteration')
    argparser.add_argument("--prcs", default=48, type=int, help='number of processes in parallel to perform forward evaluation')
    argparser.add_argument("--iterations", default=20000, type=int, help='number of iterations to update variational parameters')
    argparser.add_argument("--lr", default=0.001, type=float, help='learning rate')
    argparser.add_argument("--ini_dist", default='Normal', type=str, help='initial (base) distribution for flows-based model')
    argparser.add_argument("--sigma_scale", default=1, type=float, help='scale factor for data noise level')

    argparser.add_argument("--v0_std", default=0.15, type=float, help='Std value when defing prior informaton')
    argparser.add_argument("--smooth", default=False, type=bool, help='Whether to apply smooth factor on model vector m')
    argparser.add_argument("--smoothx", default=1, type=float, help='Smoothness parameter, smaller value means stronger smoothness')
    argparser.add_argument("--smoothy", default=1, type=float, help='Smoothness parameter, smaller value means stronger smoothness')
    argparser.add_argument("--smoothz", default=3, type=float, help='Smoothness parameter, smaller value means stronger smoothness')

    argparser.add_argument("--prior_type", default='Uniform', type=str, help='Prior pdf - either Uniform or Normal, or user-defined')
    argparser.add_argument("--prior_param", default='prior_20_top.txt', type=str, help='filename containing hyperparametes to define prior pdf')
    
    argparser.add_argument("--datatype", default='traveltime', type=str, help='Use traveltime or dispersion curves as observed data')

    argparser.add_argument("--config", default='config_grid3.ini', type=str, help='filename containing parameters for forward simulation')
    argparser.add_argument("--srcfile", default='sources_58.txt', type=str, help='filename for (x, y) source locations')
    argparser.add_argument("--recfile", default='sources_58.txt', type=str, help='filename for (x, y) source locations')
    argparser.add_argument("--datafile", default='otimes_src58', type=str, help='filename for observed dataset')

    # argparser.add_argument("--datafile", default='stimes.dat', type=str, help='filename for observed dataset')

    argparser.add_argument("--flow_init_name", type=str, default='none', help='Parameter filename for flow initial value')
    argparser.add_argument("--outdir", type=str, default='output/', help='folder path (relative to basepath) for output files')

    argparser.add_argument("--verbose", default=True, type=bool, help='Output and print intermediate inversion results')
    argparser.add_argument("--save_intermediate_result", default=True, type=bool,
                                help='Whether save intermediate training model, for resume from previous training')
    argparser.add_argument("--resume", default=False, type=bool, help='Resume previous training')
    argparser.add_argument("--nout", default=20, type=int, help='Number to print/output intermediate inversion results')


    args = argparser.parse_args()
    
    # set PyTorch default dtype to float64 to match numpy dtype
    torch.set_default_dtype(torch.float64)

    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d %H:%M:%S")
    print(f'Start VI at {current_time}...\n')
    print(f'Project basepath is {args.basepath}')
    print(f'Output folder is basepath/{args.outdir}\n')

    if args.nflow == 1:
        print(f'Transform (flow) used: {args.nflow} {args.flow} flow')
    else:
        print(f'Transform (flow) used: {args.nflow} {args.flow} flows')
    if args.flow == 'Linear':
        print(f'Covariance of Gaussian kernel is: {args.kernel}')
        if args.kernel == 'structured':
            print(f'Structured Gaussian kernel size is: {args.kernel_size}')
    print(f'The initial distribution is {args.ini_dist} distribution\n')

    ## define FMM config
    if args.datatype == 'traveltime':
        print(f'Config file for 2D FMM is: basepath/input/{args.config}')
        fmm_config_name = args.basepath + 'input/' + args.config
        config = configparser.ConfigParser()
        config._interpolation = configparser.ExtendedInterpolation()
        config.read(fmm_config_name)

    # create output folder
    Path(args.basepath + args.outdir).mkdir(parents=True, exist_ok=True)


    # load observed data
    if args.datatype == 'traveltime':
        print('Use surface wave travel time data for inversion\n')

        periods = np.array([4, 6, 8, 9, 10, 11, 12, 15])
        nperiods = len(periods)

        ## define FMM grids
        ny = config.getint('FMM','ny')
        nx = config.getint('FMM','nx')

        # define src, rec and mask
        src = np.loadtxt(args.basepath + 'input/' + args.srcfile)
        rec = np.loadtxt(args.basepath + 'input/' + args.recfile)

        ## define mask for geomotry - mimic inter-receiver interferometry
        mask = np.zeros((2,len(src)*len(rec)),dtype=np.int32)
        for i in range(len(src)):
            for j in range(len(rec)):
                if(j>i):
                    mask[0,i*len(src)+j] = 1
                    mask[1,i*len(src)+j] = i*len(src) + j + 1
        mask = np.tile(mask, (nperiods, 1))

        data_obs = np.zeros([nperiods, len(src)*len(rec)])
        sigma = np.zeros([nperiods, len(src)*len(rec)])
        
        for i in range(nperiods):
            time = np.loadtxt(args.basepath + 'input/' + args.datafile + f'_{periods[i]}s.dat')
            mask[i*2, :] = time[:,0].astype(np.int32)
            data_obs[i, :] = time[:,1]
            w = np.where(mask[i*2, :] == 0)[0]
            data_obs[i, w] = 0.
            sigma[i] = time[:,2] * args.sigma_scale
        mask = mask.reshape(nperiods, len(src)*len(rec)*2)
        print(data_obs.shape, sigma.shape)


    elif args.datatype == 'dispersion':
        print('Use surface wave dispersion curves for inversion\n')
        with open(args.basepath + 'input/' + args.datafile) as f:
            nx, ny, nperiods = np.loadtxt(f, max_rows = 1).astype('int')
            periods = np.loadtxt(f, max_rows = 1)
            data_obs = np.loadtxt(f, max_rows = nx * ny)
            sigma = np.loadtxt(f) / args.sigma_scale

    else:
        raise ValueError('Invalid data type, use either traveltime or dispersion')


    # define Bayesian prior and posterior pdf
    prior_bounds = np.loadtxt(args.basepath + 'input/' + args.prior_param)
    thickness = prior_bounds[:,0]
    nz = len(thickness)

    lower = prior_bounds[:,1].astype(np.float64)
    upper = prior_bounds[:,2].astype(np.float64)
    lower = np.broadcast_to(lower[None, None, :],(nx, ny, nz)).flatten()
    upper = np.broadcast_to(upper[None, None, :],(nx, ny, nz)).flatten()
    ndim = nx * ny * nz

    print(f'Shear wave velocity model size: nx = {nx}, ny = {ny}, nz = {nz}')
    print(f'Dimensionality of the problem: {ndim} ')

    # define smooth matrix for smooth prior information
    if args.smooth:
        L = smooth_matrix(nx, ny, nz, args.smoothx, args.smoothy, args.smoothz)
    else:
        L = None
    print(f'Smoothed prior information: {args.smooth}')

    # define Prior and Posterior pdf

    if args.prior_type == 'Uniform':
        prior = Uniform(lower = lower, upper = upper, smooth_matrix = L, top_layer_constraint = args.v0_std, nx = nx, ny = ny, nz = nz)
    elif args.prior_type == 'Normal':
        # This requires to have a loc (mean) vector and one parameter for covariance
        prior = Normal(loc = lower, std = upper)
    else:
        raise NotImplementedError("Not supported Prior distribution")
    print(f'Prior distribution is: {args.prior_type}')

    if args.datatype == 'traveltime':
        posterior = Posterior3D_tt(data_obs, config, src, rec, mask, thickness, periods, sigma = sigma, log_prior = prior.log_prob, 
                                    num_processes = args.prcs, relative_step_grad = 0.001, wave = 'love', mode = 1, velocity = 'group',
                                    lower = lower, upper = upper)
    else:
        posterior = Posterior3D_dc(data_obs.flatten(), thickness, periods, sigma = sigma.flatten(), num_processes = args.prcs, 
                                    relative_step_grad = 0.001, log_prior = prior.log_prob, wave = 'love', mode = 1, velocity = 'group')

    # define flows model
    flow = eval(args.flow)
    param = None
    if args.flow_init_name != 'none':
        # load initial value for flows
        filename = os.path.join(args.basepath, 'input/', args.flow_init_name)
        param = np.load(filename).flatten()
        print(f'Load basepath/input/{args.flow_init_name} as initial parameter value for flows model')
    if args.flow == 'Linear':
        cov_template = np.ones((args.kernel_size, args.kernel_size, args.kernel_size))
        off_diag_mask = get_offdiag_mask(cov_template, ndim, nx = nx, ny = ny, nz = nz)
        flows = [flow(dim = ndim, kernel = args.kernel, mask = off_diag_mask, param = param)
                    for _ in range(args.nflow)]

    # if the initial distribution of flow model is a Uniform distribution, 
    # then add a flow to transform from constrained to real space
    if args.ini_dist == 'Uniform':
        flows.insert(0, Constr2Real(dim = ndim, lower = 0, upper = 1))
    flows.append(Real2Constr(dim = ndim, lower = lower, upper = upper))
    variational = FlowsBasedDistribution(flows, base = args.ini_dist)

    # define VI class to perform inversion
    inversion = VariationalInversion(variationalDistribution = variational, log_posterior = posterior.log_prob)

    # optimizer = optim.Adam(variational.parameters(), lr = args.lr)
    print(f"Number of hyperparameters is: {sum(p.numel() for p in variational.parameters())}", )
    print(f'Optimising variational model for {args.iterations} iterations with {args.nsample} samples per iteration\n')


    loss_his = []
    start_ite = 0
    # if start_ite != 0, we load the previously saved model checkpoint and resume training
    if args.resume:
        name = os.path.join(args.basepath, args.outdir, f'{args.flow}_{args.kernel}_model.pt')
        try:
            checkpoint = torch.load(name)
        except:
            print('Invalid name for model checkpoint!')
        start_ite = checkpoint['iteration']
        variational.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint.keys():
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        loss_his = checkpoint['loss']
        print(f'Resume training from previous run at iteration {start_ite:4d}\n')
    else:
        print(f'Start training at iteration {start_ite}\n')

    # Perform variational inversion
    loss_his.extend(
                    inversion.update(optimizer = 'torch.optim.Adam', lr = args.lr, n_iter = args.iterations, nsample = args.nsample, 
                                n_out = args.nout, verbose = args.verbose, save_intermediate_result = args.save_intermediate_result, 
                                outpath = os.path.join(args.basepath, args.outdir))
                    )

    variational.eval()
    samples = variational.sample(3000)
    name = os.path.join(args.basepath, args.outdir, f'{args.flow}_{args.kernel}_samples.npy')
    np.save(name, samples)

    param = get_flow_param(variational.flows[-2])
    name = os.path.join(args.basepath, args.outdir, f'{args.flow}_{args.kernel}_parameter.npy')
    np.save(name, param)

    name = os.path.join(args.basepath, args.outdir, f'{args.flow}_{args.kernel}_loss.txt')
    np.savetxt(name, loss_his)

    name = os.path.join(args.basepath, args.outdir, f'{args.flow}_{args.kernel}_model.pt')
    torch.save({
                'iteration': (len(loss_his)),
                'model_state_dict': variational.state_dict(),
                'loss': loss_his,
                }, name)
