import numpy as np
import torch
import torch.optim as optim
import scipy.sparse as sparse

import os
from pathlib import Path
from argparse import ArgumentParser
import configparser
import time
from datetime import datetime

from geopvi.vi.models import VariationalDistribution, VariationalInversion
from geopvi.vi.flows import *
from geopvi.fwi2d.posterior import Posterior 
from geopvi.prior import Uniform, Normal
from geopvi.utils import smooth_matrix_2D as smooth_matrix


def get_offdiag_mask(correlation, ndim, nx = 1, nz = 1):
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


def get_flow_param(flow):
    mus = flow.u.detach().numpy()
    # sigmas = np.log(np.exp(flow.diag.detach().numpy()) + 1)
    sigmas = np.exp(flow.diag.detach().numpy())
    if flow.non_diag is None:
        return np.hstack([mus, sigmas])
    else:
        non_diag = flow.non_diag.detach().numpy().flatten()
        return np.hstack([mus, sigmas, non_diag])


if __name__ == "__main__":
    argparser = ArgumentParser(description='2D Bayesian Full waveform Inversion using GeoPVI')
    argparser.add_argument("--basepath", metavar='basepath', type=str, help='Project path',
                            default='/lustre03/other/2029iw/study/00_GeoPVI/examples/fwi2d/')

    argparser.add_argument("--flow", default='Linear', type=str)
    argparser.add_argument("--kernel", default='structured', type=str)
    argparser.add_argument("--kernel_size", default=5, type=int)
    argparser.add_argument("--nflow", default=1, type=int)
    argparser.add_argument("--nsample", default=2, type=int)
    argparser.add_argument("--prcs", default=2, type=int)
    argparser.add_argument("--iterations", default=5000, type=int)
    argparser.add_argument("--lr", default=0.001, type=float)
    argparser.add_argument("--ini_dist", default='Normal', type=str)
    argparser.add_argument("--sigma", default=0.1, type=float)

    argparser.add_argument("--smooth", default=False, type=bool)
    argparser.add_argument("--smoothx", default=500, type=float)
    argparser.add_argument("--smoothz", default=500, type=float)

    argparser.add_argument("--prior_type", default='Uniform', type=str)
    argparser.add_argument("--prior_param", default='prior.txt', type=str)
    argparser.add_argument("--fwi_config", metavar='fwi_config', default='config.ini', type=str)
    argparser.add_argument("--datafile", metavar='data_obs', default='waveform.npy', type=str)
    argparser.add_argument("--flow_init_name", type=str, default='none', 
                                help='Parameter filename for flow initial value')
    argparser.add_argument("--outdir", type=str, default='output/', 
                                help='Folder for inversion results')

    argparser.add_argument("--verbose", default=True, type=bool, help='Output intermediate results')
    argparser.add_argument("--save_intermediate_result", default=True, type=bool,
                                help='Whether save intermediate training model, for resume from previous training')
    argparser.add_argument("--resume", default=False, type=bool, help='Resume previous training')
    argparser.add_argument("--nout", default=5, type=int, help='Number to print/output intermediate inversion results')


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

    ## define FWI config
    print(f'Config file for FWI is: basepath/input/{args.fwi_config}')
    fwi_config_name = args.basepath + 'input/' + args.fwi_config
    config = configparser.ConfigParser()
    config._interpolation = configparser.ExtendedInterpolation()
    config.read(fwi_config_name)

    # create output folder
    Path(args.basepath + args.outdir).mkdir(parents=True, exist_ok=True)

    ## define FWI parameters
    nz = config.getint('FWI','nz')
    nx = config.getint('FWI','nx')
    water_layer = config.getint('FWI','layer_fixed')
    vel_water = config.getfloat('FWI','vel_fixed')
    ndim = (nz - water_layer) * nx

    print(f'FWI model size: nx = {nx}, nz = {nz}')
    print(f'Dimensionality of the problem: {ndim} ')
    print(f'Data noise is: {args.sigma}\n')

    # masked is defined to fix velocity values at their true values within the water layer 
    mask = np.full([nz, nx], True)
    mask[:water_layer] = False

    data_obs = np.load(args.basepath + 'input/' + args.datafile)
    data_obs = data_obs.flatten()

    # define Bayesian prior and posterior pdf
    prior_bounds = np.loadtxt(args.basepath + 'input/' + args.prior_param)
    lower = prior_bounds[water_layer:,0].astype(np.float64)
    upper = prior_bounds[water_layer:,1].astype(np.float64)
    lower = np.broadcast_to(lower[:,None],((nz - water_layer),nx)).flatten()
    upper = np.broadcast_to(upper[:,None],((nz - water_layer),nx)).flatten()
    
    # define smooth matrix for smooth prior information
    if args.smooth:
        L = smooth_matrix(nx, nz - water_layer, args.smoothx, args.smoothz)
    else:
        L = None
    print(f'Smoothed prior information: {args.smooth}')

    # define Prior and Posterior pdf
    if args.prior_type == 'Uniform':
        prior = Uniform(lower = lower, upper = upper, smooth_matrix = L)
    # elif args.prior_type == 'Normal':
        # This requires to have a loc (mean) vector and one parameter for covariance
        # prior = Normal()
    else:
        raise NotImplementedError("Not supported Prior distribution")
    print(f'Prior distribution is: {args.prior_type}')
    posterior = Posterior(data_obs, args, config, log_prior = prior.log_prob, mask = mask.flatten())

    # define flows model
    flow = eval(args.flow)
    param = None    
    if args.flow_init_name != 'none':
        # load initial value for flows
        filename = os.path.join(args.basepath, 'input/', args.flow_init_name)
        param = np.load(filename).flatten()
        print(f'Load basepath/input/{args.flow_init_name} as initial parameter value for flows model')
    if args.flow == 'Linear':
        cov_template = np.ones((args.kernel_size, args.kernel_size))
        off_diag_mask = get_offdiag_mask(cov_template, ndim, nx = nx, nz = nz - water_layer)
        flows = [flow(dim = ndim, kernel = args.kernel, mask = off_diag_mask, param = param)
                    for _ in range(args.nflow)]

    # if the initial distribution of flow model is a Uniform distribution, 
    # then add a flow to transform from constrained to real space
    if args.ini_dist == 'Uniform':
        flows.insert(0, Constr2Real(lower = 0, upper = 1))
    flows.append(Real2Constr(lower = lower, upper = upper))
    variational = VariationalDistribution(flows, base = args.ini_dist)

    # define VI class to perform inversion
    inversion = VariationalInversion(variationalDistribution = variational, log_posterior = posterior.log_prob)

    optimizer = optim.Adam(variational.parameters(), lr = args.lr)
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
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        loss_his = checkpoint['loss']
        print(f'Resume training from previous run at iteration {start_ite:4d}\n')
    else:
        print(f'Start training at iteration {start_ite}\n')

    # Perform variational inversion
    loss_his.append(
                    inversion.update(optimizer = optimizer, n_iter = args.iterations, nsample = args.nsample, n_out = args.nout, 
                                verbose = args.verbose, save_intermediate_result = args.save_intermediate_result)
                    )

    param = get_flow_param(variational.flows[-2])
    name = os.path.join(args.basepath, args.outdir, f'{args.flow}_{args.kernel}_parameter.npy')
    np.save(name, param)

    name = os.path.join(args.basepath, args.outdir, f'{args.flow}_{args.kernel}_loss.txt')
    np.savetxt(name, loss_his)

    name = os.path.join(args.basepath, args.outdir, f'{args.flow}_{args.kernel}_model.pt')
    torch.save({
                'iteration': (len(loss_his)),
                'model_state_dict': variational.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_his,
                }, name)
