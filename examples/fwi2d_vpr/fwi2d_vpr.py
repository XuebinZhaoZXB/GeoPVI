import numpy as np
import torch

import os
from pathlib import Path
from argparse import ArgumentParser
import configparser
import time
from datetime import datetime

from geopvi.nfvi.models import FlowsBasedDistribution, VariationalInversion
from geopvi.nfvi.flows import *
from geopvi.prior import Uniform, Normal
from posterior import Posterior 
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
    argparser = ArgumentParser(description='Variational prior replacement for 2D Bayesian FWI using GeoPVI')
    argparser.add_argument("--basepath", metavar='basepath', type=str, help='Project path', default='./')

    argparser.add_argument("--flow", default='Linear', type=str, help='Flows used to perform inversion')
    argparser.add_argument("--kernel", default='diagonal', type=str, help='Covariance kernel type if Linear flow is used')
    argparser.add_argument("--kernel_size", default=5, type=int, help='Local covariance kernel size if PSVI is performed')
    argparser.add_argument("--nflow", default=1, type=int, help='number of flows')
    argparser.add_argument("--nsample", default=10, type=int, help='number of samples for MC integration during each iteration')
    argparser.add_argument("--prcs", default=10, type=int, help='number of processes in parallel to perform forward evaluation')
    argparser.add_argument("--iterations", default=5000, type=int, help='number of iterations to update variational parameters')
    argparser.add_argument("--lr", default=0.002, type=float, help='learning rate')
    argparser.add_argument("--ini_dist", default='Normal', type=str, help='initial (base) distribution for flows-based model')
    argparser.add_argument("--sigma", default=0.1, type=float, help='data noise level')
    
    argparser.add_argument("--smooth", default=True, type=bool, help='Whether to apply smooth factor on model vector m')
    argparser.add_argument("--smoothx", default=500, type=float, help='Smoothness parameter, smaller value means stronger smoothness')
    argparser.add_argument("--smoothz", default=500, type=float, help='Smoothness parameter, smaller value means stronger smoothness')

    argparser.add_argument("--prior_type", default='Uniform', type=str, help='Prior pdf - either Uniform or Normal, or user-defined')
    argparser.add_argument("--prior_param", default='prior.txt', type=str, help='filename containing hyperparametes to define prior pdf')
    argparser.add_argument("--fwi_config", default='config.ini', type=str, help='filename containing parameters for forward simulation')
    argparser.add_argument("--flow_init_name", type=str, default='none', help='Parameter filename for flow initial value')
    argparser.add_argument("--outdir", type=str, default='output/', help='folder path (relative to basepath) for output files')

    argparser.add_argument("--postfix", default='', type=str)
    argparser.add_argument("--verbose", default=True, type=bool, help='Output intermediate results')
    argparser.add_argument("--save_intermediate_result", default=False, type=bool,
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
    elif args.prior_type == 'Normal':
        # This requires to have a loc (mean) vector and one parameter for covariance
        prior = Normal(loc = (lower + upper) / 2, std = np.sqrt((upper - lower)**2 / 12), smooth_matrix = L)
    else:
        raise NotImplementedError("Not supported Prior distribution")

    # load old posterior pdf file to define the new posterior pdf using VPR equation
    name = args.basepath + 'input/Linear_diagonal_parameter.npy'
    param = np.load(name)
    posterior = Posterior(param, lower, upper, log_prior = prior.log_prob, 
                                covariance = args.kernel, cov_inverse = True)

    # define flows model
    # if args.kernel = 'structured':
    cov_template = np.ones((args.kernel_size, args.kernel_size))
    off_diag_mask = get_offdiag_mask(cov_template, ndim, nx = nx, nz = nz - water_layer)
    flow = eval(args.flow)
    flows = [flow(dim = ndim, kernel = args.kernel, mask = off_diag_mask, param = None)
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
    # Perform VPR
    loss_his.extend(
                    inversion.update(optimizer = 'torch.optim.Adam', lr = args.lr, n_iter = args.iterations, nsample = args.nsample, 
                                n_out = args.nout, verbose = args.verbose, save_intermediate_result = args.save_intermediate_result)
                    )

    param = get_flow_param(variational.flows[-2])
    name = os.path.join(args.basepath, args.outdir, f'{args.flow}_{args.kernel}_{args.postfix}parameter.npy')
    np.save(name, param)

    name = os.path.join(args.basepath, args.outdir, f'{args.flow}_{args.kernel}_{args.postfix}loss.txt')
    np.savetxt(name, loss_his)

    name = os.path.join(args.basepath, args.outdir, f'{args.flow}_{args.kernel}_{args.postfix}model.pt')
    torch.save({
                'iteration': (len(loss_his)),
                'model_state_dict': variational.state_dict(),
                'loss': loss_his,
                }, name)