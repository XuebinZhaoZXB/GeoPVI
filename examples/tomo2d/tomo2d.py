import numpy as np
import torch

import os
from pathlib import Path
from argparse import ArgumentParser
import configparser
import time
from datetime import datetime

from geopvi.nfvi.models import FlowsBasedDistribution, VariationalInversion
from geopvi.nfvi.flows import Real2Constr, Constr2Real, Linear, NSF_CL, Permute
from geopvi.forward.tomo2d.posterior import Posterior 
from geopvi.prior import Uniform, Normal


if __name__ == "__main__":
    argparser = ArgumentParser(description='2D travel time tomography using GeoPVI')
    argparser.add_argument("--basepath", metavar='basepath', type=str, help='Project path', default='./')

    argparser.add_argument("--flow", default='NSF_CL', type=str, help='Flows used to perform inversion')
    argparser.add_argument("--kernel", default='fullrank', type=str, help='Covariance kernel type if Linear flow is used')
    argparser.add_argument("--nflow", default=6, type=int, help='number of flows')
    argparser.add_argument("--nsample", default=10, type=int, help='number of samples for MC integration during each iteration')
    argparser.add_argument("--prcs", default=10, type=int, help='number of processes in parallel to perform forward evaluation')
    argparser.add_argument("--iterations", default=3000, type=int, help='number of iterations to update variational parameters')
    argparser.add_argument("--lr", default=0.001, type=float, help='learning rate')
    argparser.add_argument("--ini_dist", default='Normal', type=str, help='initial (base) distribution for flows-based model')
    argparser.add_argument("--sigma", default=0.05, type=float, help='data noise level')

    argparser.add_argument("--prior_type", default='Uniform', type=str, help='Prior pdf - either Uniform or Normal, or user-defined')
    argparser.add_argument("--prior_param", default='prior.txt', type=str, help='filename containing hyperparametes to define prior pdf')
    argparser.add_argument("--fmm_config", default='config.ini', type=str, help='filename containing parameters for forward simulation')
    argparser.add_argument("--srcfile", default='sources.txt', type=str, help='filename for (x, y) source locations')
    argparser.add_argument("--recfile", default='sources.txt', type=str, help='filename for (x, y) receiver locations, same as sources to mimic inter-receiver interferometry')
    argparser.add_argument("--datafile", default='traveltime.txt', type=str, help='filename for observed dataset')
    argparser.add_argument("--outdir", type=str, default='output/', help='folder path (relative to basepath) for output files')

    argparser.add_argument("--verbose", default=True, type=bool, help='Output and print intermediate inversion results')
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
    print(f'The initial distribution is {args.ini_dist}\n')

    ## define FWI config
    print(f'Config file for 2D FMM is: basepath/input/{args.fmm_config}')
    fmm_config_name = args.basepath + 'input/' + args.fmm_config
    config = configparser.ConfigParser()
    config._interpolation = configparser.ExtendedInterpolation()
    config.read(fmm_config_name)

    # create output folder
    Path(args.basepath + args.outdir).mkdir(parents=True, exist_ok=True)

    ## define FWI parameters
    ny = config.getint('FMM','ny')
    nx = config.getint('FMM','nx')
    ndim = ny * nx

    print(f'FMM model size: nx = {nx}, ny = {ny}')
    print(f'Dimensionality of the problem: {ndim} ')
    print(f'Data noise is: {args.sigma}\n')

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

    data_obs = np.loadtxt(args.basepath + 'input/' + args.datafile)
    data_obs = data_obs.flatten()

    # define Bayesian prior and posterior pdf
    prior_bounds = np.loadtxt(args.basepath + 'input/' + args.prior_param)
    lower = prior_bounds[:,0].astype(np.float64)
    upper = prior_bounds[:,1].astype(np.float64)

    # define Prior and Posterior pdf
    if args.prior_type == 'Uniform':
        prior = Uniform(lower = lower, upper = upper)
    elif args.prior_type == 'Normal':
        # This requires to have a loc (mean) vector and one parameter for covariance
        prior = Normal(loc = lower, std = upper)
    else:
        raise NotImplementedError("Not supported Prior distribution")
    print(f'Prior distribution is: {args.prior_type}')
    posterior = Posterior(data_obs, config, src, rec, mask = mask, sigma = args.sigma, 
                                log_prior = prior.log_prob, num_processes = args.prcs)

    # define flows model
    flow = eval(args.flow)
    # flows = [flow(dim = ndim, kernel = args.kernel) for _ in range(args.nflow)]
    # flows = [Linear(dim = ndim, param = np.hstack([np.zeros(ndim), np.full((ndim,), 1.6)]), trainable = False)]
    flows = []
    for i in range(args.nflow):
        flows += [Permute(dim = ndim), flow(dim = ndim, K = 8, B = 3, hidden_dim = [64, 128])]

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
        name = os.path.join(args.basepath, args.outdir, f'{args.flow}_model.pt')
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
                            save_intermediate_result = args.save_intermediate_result, n_out = args.nout, verbose = args.verbose)
                    )

    variational.eval()
    samples = variational.sample(3000)
    name = os.path.join(args.basepath, args.outdir, f'{args.flow}_samples.npy')
    np.save(name, samples)

    name = os.path.join(args.basepath, args.outdir, f'{args.flow}_loss.txt')
    np.savetxt(name, loss_his)

    name = os.path.join(args.basepath, args.outdir, f'{args.flow}_model.pt')
    torch.save({
                'iteration': (len(loss_his)),
                'model_state_dict': variational.state_dict(),
                'loss': loss_his,
                }, name)
