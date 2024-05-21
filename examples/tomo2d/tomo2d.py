import numpy as np
import torch
import torch.optim as optim

import os
from pathlib import Path
from argparse import ArgumentParser
import configparser
import time
from datetime import datetime

from geopvi.vi.models import VariationalModel
from geopvi.vi.flows import *
from geopvi.tomo2d.posterior import Posterior 
from geopvi.prior import Uniform, Normal


def gen_sample(n = 512, dim = 1, para1 = 0., para2 = 1., ini = 'Normal'):
    """
    The initial distribution: q_0(z_0)
    (Often a known simple distribution, e.g., a Uniform or Standard Gaussian)
    """
    if ini == 'Normal':
        return np.random.normal(0., 1., size = (n, dim))
    if ini == 'Uniform':
        eps = np.finfo(np.float32).eps
        return np.random.uniform(para1 + eps, para2 - eps, size = (n, dim))

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
    argparser = ArgumentParser(description='2D travel time tomography using GeoVI')
    argparser.add_argument("--basepath", metavar='basepath', type=str, help='Project path',
                            default='/lustre03/other/2029iw/study/00_GeoPVI/examples/tomo2d/')

    argparser.add_argument("--flow", default='Linear', type=str)
    argparser.add_argument("--kernel", default='fullrank', type=str)
    argparser.add_argument("--nflow", default=1, type=int)
    argparser.add_argument("--nsample", default=10, type=int)
    argparser.add_argument("--prcs", default=10, type=int)
    argparser.add_argument("--iterations", default=5000, type=int)
    argparser.add_argument("--lr", default=0.001, type=float)
    argparser.add_argument("--ini_dist", default='Normal', type=str)
    argparser.add_argument("--sigma", default=0.05, type=float)

    argparser.add_argument("--smooth", default=False, type=bool)
    argparser.add_argument("--smoothx", default=500, type=float)
    argparser.add_argument("--smoothy", default=500, type=float)

    argparser.add_argument("--prior_type", default='Uniform', type=str)
    argparser.add_argument("--prior_param", default='prior.txt', type=str)
    argparser.add_argument("--fmm_config", metavar='fmm_config', default='config.ini', type=str)
    argparser.add_argument("--datafile", metavar='data_obs', default='traveltime.txt', type=str)
    argparser.add_argument("--outdir", type=str, default='output/', help='Folder for inversion results')

    argparser.add_argument("--verbose", default=True, type=bool, help='Output intermediate results')
    argparser.add_argument("--save_intermediate_result", default=False, type=bool,
                                help='Whether save intermediate training model, for resume from previous training')
    argparser.add_argument("--resume", default=False, type=bool, help='Resume previous training')
    argparser.add_argument("--output_interval", default=1000, type=int, help='frequency for output model parameters')


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
    angle = np.arange(0., 2.*np.pi, np.pi/8)
    src = np.array([[4.*np.sin(x), 4.*np.cos(x)] for x in angle])
    rec = src
    srcx = np.ascontiguousarray(src[:,0])
    srcy = np.ascontiguousarray(src[:,1])
    recx = np.ascontiguousarray(rec[:,0])
    recy = np.ascontiguousarray(rec[:,1])

    ## define mask for geomotry
    mask = np.zeros((2,len(srcx)*len(recx)),dtype=np.int32)
    for i in range(len(srcx)):
        for j in range(len(recx)):
            if(j>i):
                mask[0,i*len(srcx)+j] = 1
                mask[1,i*len(srcx)+j] = i*len(srcx) + j + 1

    data_obs = np.loadtxt(args.basepath + 'input/' + args.datafile)
    data_obs = data_obs.flatten()

    # define Bayesian prior and posterior pdf
    prior_bounds = np.loadtxt(args.basepath + 'input/' + args.prior_param)
    lower = prior_bounds[:,0].astype(np.float64)
    upper = prior_bounds[:,1].astype(np.float64)
    
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
    posterior = Posterior(data_obs, config, src, rec, mask = mask, sigma = args.sigma, 
                                log_prior = prior.log_prob, num_processes = args.prcs)

    # define flows model
    flow = eval(args.flow)
    flows = [flow(dim = ndim, kernel = args.kernel) for _ in range(args.nflow)]

    # if the initial distribution of flow model is a Uniform distribution, 
    # then add a flow to transform from constrained to real space
    if args.ini_dist == 'Uniform':
        flows.insert(0, Constr2Real(lower = 0, upper = 1))
    flows.append(Real2Constr(lower = lower, upper = upper))

    model = VariationalModel(flows)

    optimizer = optim.Adam(model.parameters(), lr = args.lr)
    loss_his = []

    print(f"Number of flow parameters is: {sum(p.numel() for p in model.parameters())}", )
    print(f'Optimising ADVI for {args.iterations} iterations with {args.nsample} samples per iteration\n')

    start = time.time()

    start_ite = 0
    # if start_ite != 0, we load the previously saved model checkpoint and resume training
    if args.resume:
        name = os.path.join(args.basepath, args.outdir, f'{args.flow}_{args.kernel}_model.pt')
        try:
            checkpoint = torch.load(name)
        except:
            print('Invalid name for model checkpoint!')
        start_ite = checkpoint['iteration']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        loss_his = checkpoint['loss']
        print(f'Resume training from previous run at iteration {start_ite:4d}\n')
    else:
        print(f'Start training at iteration {start_ite}\n')
    print('----------------------------------------')

    for i in range(start_ite, args.iterations):
        optimizer.zero_grad()
        # model.train()
        x = torch.as_tensor(gen_sample(args.nsample, ndim, para1 = lower, para2 = upper, ini=args.ini_dist))
        z, log_det = model(x)
        logp = posterior.log_prob(z)

        loss = -torch.mean(logp + log_det) # mean: Expectation term using Monte Carlo
        loss.backward()
        optimizer.step()
        loss_his.append(loss.data.numpy())

        if i % args.output_interval == 0 and args.verbose:
            # save intermediate normalising flows model
            if args.save_intermediate_result:
                param = get_flow_param(model.flows[-2])
                name = os.path.join(args.basepath, args.outdir, f'{args.flow}_{args.kernel}_ite{i}_parameter.npy')
                np.save(name, param)

                name = os.path.join(args.basepath, args.outdir, f'{args.flow}_{args.kernel}_loss.txt')
                np.savetxt(name, loss_his)
                name = os.path.join(args.basepath, args.outdir, f'{args.flow}_{args.kernel}_model.pt')
                torch.save({
                            'iteration': i,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': loss_his,
                            }, name)
                
            print(f'Iteration: {i:>5d},\tLoss: {loss.data:>10.2f}')
            end = time.time()
            print(f'The elapsed time is: {end-start:.2f} s')

    print(f'Iteration: {args.iterations:>5d},\tLoss: {loss.data:>10.2f}')
    end = time.time()
    print(f'The elapsed time is: {end-start:.2f} s')
    print('----------------------------------------\n')
    
    print('Finish training!')

    param = get_flow_param(model.flows[-2])
    name = os.path.join(args.basepath, args.outdir, f'{args.flow}_{args.kernel}_parameter.npy')
    np.save(name, param)

    name = os.path.join(args.basepath, args.outdir, f'{args.flow}_{args.kernel}_loss.txt')
    np.savetxt(name, loss_his)

    name = os.path.join(args.basepath, args.outdir, f'{args.flow}_{args.kernel}_model.pt')
    torch.save({
                'iteration': i,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_his,
                }, name)
