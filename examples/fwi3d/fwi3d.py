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

from geopvi.prior import Uniform, Normal
from geopvi.vi.models import VariationalModel
from geopvi.vi.flows import *
from geopvi.fwi3d.posterior import Posterior
import geopvi.fwi3d.dask_utils as du 


def delta(n):
    diag0 = np.full((n,),fill_value=-2); diag0[0]=-1; diag0[-1]=-1
    diag1 = np.full((n-1,),fill_value=1)
    diagonals = [diag0,diag1,diag1]
    D = sparse.diags(diagonals,[0,-1,1]).tocsc()
    return D
    
def smooth_matrix(nx, ny, nz, smoothx, smoothy, smoothz):
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

def init_vfwi(args, config):
    Path(args.basepath + args.outdir).mkdir(parents=True, exist_ok=True)
    Path(config.get('path', 'inpath')).mkdir(parents=True, exist_ok=True)
    Path(config.get('path', 'outpath')).mkdir(parents=True, exist_ok=True)
    Path(config.get('dask', 'daskpath')).mkdir(parents=True, exist_ok=True)

def get_offdiag_mask(correlation, ndim, nx = 1, ny = 1, nz = 1):
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

def gen_sample(n = 1, dim = 1, para1 = 0., para2 = 1., ini = 'Normal'):
    """
    The initial distribution: q_0(z_0)
    (Often a known simple and analytically known distribution, like Uniform or Standard Gaussian)
    """
    if ini == 'Normal':
        return np.random.normal(0., 1., size = (n, dim))
    if ini == 'Uniform':
        eps = np.finfo(np.float32).eps
        return np.random.uniform(para1 + eps, para2 - eps, size = (n, dim))

def get_flow_param(flow):
    mus = flow.u.detach().numpy()
    sigmas = np.exp(flow.diag.detach().numpy())
    if flow.non_diag is None:
        return np.hstack([mus, sigmas])
    else:
        non_diag = flow.non_diag.detach().numpy().flatten()
        return np.hstack([mus, sigmas, non_diag])


if __name__ == "__main__":
    argparser = ArgumentParser(description='3D Bayesian Full waveform Inversion using GeoVI')
    argparser.add_argument("--basepath", metavar='basepath', type=str, help='Project path',
                            default='/home/user/GeoPVI/examples/fwi3d/')

    argparser.add_argument("--flow", default='Linear', type=str)
    argparser.add_argument("--kernel", default='structured', type=str)
    argparser.add_argument("--kernel_size", default=5, type=int)
    argparser.add_argument("--nflow", default=1, type=int)
    argparser.add_argument("--nsample", default=5, type=int)
    argparser.add_argument("--iterations", default=1000, type=int)
    argparser.add_argument("--lr", default=0.005, type=float)
    argparser.add_argument("--ini_dist", default='Normal', type=str)
    argparser.add_argument("--sigma", default=2e-7, type=float)

    argparser.add_argument("--smooth", default=False, type=bool)
    argparser.add_argument("--smoothx", default=1000, type=float)
    argparser.add_argument("--smoothy", default=1000, type=float)
    argparser.add_argument("--smoothz", default=1000, type=float)

    argparser.add_argument("--prior_type", default='Uniform', type=str)
    argparser.add_argument("--prior_param", default='Uniform_prior.txt', type=str)
    argparser.add_argument("--fwi_config", default='config.ini', type=str, help='configure file for FWI')
    argparser.add_argument("--flow_init_name", type=str, default='none', 
                                help='Parameter filename for flow initial value')
    argparser.add_argument("--outdir", type=str, default='output/', 
                                help='Folder for inversion results')
    
    argparser.add_argument("--verbose", default=True, type=bool, help='Output intermediate results')
    argparser.add_argument("--save_intermediate_result", default=True, type=bool,
                                help='Whether save intermediate training model, for resume from previous training')
    argparser.add_argument("--resume", default=False, type=bool, help='Resume previous training')
    argparser.add_argument("--output_interval", default=100, type=int, help='frequency for output model parameters')


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

    init_vfwi(args, config)

    ## define FWI parameters
    nx = config.getint('FWI','nx')
    ny = config.getint('FWI','ny')
    nz = config.getint('FWI','nz')
    water_layer = config.getint('FWI','water_layer')
    ndim = (nz - water_layer) * nx * ny

    print(f'FWI model size: ny = {ny}, nx = {nx}, nz = {nz}')
    print(f'Dimensionality of the problem: {ndim} ')
    print(f'Data noise is: {args.sigma}\n')

    # masked is defined to fix velocity values at their true values within the water layer 
    mask = np.full([ny, nx, nz], True)
    mask[:,:,:water_layer] = False

    # define dask cluster for parallelisation computation
    daskpath = config.get('dask','daskpath')
    print(f'Create dask cluster at: {daskpath}\n')
    cluster, client = du.dask_init(config.get('dask','pe'), config.getint('dask','nnodes'),
                                   nworkers=config.getint('dask','nworkers'),
                                   ph=config.getint('dask','ph'), odask=daskpath)

    # define Bayesian prior and posterior pdf
    prior_bounds = np.loadtxt(args.basepath + 'input/' + args.prior_param)
    lower = prior_bounds[water_layer:,0].astype(np.float64)
    upper = prior_bounds[water_layer:,1].astype(np.float64)
    lower = np.broadcast_to(lower[None, None, :],(ny, nx, (nz - water_layer))).flatten()
    upper = np.broadcast_to(upper[None, None, :],(ny, nx, (nz - water_layer))).flatten()

    # define smooth matrix for smooth prior information
    if args.smooth:
        L = smooth_matrix(nx, ny, nz - water_layer, args.smoothx, args.smoothy, args.smoothz)
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
    posterior = Posterior(args, config, log_prior = prior.log_prob, mask = mask.flatten(), client = client)

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
        off_diag_mask = get_offdiag_mask(cov_template, ndim, nx = nx, ny = ny, nz = nz - water_layer)
        flows = [flow(dim = ndim, kernel = args.kernel, mask = off_diag_mask, param = param)
                    for _ in range(args.nflow)]

    # if the initial distribution of flow model is a Uniform distribution, 
    # then add a flow to transform from constrained to real space
    if args.ini_dist == 'Uniform':
        flows.insert(0, Constr2Real(lower = lower, upper = upper))
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
            # Save intermediate model parameters
            param = get_flow_param(model.flows[-2])
            name = os.path.join(args.basepath, args.outdir, f'{args.flow}_{args.kernel}_ite{i}_parameter.npy')
            np.save(name, param)

            name = os.path.join(args.basepath, args.outdir, f'{args.flow}_{args.kernel}_loss.txt')
            np.savetxt(name, loss_his)

            # # If you want to get posterior samples and save them, you can use the following:
            # x = torch.as_tensor(gen_sample(2000, ndim, para1 = lower, para2 = upper, ini=args.ini_dist))
            # z = model.sample(x)
            # z = z.data.numpy()
            # name = os.path.join(args.basepath, args.outdir, f'{args.flow}_{args.kernel}_ite{i}_sample.npy')
            # np.save(name, z)

            # save intermediate normalising flows model
            if args.save_intermediate_result:
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

    du.dask_del(cluster, client, odask=daskpath)

    # # save posterior samples from trained model
    # x = torch.as_tensor(gen_sample(2000, ndim, para1 = lower, para2 = upper, ini=args.ini_dist))
    # z = model.sample(x)
    # z = z.data.numpy()
    # name = os.path.join(args.basepath, args.outdir, f'{args.flow}_{args.kernel}_sample.npy')
    # np.save(name, z)
