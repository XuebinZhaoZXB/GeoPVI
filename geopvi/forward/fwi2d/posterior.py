import numpy as np
import torch
from torch.autograd import Function
from torch.multiprocessing import Pool

from geopvi.forward.fwi2d import aco2d

class ForwardModel(Function):
    @staticmethod
    def forward(ctx, input, func):
        output, grad = func(input)
        ctx.save_for_backward(input, torch.tensor(grad))
        return torch.tensor(output)

    @staticmethod
    def backward(ctx, grad_output):
        '''
        this function returns the gradient w.r.t the input tensor in the forward function
        therefore, the return shape should be the same as the shape of input tensor
        '''
        input, grad = ctx.saved_tensors
        # grad_input = (grad_output[...,None] * grad).sum(axis = -2)
        grad_input = (grad_output[...,None] * grad)
        return grad_input, None


def prepare_fwi_parameters(paramfile, config):
    with open(paramfile,'w') as f:
        f.write('--grid points in x direction (nx)\n')
        f.write(config.get('FWI','nx')+'\n')
        f.write('--grid points in z direction (nz)\n')
        f.write(config.get('FWI','nz')+'\n')
        f.write('--pml points (pml0)\n')
        f.write(config.get('FWI','pml')+'\n')
        f.write('--Finite difference order (Lc)\n')
        f.write(config.get('FWI','Lc')+'\n')
        f.write('--Method to calculate Laplace operator: 0 for FDM and 1 for PSM, (laplace_slover)\n')
        f.write(config.get('FWI','laplace_slover')+'\n')
        f.write('--Total number of sources (ns)\n')
        f.write(config.get('FWI','ns')+'\n')
        f.write('--Total time steps (nt)\n')
        f.write(config.get('FWI','nt')+'\n')
        f.write('--Shot interval in grid points (ds)\n')
        f.write(config.get('FWI','ds')+'\n')
        f.write('--Grid number of the first shot to the left of the model (ns0)\n')
        f.write(config.get('FWI','ns0')+'\n')
        f.write('--Depth of source in grid points (depths)\n')
        f.write(config.get('FWI','depths')+'\n')
        f.write('--Depth of receiver in grid points (depthr)\n')
        f.write(config.get('FWI','depthr')+'\n')
        f.write('--Total number of receivers (nr)\n')
        f.write(config.get('FWI','nr')+'\n')
        f.write('--Receiver interval in grid points (dr)\n')
        f.write(config.get('FWI','dr')+'\n')
        f.write('--Grid number of the first receiver to the left of the model (nr0)\n')
        f.write(config.get('FWI','nr0')+'\n')
        f.write('--Time step interval of saved wavefield during forward (nt_interval)\n')
        f.write(config.get('FWI','nt_interval')+'\n')
        f.write('--Grid spacing in x direction (dx)\n')
        f.write(config.get('FWI','dx')+'\n')
        f.write('--Grid spacing in z direction (dz)\n')
        f.write(config.get('FWI','dz')+'\n')
        f.write('--Time step (dt)\n')
        f.write(config.get('FWI','dt')+'\n')
        f.write('--Dominant frequency (f0)\n')
        f.write(config.get('FWI','f0')+'\n')
        # Omission writes 19 pairs: both native solvers now default to CPML.
        if config.has_option('FWI', 'boundary_type'):
            boundary = config.getint('FWI', 'boundary_type')
            if boundary not in (0, 1):
                raise ValueError('boundary_type must be 0 (Liao) or 1 (CPML)')
            f.write('--boundary_type: 0=Liao, 1=CPML\n')
            f.write(str(boundary)+'\n')

    return


class Posterior():
    def __init__(self, data, config, vel_fixed = 1950, sigma = 0.1, num_processes = 1, log_prior = None, 
                            mask = None, data_mask = None, paramfile = 'input_params.txt'):
        '''
        data: observed data with shape: (ns*nt*nr,)
        config: configure file used to define nuisance parameters for 2D fwi code
        vel_fixed: velocity value for fixed (normally water layer) regions
        log_prior: a function that takes samples as input and calculates their log-prior values (using PyTorch)
                    Return: y = log_prior(x)
        mask: a mask array where the parameters with mask = 0 will be fixed 
        data_mask: a mask array that defines which trace (seismogram) is used for fwi: data_mask = 0 means the trace is not used
        sigma: data error/uncertainty
        num_processes: number of process for parallelisation
        paramfile: filename (and full path) that defines parameters for 2D FWI
        '''
        self.data = data
        self.mask = mask
        self.data_mask = data_mask
        self.log_prior = log_prior
        self.num_processes = num_processes
        self.sigma = sigma
        self.vel_water = vel_fixed
        self.paramfile = paramfile
        # create mask matrix for model parameters that are fixed during inversion (e.g., water layer)
        if mask is None:
            nx = config.getint('FWI','nx')
            nz = config.getint('FWI','nz')
            mask = np.full((nx*nz),True)
        self.mask = mask

        prepare_fwi_parameters(self.paramfile, config)

    def solver(self, vel):
        """
        warpper function for multi-processing.pool
        data_syn: synthetic waveform data with shape of (ns*nt*nr,) 1 dim array
        grad: gradient of l2_loss w.r.t. input velocity model with shape of (nz, nx)
        """
        data_syn, grad = aco2d.fwi(vel, self.data, paramfile = self.paramfile, data_mask = self.data_mask)
        grad = grad[self.mask]
        loss = 0.5 * np.sum((data_syn - self.data)**2)
        return loss, grad

    def fwi(self, x):
        '''
        Calculate modelled waveform data and fwi data-model gradient using finite difference
        '''
        m, _ = x.shape
        loss = np.zeros(m)
        grad = np.zeros(x.shape)
        if not isinstance(x, np.ndarray):
            x = x.detach().numpy().astype(np.float64)
        # vel = np.full(self.mask.shape, self.vel_water, dtype = np.float64)
        vel = np.full([m, self.mask.size], self.vel_water, dtype = np.float64)
        vel[:,self.mask] = x
        for i in range(m):
            loss[i], grad[i] = self.solver(vel[i])
            # data_syn, gradient = aco2d.fwi(vel[i], self.data, paramfile = self.paramfile)
            # grad[i] = gradient[self.mask]
            # loss[i] = 0.5 * np.sum((data_syn - self.data)**2)
        return loss, grad

    def fwi_parallel(self, x):
        '''
        Parallelised version of self.fwi using torch.multiprocessing
        '''
        m, _ = x.shape
        loss = np.zeros(m)
        grad = np.zeros(x.shape)
        if not isinstance(x, np.ndarray):
            x = x.detach().numpy().astype(np.float64)
        vel = np.full([m, self.mask.size], self.vel_water, dtype = np.float64)
        vel[:,self.mask] = x

        with Pool(processes = self.num_processes) as pool:
            results = pool.map(self.solver, [vel[i] for i in range(m)])

        for i in range(m):
            loss[i] = results[i][0]
            grad[i] = results[i][1]
        return loss, grad

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
        if self.num_processes == 1:
            loss = ForwardModel.apply(x, self.fwi)
        else:
            loss = ForwardModel.apply(x, self.fwi_parallel)
        logp = -loss/self.sigma**2 + self.log_prior(x)
        return logp


class PosteriorVariableDensity():
    """Bayesian posterior for joint velocity/density acoustic FWI.

    Each row of ``x`` is ordered according to ``parameter_order``.  The
    supported orders are ``'velocity_density'`` (the historical default)
    and ``'density_velocity'``.  The two masks may differ, which makes it
    possible to fix (for example) the water-layer velocity and density
    independently.
    """

    def __init__(self, data, config, vel_fixed=1950.0, density_fixed=1000.0,
                 sigma=0.1, num_processes=1, log_prior=None,
                 velocity_mask=None, density_mask=None, data_mask=None,
                 paramfile='input_params.txt',
                 parameter_order='velocity_density'):
        """Create a variable-density posterior.

        Parameters are analogous to :class:`Posterior`, except that separate
        masks and fixed/background values are accepted for velocity and
        density.  ``vel_fixed`` and ``density_fixed`` may be scalars or flat
        arrays of length ``nx*nz``.
        """
        nx = config.getint('FWI', 'nx')
        nz = config.getint('FWI', 'nz')
        self.model_size = nx * nz

        self.data = data
        expected_data_size = (config.getint('FWI', 'ns') *
                              config.getint('FWI', 'nt') *
                              config.getint('FWI', 'nr'))
        if self.data.size != expected_data_size:
            raise ValueError('data length must equal ns*nt*nr')
        if not np.all(np.isfinite(self.data)):
            raise ValueError('data must contain only finite values')
        if sigma <= 0:
            raise ValueError('sigma must be positive')
        if num_processes < 1:
            raise ValueError('num_processes must be at least one')

        self.velocity_mask = self._prepare_mask(velocity_mask, 'velocity_mask')
        self.density_mask = self._prepare_mask(density_mask, 'density_mask')
        self.vel_fixed = self._prepare_fixed_model(vel_fixed, 'vel_fixed')
        self.density_fixed = self._prepare_fixed_model(
            density_fixed, 'density_fixed'
        )
        self.ndim = (np.count_nonzero(self.velocity_mask) +
                     np.count_nonzero(self.density_mask))
        valid_orders = ('velocity_density', 'density_velocity')
        if parameter_order not in valid_orders:
            valid = ', '.join(valid_orders)
            raise ValueError(f"parameter_order must be one of: {valid}")
        self.parameter_order = parameter_order

        if data_mask is None:
            self.data_mask = None
        else:
            self.data_mask = np.ascontiguousarray(
                data_mask, dtype=np.float32
            ).reshape(-1)
            if self.data_mask.size != self.data.size:
                raise ValueError('data_mask length must equal data length')
            if not np.all(np.isfinite(self.data_mask)):
                raise ValueError('data_mask must contain only finite values')

        self.log_prior = log_prior
        self.num_processes = num_processes
        self.sigma = sigma
        self.paramfile = paramfile
        prepare_fwi_parameters(self.paramfile, config)

    def _prepare_mask(self, mask, name):
        if mask is None:
            return np.ones(self.model_size, dtype=bool)
        mask = np.asarray(mask, dtype=bool).reshape(-1)
        if mask.size != self.model_size:
            raise ValueError('%s length must equal nx*nz' % name)
        return mask

    def _prepare_fixed_model(self, value, name):
        value = np.asarray(value, dtype=np.float64)
        if value.ndim == 0:
            model = np.full(self.model_size, value.item(), dtype=np.float64)
        else:
            model = np.ascontiguousarray(value, dtype=np.float64).reshape(-1)
            if model.size != self.model_size:
                raise ValueError('%s length must equal nx*nz' % name)
            model = model.copy()
        if not np.all(np.isfinite(model)) or np.any(model <= 0.0):
            raise ValueError('%s must contain finite positive values' % name)
        return model

    def _unpack_models(self, x):
        """Expand concatenated active parameters into full physical models."""
        n_velocity = np.count_nonzero(self.velocity_mask)
        n_density = np.count_nonzero(self.density_mask)
        velocity = self.vel_fixed.copy()
        density = self.density_fixed.copy()
        if self.parameter_order == 'velocity_density':
            velocity[self.velocity_mask] = x[:n_velocity]
            density[self.density_mask] = x[n_velocity:n_velocity + n_density]
        else:  # density_velocity
            density[self.density_mask] = x[:n_density]
            velocity[self.velocity_mask] = x[n_density:n_density + n_velocity]
        return velocity, density

    def solver(self, models):
        """Return the data misfit and joint active-parameter gradient."""
        velocity, density = models
        data_syn, grad_velocity, grad_density = \
            aco2d.fwi_variable_density(
                velocity, density, self.data, paramfile=self.paramfile,
                data_mask=self.data_mask
            )
        for name, value in (('synthetic data', data_syn),
                            ('velocity gradient', grad_velocity),
                            ('density gradient', grad_density)):
            if not np.all(np.isfinite(value)):
                raise FloatingPointError('Nonfinite variable-density FWI %s' % name)
        residual = data_syn.astype(np.float64) - self.data.astype(np.float64)
        if self.data_mask is not None:
            residual *= self.data_mask.astype(np.float64)
        loss = 0.5 * np.dot(residual, residual)
        if not np.isfinite(loss):
            raise FloatingPointError('Nonfinite variable-density FWI loss')
        if self.parameter_order == 'velocity_density':
            grad = np.concatenate((grad_velocity[self.velocity_mask],
                                   grad_density[self.density_mask]))
        else:  # density_velocity
            grad = np.concatenate((grad_density[self.density_mask],
                                   grad_velocity[self.velocity_mask]))
        return loss, grad

    def fwi(self, x):
        """Evaluate misfit and gradient for a batch of joint models."""
        if not isinstance(x, np.ndarray):
            x = x.detach().cpu().numpy()
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.ndim:
            raise ValueError('x must have shape (nsamples, %d)' % self.ndim)
        if not np.all(np.isfinite(x)):
            raise ValueError('x must contain only finite values')

        models = [self._unpack_models(sample) for sample in x]
        loss = np.zeros(x.shape[0], dtype=np.float64)
        grad = np.zeros(x.shape, dtype=np.float64)
        for index, model in enumerate(models):
            loss[index], grad[index] = self.solver(model)
        return loss, grad

    def fwi_parallel(self, x):
        """Parallel version of :meth:`fwi` using torch multiprocessing."""
        if not isinstance(x, np.ndarray):
            x = x.detach().cpu().numpy()
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.ndim:
            raise ValueError('x must have shape (nsamples, %d)' % self.ndim)
        if not np.all(np.isfinite(x)):
            raise ValueError('x must contain only finite values')

        models = [self._unpack_models(sample) for sample in x]
        with Pool(processes=self.num_processes) as pool:
            results = pool.map(self.solver, models)
        loss = np.asarray([result[0] for result in results], dtype=np.float64)
        grad = np.asarray([result[1] for result in results], dtype=np.float64)
        return loss, grad

    def log_prob(self, x):
        """Return the unnormalised log-posterior for each row of ``x``."""
        if not torch.isfinite(x).all():
            raise FloatingPointError('Nonfinite variable-density posterior input samples')
        fwi_function = self.fwi if self.num_processes == 1 else self.fwi_parallel
        loss = ForwardModel.apply(x, fwi_function)
        logp = -loss / self.sigma**2
        if self.log_prior is not None:
            logp = logp + self.log_prior(x)
        return logp
