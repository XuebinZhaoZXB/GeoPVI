#####################################################################
GeoPVI (Geophysical Inversion using Parametric Variational Inference)
#####################################################################

This package solves fully nonlinear (Bayesian) **Geo**\ physical inverse problems using **P**\ arametric **V**\ ariational **I**\ nference, 
in which a variational distribution is defined to approximate the Bayesian posterior probability distribution function (pdf) and is represented
by parametric (analytic) expressions. GeoPVI currently features mean field and full rank automatic differentiation variational inference (ADVI), 
physically structured variational inference (PSVI), normalising flows, and boosting variational inference (BVI).


Requirements
--------------
numpy, torch, cython, dask, scipy


Installation
------------

In the ``GeoPVI`` folder, run

.. code-block:: sh

    sh setup.sh install

If you don't have permission to install GeoPVI into your Python environment, simply replace 

.. code-block:: sh

    pip install --user -e .

in ``setup.sh``.

The package is still in heavy development and can change rapidly. Therefore, it is recommended to install GeoPVI in an editable mode. 
Alternatively, if you do not want to install the package, run

.. code-block:: sh

    sh setup.sh

Then, you need to tell scripts which use the GeoPVI package where the package is. For example, run a script with

.. code-block:: python

    PYTHONPATH=/your/GeoPVI/path python fwi.py

See examples in ``examples`` folder. 


Get started
---------------------
To perform Bayesian inversion using variational inference methods, you need to define two main components: 
a variational distribution and a function to estimate posterior probability value.
.. code-block:: python
    
    def log_prob(m):
    # Input samples m has a shape of (nsamples, ndim)
    # Output the log-posterior values for m
    logp = log_prior + log_like
    return logp

To define a variational distribution
.. code-block:: python

    from geopvi.nfvi.models import FlowsBasedDistribution
    from geopvi.nfvi.flows import Linear, Real2Constr

    flows = [Linear(dim, kernel = 'diagonal')]
    flows.append(Real2Constr(lower = lowerbound , upper = upperbound))
    variational_pdf = FlowsBasedDistribution(flows , base = 'Normal')

This defines a transformed diagonal Gaussian distribution as a variational distribution, corresponding to mean field ADVI.

GeoPVI provides a wrapper to perform variational inversion:
.. code-block:: python

    from geopvi.nfvi.models import VariationalInversion

    inversion = VariationalInversion(variationalDistribution = variational_pdf, log_posterior = log_prob)
    negative_elbo = inversion.update(n_iter = 1000, nsample = 10)

which updates the variational distribution for 1000 iterations and with 10 samples per iteration for Monte Carlo integration.
This returns the ``negative_elbo`` value for each iteration. 

After training, posterior samples can be obtained by
.. code-block:: python

    samples = variational_pdf.sample(nsample = 2000)

For comprehensive guides and examples on using GeoPVI, please check out GeoPVI user manual in ``doc`` folder and tutorials in``examples``.


Examples
---------
- For a complete 2D travel time tomography example, please see the example in ``examples/tomo2d``. 
- For a complete 2D full waveform inversion example, please see the example in ``examples/fwi2d``. 
- For a complete 3D full waveform inversion example using the **BP** ``tdwi`` forward modeller, please see the example in ``examples/fwi3d_bp``.
- For an example implementation of 3D full waveform inversion, please see the example in ``examples/fwi3d``. Note
  that this requires users to provide an external 3D FWI code to calculate misfit values and gradients. See details
  in ``geopvi/fwi3d``.
- Other implementation examples can be found in ``example/tutorials``.


Specifically for BP HPC server
-------------------------------
GeoPVI are tested using ``intel-2019`` and ``intel-2020`` conda environments.
To perform 3D FWI using BP's server and the ``tdwi`` solver, please use codes in ``geopvi/forward/fwi3d_bp``.


References
----------
- Zhao, X., Curtis, A. & Zhang, X. (2022). Bayesian seismic tomography using normalizing flows. Geophysical Journal International, 228 (1), 213-239.
- Zhao, X., & Curtis, A. (2024). Bayesian inversion, uncertainty analysis and interrogation using boosting variational inference. Journal of Geophysical Research: Solid Earth 129 (1), e2023JB027789