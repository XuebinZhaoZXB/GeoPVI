#####################################################################
GeoPVI
#####################################################################

This package solves fully nonlinear Bayesian **Geo**\ scientific inverse problems using **P**\ arametric **V**\ ariational **I**\ nference methods.

In GeoPVI, a variational distribution is defined to approximate the Bayesian posterior probability distribution function (pdf) and is represented
by parametric (explicit mathematical) expressions. GeoPVI currently features automatic differentiation variational inference (ADVI), 
physically structured variational inference (PSVI), normalising flows, and boosting variational inference (BVI). 
Future updates will expand this package to incorporate other parametric variational methods that have been tested in geophysics. 


Requirements
--------------
numpy, torch, cython, dask, scipy

(Note: to run the tutorial examples in ``examples/tutorials``, matplotlib and jupyter notebook are also required.)


Installation
------------

In the ``GeoPVI`` folder, run

.. code-block:: sh

    sh setup.sh install

If you don't have permission to install GeoPVI into your Python environment, simply replace 

.. code-block:: sh

    pip install --user -e .

in ``setup.sh``. The package is still in heavy development and can change rapidly. Therefore, it is recommended to install GeoPVI in an editable mode. 

Alternatively, if you do not want to install the package, run

.. code-block:: sh

    sh setup.sh

Then, you need to tell scripts which use the GeoPVI package where the package is. For example, run a script with

.. code-block:: python

    PYTHONPATH=/your/GeoPVI/path python fwi.py

See examples in ``examples`` folder. 


Get started
---------------------
Two main components are required to perform Bayesian inversion using GeoPVI: 
a function to estimate the posterior probability values and a variational distribution.

.. code-block:: python
    
    def log_prob(m):
        # Input array of samples m has a shape of (nsamples, ndim)
        # This function outputs the log-posterior values for m
        # by summing logarithmic prior and likelihood values for m
        logp = log_prior(m) + log_like(m)
    return logp

To define a variational distribution

.. code-block:: python

    from geopvi.nfvi.models import FlowsBasedDistribution
    from geopvi.nfvi.flows import Linear, Real2Constr

    flows = [Linear(dim, kernel = 'diagonal')]
    flows.append(Real2Constr(lower = lowerbound , upper = upperbound))
    variational_pdf = FlowsBasedDistribution(flows , base = 'Normal')

This defines a variational distribution represented by mean field ADVI.

GeoPVI provides a wrapper to perform variational inversion:

.. code-block:: python

    from geopvi.nfvi.models import VariationalInversion

    inversion = VariationalInversion(variationalDistribution = variational_pdf, log_posterior = log_prob)
    negative_elbo = inversion.update(n_iter = 1000, nsample = 10)

which updates the variational distribution for 1000 iterations, with 10 samples per iteration for Monte Carlo integration.
This returns the ``negative_elbo`` value for each iteration. 

After training, posterior samples can be obtained by

.. code-block:: python

    samples = variational_pdf.sample(nsample = 2000)


Documentation
---------------
For comprehensive guides and examples on using GeoPVI, please check out GeoPVI user manual in ``doc`` folder and tutorials in ``examples/tutorials``.


Examples
---------
- For a complete 2D travel time tomography example, please see the example in ``examples/tomo2d``. 
- For a complete 2D full waveform inversion example, please see the example in ``examples/fwi2d``.
- For the constant- and variable-density 2D acoustic kernels, their small
  demos, validation tests, and benchmark, see ``geopvi/forward/fwi2d``.
- For a complete 3D surface wave inversion, please see the example in ``examples/swi3d``. 
- For a complete example of performing **variational prior replacement (VPR)** to update prior information post Bayesian inversion, please see the example in ``examples/fwi2d_vpr``. 
  In this example, a uniform prior probability distribution is replaced by a smoothed version of the uniform prior pdf, with almost zero additional computational cost.
- For an example implementation of 3D full waveform inversion, please see the example in ``examples/fwi3d``. Note
  that this requires users to provide an external 3D FWI code to calculate misfit values and gradients. See details
  in ``geopvi/fwi3d``.
- Other implementation examples (currently including 1D surface wave dispersion inversion and vertical electrical sounding inversion) can be found in ``examples/tutorials``.


Citation
----------
If you use GeoPVI in your research, please cite our paper that introduces the package:

- Zhao, X., & Curtis, A. (2026). GeoPVI: a Python Package for Geoscientific Inversion using Parametric Variational Inference. Geophysical Journal International, in press.


References
----------
Below are relevant papers that introduce the variational methods implemented in GeoPVI and their applications in geophysics. 
Please also consider citing these papers if you use the corresponding methods in your research:

- Zhao, X., Curtis, A. & Zhang, X. (2022). Bayesian seismic tomography using normalizing flows. Geophysical Journal International, 228 (1), 213-239.
- Zhao, X., & Curtis, A. (2024). Bayesian inversion, uncertainty analysis and interrogation using boosting variational inference. Journal of Geophysical Research: Solid Earth 129 (1), e2023JB027789.
- Zhao, X., & Curtis, A. (2024). Physically Structured Variational Inference for Bayesian Full Waveform Inversion. Journal of Geophysical Research: Solid Earth 129 (11), e2024JB029557.
- Zhao, X., & Curtis, A. (2024). Variational prior replacement in Bayesian inference and inversion. Geophysical Journal International, 239 (2), 1236-1256.
- Zhao, X., & Curtis, A. (2025). Efficient Bayesian Full Waveform Inversion and Analysis of Prior Hypotheses in 3D. Geophysics, 90 (6), R373-R388.
- Zhao, X., Irvin, L., Galetti, E., & Curtis, A. (2026). Direct-3D Variational Bayesian Surface Wave Inversion and Its Application to Ambient Noise Tomography beneath Great Britain. Geophysical Journal International, 245 (2).
