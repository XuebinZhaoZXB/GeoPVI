#####################################################################
GeoPVI (Geophysical Inversion using Parametric Variational Inference)
#####################################################################

This package solves fully nonlinear geophysical inverse problems using parametric variational inference, specifically including normalising flows, boosting variational inference, and physically structured variational inference.


Author
----------
 - Xuebin Zhao <xuebin.zhao@ed.ac.uk>

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

We recommend to install GeoPVI in an editable mode. Alternatively, if you do not want to install the package, simply do

.. code-block:: sh

    sh setup.sh

Then, you need to tell scripts which use the GeoPVI package where the package is. For example, simply run a script with

.. code-block:: python
    
    PYTHONPATH=/your/GeoPVI/path python fwi.py

See examples in ``examples`` folder. 


Examples
---------
- For a complete 2D full waveform inversion example, please see the example in ``examples/fwi2d``. 
- For an example implementation of 3D full waveform inversion, please see the example in ``examples/fwi3d``. Note
  that this requires users to provide an external 3D FWI code to calculate misfit values and gradients. See details
  in ``geopvi/fwi3d``.

References
----------
- Zhao, X., Curtis, A. & Zhang, X. (2022). Bayesian seismic tomography using normalizing flows. Geophysical Journal International, 228 (1), 213-239.
- Zhao, X., & Curtis, A. (2024). Bayesian inversion, uncertainty analysis and interrogation using boosting variational inference. Journal of Geophysical Research: Solid Earth 129 (1), e2023JB027789