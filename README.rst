.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black

.. image:: pipeline.svg

.. image:: interrogate.svg

===============================
Documentation of ADDA
===============================

ADDA (Automatic Differentiation for Data Assimilation) is a flexible PyTorch framework for performing data assimilation in various contexts.
Its main distinguishing feature is the use of automatic differentiation (in PyTorch, with JAX compatibility) for 4D-Var methods.

Full documentation at https://m-dml.pages.hzdr.de/adda/

Many pedagogical examples of usage can be found in the "notebooks" folder. For example, the following illustration is taken from our Lorenz-63 notebook.

.. image:: L63_sc4dvar.png

The data assimilation methods included in ADDA are:

- Kalman filter, fixed-lag smoother and RTS smoother
- Ensemble Kalman Filter and Smoother
- 4D-Var, in its hard-constraint and weak-constraint variants, with either a single window or a sliding window approach.
- Latent data assimilation, as in `this paper <https://www.science.org/doi/full/10.1126/sciadv.aea4248>`_

Our exemple notebooks include (non-exhaustive list):

- 4D-Var in different observation scenarios on the Lorenz-63 system
- Ensemble Kalman Filter and Smoother on the Lorenz-96 system
- 4D-Var with a neural emulator of the Kolmogorov flow
- Latent data assimilation on the Kuramoto-Sivashinsky system
- 4D-Var on the 2D Kuramoto-Sivashinsky system leveraging the Exponax library with a PyTorch-JAX bridge

--------------------------------------------

Installation
------------
To install you have to clone the repository and install it with pip:

.. code-block:: bash

   git clone https://github.com/anthony-frion/ADDA
   cd adda
   pip install -e .

--------------------------------------------
