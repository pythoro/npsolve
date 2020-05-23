.. npsolve documentation master file, created by
   sphinx-quickstart on Mon Sep  2 16:12:17 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to npsolve's documentation!
===================================

The *npsolve* package is a small, simple package built on *numpy* and
*fastwire* to make it easy to use object-oriented classes and methods for
the calculation step for numerical solvers.

Many numerical solvers (like those in *scipy*) provide candidate solutions as
a numpy ndarray. They often also require a numpy ndarray as a return value
(e.g. an array of derivatives) during the solution. These requirements can make
it difficult to use an object oriented approach to performing the calculations.
Usually, we end up with script-like code that looses many of the benefits
of object-oriented programming.

The npsolve framework links a solver with multiple classes that handle the
calculations for each step in the algorithm. It allows different parts of 
the calculations to be encapsulated and polymorphic, and makes the code 
much easier to modify and maintain.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   tutorials
   package

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
