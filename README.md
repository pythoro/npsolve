# npsolve

Many numerical solvers (like those in scipy) provide candidate solutions as a numpy array. They often also require a numpy array as a return value (e.g. an array of derivatives) during the solution. These requirements can make it difficult to use an object oriented approach to performing the calculations. 

Enter *npsolve* - a small, simple package built on *fastwire* to make it easy to use object-oriented code for the calculation step for numerical solvers.
