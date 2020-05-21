# npsolve

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


## Basic usage tutorial

Let's use npsolve to do some integration through time, like you would to
solve an ODE. Instead of equations, though, we're using class methods.

First, setup some classes that you want to do calculations with. We do this
by using the `add_var` method to setup variables and their initial values.

```python

import numpy as np
import npsolve

class Component1(npsolve.Partial):
    def __init__(self):
        super().__init__() # Don't forget to call this!
        self.add_var('position', init=0.1)
        self.add_var('velocity', init=0.3)
    

class Component2(npsolve.Partial):
    def __init__(self):
        super().__init__()  # Don't forget to call this!
        self.add_var('force', init=-0.1)

```

All the variables are made available to all Partial instances automatically
through their `state` attribute. It's a dictionary. The `add_var` method 
sets initial values into the instance's state dictionary. Later, the `Solver`
will ultimately replace the `state` attribute with a new dictionary that
contains all variables from all the Partial classes.

Next, we'll tell these classes how to do some calculations during each time
step. The `step` method is called automatically and expects a dictionary of
return values (e.g. derivatives). We'll use that one here. The state
dictionary is given again as the first argument, but we're going to use the
internal `state` attribute instead. So, we'll add some more methods:

```python

class Component1(npsolve.Partial):
    def __init__(self):
        super().__init__() # Don't forget to call this!
        self.add_var('position', init=0.1)
        self.add_var('velocity', init=0.3)
    
    def step(self, state_dct, t, *args):
        ''' Called by the solver at each time step 
        Calculate acceleration based on the 
        '''
        acceleration = 1.0 * self.state['force']
        derivatives = {'position': self.velocity,
                       'velocity': acceleration}
        return derivatives


class Component2(npsolve.Partial):
    def __init__(self):
        super().__init__()  # Don't forget to call this!
        self.add_var('force', init=-0.1)

    def calculate(self, t):
        ''' Some arbitrary calculations based on current time t
        and the position at that time calculated in Component1.
        This returns a derivative for variable 'c'
        '''
        dc = 1.0 * np.cos(2*t) * self.state['position']
        derivatives = {'force': dc}
        return derivatives
    
    def step(self, state_dct, t, *args):
        ''' Called by the solver at each time step '''
        return self.calculate(t)
        
		
```

Now, we'll set up the solver. For this example, we'll use the odeint solver
from Scipy. Here's what it looks like:


```python

from scipy.integrate import odeint

class Solver(npsolve.Solver):
    def solve(self):
        self.npsolve_init() # Initialise
        self.t_vec = np.linspace(0, 10, 1001)
        result = odeint(self.step, self.npsolve_initial_values, self.t_vec)
        return result

```

Let's look at what's going on in the `solve` method. By default, Solvers
have a `step` method that's ready to use. (They also have a `one_way_step`
method that doesn't expect return values from the Partials, and a `tstep` 
method that expects a time value as the first argument.) After initialisation,
the initial values set by the Partial classes are captured in the
`npsolve_initial_values` attribute. By default, the Solver's `step` method
returns a vector of all the return values, the same size as the Solver's
`npsolve_initial_values` array. So most of the work is done for us here
already. 

Note here that we don't need to know anything about the model or
the elements in the model. This allows us to decouple the model and Partials
from the solver. We can pass in different models, or pass models to different
solvers. We can make models with different components. It's flexible and easy
to maintain!

To run, we just have to instantiate the Solver and Partial instances,
then pass a list or dictionary of the Partial instances to the `connect` 
method of the Solver. They'll link up automatically through *fastwire*.


```python
    
def run():
    solver = Solver()
    partials = [Component1(), Component2()]
    solver.connect(partials)
    res = solver.solve()
    return res, solver

```

Let's set up a plot to see the results. Use the `npsolve_slices` attribute of
the Solver to get the right columns.

```python

import matplotlib.pyplot as plt

def plot(res, s):
    slices = s.npsolve_slices
    
    plt.plot(s.t_vec, res[:,slices['position']], label='position')
    plt.plot(s.t_vec, res[:,slices['velocity']], label='velocity')
    plt.plot(s.t_vec, res[:,slices['force']], label='force')
    plt.legend()

```

Run it and see what happens!

```python

res, s = run()
plot(res, s)

```

