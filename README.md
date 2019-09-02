# npsolve

Many numerical solvers (like those in scipy) provide candidate solutions as a numpy array. They often also require a numpy array as a return value (e.g. an array of derivatives) during the solution. These requirements can make it difficult to use an object oriented approach to performing the calculations. 

The *npsolve* package is a small, simple package built on *numpy* and *fastwire* to make it easy to use object-oriented code for the calculation step for numerical solvers.


## Basic usage tutorial


First, setup some classes that you want to do calculations with. We do this by using the *add_var* method to setup variables and their initial values.

```python

import numpy as np
import npsolve

class Component1(npsolve.Partial):
    def __init__(self):
        super().__init__()
        self.add_var('position', init=0.1)
        self.add_var('velocity', init=0.3)
    
class Component2(npsolve.Partial):
    def __init__(self):
        super().__init__()
        self.add_var('force', init=-0.1)

```

Next override the *set_vectors* method to store views you might want. In this case, we'll just save the variables as attributes. Note that these are actually views, that are automatically updated by the solver. We'll do it differently with Component2.

```python


class Component1(npsolve.Partial):
    def __init__(self):
        super().__init__()
        self.add_var('position', init=0.1)
        self.add_var('velocity', init=0.3)
    
    def set_vectors(self, state_dct, ret_dct):
        ''' Set some state views for use during calculations '''
        self.position = state_dct['position']
        self.velocity = state_dct['velocity']
        self.force = state_dct['force']
    

class Component2(npsolve.Partial):
    def __init__(self):
        super().__init__()
        self.add_var('force', init=-0.1)

```

Note that variables are made available to all Partial instances automatically.

Then, we'll tell them how to do the calculations. The *step* method is called automatically and expects a dictionary of return values (e.g. derivatives). A dictionary of the current state values is provided (again), but we're going to use the views we set in the *set_vectors* method.

```python

class Component1(npsolve.Partial):
    def __init__(self):
        super().__init__()
        self.add_var('position', init=0.1)
        self.add_var('velocity', init=0.3)
    
    def set_vectors(self, state_dct, ret_dct):
        ''' Set some state views for use during calculations '''
        self.position = state_dct['position']
        self.velocity = state_dct['velocity']
        self.force = state_dct['force']
    
    def step(self, state_dct, *args):
        ''' Called by the solver at each time step 
        Calculate acceleration based on the 
        '''
        acceleration = 1.0 * self.force
        derivatives = {'position': self.velocity,
                       'velocity': acceleration}
        return derivatives
		

class Component2(npsolve.Partial):
    def __init__(self):
        super().__init__()
        self.add_var('force', init=-0.1)

    def calculate(self, state_dct, t):
        ''' Some arbitrary calculations based on current time t
        and the position at that time calculated in Component1.
        This returns a derivative for variable 'c'
        '''
        dc = 1.0 * np.cos(2*t) * state_dct['position']
        derivatives = {'force': dc}
        return derivatives
    
    def step(self, state_dct, t, *args):
        ''' Called by the solver at each time step '''
        return self.calculate(state_dct, t)
        
		
```

Now, we'll set up the solver. By default, Solvers have a *step* method that's ready to use, and after initialisation, the initial values set by the Partial classes are captured in the *npsolve_initial_values* attribute. By default, the Solver's *step* method returns a vector of all the return values, the same size as the Solver's npsolve_initial_values array.


```python

from scipy.integrate import odeint

class Solver(npsolve.Solver):
    def solve(self):
        self.t_vec = np.linspace(0, 5, 1001)
        result = odeint(self.step, self.npsolve_initial_values, self.t_vec)
        return result
```


To run, we just have to instantiate the Solver before the Partials that use it, then call the *npsolve_init* method. It doesn't matter where in the code we create the Solver and Partial instances - they'll link up automatically through *fastwire*.


```python

def run():
    s = Solver()
    c1 = Component1()
    c2 = Component2()
	
	# Now we connect the components
    s.setup_signals() # Always call this before calling connect on the Partial classes.
    c1.connect()
    c2.connect()
	
	# Get the solver ready
    s.npsolve_init()
	
    # Now we can run!
    res = s.solve()
    return res, s

```

Let's set up a plot to see the results. Use the *npsolve_slices* attribute of the Solver to get the right columns.

```python

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







