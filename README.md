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
        super().__init__() # Don't forget to call this!
        self.add_var('position', init=0.1)
        self.add_var('velocity', init=0.3)
    
class Component2(npsolve.Partial):
    def __init__(self):
        super().__init__() # Don't forget to call this!
        self.add_var('force', init=-0.1)

```

Next override the *set_vectors* method to store views you might want. In this case, we'll just save the variables as attributes. Note that these are actually views, that are automatically updated by the solver. We'll do it differently with Component2.

```python


class Component1(npsolve.Partial):
    def __init__(self):
        super().__init__() # Don't forget to call this!
        self.add_var('position', init=0.1)
        self.add_var('velocity', init=0.3)
    
    def set_vectors(self, state_dct, ret_dct):
        ''' Set some state views for use during calculations '''
        self.position = state_dct['position']
        self.velocity = state_dct['velocity']
        self.force = state_dct['force']
    

class Component2(npsolve.Partial):
    def __init__(self):
        super().__init__() # Don't forget to call this!
        self.add_var('force', init=-0.1)

```

Note that variables are made available to all Partial instances automatically.

Then, we'll tell them how to do the calculations. The *step* method is called automatically and expects a dictionary of return values (e.g. derivatives). We'll use that one here. A dictionary of the current state values is provided (again), but we're going to use the views we set in the *set_vectors* method.

```python

class Component1(npsolve.Partial):
    def __init__(self):
        super().__init__() # Don't forget to call this!
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
        super().__init__() # Don't forget to call this!
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


Now let's make a simple model that gathers together our Partials. 

```python

class Model():
    def __init__(self):
        self.elements = {}
		
    def add_element(self, key, element):
        self.elements[key] = element

```

Now, we'll set up the solver. For this example, we'll use the odeint solver from Scipy. Here's what it looks like:


```python

from scipy.integrate import odeint

class Solver(npsolve.Solver):
    def solve(self):
        self.t_vec = np.linspace(0, 10, 1001)
        result = odeint(self.step, self.npsolve_initial_values, self.t_vec)
        return result
    
    def set_model(self, model):
        self.model = model
        self.connect(model)
		
    def connect(self, model):
        self.remove_signals()
        self.setup_signals()
        for k, e in model.elements.items():
            e.connect()
        self.close_signals()

```

Let's look at what's going on, starting with the `solve` method. By default, Solvers have a *step* method that's ready to use. (They also have a *one_way_step* method that doesn't expect return values from the Partials, and a *tstep* method that has a time value as the first argument.) After initialisation, the initial values set by the Partial classes are captured in the *npsolve_initial_values* attribute. By default, the Solver's *step* method returns a vector of all the return values, the same size as the Solver's npsolve_initial_values array. So most of the work is done for us here already. Note here that we don't need to know anything about the model or the elements in the model.


We'll pass a model into our solver, and we need to connect the model elements to the solver. In the `connect` method above, we're using this typical sequence of calls:
  * Solver.remove_signals: To clean up any signals from previous `connect` calls
  * Solver.setup_signals: Create a new set of signals.
  * Partial.connect: We call `connect` on each partial *after* calling `setup_signals` on the Solver.
  * Solver.close_signals: Close the signals so they aren't accidentally used by anything else.

This allows us to decouple the model and Partials from the solver. We can pass in different models, or pass models to different solvers. We can make models with different components. It's flexible and easy to maintain!

To run, we just have to instantiate the Solver before the Partials that use it, then call the *npsolve_init* method. It doesn't matter where in the code we create the Solver and Partial instances - they'll link up automatically through *fastwire*.


```python
    
def make_model():
    m = Model()
    m.add_element('component 1', Component1())
    m.add_element('component 2', Component2())
    return m
	

def make_solver():
    return Solver()

def run():
    solver = make_solver()
    model = make_model()
    solver.set_model(model)

    # Initialise the solver
    solver.npsolve_init()
	
    # Now we can run!
    res = solver.solve()
    return res, solver

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







