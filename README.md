# npsolve

The *npsolve* package is a small, simple package built on *numpy* to make it
easy to use object-oriented classes and methods for the calculation step for
numerical solvers.

Many numerical solvers (like those in *scipy*) provide candidate solutions as
a numpy ndarray. They often also require a numpy ndarray as a return value
(e.g. an array of derivatives) during the solution. These requirements can make
it difficult to use an object oriented approach to performing the calculations.
Usually, we end up with script-like code that looses many of the benefits
of object-oriented programming.

The npsolve framework sets up an intermediate object, a Package, that 
translates between unnamed vectors and object-oriented classes. It facilitates
both simple and complex inter-dependencies and keeps code modular and 
maintainable.

Advantages:
* Introduces very little overhead in calculation time
* Explicit, customisable steps in the calculation for each time step
* Able to be used with any solver


## Basic usage tutorial

Let's use npsolve to do some integration through time, like you would to
solve an ODE. Instead of equations, though, we're using class methods. The
code for all the tutorials is available in the repository under 'examples'.

We'll assume there is some interdependency between different components. To
begin, let's set up some constaint variable names.

```python

import numpy as np
import npsolve
import matplotlib.pyplot as plt

# Unique variable names
COMP1_POS = 'position1'
COMP1_VEL = 'velocity1'
COMP2_VALUE = 'component2_value'
COMP2_FORCE = 'comp2_force'

```

Now we'll set up a class for each component in our model.

```python

class Component1():
    def set_comp2_force(self, force):
        self._comp2_force = force

    def get_pos(self, state):
        return state[COMP1_POS]

    def step(self, state, t, log):
        """Called by the solver at each time step

        Calculate acceleration based on the net component2_value.
        """
        acceleration = self._comp2_force * 1.0
        derivatives = {
            "position1": state[COMP1_VEL],
            "velocity1": acceleration,
        }
        return derivatives

class Component2:
    def get_force(self, state):
        return 1.0 * state[COMP2_VALUE]

    def set_comp1_pos(self, pos):
        self._comp1_pos = pos

    def calculate(self, state, t):
        """Some arbitrary calculations based on current time t
        and the position at that time calculated in Component1.
        This returns a derivative for variable 'c'
        """
        dc = 1.0 * np.cos(2 * t) * self._comp1_pos
        derivatives = {COMP2_VALUE: dc}
        return derivatives

    def step(self, state, t, log):
        """Called by the solver at each time step"""
        return self.calculate(state, t)

```

Note that external dependencies will be injected via the `set_comp1_pos` and 
`set_comp2_force` methods. The classes also provide some methods to return
certain parameters in the `get_pos` and `get_force` methods.

They each have a method we've called 'step', which will be called by the
Package. Whatever their name, these methods must accept three (or more) 
parameters, `state`, `t`, and `log`. These are passed in by the Package
at each time step.

- state (dict): A dictionary that contains the current values for all state
    variables. We're get to how they are set up soon.
- t (float): The current time step.
- log (dict | None): If log is not None, usually at the completion of a
    time step, this log dictionary can be added to to log additional values.

Next, we'll make a class to handle inter-dependencies.

```python

class Assembly:
    """Handle inter-dependencies."""
    def __init__(self, comp1, comp2):
        self.comp1 = comp1
        self.comp2 = comp2

    def precalcs(self, state, t, log):
        """Inject dependencies for later calculations in 'step' methods."""
        comp1 = self.comp1
        comp2 = self.comp2
        comp1_pos = comp1.get_pos(state)
        comp2_force = comp2.get_force(state)
        if log:
            # Log whatever we want here into a dictionary.
            log[COMP2_FORCE] = comp2_force
        comp1.set_comp2_force(comp2_force)
        comp2.set_comp1_pos(comp1_pos)

```

We're simply making a class that accepts instances of our two components,
and then provides a method that injects their inter-dependencies, which
we've called `precalcs` in this case. This method must accept three (or more) 
parameters, `state`, `t`, and `log` because it will be called by the
Package. 

Now, we need to make a function to create a Package instance.

```python
    
def get_package():
    component1 = Component1()
    component2 = Component2()
    assembly = Assembly(component1, component2)
    package = npsolve.Package()
    package.add_component(component1, 'comp1', 'step')
    package.add_component(component2, 'comp2', 'step')
    package.add_component(assembly, 'assembly', None)
    package.set_stage_calls(
        [('assembly', 'precalcs')]
    )
    return package

```

Here, we're creating instances of our components and our assembly.
Then, we're adding them to a new Package instance. When we use
`add_component`, we pass in the instance object, a unique name, and the
method to call to finish each time step and get state derivatives. If the
derivatives method is set to None, the derivatives will default to 0.0.

We're also setting up stage calls that happen prior to the final calls to
get derivatives. We can pass in many different calls to different components.
Here, we just specify one call to the `precalcs` method in the component
we've named 'assembly'.

Note also that we've set the final derivative call for the assembly 
component to None, so it won't be called at the end of the current timestep.

To perform the integration, we'll use the inbuilt ODEIntegrator class.

```python

def solve(package, t_end=10):
    ode_integrator = npsolve.solvers.ODEIntegrator()
    dct = ode_integrator.run(package, t_end)
    return dct

```

Now, we are ready to run. To run, we need to create a dictionary that 
contains initial values for all our state variables. Any missing ones
will not be found by any components that depend on them. Then we setup
the Package by passing the initial values dictionary to its `setup` method.

```python

def run():
    package = get_package()
    inits = {COMP1_POS: 0.1,
             COMP1_VEL: 0.3,
             COMP2_VALUE: -0.1}
    package.setup(inits)
    dct = solve(package)
    return dct

```

Lastly, we'll add functions to plot results and execute the script.

```python

def plot(dct):
    plt.figure(1)
    dct2 = dct.copy()
    t_vec = dct2.pop('time')
    for var_name, values in dct2.items():
        plt.plot(t_vec, values, label=var_name)
    plt.legend()
    plt.show()


def execute():
    dct = run()
    plot(dct)


if __name__ == '__main__':
    execute()

```

Run it and see the results!

## Tutorials

Check out the tutorials in the examples folder to learn the basics and 
learn about some more advanced features like the the soft_functions.