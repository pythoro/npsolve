Tutorial 3 - Timeseries input
=============================

We might have inputs we want to use that aren't functions. They might be 
measured timeseries data, for example, such as position over time. Or, they
could be hypothetical data. For this tutorial, we're just use some random
numbers to make a particle move about in a 2-dimensional space.

In general, we just need to set up an interpolation function to turn the 
discrete timeseries data into continuous values.

First, the setup:

::

    import numpy as np
    from scipy.integrate import odeint
    from scipy.interpolate import make_interp_spline
    import matplotlib.pyplot as plt

    import npsolve

    POS = "position"

We'll use just one variable name, POS, which will have values for x and y
dimensions.

The Particle class
------------------

Now let's start to write a Particle class:

:: 

    class Particle:
        def __init__(self, time_points, positions):
            self.time_points = time_points
            self.positions = positions
            self._xts = make_interp_spline(time_points, positions[:, 0])
            self._yts = make_interp_spline(time_points, positions[:, 1])
            
We'll pass in some timeseries data for the `time_points` and `positions`
attributes. We're creating some splines to interpolate the data in both
x and y dimensions.

Let's add a method to return the initial position.

:: 

    class Particle():
        # ...
    
        def get_init_pos(self):
            return np.array([self._xts(0.0), self._yts(0.0)])

Here, we're simply returning the x and y coordinate at time=0.

Now let's add a step method to be called during integration.

:: 

    class Particle():
        # ...
    
        def step(self, state, t, log):
            """Called by the solver at each time step
            Calculate acceleration based on the
            """
            velocity = np.array([self._xts(t, nu=1), self._yts(t, nu=1)])
            derivatives = {POS: velocity}
            return derivatives


We're getting the velocity in each axis by calling the spline interpolators
with the current time and passing nu=1 to get the first derivative. 

The System
----------

We'll add a function to create the simple System configuration.


::

    def get_system(time_points, positions):
        particle = Particle(time_points, positions)
        system = npsolve.System()
        system.add_component(particle, "particle", "step")
        return system

Integrating
-----------

Now, we can can create a function to run the integration. For the initial
values, we'll call the `get_init_pos` method on the Particle object. We can
get it from the System object, which provides dictionary-like access to the
components added to it. The key is the name we defined earlier, 'particle'.

We're also return the particle object for use later.

::

    def run(t_end=1.0, n=100001):
        np.random.seed(0)
        time_points = np.linspace(0, 1, 51)
        positions = np.random.rand(51, 2) * 10
        system = get_system(time_points, positions)
        particle: Particle = system["particle"]
        inits = {POS: particle.get_init_pos()}
        system.setup(inits)
        dct = npsolve.integrate(system, t_end=t_end, framerate=(n - 1) / t_end)
        return particle, dct

Plotting
--------

We'll add a few functions to plot results. We're going to add the particle 
object as an argument so that we can plot the positions used when creating
the particle.

::

    def plot(dct, particle):
        plt.figure(1)
        plt.plot(dct[POS][:, 0], dct[POS][:, 1], linewidth=0.5)
        plt.scatter(
            particle.positions[:, 0], particle.positions[:, 1], c="r", marker="."
        )
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()


    def plot_vs_time(dct, particle):
        fig, axes = plt.subplots(2, 1, sharex=True, num=2)
        axes[0].plot(dct["time"], dct[POS][:, 0], linewidth=0.5)
        axes[0].scatter(
            particle.time_points, particle.positions[:, 0], c="r", marker="."
        )
        axes[0].set_xlabel("time")
        axes[0].set_ylabel("x")
        axes[1].plot(dct["time"], dct[POS][:, 1], linewidth=0.5)
        axes[1].scatter(
            particle.time_points, particle.positions[:, 1], c="r", marker="."
        )
        axes[1].set_xlabel("time")
        axes[1].set_ylabel("y")
        plt.show()



Execution
---------

A few functions will run everything we need.

::

    def execute():
        particle, dct = run()
        plot(dct, particle)
        plot_vs_time(dct, particle)


    if __name__ == "__main__":
        execute()

Results
-------

Here's how our particle has moved...

.. image:: ../../examples/tutorial_3a.png
    :width: 600

And we can see how the interpolation spline has controlled the velocity, 
and hence position, over time.

.. image:: ../../examples/tutorial_3b.png
    :width: 600
    
