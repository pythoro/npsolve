Tutorial 1b - An alternative
============================

The state dictionary always contains current values for all variables. This
is why, in Tutorial 1, we could pass it from the Assembly to each component.

Here, we'll see a different way to achieve the same result.

We'll make the components store the state first in their own stage calls,
to provide it to the Assembly on request.

::

    class Component1:
        def set_state(self, state, t, log):
            self._state = state

        def get_pos(self):
            return self._state[COMP1_POS]

        def set_comp2_force(self, force):
            self._comp2_force = force

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
        def set_state(self, state, t, log):
            self._state = state

        def get_force(self):
            return 1.0 * self._state[COMP2_VALUE]

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
            

Now, we'll modify Assembly as follows.

::

    class Assembly:
        """Handle inter-dependencies."""

        def __init__(self, comp1, comp2):
            self.comp1 = comp1
            self.comp2 = comp2

        def precalcs(self, state, t, log):
            """Inject dependencies for later calculations in 'step' methods."""
            comp1 = self.comp1
            comp2 = self.comp2
            comp1_pos = comp1.get_pos()
            comp2_force = comp2.get_force()
            if log:
                # Log whatever we want here into a dictionary.
                log[COMP2_FORCE] = comp2_force
            comp1.set_comp2_force(comp2_force)
            comp2.set_comp1_pos(comp1_pos)


Finally, we'll add stage calls to the components before the call to `precals`:

::

    def get_system():
        component1 = Component1()
        component2 = Component2()
        assembly = Assembly(component1, component2)
        system = npsolve.System()
        system.add_component(component1, "comp1", "step")
        system.add_component(component2, "comp2", "step")
        system.add_component(assembly, "assembly", None)
        system.set_stage_calls([
            ('comp1', 'set_state'),
            ('comp2', 'set_state'),
            ("assembly", "precalcs"),
            ])
        return system


This method is arguably more flexible as it allows for logging when
setting the state.