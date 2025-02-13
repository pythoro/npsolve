"""The core functionality of npsolve.

Created on Mon Aug  5 14:34:54 2019

@author: Reuben

Npsolve has a simple, small core. It's designed to give good flexibility,
without compromising on performance.

"""

from __future__ import annotations

import numpy as np


class Slicer:
    """Manage variable arrays and views of those arrays.

    The main function is to take dictionaries of variables and create numpy
    arrays from them. Then, it creates a dictionary of numpy views
    of those arrays for each variable.

    The input dictionaries can include floats as well as np.ndarrays of
    shape (n,) for each variable.

    """

    def __init__(self, dct: dict[str : np.ndarray] | None = None) -> None:
        """Create a new instance.

        Args:
            dct (optional, dict | None): A dictionary of string-value pairs
                to add.
        """
        self._slices = {}
        self._i = 0
        if dct is not None:
            self.add_dict(dct)

    @property
    def slices(self) -> dict[str:slice]:
        """Return the slice mapping for a full array."""
        return self._slices.copy()

    @property
    def length(self) -> int:
        """Return the total number of parameters."""
        return self._i

    def add_dict(self, dct: dict) -> None:
        """Add a template dictionary.

        Args:
            dct (dict): The template dictionary.
        """
        for key, val in dct.items():
            self.add(key, val)

    def add(self, key: str, val: np.ndarray | float) -> None:
        """Add a key with a template value.

        Args:
            key (str): The key.
            val (np.ndarray | float): The template value used to configure
                the view slices.

        """
        vec = np.atleast_1d(val)
        n = len(vec)
        self._slices[key] = slice(self._i, self._i + n)
        self._i += n

    def get_slice(self, key: str) -> slice:
        """Return the slice for a given key.

        Args:
            key (str): The variable key.

        Returns:
            slice: The slice for the main array corresponding to this
            variable.
        """
        return self._slices[key]

    def get_view(
        self, vec: np.ndarray, key: str, writeable: bool
    ) -> np.ndarray[float]:
        """Return a view of the full vector for a specific key.

        Args:
            vec (np.ndarray): The full vector, a (n,) np.ndarray.
            key (str): The key for the value.
            writeable (optional, bool): If True, the view is writeable.
                Defaults to False.

        Returns:
            np.ndarray: A view of the full vector, vec, corresponding to the
            key.
        """
        view = vec[self.get_slice(key)]
        view.flags.writeable = writeable
        return view

    def get_state_vec(self, dct: dict[str : np.ndarray | float]) -> np.ndarray:
        """Return a full vector filled with entries from a dictionary.

        Args:
            dct (dict): The dictionary of values, where keys correspond with
                those previously added to the Slicer.

        Returns:
            np.ndarray: The full np.ndarray containing the data from the
            dictionary in the appropriate indices.
        """
        state = np.zeros(self._i)
        for key, val in dct.items():
            state[self.get_slice(key)] = val
        return state

    def get_state(
        self, state_vec: np.ndarray[float], writeable: bool = False
    ) -> dict[str : np.ndarray]:
        """Return a state dictionary, given a full vector.

        Args:
            state_vec (np.ndarray[float]): The full vector, containing all
                entries.
            writeable (bool): If True, make the views for each variable
                writeable, instead of read-only. Defaults to False.
        """
        state = {}
        for key in self._slices:
            state[key] = self.get_view(state_vec, key, writeable=writeable)
        return state


class System:
    """The interface between an integrator and objects in the system."""

    def __init__(self) -> None:
        """Create a new instance."""
        self._stage_calls = []
        self._initialise_calls = []
        self._components = {}
        self._deriv_methods = {}

    def __getitem__(self, component_name: str) -> object:
        """Return the named component, if it is found."""
        return self._components[component_name]

    @property
    def components(self) -> dict[str:object]:
        """Return the dictionary of all added components, keyed by name."""
        return self._components

    @property
    def slices(self) -> dict[str:slice]:
        """Dictionary of all variables and thier slices of the state vec."""
        return self.slicer.slices

    def add_component(
        self, component: object, name: str, deriv_method_name: str | None
    ) -> None:
        """Add a new component.

        Args:
            component (object): The object to add.
            name (str): The component name.
            deriv_method_name (str | None): The name of the method called at
                the end of each time step to return the derivatives. If None,
                the component is not called.

        Raises:
            KeyError: If the component name has already been added.

        """
        if name in self._components:
            raise KeyError("Component name cannot be added twice: " + name)
        self._components[name] = component
        if deriv_method_name is not None:
            try:
                method = getattr(component, deriv_method_name)
            except AttributeError as e:
                raise AttributeError(
                    "Derivative method not found for component: '"
                    + deriv_method_name
                    + "'"
                ) from e
            self._deriv_methods[name] = method

    def set_stage_calls(self, stage_calls: list[str, str]) -> None:
        """Set all stage calls during a time step.

        Args:
            stage_calls (list[str, str]): A list of tuples, where the first
                entry is the component name, and the second entry is the
                method name to call. The Package will call each method in the
                same order as entered here.
        """
        self._stage_calls = []
        for component_name, method_name in stage_calls:
            self.add_stage_call(component_name, method_name)

    def add_stage_call(self, component_name: str, method_name: str) -> None:
        """Append a stage call to the end of the stage calls list.

        Args:
            component_name (str): The name of the component with the method.
            method_name (str): The method name to call.
        """
        try:
            component = self._components[component_name]
        except KeyError as e:
            raise KeyError(
                "Component not found in System: '"
                + component_name
                + "'. Use the add_component method to add it first."
            ) from e
        try:
            method = getattr(component, method_name)
        except AttributeError as e:
            raise AttributeError(
                "Method not found for component '"
                + component_name
                + "': '"
                + method_name
                + "'"
            ) from e
        self._stage_calls.append((component_name, method))

    def set_initialise_calls(self, init_calls: list[str, str]) -> None:
        """Set all initialise calls, prior to the first time step.

        Args:
            init_calls (list[str, str]): A list of tuples, where the first
                entry is the component name, and the second entry is the
                method name to call. The Package will call each method in the
                same order as entered here.

        Note:
            Initialise calls are called only once prior to commencing the
            integration.
        """
        self._initialise_calls = []
        for component_name, method_name in init_calls:
            self.add_initialise_call(component_name, method_name)

    def add_initialise_call(
        self, component_name: str, method_name: str
    ) -> None:
        """Append an initialise call to the end of the initialise calls list.

        Args:
            component_name (str): The name of the component with the method.
            method_name (str): The method name to call.

        Note:
            Initialise calls are called only once prior to commencing the
            integration.
        """
        try:
            component = self._components[component_name]
        except KeyError as e:
            raise KeyError(
                "Component not found in System: '"
                + component_name
                + "'. Use the add_component method to add it first."
            ) from e
        try:
            method = getattr(component, method_name)
        except AttributeError as e:
            raise AttributeError(
                "Method not found for component '"
                + component_name
                + "': '"
                + method_name
                + "'"
            ) from e
        self._initialise_calls.append((component_name, method))

    def _initialise_components(
        self,
        state: dict[str : np.ndarray],
        t: float,
        log: dict[str : np.ndarray | float] | None,
        *args: float,
        **kwargs: float,
    ) -> None:
        """Initialise components if required.

        Args:
            state (dict[str: np.ndarray]): The dictionary of state values.
                This will be equal to the initial values.
            t (float): The current time.
                This will be 0.0.
            log (dict[str: np.ndarray | float] | None): May be None, or a
                mutatable dictionary to add values too.
            *args (optional, float): Optional positional arguments to pass
                to the method.
            **kwargs (optional, float): Optional keyword arguments to pass
                to the method.
        """
        for _, method in self._initialise_calls:
            method(state, t, log, *args, **kwargs)

    def setup(self, inits: dict[str : np.ndarray | float]) -> None:
        """Setup the System.

        Args:
            inits (dict[str: np.ndarray | float]): A dictionary where each
                key is a variable name and each value is the corresponding
                initial value. The inits dict must include all variables
                required by called components in the system.

        Note:
            This call sets up the System slices, to translate between the
            components and the integrator.
        """
        slicer = Slicer(inits)
        state_vec = slicer.get_state_vec(inits)
        ret_vec = np.zeros_like(state_vec)
        keys = list(inits.keys())
        state = slicer.get_state(state_vec, writeable=False)
        ret = slicer.get_state(ret_vec, writeable=True)
        self.inits = inits
        self.init_vec = state_vec.copy()
        self._state_vec = state_vec
        self._ret_vec = ret_vec
        self._state = state
        self._ret = ret
        self.keys = keys
        self.slicer = slicer
        self._initialise_components(state, 0.0, None)

    def step(
        self,
        vec: np.ndarray[float],
        t: float,
        log: dict | None = None,
        *args: float,
        **kwargs: float,
    ) -> np.ndarray[float]:
        """Call the components for the time step and return derivatives.

        Args:
            vec (np.ndarray[float]]): State values for the current trial
                in a single numpy ndarray.
            t (float): The current time through the integration, starting
                from 0.0.
            log (dict[str: np.ndarray | float] | None): May be None, or a
                mutatable dictionary to add values too.
            *args (optional, float): Optional positional arguments to pass
                to the method.
            **kwargs (optional, float): Optional keyword arguments to pass
                to the method.

        Returns:
            np.ndarray: An array of derivatives for the state values.
        """
        self._state_vec[:] = vec
        ret = self._ret
        state = self._state
        for _, method in self._stage_calls:
            method(state, t, log, *args, **kwargs)
        for deriv_method in self._deriv_methods.values():
            for name, val in deriv_method(
                state, t, log, *args, **kwargs
            ).items():
                ret[name][:] = val  # sets values efficiently in self._ret
        return self._ret_vec

    def tstep(
        self,
        t: float,
        vec: np.ndarray[float],
        log: dict | None = None,
        *args: float,
        **kwargs: float,
    ) -> np.ndarray[float]:
        """Call the components for the time step and return derivatives.

        Args:
            t (float): The current time through the integration, starting
                from 0.0.
            vec (np.ndarray[float]]): State values for the current trial
                in a single numpy ndarray.
            log (dict[str: np.ndarray | float] | None): May be None, or a
                mutatable dictionary to add values too.
            *args (optional, float): Optional positional arguments to pass
                to the method.
            **kwargs (optional, float): Optional keyword arguments to pass
                to the method.

        Returns:
            np.ndarray: An array of derivatives for the state values.

        Note:
            This method is identical to :meth:`step` except the order of the
            `vec` and `t` arguments are reversed.
        """
        self._state_vec[:] = vec
        ret = self._ret
        state = self._state
        for _, method in self._stage_calls:
            method(state, t, log, *args, **kwargs)
        for deriv_method in self._deriv_methods.values():
            for name, val in deriv_method(
                state, t, log, *args, **kwargs
            ).items():
                ret[name][:] = val  # sets values efficiently in self._ret
        return self._ret_vec

    def get_state(
        self, state_vec: np.ndarray[float]
    ) -> dict[str : np.ndarray]:
        """Get the state dictionary, given the state vec.

        Args:
            state_vec (np.ndarray[float]): The full state vector.

        Returns:
            dict[str: np.ndarray]: The state dictionary, where keys are
            variable names and values are their corresponding values.
        """
        return self.slicer.get_state(state_vec.copy())
