# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 14:34:54 2019

@author: Reuben

Npsolve has a simple, small core. It's designed to give
good flexibility without compromising on performance.

"""

from __future__ import annotations

import numpy as np
import traceback
import typing


class Partial:
    """A base class responsible for a set of variables

    Note:
        __init__ method must be called.

    """

    def __init__(self):
        self.npsolve_vars = {}
        self.state = {}
        self.__cache_methods = self._get_cached_methods()
        self.__cache_clear_functions = self._get_cache_clear_functions()
        self.cache_clear()  # Useful for iPython console autoreload.

    def connect_solver(self, solver: "Solver") -> None:
        """Register connection with the solver"""
        solver.connect_partial(self)

    def set_vectors(self, state_dct: dict, ret_dct: dict) -> None:
        """Override to set up views of the state vector

        Args:
            state_dct (dict): A dictionary of numpy array views for the state
            of all variables. Provided by the Solver.
            ret_dct (dict): A similar dictionary of return values. Not
            usually used.
        """
        pass

    def _set_state(self, state: dict) -> None:
        """Set the state dictionary

        Args:
            state (dict): The state dictionary

        Note:
            The state dictionary de-numpify's scalars by default.
        """
        self.state = state

    def _get_vars(self) -> dict:
        return self.npsolve_vars

    def set_meta(self, name: str, **kwargs) -> None:
        """Set meta information for a variable

        Args:
            name (str): The name of the variable.
            **kwargs: Key word attributes for the variable.
        """
        self.npsolve_vars[name].update(kwargs)

    def set_init(
        self, name: str, init: typing.Union[float, int, np.ndarray]
    ) -> None:
        """Set the initial value for a variable

        Args:
            name (str): The variable name
            init (array-like): The initial value(s). Can be a scalar or 1D
                ndarray.
        """
        self.npsolve_vars[name]["init"] = np.atleast_1d(init)

    def get_init(self, name: str) -> typing.Union[float, int, np.ndarray]:
        """Get the initial value for a variable

        Args:
            name (str): The variable name
        """
        return self.npsolve_vars[name]["init"]

    def add_var(
        self,
        name: str,
        init: typing.Union[float, int, np.ndarray],
        safe: bool = True,
        live: bool = True,
        **kwargs,
    ) -> None:
        """Add a new variable

        Args:
            name (str): The variable name
            init (array-like): The initial value(s). Can be a scalar or 1D
                ndarray.
            safe (bool): If true, ensures that variable name does not already
                exist.
            live (bool): Deprecated. Always true.
            **kwargs: Optional kew word attributes for the variable.
        """
        if safe and name in self.npsolve_vars:
            raise KeyError(str(name) + " already exists")
        self.state[name] = np.atleast_1d(init)
        if live:
            self.npsolve_vars[name] = {}
            self.set_init(name, init)
            self.set_meta(name, **kwargs)

    def clear_vars(self) -> None:
        self.npsolve_vars = {}

    def add_vars(self, dct: dict) -> None:
        """Add multiple variables

        Args:
            dct (dict): A dictionary in which keys are variable names and
                values are dictionaries with name, initial value, etc.
        """
        for name, d in dct.items():
            self.add_var(name, **d)

    def _get_cached_methods(self) -> list[typing.Callable]:
        """Return a list of the cached methods

        Note:
            The solver clears the cache for each method with each step.

        Returns:
            list: A list of the methods.
        """
        functions = []
        for name in dir(self):
            if name.startswith("__") and name.endswith("__"):
                continue
            func = getattr(self, name, None)
            if hasattr(func, "cacheable"):
                functions.append(func)
        return functions

    def _get_cache_clear_functions(self) -> list[typing.Callable]:
        """Get the cache_clear functions for cached methods

        Returns:
            list: A list of functions
        """
        return [func.cache_clear for func in self.__cache_methods]

    def cache_clear(self) -> None:
        """Clear the cache for all cached methods"""
        [f() for f in self.__cache_clear_functions]

    def _set_caching(self, enable: bool) -> None:
        """Enable / disable caching in cached methods"""
        [f.set_caching(enable) for f in self.__cache_methods]

    def _get_step_method(self) -> typing.Callable:
        """Return the step method"""
        return self.step

    def step(self, state_dct: dict, *args) -> dict:
        """Takes the current state and returns a dictionary.

        The dictionary should contain keys for each of the variables
        declared in the instance, and each value is usually a derivative.
        """
        raise NotImplementedError("The step method must be implemented.")

    def enable_caching(self):
        """Enable caching"""
        self._set_caching(enable=True)

    def disable_caching(self):
        """Enable caching"""
        self._set_caching(enable=False)


class Solver:
    """The solver that pulls together the partials and allows solving"""

    def __init__(self):
        self._cache_clear_functions = []
        self._container = None
        self.state = {}
        self._partials = []

    def _setup_vecs(self, dct: dict) -> (dict, np.ndarray, np.ndarray):
        """Create vectors and slices based on a dictionary of variables

        Args:
            dct (dict): A dictionary in which keys are variable names and
                values are dictionaries of attributes, which include an
                'init' entry for initial value.

        Returns:
            dict: A dictionary of slices. Each slice corresponds to values
            for a given variable in the state and return vectors.
            ndarray: A 1d state vector.
            ndarray: A 1d vector for return values
        """
        slices = {}
        meta = {}
        i = 0
        for key, item in dct.items():
            n = len(item["init"])
            slices[key] = slice(i, i + n)
            meta[key] = item
            i += n
        state = np.zeros(i)
        for key, slc in slices.items():
            state[slc] = np.atleast_1d(dct[key]["init"])
        ret = np.zeros(i)
        return slices, state, ret

    def _make_dcts(
        self, slices: dict, state: np.ndarray, ret: np.ndarray
    ) -> (dict, dict):
        """Create dictionaries of numpy views for all variables"""
        state_dct = {}
        ret_dct = {}
        for name, slc in slices.items():
            state_view = state[slc]
            state_view.flags["WRITEABLE"] = False
            state_dct[name] = state_view

            ret_view = ret[slc]
            ret_dct[name] = ret_view
        return state_dct, ret_dct

    def _fetch_vars(self) -> dict:
        """Collect variable data from connected Partial instances"""
        dct = {}
        dicts = [partial._get_vars() for partial in self._partials]
        for d in dicts:
            for key in d.keys():
                if key in dct:
                    raise KeyError(
                        'Variable "'
                        + str(key)
                        + '" is defined '
                        + "by more than one Partial class."
                    )
            dct.update(d)
        return dct

    def _emit_vectors(self) -> None:
        """Pass out vectors and slices to connected Partial instances"""
        for partial in self._partials:
            partial.set_vectors(
                state_dct=self.npsolve_state_dct, ret_dct=self.npsolve_ret_dct
            )

    def _emit_state(self) -> None:
        """Pass out vectors and slices to connected Partial instances"""
        for partial in self._partials:
            partial._set_state(state=self.npsolve_state_dct)

    def _fetch_step_methods(self) -> list[typing.Callable]:
        lst = [partial._get_step_method() for partial in self._partials]
        out = []
        for ret in lst:
            if isinstance(ret, list):
                out.extend(ret)
            else:
                out.append(ret)
        return out

    def fetch_partials(self) -> list[Partial]:
        """Fetch a dictionary of all connected Partial instances"""
        return {partial.npsolve_name: partial for partial in self._partials}

    def _fetch_cache_clears(self) -> list[typing.Callable]:
        lst = [
            partial._get_cache_clear_functions() for partial in self._partials
        ]
        out = []
        for l in lst:
            out.extend(l)
        return out

    def npsolve_init(self) -> None:
        """Initialise the Partials and be ready to solve"""
        dct = self._fetch_vars()
        slices, state, ret = self._setup_vecs(dct)
        state_dct, ret_dct = self._make_dcts(slices, state, ret)
        self.npsolve_variables = dct
        self.npsolve_slices = slices
        self.npsolve_state = state
        self.npsolve_initial_values = state.copy()
        self.npsolve_ret = ret
        self.npsolve_state_dct = state_dct
        self.npsolve_ret_dct = ret_dct
        self._emit_vectors()
        self._emit_state()
        self._step_methods = self._fetch_step_methods()
        self._cache_clear_functions = self._fetch_cache_clears()
        for partial in self._partials:
            partial._set_caching(enable=True)

    def npsolve_finish(self) -> None:
        """Tidy up after a round of solving"""
        for partial in self._partials:
            partial._set_caching(enable=False)

    def one_way_step(self, vec: np.ndarray, *args, **kwargs) -> None:
        """The method to be called every iteration with no return val

        Args:
            vec (ndarray): The state vector (passed in by the solver)
            args: Optional arguments passed to step method in each Partial.
            kwargs: Optional keyword arguments for each step method call.

        Returns:
            None

        Note: This method relies on other methods being used to inform the
            solver during its iteration.
        """
        self.npsolve_state[:] = vec
        state_dct = self.npsolve_state_dct
        for f in self._cache_clear_functions:
            f()
        for step in self._step_methods:
            step(state_dct, *args, **kwargs)

    def step(self, vec: np.ndarray, *args, **kwargs) -> np.ndarray:
        """The method to be called every iteration by the numerical solver

        Args:
            vec (ndarray): The state vector (passed in by the solver)
            args: Optional arguments passed to step method in each Partial.
            kwargs: Optional keyword arguments for each step method call.

        Returns:
            np.ndarray: A vector passed back to the solver. This will often
            contain derivatives for integration problems and error or cost
            values for optimisation problems.

        """
        self.npsolve_state[:] = vec
        state_dct = self.npsolve_state_dct
        ret_dct = self.npsolve_ret_dct
        for f in self._cache_clear_functions:
            f()
        for step in self._step_methods:
            for name, val in step(state_dct, *args, **kwargs).items():
                ret_dct[name][:] = val
        return self.npsolve_ret

    def tstep(self, t: float, vec: np.ndarray, *args, **kwargs) -> np.ndarray:
        """The method to be called every iteration by the numerical solver

        Args:
            vec (ndarray): The state vector (passed in by the solver)
            args: Optional arguments passed to step method in each Partial.
            kwargs: Optional keyword arguments for each step method call.

        Returns:
            dict: A dictionary containing keys for each variable. The values
                must match the shape of the state. These will often contain
                derivatives for integration problems and error or cost values
                for optimisation problems.

        Note:
            This method is similar ot the :meth:`step` method, but is used
            where a time value is passed as the first argument.
        """
        self.npsolve_state[:] = vec
        state_dct = self.npsolve_state_dct
        ret_dct = self.npsolve_ret_dct
        for f in self._cache_clear_functions:
            f()
        for step in self._step_methods:
            try:
                ret = step(state_dct, t, *args, **kwargs)
            except TypeError as e:
                traceback.print_exc()
                raise TypeError("Error from " + str(step) + ": " + e.args[0])
            if not isinstance(ret, dict):
                raise ValueError(
                    str(step)
                    + " did not return a dictionary of "
                    + "derivatives."
                )
            for name, val in ret.items():
                ret_dct[name][:] = val
        return self.npsolve_ret

    def as_dct(self, sol: np.ndarray) -> dict[str, np.array]:
        """Split out solution array into dictionary of values

        Args:
            sol (ndarray): A 1D or 2D array where columns correspond to state
                values

        This convenience method splits out a 2D array into a dictionary of
        vectors or arrays, with variables as keys.
        """
        d = {}
        if sol.ndim == 1:
            for key, slc in self.npsolve_slices.items():
                d[key] = sol[slc]
        if sol.ndim == 2:
            for key, slc in self.npsolve_slices.items():
                d[key] = sol[:, slc]
        return d

    def get_state_dct(self, squeeze=True, unitise=True) -> dict:
        """Return the current state dictionary

        Args:
            squeeze (bool): Squeeze np.ndarrays to minimal dimensions.
            unitize (bool): Convert size-1 np.ndarrays to python floats.
        """
        dct = self.npsolve_state_dct.copy()
        if squeeze:
            for k in dct.keys():
                dct[k] = np.squeeze(dct[k])
        if unitise:
            for k in dct.keys():
                v = np.atleast_1d(dct[k])
                if len(v) == 1:
                    dct[k] = v.item()
        return dct

    def connect_partial(self, partial: Partial) -> None:
        """Connect a Partial instance"""
        self._partials.append(partial)

    def connect_partials(
        self, partials: typing.Union[dict[str, Partial], list[Partial]]
    ) -> None:
        """Connect a dict or list of partials to the Solver instance

        Args:
            partials (list, dict, Partial): A list or dictionary of Partial
                instances.

        """
        if isinstance(partials, dict):
            for partial in partials.values():
                partial.connect_solver(self)
        elif isinstance(partials, list):
            for partial in partials:
                partial.connect_solver(self)
        elif isinstance(partials, Partial):
            partials.connect_solver(self)
        else:
            raise ValueError(
                "partials argument must be a list of list or dict of "
                + " npsolve.Partial instances."
            )

    connect = connect_partials




class Slicer:
    def __init__(self, dct: dict | None = None):
        self._slices = {}
        self._i = 0
        if dct is not None:
            self.add_dict(dct)

    @property
    def slices(self):
        return self._slices.copy()

    @property
    def length(self):
        return self._i

    def add_dict(self, dct: dict):
        for key, val in dct.items():
            self.add(key, val)

    def add(self, key: str, val: np.ndarray | float):
        vec = np.atleast_1d(val)
        n = len(vec)
        self._slices[key] = slice(self._i, self._i + n)
        self._i += n

    def get_slice(self, key: str):
        return self._slices[key]

    def get_view(self, vec: np.ndarray, key: str, writeable: bool):
        view = vec[self.get_slice(key)]
        view.flags.writeable = writeable
        return view

    def get_state_vec(self, dct: dict):
        state = np.zeros(self._i)
        for key, val in dct.items():
            state[self.get_slice(key)] = val
        return state

    def get_state(self, state_vec, keys, writeable=False):
        state = {}
        for key in keys:
            state[key] = self.get_view(state_vec, key, writeable=writeable)
        return state


class Package:
    def __init__(self):
        self._stage_calls = []
        self._components = {}
        self._deriv_methods = {}

    @property
    def components(self):
        return self._components

    @property
    def slices(self):
        return self.slicer.slices

    def add_component(self, component: object, name: str, deriv_method_name: str | None) -> None:
        self._components[name] = component
        if deriv_method_name is not None:
            try:
                method = getattr(component, deriv_method_name)
            except AttributeError as e:
                raise AttributeError("Derivative method not found for component: '" +
                + deriv_method_name + "'")
            self._deriv_methods[name] = method

    def set_stage_calls(self, lst: list[str, str]):
        self._stage_calls = []
        for component_name, method_name in lst:
            self.add_stage_call(component_name, method_name)

    def add_stage_call(self, component_name: str, method_name: str):
        try:
            component = self._components[component_name]
        except KeyError as e:
            raise KeyError("Component not found in Package: '" + component_name 
            + "'. Use the add_component method to add it first.") from e
        try:
            method = getattr(component, method_name)
        except AttributeError as e:
            raise AttributeError("Method not found for component '" + component_name +
            "': '" + method_name + "'")
        self._stage_calls.append((component_name, method))

    def setup(self, inits):
        slicer = Slicer(inits)
        state_vec = slicer.get_state_vec(inits)
        ret_vec = np.zeros_like(state_vec)
        state = slicer.get_state(state_vec, inits.keys(), writeable=False)
        ret = slicer.get_state(ret_vec, inits.keys(), writeable=True)
        self.inits = inits
        self.init_vec = state_vec.copy()
        self._state_vec = state_vec
        self._ret_vec = ret_vec
        self._state = state
        self._ret = ret
        self.slicer = slicer

    def step(self, vec, t, *args, log=None, **kwargs):
        self._state_vec[:] = vec
        ret = self._ret
        state = self._state
        for component_name, method in self._stage_calls:
            method(state, t, *args, log=log, **kwargs)
        for deriv_method in self._deriv_methods.values():
            for name, val in deriv_method(state, t, *args, log=log, **kwargs).items():
                ret[name][:] = val  # sets values efficiently in self._ret
        return self._ret_vec

    def get_state(self, state_vec):
        return self.slicer.get_state(state_vec.copy())