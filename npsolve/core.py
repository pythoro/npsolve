# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 14:34:54 2019

@author: Reuben

Npsolve has a simple, small core. It's designed to give
good flexibility without compromising on performance.

"""

from __future__ import annotations

import numpy as np


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

    def __getitem__(self, component_name: str) -> object:
        return self._components[component_name]

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
        keys = list(inits.keys())
        state = slicer.get_state(state_vec, keys, writeable=False)
        ret = slicer.get_state(ret_vec, keys, writeable=True)
        self.inits = inits
        self.init_vec = state_vec.copy()
        self._state_vec = state_vec
        self._ret_vec = ret_vec
        self._state = state
        self._ret = ret
        self.keys = keys
        self.slicer = slicer

    def step(self, vec, t, log=None, *args, **kwargs):
        self._state_vec[:] = vec
        ret = self._ret
        state = self._state
        for component_name, method in self._stage_calls:
            method(state, t, log=log, **kwargs)
        for deriv_method in self._deriv_methods.values():
            for name, val in deriv_method(state, t, log, *args, **kwargs).items():
                ret[name][:] = val  # sets values efficiently in self._ret
        return self._ret_vec

    def tstep(self, t, vec, log=None, *args, **kwargs):
        self._state_vec[:] = vec
        ret = self._ret
        state = self._state
        for component_name, method in self._stage_calls:
            method(state, t, log=log, **kwargs)
        for deriv_method in self._deriv_methods.values():
            for name, val in deriv_method(state, t, log, *args, **kwargs).items():
                ret[name][:] = val  # sets values efficiently in self._ret
        return self._ret_vec

    def get_state(self, state_vec):
        return self.slicer.get_state(state_vec.copy())