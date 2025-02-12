# -*- coding: utf-8 -*-
"""
Created on Fri May 22 10:36:19 2020

@author: Reuben

This module contains more specialised solvers based on scipy.

"""

import numpy as np

try:
    from scipy.integrate import ode

    scipy_found = True
except ModuleNotFoundError:
    scipy_found = False


FINAL = "final"
STOP = "stop"


class Logger:
    def __init__(self, system, x_name="time", squeeze=True):
        self.system = system
        self._x_name = x_name
        self._squeeze = squeeze
        self._log_data = []

    def log(self, state_vec, t):
        log = {"stop": False, self._x_name: t}
        system = self.system
        slicer = system.slicer
        system.step(state_vec, t, log)
        state = slicer.get_state(state_vec, system.keys)
        state.update(log)
        self._log_data.append(state)
        return log

    def get_data_dct(self):
        keys = list(self._log_data[0].keys())
        data_dct = {}
        for key in keys:
            if self._squeeze:
                data = np.array(
                    [np.squeeze(row[key]) for row in self._log_data]
                )
            else:
                data = np.array([row[key] for row in self._log_data])
            data_dct[key] = data
        return data_dct


class ODEIntegrator:
    def __init__(
        self,
        framerate=60.0,
        interface_cls=None,
        integrator_name="lsoda",
        integrator_kwargs=None,
    ):
        self.framerate = framerate
        self._int_kwargs = (
            {} if integrator_kwargs is None else integrator_kwargs
        )
        if interface_cls is None:
            if scipy_found:
                interface_cls = ode
            else:
                raise ImportError("Scipy not found for default integrator.")
        self._interface_cls = interface_cls
        self._integrator_name = integrator_name

    def _setup_integrator(self, system):
        """Set up the integrator"""
        integrator = self._interface_cls(system.tstep)
        integrator.set_integrator(self._integrator_name, **self._int_kwargs)
        integrator.set_initial_value(system.init_vec)
        return integrator

    def _make_x_vector(self, end):
        """Make a regular x vector

        Args:
            end (float): The end of the integration.

        Returns:
            ndarray: The x vector, rounded to nearest whole frame
        """
        rem = end % (1 / self.framerate)
        x_end = end - rem
        n = int(x_end * self.framerate)
        return np.linspace(0, x_end, n)

    def run(self, system, end, x_name="time", squeeze=True, **kwargs):
        """Run the solver

        Args:
            end (float): The end point for the integration. Integration starts
                from 0 and will end at this value. Often this is a time.

        Returns:
            dict: A dictionary where keys are the variable names and
            other logged names, and the values are ndarrays of the values
            through time.
        """
        x_vec = self._make_x_vector(end)
        integrator = self._setup_integrator(system)
        dt = x_vec[1] - x_vec[0]
        t = 0.0
        logger = Logger(system, x_name=x_name, squeeze=squeeze)
        logger.log(system.init_vec, 0)
        stop = False
        while integrator.successful() and t < end and not stop:
            t = t + dt
            state_vec = integrator.integrate(t)
            log = logger.log(state_vec, t)
            if log[STOP]:
                stop = True
        data_dct = logger.get_data_dct()
        return data_dct
