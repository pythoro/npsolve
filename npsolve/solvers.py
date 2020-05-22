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
except:
    scipy_found = False

from . import core

FINAL = 'final'
STOP = 'stop'


class Integrator(core.Solver):
    """ A versatile integrator, with extra logging and stop flag
    
    This integrator allows variables to be logged during the integration,
    which are then included in the output. In addition, Partial instances
    can set a flag to stop the integration at any point.
    
    Args:
        status (defaultdict): A dictionary that contains status flags. Key
            flags are npsolve.solvers.FINAL and npsolve.solvers.STOP (which are
            strings). The default should be a function that returns None.
            Obtain one by calling `npsolve.get_status(<name>)`.
        logger (defaultdict): A dictionary in which the 
            default values are lists. Obtain one by calling
            `npsolve.get_solver(<name>)`.
        framerate (float): [OPTIONAL] The number of return values per unit x
            (which is often time). Defaults to 60.0.
        interface_cls (class): [OPTIONAL] The class of interface to use for
            integrator algorithms. Defaults to scipy.intergrate.ode if 
            scipy is found.
        integrator_name (str): [OPTIONAL] The name of the integrator to use.
            Defaults to 'lsoda'.
        squeeze (bool): [OPTIONAL] Ensure output arrays are squeezed. Defaults
            to True.
        x_name (str): [OPTIONAL]: The name for the x value, which is logged
            in the outputs. Defaults to 'time'.
        update_inits (bool): [OPTIONAL] Update the initial values of the 
            Partial instances with the solution at the end. Defaults to False.
        Other keyword arguments: [OPTIONAL] Are passed to the integrator by 
            the call `interface_cls.set_integrator(integrator_name, **kwargs)`.

    Returns:
        dict: A dictionary of integrated values. The values are ndarrays, which
        are at the framerate specified by the 'framerate' argument.
    
    Adding logged variables:
        Only log variables when the solver has finalised the current frame.
        Integrators like scipy's ode 'lsoda' use variable time steps, and take
        numerous guesses at the state as they jump around in time. Once it has
        reached an accurate state for the x value at the end of the frame, 
        status[npsolve.solvers.FINAL] is set to True. Only log values when this
        flat is True. An example:
        
        ::
            
            if status[FINAL]:
                logger['variable_name_1'] = current_value
                
    Stopping the integration:
        Stop the integration by setting status[npsolve.solvers.STOP] to True.
            
    
    """
    def __init__(self,
                 status=None,
                 logger=None,
                 framerate=60.0,
                 interface_cls=None,
                 integrator_name='lsoda',
                 squeeze=True,
                 x_name='time',
                 update_inits=False,
                 **kwargs):
        super().__init__()
        self.status = status
        self.logger = logger
        self.framerate = framerate
        self.kwargs = kwargs
        if interface_cls is None:
            if scipy_found:
                interface_cls = ode
            else:
                raise ImportError('Scipy not found for default integrator.')
        self._interface_cls = interface_cls
        self._integrator_name = integrator_name
        self._squeeze = squeeze
        self._x_name = x_name
        self._update_inits = update_inits
        
    def _setup_integrator(self):
        """ Set up the integrator """
        self.npsolve_init()
        integrator = self._interface_cls(self.tstep)
        integrator.set_integrator(self._integrator_name, **self.kwargs)
        integrator.set_initial_value(self.npsolve_initial_values)
        return integrator
        
    def _log_initial_step(self):
        """ This needs to be called to log the first step """
        self.logger.clear()
        self.status[STOP] = False
        self.status[FINAL] = True
        self.tstep(0, self.npsolve_initial_values)
        self.status[FINAL] = False
        self.logger[self._x_name].append(0)
        return self.npsolve_initial_values.copy()

    def _make_x_vector(self, end):
        """ Make a regular x vector 
        
        Args:
            end (float): The end of the integration.
            
        Returns:
            ndarray: The x vector, rounded to nearest whole frame
        """
        rem = end % (1 / self.framerate)
        x_end = end - rem
        n = int(x_end * self.framerate)
        return np.linspace(0, x_end, n)
    
    def _vectorise(self, dct):
        """ Make the outputs numpy arrays 
        
        Args:
            dct (dict): A dictionary of lists.
            
        Returns:
            dict: A dictionary of ndarrays
        """
        if self._squeeze:
            return {k: np.squeeze(np.array(v)) for k, v in dct.items()}
        else:
            return {k: np.array(v) for k, v in dct.items()}
    
    def _update_initial_values(self):
        """ Update the initial values to the end values """
        for name, partial in self.fetch_partials().items():
            for var in partial.npsolve_vars:
                partial.set_init(var, self.npsolve_state_dct[var])
    
    def run(self, end, **kwargs):
        """ Run the solver 
        
        Args:
            end (float): The end point for the integration. Integration starts
                from 0 and will end at this value. Often this is a time.
        
        Returns:
            dict: A dictionary where keys are the variable names and
            other logged names, and the values are ndarrays of the values
            through time. 
        """
        status = self.status
        logger = self.logger
        x_name = self._x_name
        x_vec = self._make_x_vector(end)
        i = self._setup_integrator(**kwargs)
        solution = [self._log_initial_step()]
        x_end = x_vec[-1]
        dt = x_vec[1] - x_vec[0]
        t = 0.0
        while i.successful() and t < end and not status[STOP]:
            t = t + dt
            vec = i.integrate(t)
            solution.append(vec)
            logger[x_name].append(t)
            status[FINAL] = True
            self.tstep(t, vec)
            status[FINAL] = False
        self.step(solution[-1], x_end) # Leave in last time step state.
        if self._update_inits:
            self._update_initial_values()
        solution_arr = np.array(solution)
        dct = self.as_dct(solution_arr)
        dct.update(self._vectorise(logger))
        status[STOP] = False
        self.npsolve_finish()
        return dct
    
