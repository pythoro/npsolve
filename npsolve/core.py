# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 14:34:54 2019

@author: Reuben

Npsolve has a simple, small core built on fastwire. It's designed to give
good flexibility without compromising on performance.

"""

import numpy as np
import fastwire as fw
import traceback

from contextlib import contextmanager
from . import settings

sb = fw.SignalBox()

EMIT_VECTORS = 'EMIT_VECTORS'
GET_VARS = 'GET_VARS'
GET_STEP_METHODS = 'GET_STEP_METHODS'
GET_PARTIALS = 'GET_PARTIALS'
SET_CACHING = 'SET_CACHING'
GET_CACHE_CLEARS = 'GET_CACHE_CLEARS'

class Partial():
    ''' A base class responsible for a set of variables 
    
    Note:
        __init__ method must be called.
        
    '''
    
    def __init__(self):
        self.npsolve_vars = {}
        self.__cache_methods = self._get_cached_methods()
        self.__cache_clear_functions = self._get_cache_clear_functions()
        self.cache_clear() # Useful for iPython console autoreload.
        if settings.AUTO_CONNECT:
            self.connect()
        
    def connect(self, cid=None):
        ''' Connect this instance to the Solver instance
        
        Args:
            cid (int): The container id provided the setup_signals method
            of the Solver instance.
        '''
        try:
            c = sb.get_container(cid)
            c.get(EMIT_VECTORS, must_exist=True).connect(self.set_vectors)
            c.get(GET_VARS, must_exist=True).connect(self._get_vars)
            c.get(GET_STEP_METHODS, must_exist=True).connect(self._get_step_methods)
            c.get(GET_PARTIALS, must_exist=True).connect(self._get_self)
            c.get(SET_CACHING, must_exist=True).connect(self._set_caching)
            c.get(GET_CACHE_CLEARS,
                  must_exist=True).connect(self._get_cache_clear_functions)
        except KeyError:
            raise KeyError('Solver must be created before Partial instance.')
    
    def _get_self(self):
        return self
    
    def set_vectors(self, state_dct, ret_dct):
        ''' Override to set up views of the state vector 
        
        Args:
            state_dct (dict): A dictionary of numpy array views for the state
            of all variables. Provided by the Solver.
            ret_dct (dict): A similar dictionary of return values. Not
            usually used.
        '''
        pass
    
    def _get_vars(self):
        return self.npsolve_vars

    def set_meta(self, name, **kwargs):
        ''' Set meta information for a variable 
        
        Args:
            **kwargs: Key word attributes for the variable.
        '''
        self.npsolve_vars[name].update(kwargs)

    def set_init(self, name, init):
        ''' Set the initial value for a variable 
        
        Args:
            name (str): The variable name
            init (array-like): The initial value(s). Can be a scalar or 1D
                ndarray.
        '''
        self.npsolve_vars[name]['init'] = np.atleast_1d(init)

    def add_var(self, name, init, **kwargs):
        ''' Add a new variable 
        
        Args:
            name (str): The variable name
            init (array-like): The initial value(s). Can be a scalar or 1D
                ndarray.
            **kwargs: Optional kew word attributes for the variable.
        '''
        if name in self.npsolve_vars:
            raise KeyError(str(name) + ' already exists')
        self.npsolve_vars[name] = {}
        self.set_init(name, init)
        self.set_meta(name, **kwargs)
        
    def add_vars(self, dct):
        for name, d in dct.items():
            self.add_var(name, **d)
    
    def _get_cached_methods(self):
        functions = []
        for name in dir(self):
            if name.startswith('__') and name.endswith('__'):
                continue
            func = getattr(self, name, None)
            if hasattr(func, 'cacheable'):
                functions.append(func)
        return functions
    
    def _get_cache_clear_functions(self):
        return [func.cache_clear for func in self.__cache_methods]
    
    def cache_clear(self):
        [f() for f in self.__cache_clear_functions]
    
    def _set_caching(self, enable):
        [f.set_caching(enable) for f in self.__cache_methods]
    
    def _get_step_methods(self):
        return self.step
    
    def step(self, state_dct, *args):
        raise NotImplementedError('The step method must be implemented.')
        # return dict with key and return values.
        
    def enable_caching(self):
        [f.cache_enable() for f in self._get_cached_methods()]


class Solver():
    ''' The solver that pulls together the partials and allows solving '''
    
    def __init__(self):
        self._cache_clear_functions = []
        if settings.AUTO_CONNECT:
            self.setup_signals()
        
    def setup_signals(self):
        ''' Setup the fastwire signals that Partial instances will require 
        
        Returns:
            int: The container id for the signals.
        '''
        self._container = sb.add(activate=True, remove_with=self)
        signals = [EMIT_VECTORS, GET_VARS, GET_STEP_METHODS, GET_PARTIALS,
                   SET_CACHING, GET_CACHE_CLEARS]
        self._signals = {name: sb.get(name) for name in signals}
        return self._container.id
    
    def close_signals(self):
        sb.deactivate(self._container.id)
        
    def remove_signals(self):
        sb.remove(self._container.id)
    
    def _setup_vecs(self, dct, pinned=None):
        ''' Create vectors and slices based on a dictionary of variables 
        
        Args:
            dct (dict): A dictionary in which keys are variable names and
                values are dictionaries of attributes, which include an
                'init' entry for initial value.
        '''
        slices = {}
        pinned = {} if pinned is None else pinned
        meta = {}
        i = 0
        for key, item in dct.items():
            if key in pinned:
                continue
            n = len(item['init'])
            slices[key] = slice(i, i+n)
            meta[key] = item
            i += n
        state = np.zeros(i)
        for key, slc in slices.items():
            state[slc] = np.atleast_1d(dct[key]['init'])
        ret = np.zeros(i)
        return slices, state, ret
        
    def _make_dcts(self, slices, state, ret, pinned=None):
        ''' Create dictionaries of numpy views for all variables 
        
        '''
        state_dct = {}
        ret_dct = {}
        for name, slc in slices.items():
            state_view = state[slc]
            state_view.flags['WRITEABLE'] = False
            state_dct[name] = state_view

            ret_view = ret[slc]
            ret_dct[name] = ret_view
        if pinned is not None:
            pinned_update = {k: np.atleast_1d(v) for k, v in pinned.items()}
            state_dct.update(pinned_update)
            ret_update = {k: np.zeros_like(v) for k, v in pinned.items()}
            ret_dct.update(ret_update)
        return state_dct, ret_dct
    
    def _fetch_vars(self):
        ''' Collect variable data from connected Partial instances '''
        dct = {}
        dicts = self._signals[GET_VARS].fetch_all()
        for d in dicts:
            for key in d.keys():
                if key in dct:
                    raise KeyError('Variable "' + str(key) + '" is defined ' +
                                   'by more than one Partial class.')
            dct.update(d)
        return dct
        
    def _emit_vectors(self):
        ''' Pass out vectors and slices to connected Partial instances '''
        self._signals[EMIT_VECTORS].emit(
                state_dct=self.npsolve_state_dct,
                ret_dct=self.npsolve_ret_dct)

    def freeze(self):
        ''' Give static copies of vectors to connected Partial instances '''
        state_dct, ret_dct = self._make_dcts(self.npsolve_slices,
                                             self.npsolve_state.copy(),
                                             self.npsolve_ret.copy())
        self._signals[EMIT_VECTORS].emit(
                state_dct=state_dct,
                ret_dct=ret_dct)
        return self.npsolve_state.copy()
        
    def unfreeze(self, state=None):
        ''' Give 'live' vectors to connected Partial instances '''
        if state is not None:
            self.npsolve_state[:] = state
        self._emit_vectors()

    def _fetch_step_methods(self):
        lst = self._signals[GET_STEP_METHODS].fetch_all()
        out = []
        for ret in lst:
            if isinstance(ret, list):
                out.extend(ret)
            else:
                out.append(ret)
        return out

    def fetch_partials(self):
        ''' Fetch a dictionary of all connected Partial instances '''
        lst = self._signals[GET_PARTIALS].fetch_all()
        dct = {}
        for partial in lst:
            try:
                name = str(partial.npsolve_name)
            except AttributeError:
                name = str(partial)
            dct[name] = partial
        return dct

    def _fetch_cache_clears(self):
        lst = self._signals[GET_CACHE_CLEARS].fetch_all()
        out = []
        for l in lst:
            out.extend(l)
        return out

    def npsolve_init(self, pinned=None):
        ''' Initialise the Partials and be ready to solve 
        
        Args:
            pinned (dict): A dictionary of variable-value pairs to hold 
                constant during stepping.
        '''
        dct = self._fetch_vars()
        slices, state, ret = self._setup_vecs(dct, pinned)
        state_dct, ret_dct = self._make_dcts(slices, state, ret, pinned)
        self.npsolve_variables = dct
        self.npsolve_slices = slices
        self.npsolve_state = state
        self.npsolve_initial_values = state.copy()
        self.npsolve_ret = ret
        self.npsolve_state_dct = state_dct
        self.npsolve_ret_dct = ret_dct
        self._emit_vectors()
        self._step_methods = self._fetch_step_methods()
        self._cache_clear_functions = self._fetch_cache_clears()
        self._signals[SET_CACHING].emit(enable=True)

    @contextmanager
    def pinned(self, dct):
        ''' A context manager that unpinned all variables on exit '''
        self.npsolve_init(pinned=dct)
        yield
        state = self.npsolve_state.copy()
        self.npsolve_init()
        self.npsolve_state[:] = state

    def one_way_step(self, vec, *args, **kwargs):
        ''' Method to be called every iteration with no return val 
        
        Note: This method relies on other methods being used to inform the
        solver during its iteration.
        '''
        self.npsolve_state[:] = vec
        state_dct = self.npsolve_state_dct
        for f in self._cache_clear_functions:
            f()
        for step in self._step_methods:
            step(state_dct, *args, **kwargs)
                    
    def step(self, vec, *args, **kwargs):
        ''' The method to be called every iteration by the numerical solver '''
        self.npsolve_state[:] = vec
        state_dct = self.npsolve_state_dct
        ret_dct = self.npsolve_ret_dct
        for f in self._cache_clear_functions:
            f()
        for step in self._step_methods:
            for name, val in step(state_dct, *args, **kwargs).items():
                if self.npsolve_isolate is None:
                    ret_dct[name][:] = val
                else:
                    if name in self.npsolve_isolate:
                        ret_dct[name][:] = val
        return self.npsolve_ret

    def tstep(self, t, vec, *args, **kwargs):
        ''' The method to be called every iteration by the numerical solver '''
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
                raise TypeError('Error from ' + str(step) + ': ' + e.args[0])
            if not isinstance(ret, dict):
                raise ValueError(str(step) + ' did not return a dictionary of '
                                 + 'derivatives.')
            for name, val in ret.items():
                if self.npsolve_isolate is None:
                    ret_dct[name][:] = val
                else:
                    if name in self.npsolve_isolate:
                        ret_dct[name][:] = val
        return self.npsolve_ret
        
    def as_dct(self, sol):
        ''' Split out solution array into dictionary of values '''
        d = {}
        for key, slc in self.npsolve_slices.items():
            d[key] = sol[:,slc]
        return d