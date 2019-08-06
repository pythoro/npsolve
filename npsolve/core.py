# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 14:34:54 2019

@author: Reuben
"""

import numpy as np
import fastwire as fw

sb = fw.SignalBox()

EMIT_VECTORS = 'EMIT_VECTORS'
GET_VARS = 'GET_VARS'


class Partial():
    ''' A base class responsible for a set of variables 
    
    Note:
        __init__ method must be called.
        
    '''
    
    def __init__(self):
        self.npsolve_vars = {}
        self.__cache_clear_functions = self._get_cache_clear_functions()
        try:
            sb.get(EMIT_VECTORS, must_exist=True).connect(self.set_vectors)
            sb.get(GET_VARS, must_exist=True).connect(self._get_vars)
        except KeyError:
            raise KeyError('Solver must be created before Partial instance.')
    
    def set_vectors(self, state, ret, slices):
        ''' Override to set up views of the state vector '''
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

    def get_state(self, name, state, slices):
        ''' Copy the value for any state variable 
        
        Args:
            name (str): The variable name
            
        Returns:
            ndarray: A copy of the variable values.

        '''
        return state[slices[name]].copy()

    def get_state_view(self, name, state, slices):
        ''' Get the numpy view of any state variable 
        
        Args:
            name (str): The variable name
            
        Returns:
            ndarray: A (non-writable) view of the variable values.
        
        Note:
            The values are 'live'.
        '''
        view = state[slices[name]]
        view.flags['WRITEABLE'] = False
        return view
        
    def _get_cache_clear_functions(self):
        functions = []
        for name in dir(self):
            if name.startswith('__') and name.endswith('__'):
                continue
            func = getattr(self, name, None)
            if hasattr(func, 'cacheable'):
                functions.append(func.cache_clear)
        return functions
    
    def cache_clear(self):
        for f in self.__cache_clear_functions:
            f()
            
    def step(self, state, slices, t):
        self.cache_clear()
        # return dict with key and return values.


class Solver():
    
    def __init__(self):
        self._container_id = sb.add(remove_with=self)
        self._setup_signals()
        self._cache_clear_functions = []
        
    def _setup_signals(self):
        ''' Setup the signals that Partial instances will require '''
        signals = [EMIT_VECTORS, GET_VARS]
        self._signals = {name: sb.get(name) for name in signals}
    
    def _setup_vecs(self, dct):
        ''' Create vectors and slices based on a dictionary of variables 
        
        Args:
            dct (dict): A dictionary in which keys are variable names and
                values are dictionaries of attributes, which include an
                'init' entry for initial value.
        '''
        slices = {}
        meta = {}
        i = 0
        for key, item in dct.items():
            n = len(item['init'])
            slices[key] = slice(i, i+n)
            meta[key] = item
            i += n
        state = np.zeros(i)
        for key, slc in slices.items():
            state[slc] = dct[key]['init']
        ret = np.zeros(i)
        return slices, state, ret
        
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
        self.npsolve_slices, self.npsolve_state, self.npsolve_ret = self._setup_vecs(dct)

    def _emit_vectors(self):
        ''' Pass out vectors and slices to connected Partial instances '''
        self._signals[EMIT_VECTORS].emit(state=self.npsolve_state,
                                   ret=self.npsolve_ret,
                                   slices=self.npsolve_slices)

    def npsolve_init(self):
        ''' Initialise the Partials and be ready to solve '''
        self._fetch_vars()
        self._emit_vectors()
                    
    def step(self, vec):
        pass
        