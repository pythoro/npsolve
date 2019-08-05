# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 14:34:54 2019

@author: Reuben
"""

import numpy as np
import fastwire as fw

sb = fw.SignalBox()

SET_VECTORS = 'SET_VECTORS'
GET_VARS = 'GET_VARS'
VECTORS_SET = 'VECTORS_SET'

class Partial():
    ''' A base class responsible for a set of variables 
    
    Note:
        __init__ method must be called.
        
    '''
    
    def __init__(self):
        self.npsolve_vars = {}
        try:
            sb.get(SET_VECTORS, must_exist=True).connect(self._set_vectors)
            sb.get(GET_VARS, must_exist=True).connect(self._get_init)
            sb.get(VECTORS_SET, must_exist=True).connect(self._vectors_set)
        except KeyError:
            raise KeyError('Solver must be created before Partial instance.')
    
    def _set_vectors(self, state, ret, slices):
        self._npsolve_slices = {n: slices[n] for n in self.npsolve_vars.keys()}
        self.__npsolve_ret = ret
        self.__npsolve_state = state
    
    def _vectors_set(self):
        ''' Called after vectors are set 
        
        This provides opportunity to use methods such as get_state_view.
        '''
        pass
    
    def _get_value(self, name):
        ''' Method to get value of named variable
        
        Note:
            Can be overriden to customise this behaviour.
        '''
        return getattr(self, name)
    
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

    def get_state(self, name):
        ''' Copy the value for any state variable 
        
        Args:
            name (str): The variable name
            
        Returns:
            ndarray: A copy of the variable values.

        '''
        return self.__npsolve_state[self._npsolve_slices[name]].copy()

    def get_state_view(self, name):
        ''' Get the numpy view of any state variable 
        
        Args:
            name (str): The variable name
            
        Returns:
            ndarray: A (non-writable) view of the variable values.
        
        Note:
            The values are 'live'.
        '''
        view = self.__npsolve_state[self._npsolve_slices[name]]
        view.flags['WRITEABLE'] = False
        return view
    
    def set_return(self, name, value):
        ''' Set the return value for a variable 
        
        Args:
            name (str): The variable name
            value (array-like): The value(s) to set.
            
        Note:
            value must be the same shape as the 'init' value of the variable.
        '''
        self.__npsolve_ret[self._npsolve_slices[name]] = value
    
    
class Solver():
    
    def __init__(self):
        self._container_id = sb.add(remove_with=self)
        self._setup_signals()
        
    def _setup_signals(self):
        signals = [SET_VECTORS, GET_VARS, VECTORS_SET]
        self._signals = {name: sb.get(name) for name in signals}
    
    def _setup_vecs(self, dct):
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
        
    def _get_vars(self):
        dct = {}
        dicts = self._signals[GET_VARS].fetch_all()
        for d in dicts:
            for key in d.keys():
                if key in dct:
                    raise KeyError('Variable "' + str(key) + '" is defined ' +
                                   'by more than one Partial class.')
            dct.update(d)
        self.npsolve_slices, self.npsolve_state, self.npsolve_ret = self._setup_vecs(dct)

    def _set_vectors(self):
        self._signals[SET_VECTORS].emit(state=self.npsolve_state,
                                   ret=self.npsolve_ret,
                                   slices=self.npsolve_slices)

    def npsolve_init(self):
        self._get_vars()
        self._set_vectors()
        self._signals[VECTORS_SET].emit()