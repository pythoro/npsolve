# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 14:34:54 2019

@author: Reuben
"""

import numpy as np
import fastwire as fw

sb = fw.SignalBox()

SET_VECTORS = 'SET_VECTORS'
GET_INIT = 'GET_INIT'
VECTORS_SET = 'VECTORS_SET'

class Partial():
    
    def __init__(self):
        self._names = {}
        try:
            sb.get(SET_VECTORS, must_exist=True).connect(self._set_vectors)
            sb.get(GET_INIT, must_exist=True).connect(self._get_init)
            sb.get(VECTORS_SET, must_exist=True).connect(self._vectors_set)
        except KeyError:
            raise KeyError('Solver must be created before Partial instance.')
    
    def _set_vectors(self, state, ret, slices):
        self._npsolve_slices = {n: slices[n] for n in self._names.keys()}
        self.__npsolve_ret = ret
        self.__npsolve_state = state
    
    def _vectors_set(self):
        pass
    
    def _get_value(self, name):
        ''' Method to get value of named variable
        
        Note:
            Can be overriden to customise this behaviour.
        '''
        return getattr(self, name)
    
    def _get_init(self):
        return self._names

    def set_meta(self, name, **kwargs):
        self._names[name].update(kwargs)

    def set_init(self, name, init):
        self._names[name]['init'] = np.atleast_1d(init)

    def add_name(self, name, init, **kwargs):
        if name in self._names:
            raise KeyError(str(name) + ' already exists')
        self._names[name] = {}
        self.set_init(name, init)
        self.set_meta(name, **kwargs)

    def get_state(self, name):
        ''' Copy the value for any state variable '''
        return self.__npsolve_state[self._npsolve_slices[name]].copy()

    def get_state_view(self, name):
        ''' Get the numpy view of any state variable 
        
        Note:
            The values are 'live'.
        '''
        view = self.__npsolve_state[self._npsolve_slices[name]]
        view.flags['WRITEABLE'] = False
        return view
    
    def set_return(self, name, value):
        ''' Set the return value for a variable '''
        self.__npsolve_ret[self._npsolve_slices[name]] = value
    
    
class Solver():
    
    def __init__(self):
        self._container_id = sb.add(remove_with=self)
        self._setup_signals()
        
    def _setup_signals(self):
        signals = [SET_VECTORS, GET_INIT, VECTORS_SET]
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
        
    def _get_init(self):
        dct = {}
        dicts = self._signals[GET_INIT].fetch_all()
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
        self._get_init()
        self._set_vectors()
        self._signals[VECTORS_SET].emit()