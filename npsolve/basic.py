# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 10:41:13 2020

@author: Reuben

A lightweight module for simplified array index management.

"""

import numpy as np

class V_Set():
    def __init__(self, names=None, sizes=None):
        self._dct = {}
        self._n = 0
        if names is not None:
            self.add(names, sizes)
    
    def add(self, names, sizes=None):
        """ Add another variable to the set 
        
        
        """
        if isinstance(names, str):
            names = names.split(' ')
        sizes = [1] * len(names) if sizes is None else sizes
        for name, size in zip(names, sizes):
            if name not in self._dct:
                if size == 1:
                    self._dct[name] = self._n
                else:
                    self._dct[name] = slice(self._n, self._n + size)
                self._n += size
            else:
                ValueError(name, 'already exists in the set.')
                
    def ind(self, names=None):
        """ Return indices or slices corresponding to variable names 
        
        Args:
            names (str, list): A single name, a series of names separated by
                spaces, or a list of names.
        
        Returns:
            tuple: A tuple of indices or slices corresponding to variable
            names.
        
        """
        if names is None:
            lst = list(self._dct.keys()) if names is None else names
        elif isinstance(names, str):
            lst = names.split(' ')
        elif isinstance(names, list):
            lst = names
        else:
            raise KeyError("Argument 'names' 'not understood.")
        if len(lst) == 1:
            return self._dct[lst[0]]
        else:
            return tuple(self._dct[n] for n in lst)
    
    def __getitem__(self, name):
        return self.ind(name)
            
    def array(self, dct=None, **kwargs):
        try:
            dct = kwargs if dct is None else dct
            return np.hstack([dct[n] for n in self._dct.keys()])
        except KeyError as e:
            raise ValueError('Value not provided for required key: ' + str(e))

    def __str__(self):
        return ', '.join(self._dct.keys())
    
    def __repr__(self):
        return "V_Set: " + ', '.join(self._dct.keys())