# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 10:41:13 2020

@author: Reuben

A lightweight module for simplified array index management.

"""

import numpy as np

class V_Set():
    """ A class that helps manage indices for array operations 
    
    Args:
        names (str, list): A single name, a series of names separated by
            spaces, or a list of names.
        sizes (list): [Optional] A list of integers indicating how many
            values need to be allocated for each name. Defaults to 1 for
            all names.

    """
    
    def __init__(self, names=None, sizes=None):
        self._dct = {}
        self._n = 0
        self._locked = False
        if names is not None:
            self.add(names, sizes)
            self.lock()
    
    def add(self, names, sizes=None):
        """ Add another variable to the set 
        
        Args:
            names (str, list): A single name, a series of names separated by
                spaces, or a list of names.
            sizes (list): [Optional] A list of integers indicating how many
                values need to be allocated for each name. Defaults to 1 for
                all names.
        
        """
        if self._locked:
            raise RuntimeError('Set is locked and cannot be added to.')
        if isinstance(names, str):
            names = names.split(' ')
        if len(names) == 1 and isinstance(sizes, int):
            sizes = [sizes]
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
               
    def lock(self) :
        self._locked = True
               
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
        """ Assemble values for each variable into a 1d array 
        
        Args:
            dct (dict): [Optional] A dictionary of values for all names.
            **kwargs: Optional name-value keyword arguments, which can be
                used instead of the dct argument.
        
        Returns:
            ndarray: A 1d numpy array.
        
        This method puts thge values in the right places within the
        array.
        
        """
        try:
            dct = kwargs if dct is None else dct
            return np.hstack([dct[n] for n in self._dct.keys()])
        except KeyError as e:
            raise ValueError('Value not provided for required key: ' + str(e))

    def __str__(self):
        return ', '.join(self._dct.keys())
    
    def __repr__(self):
        return "V_Set: " + ', '.join(self._dct.keys())