# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 08:31:52 2019

@author: Reuben

This module is about logging extra variables during solving.

The recommended way is to use a List_Container.

"""

from collections import defaultdict
import numpy as np
try:
    from scipy.interpolate import splrep, splev, splder, splantider
    scipy_found = True
except:
    scipy_found = False


def none():
    return None

class Dict_Container(dict):
    def __missing__(self, key):
        self[key] = defaultdict(none)
        return self[key]

class List_Container(dict):
    def __missing__(self, key):
        self[key] = []
        return self[key]

class Set_Container(dict):
    def __missing__(self, key):
        self[key] = set()
        return self[key]

class List_Container_Container(dict):
    def __missing__(self, key):
        self[key] = List_Container()
        return self[key]



dict_container = Dict_Container()
list_container = List_Container()
set_container = Set_Container()
list_container_container = List_Container_Container()

def get_dict(name):
    return dict_container[name]

def get_list(name):
    return list_container[name]

def get_set(name):
    return set_container[name]

def get_list_container(name):
    return list_container_container[name]

get_status = get_dict
get_logger = get_list_container



class Timeseries():
    """ A utility class to specify values from time-series data 
    
    Args:
        xs (ndarray): A 1D array of x values. Must be monotonically increasing.
        ys (ndarray): A 1D array of y values
        
    Usage:
        The Timeseries class is callable. It interpolates values smoothly
        between the inputs using splines. It offers the 1st integral 
        to the 3rd derivative of values.
        
        ::
            
            timeseries = Timeseries(xs, ys) # Create
            timeseries(5) # Value at x=5
            timeseries(5, -1) # 1st integral (antiderivative) at x=5
            timeseries(5, 0) # Value at x=5
            timeseries(5, 1) # 1st derivative at x=5
            timeseries(5, 2) # 2nd derivative at x=5
            timeseries(5, 3) # 3rd derivative at x=5
    
    Note:
        Use `Timeseries.from_csv` to generate values from a csv file.
    
    """
    def __init__(self, xs, ys):
        if not scipy_found:
            raise ImportError('Scipy needed for Timeseries class.')
        self.xs = xs
        self.ys = ys
        self._tcks = self._make_splines(xs, ys)

    @classmethod
    def _read_from_csv(cls, **kwargs):
        arr = np.genfromtxt(**kwargs)
        xs = arr[:,0]
        ys = arr[:,1]
        return xs, ys
        
    @classmethod
    def from_csv(cls, fname, x_col=0, y_col=1, skip_header=0, 
                 delimiter=',', **kwargs):
        """ Create a Timeseries from csv data 
        
        Args:
            fname (str): The filename
            x_col (int): The column index of the x data (0 is the first column)
            y_col (int): The column index of the y data
            skip_header (int): [Optional] Number of header rows to skip. 
                Defaults to 0.
            delimiter (str): [Optional] Delimiter. Defaults to ','.
            **kwargs: [Optional] Other keyword arguments passed to 
                numpy.genfromtxt.
        """
        usecols = (x_col, y_col)
        xs, ys = cls._read_from_csv(fname=fname, usecols=usecols,
                        skip_header=skip_header, delimiter=delimiter,
                     **kwargs)
        return cls(xs, ys)
        
    def _make_splines(self, xs, ys):
        """ Make the splines """
        base = splrep(xs, ys)
        tcks = {
                -1: splantider(base, 1),
                0: base,
                1: splder(base, 1),
                2: splder(base, 2),
                3: splder(base, 3)
                }
        return tcks
    
    def get(self, x, der=0, ext=3):
        """ Get an interpolated value 
        
        Args:
            x (float, ndarray): The x value(s).
            der (int): [Optional] The derivative number. Defaults to 0.
            ext (int): [Optional] What to do outside the range of xs. See 
                scipy.interpolate.splev for details. Defaults to 3, which means
                to return the boundary values.
        
        Returns:
            ndarray: The y value(s)
        
        """
        return splev(x, self._tcks[der], ext=ext)
    
    def __call__(self, x, der=0, ext=3):
        """ Get an interpolated value 
        
        Args:
            x (float, ndarray): The x value(s).
            der (int): [Optional] The derivative number. Defaults to 0.
            ext (int): [Optional] What to do outside the range of xs. See 
                scipy.interpolate.splev for details. Defaults to 3, which means
                to return the boundary values.
        
        Returns:
            ndarray: The y value(s)
        
        """        
        return splev(x, self._tcks[der], ext=ext)
        