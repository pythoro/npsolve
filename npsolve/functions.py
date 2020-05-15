# -*- coding: utf-8 -*-
"""
Created on Fri May 15 16:17:04 2020

@author: Reuben

These functions can be used to prevent discontinuities, which can cause
trouble for numerical methods.
"""

import numpy as np
from math import exp, log

DEFAULT_SCALE = 1e-3

def soft_limit(value, limit=0.0, side=1, scale=DEFAULT_SCALE):
    """ Limit the value softly to prevent discontinuous gradient
    
    Args:
        value (int, float, ndarray): The value(s) to soft limit
        limit (float): The value to limit at
        side (int): 1 for min, -1 for max
        scale: A scale factor for the softening
    
    Returns:
        float, ndarray: The limited value(s)
        
    Note:
        This function uses a softplus function to perform smoothing.
        See https://en.wikipedia.org/wiki/Activation_function. Values for the
        calculation are clipped to avoid overflow errors.
    """
    rel = (value - limit) / scale * side
    if isinstance(value, np.ndarray):
        clipped = np.clip(rel, -700, 700)
        soft_plus = np.log(1 + np.exp(clipped)) * scale
        return limit + soft_plus * side
    clipped = min(max(rel, -700), 700)
    soft_plus = log(1 + exp(clipped)) * scale
    return limit + soft_plus * side

def soft_min(value, limit=0.0, scale=DEFAULT_SCALE):
    """ Limit value to a minimum softly to to prevent discontinuous gradient
    
    Args:
        value (int, float, ndarray): The value(s) to soft limit
        limit (float): The value to limit at
        scale: A scale factor for the softening
    
    Returns:
        float, ndarray: The limited value(s)
        
    See also:
        soft_limit
    """
    rel = (value - limit) / scale
    if isinstance(value, np.ndarray):
        clipped = np.clip(rel, -700, 700)
        soft_plus = np.log(1 + np.exp(clipped)) * scale
        return limit + soft_plus
    clipped = min(max(rel), -700, 700)
    soft_plus = log(1 + exp(clipped)) * scale
    return limit + soft_plus

def soft_max(value, limit=0.0, scale=DEFAULT_SCALE):
    """ Limit value to a maximum softly to to prevent discontinuous gradient
    
    Args:
        value (int, float, ndarray): The value(s) to soft limit
        limit (float): The value to limit at
        scale: A scale factor for the softening
    
    Returns:
        float, ndarray: The limited value(s)
        
    See also:
        soft_limit
    """
    rel = -(value - limit) / scale
    if isinstance(value, np.ndarray):
        clipped = np.clip(rel, -700, 700)
        soft_plus = np.log(1 + np.exp(clipped)) * scale
        return limit - soft_plus
    clipped = min(max(rel, -700), 700)
    soft_plus = log(1 + exp(clipped)) * scale
    return limit - soft_plus

def soft_range(value, lower, upper, scale=DEFAULT_SCALE):
    """ Limit value to a range softly to to prevent discontinuous gradient
    
    Args:
        value (int, float, ndarray): The value(s) to soft limit
        lower (float): The lower threshold
        upper (float): The upper threshold
        scale: A scale factor for the softening
    
    Returns:
        float, ndarray: The limited value(s)
        
    See also:
        soft_limit
    """
    capped = soft_max(value, limit=upper, scale=scale)
    return soft_min(capped, limit=lower, scale=scale)

def soft_step(value, limit=0.0, side=1, scale=DEFAULT_SCALE):
    """ A smooth step to prevent discontinuous gradient
    
    Args:
        value (int, float, ndarray): The value(s)
        limit (float): The value to step at
        side (int): 1 for min, -1 for max
        scale: A scale factor for the softening
    
    Returns:
        float, ndarray: Value(s) between 0 and 1
        
    Note:
        This function uses a sigmoid function to perform smoothing. See
        https://en.wikipedia.org/wiki/Sigmoid_function. Values for the
        calculation are clipped to avoid overflow errors.
    """
    rel = (value - limit) / scale * side
    if isinstance(value, np.ndarray):
        clipped = np.clip(rel, -700, 700)
        return 1/(1 + np.exp(-clipped))
    clipped = min(max(rel, -700), 700)
    return 1/(1 + exp(-clipped))
    
def soft_above(value, limit=0.0, scale=DEFAULT_SCALE):
    """ A smooth step from 0 below a limit to 1 above it
    
    Args:
        value (int, float, ndarray): The value(s)
        limit (float): The value to step at
        scale: A scale factor for the softening
    
    Returns:
        float, ndarray: Value(s) between 0 and 1
        
    See also:
        soft_step
    """
    rel = (value - limit) / scale
    if isinstance(value, np.ndarray):
        clipped = np.clip(rel, -700, 700)
        return 1/(1 + np.exp(-clipped))
    clipped = min(max(rel, -700), 700)
    return 1/(1 + exp(-clipped))


def soft_below(value, limit=0.0, scale=DEFAULT_SCALE):
    """ A smooth step from 1 below a limit to 0 above it
    
    Args:
        value (int, float, ndarray): The value(s)
        limit (float): The value to step at
        scale: A scale factor for the softening
    
    Returns:
        float, ndarray: Value(s) between 0 and 1
        
    See also:
        soft_step
    """
    rel = -(value - limit) / scale
    if isinstance(value, np.ndarray):
        clipped = np.clip(rel, -700, 700)
        return 1/(1 + np.exp(-clipped))
    clipped = min(max(rel, -700), 700)
    return 1/(1 + exp(-clipped))

def soft_within(value, lower, upper, scale=DEFAULT_SCALE):
    """ Steps smoothly from 0 outside a range to 1 inside it
    
    Args:
        value (int, float, ndarray): The value(s)
        lower (float): The lower threshold
        upper (float): The upper threshold
        scale: A scale factor for the softening
    
    Returns:
        float, ndarray: Value(s) between 0 and 1
        
    See also:
        soft_step
    """
    below = soft_below(value, limit=upper, scale=scale)
    above = soft_above(value, limit=lower, scale=scale) 
    return below * above

def soft_outside(value, lower, upper, scale=DEFAULT_SCALE):
    """ Steps smoothly from 1 outside a range to 0 inside it
    
    Args:
        value (int, float, ndarray): The value(s)
        lower (float): The lower threshold
        upper (float): The upper threshold
        scale: A scale factor for the softening
    
    Returns:
        float, ndarray: Value(s) between 0 and 1
        
    See also:
        soft_step
    """
    below = soft_below(value, limit=lower, scale=scale)
    above = soft_above(value, limit=upper, scale=scale) 
    return below + above

def soft_sign(value, scale=DEFAULT_SCALE):
    """ A smooth step from -1 below 0 to +1 above it
    
    Args:
        value (int, float, ndarray): The value(s)
        scale: A scale factor for the softening
    
    Returns:
        float, ndarray: Value(s) between 0 and 1
        
    Note:
        This function uses a sigmoid function to perform smoothing. See
        https://en.wikipedia.org/wiki/Sigmoid_function. Values for the
        calculation are clipped to avoid overflow errors.
    """
    if isinstance(value, np.ndarray):
        clipped = np.clip(value / scale, -700, 700)
        return 2/(1 + np.exp(-clipped)) - 1
    clipped = min(max(value / scale, -700), 700)
    return 2/(1 + exp(-clipped)) - 1
