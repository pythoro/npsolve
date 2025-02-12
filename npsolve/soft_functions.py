# -*- coding: utf-8 -*-
"""
Created on Fri May 15 16:17:04 2020

@author: Reuben

These functions can be used to prevent discontinuities, which can cause
trouble for numerical methods.
"""

import numpy as np
from math import exp, log

DEFAULT_SCALE = 1e-4
SCALARISE = True


def lim(x, limit=0.0, side=1, scale=DEFAULT_SCALE):
    """Limit the value softly to prevent discontinuous gradient

    Args:
        x (int, float, ndarray): The value(s) to soft limit
        limit (float): [OPTIONAL] The value to limit at. Defaults to 0.
        side (int): [OPTIONAL] 1 for min, -1 for max. Defaults to 1.
        scale (float): [OPTIONAL] A scale factor for the softening

    Returns:
        float, ndarray: The limited value(s)

    Note:
        This function uses a softplus function to perform smoothing.
        See https://en.wikipedia.org/wiki/Activation_function. Values for the
        calculation are clipped to 700 avoid overflow errors, as the max
        value for a float is exp(709.782).
    """
    if isinstance(x, np.ndarray):
        if x.size > 1 or not SCALARISE:
            rel = (x - limit) / scale * side
            clipped = np.minimum(rel, 700)
            soft_plus = np.log(1 + np.exp(clipped)) * scale
            out = limit + soft_plus * side
            filt = rel > 699
            out[filt] = x[filt]
            return out
        else:
            x = x.item()
    rel = (x - limit) / scale * side
    if rel > 700:
        return x
    soft_plus = log(1 + exp(rel)) * scale
    return limit + soft_plus * side


def floor(x, limit=0.0, scale=DEFAULT_SCALE):
    """Limit value to a minimum softly to to prevent discontinuous gradient

    Args:
        x (int, float, ndarray): The value(s) to soft limit
        limit (float): [OPTIONAL] The value to limit at. Defaults to 0.
        scale (float): [OPTIONAL] A scale factor for the softening

    Returns:
        float, ndarray: The limited value(s)

    See also:
        soft_limit
    """
    if isinstance(x, np.ndarray):
        if x.size > 1 or not SCALARISE:
            rel = (x - limit) / scale
            clipped = np.minimum(rel, 700)
            soft_plus = np.log(1 + np.exp(clipped)) * scale
            out = limit + soft_plus
            filt = rel > 699
            out[filt] = x[filt]
            return out
        else:
            x = x.item()
    rel = (x - limit) / scale
    if rel > 700:
        return x
    soft_plus = log(1 + exp(rel)) * scale
    return limit + soft_plus


def ceil(x, limit=0.0, scale=DEFAULT_SCALE):
    """Limit value to a maximum softly to to prevent discontinuous gradient

    Args:
        x (int, float, ndarray): The value(s) to soft limit
        limit (float): [OPTIONAL] The value to limit at. Defaults to 0.
        scale (float): [OPTIONAL] A scale factor for the softening

    Returns:
        float, ndarray: The limited value(s)

    See also:
        soft_limit
    """

    if isinstance(x, np.ndarray):
        if x.size > 1 or not SCALARISE:
            rel = -(x - limit) / scale
            clipped = np.minimum(rel, 700)
            soft_plus = np.log(1 + np.exp(clipped)) * scale
            out = limit - soft_plus
            filt = rel > 699
            out[filt] = x[filt]
            return out
        else:
            x = x.item()
    rel = -(x - limit) / scale
    if rel > 700:
        return x
    soft_plus = log(1 + exp(rel)) * scale
    return limit - soft_plus


def clip(x, lower, upper, scale=DEFAULT_SCALE):
    """Limit value to a range softly to to prevent discontinuous gradient

    Args:
        x (int, float, ndarray): The value(s) to soft limit
        lower (float): The lower threshold
        upper (float): The upper threshold
        scale: A scale factor for the softening

    Returns:
        float, ndarray: The limited value(s)

    See also:
        soft_limit
    """
    capped = ceil(x, limit=upper, scale=scale)
    return floor(capped, limit=lower, scale=scale)


def posdiff(x, limit=0.0, scale=DEFAULT_SCALE):
    """Positive-only difference (0 below limit to difference above limit)

    Args:
        x (int, float, ndarray): The value(s) to soft limit
        limit (float): [OPTIONAL] The value to limit at. Defaults to 0.
        scale (float): [OPTIONAL] A scale factor for the softening

    Returns:
        float, ndarray: The limited value(s)

    See also:
        soft_limit
    """
    if isinstance(x, np.ndarray):
        if x.size > 1 or not SCALARISE:
            rel = (x - limit) / scale
            clipped = np.minimum(rel, 700)
            out = np.log(1 + np.exp(clipped)) * scale
            filt = rel > 699
            out[filt] = x[filt] - limit
            return out
        else:
            x = x.item()
    rel = (x - limit) / scale
    if rel > 700:
        return x - limit
    soft_plus = log(1 + exp(rel)) * scale
    return soft_plus


def negdiff(x, limit=0.0, scale=DEFAULT_SCALE):
    """Negative-only difference (difference below limit to 0 above limit)

    Args:
        x (int, float, ndarray): The value(s) to soft limit
        limit (float): [OPTIONAL] The value to limit at. Defaults to 0.
        scale (float): [OPTIONAL] A scale factor for the softening

    Returns:
        float, ndarray: The limited value(s)

    See also:
        soft_limit
    """

    if isinstance(x, np.ndarray):
        if x.size > 1 or not SCALARISE:
            rel = -(x - limit) / scale
            clipped = np.minimum(rel, 700)
            out = np.log(1 + np.exp(clipped)) * scale
            filt = rel > 699
            out[filt] = x[filt] - limit
            return -out
        else:
            x = x.item()
    rel = -(x - limit) / scale
    if rel > 700:
        return x - limit
    soft_plus = log(1 + exp(rel)) * scale
    return -soft_plus


def step(x, limit=0.0, side=1, scale=DEFAULT_SCALE):
    """A smooth step to prevent discontinuous gradient

    Args:
        x (int, float, ndarray): The value(s)
        limit (float): [OPTIONAL] The value to step at. Defaults to 0.
        side (int): [OPTIONAL] 1 for min, -1 for max. Defaults to 1.
        scale (float): [OPTIONAL] A scale factor for the softening

    Returns:
        float, ndarray: Value(s) between 0 and 1

    Note:
        This function uses a sigmoid function to perform smoothing. See
        https://en.wikipedia.org/wiki/Sigmoid_function. Values for the
        calculation are clipped to avoid overflow errors.
    """
    if isinstance(x, np.ndarray):
        if x.size > 1 or not SCALARISE:
            rel = (x - limit) / scale * side
            clipped = np.maximum(rel, -700)
            return 1 / (1 + np.exp(-clipped))
        else:
            x = x.item()
    rel = (x - limit) / scale * side
    clipped = max(rel, -700)
    return 1 / (1 + exp(-clipped))


def above(x, limit=0.0, scale=DEFAULT_SCALE):
    """A smooth step from 0 below a limit to 1 above it

    Args:
        x (int, float, ndarray): The value(s)
        limit (float): [OPTIONAL] The value to step at. Defaults to 0.
        scale (float): [OPTIONAL] A scale factor for the softening

    Returns:
        float, ndarray: Value(s) between 0 and 1

    See also:
        soft_step
    """
    if isinstance(x, np.ndarray):
        if x.size > 1 or not SCALARISE:
            rel = (x - limit) / scale
            clipped = np.maximum(rel, -700)
            return 1 / (1 + np.exp(-clipped))
        else:
            x = x.item()
    rel = (x - limit) / scale
    clipped = max(rel, -700)
    return 1 / (1 + exp(-clipped))


def below(x, limit=0.0, scale=DEFAULT_SCALE):
    """A smooth step from 1 below a limit to 0 above it

    Args:
        x (int, float, ndarray): The value(s)
        limit (float): [OPTIONAL] The value to step at. Defaults to 0.
        scale (float): [OPTIONAL] A scale factor for the softening

    Returns:
        float, ndarray: Value(s) between 0 and 1

    See also:
        soft_step
    """
    if isinstance(x, np.ndarray):
        if x.size > 1 or not SCALARISE:
            rel = -(x - limit) / scale
            clipped = np.maximum(rel, -700)
            return 1 / (1 + np.exp(-clipped))
        else:
            x = x.item()
    rel = -(x - limit) / scale
    clipped = max(rel, -700)
    return 1 / (1 + exp(-clipped))


def within(x, lower, upper, scale=DEFAULT_SCALE):
    """Steps smoothly from 0 outside a range to 1 inside it

    Args:
        x (int, float, ndarray): The value(s)
        lower (float): The lower threshold
        upper (float): The upper threshold
        scale (float): [OPTIONAL] A scale factor for the softening

    Returns:
        float, ndarray: Value(s) between 0 and 1

    See also:
        soft_step
    """
    b = below(x, limit=upper, scale=scale)
    a = above(x, limit=lower, scale=scale)
    return b * a


def outside(x, lower, upper, scale=DEFAULT_SCALE):
    """Steps smoothly from 1 outside a range to 0 inside it

    Args:
        x (int, float, ndarray): The value(s)
        lower (float): The lower threshold
        upper (float): The upper threshold
        scale (float): [OPTIONAL] A scale factor for the softening

    Returns:
        float, ndarray: Value(s) between 0 and 1

    See also:
        soft_step
    """
    b = below(x, limit=lower, scale=scale)
    a = above(x, limit=upper, scale=scale)
    return b + a


def sign(x, scale=DEFAULT_SCALE):
    """A smooth step from -1 below 0 to +1 above it

    Args:
        x (int, float, ndarray): The value(s)
        scale (float): [OPTIONAL] A scale factor for the softening

    Returns:
        float, ndarray: Value(s) between 0 and 1

    Note:
        This function uses a sigmoid function to perform smoothing. See
        https://en.wikipedia.org/wiki/Sigmoid_function. Values for the
        calculation are clipped to avoid overflow errors.
    """
    if isinstance(x, np.ndarray):
        if x.size > 1 or not SCALARISE:
            clipped = np.maximum(x / scale, -700)
            return 2 / (1 + np.exp(-clipped)) - 1
        else:
            x = x.item()
    clipped = max(x / scale, -700)
    return 2 / (1 + exp(-clipped)) - 1


def gaussian(x, center=0.0, scale=DEFAULT_SCALE):
    """A gaussian function, with a peak of 1.0

    Args:
        x (int, float, ndarray): The value(s)
        center (float): [OPTIONAL] The x-position of the peak center
        scale (float): [OPTIONAL] A scale factor for the curve.


    """
    if isinstance(x, np.ndarray):
        if x.size > 1 or not SCALARISE:
            clipped = np.maximum((x - center) ** 2 / scale, -700)
            return np.exp(-clipped)
        else:
            x = x.item()
    clipped = max((x - center) ** 2 / scale, -700)
    return exp(-clipped)
