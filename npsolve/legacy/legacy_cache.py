# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 07:06:54 2019

@author: Reuben

This module is deprecated and will be removed in a future version!

Simple caching inspired by functools.lru_cache.

Sometimes, Partial instances may need to call each other (via a fastwire fetch
is a good method). Caching allows a way to reuse the computations for each step
if needed, to avoid having to double-up in those cases.

"""

import functools

def multi_cached():
    ''' A cache method that considers arguments '''
    def decorator(user_function):
        # Needs to be inside a decorating function
        sentinel = object()                 # unique object used to signal cache misses
        make_key = functools._make_key      # build a key from the function arguments
        cache_enabled = False
        cache = {}
        cache_get = cache.get    # bound method to lookup a key or return None

        @functools.wraps(user_function)        
        def wrapper(*args, **kwds):
            if not cache_enabled:
                return user_function(*args, **kwds)
            key = make_key(args, kwds, typed=False)
            result = cache_get(key, sentinel)
            if result is not sentinel:
                return result
            result = user_function(*args, **kwds)
            cache[key] = result
            return result
    
        def cache_enable():
            nonlocal cache_enabled
            cache_enabled = True
    
        def cache_disable():
            nonlocal cache_enabled
            cache_enabled = False
            cache.clear()
            
        def set_caching(enable):
            if enable == True:
                cache_enable()
            else:
                cache_disable()
        
        wrapper.cache_clear = cache.clear
        wrapper.cache_enable = cache_enable
        wrapper.cache_disable = cache_disable
        wrapper.set_caching = set_caching
        wrapper.cacheable = True
        return wrapper
    return decorator


def mono_cached():
    ''' A cache method that only considers the 'self' argument 
    
    This works very similar to multi-cache but doesn't use the make_key 
    function from functools to save a little bit of time.
    '''
    def decorator(user_function):
        # Needs to be inside a decorating function
        sentinel = object()                 # unique object used to signal cache misses
        sentinel_hash = hash(sentinel)
        cache_enabled = False
        cache = {}
        cache_get = cache.get    # bound method to lookup a key or return None

        @functools.wraps(user_function)        
        def wrapper(*args, **kwds):
            if not cache_enabled:
                return user_function(*args, **kwds)
            if len(args) == 0:
                key = sentinel_hash
            else:
                key = hash(args[0])
            result = cache_get(key, sentinel)
            if result is not sentinel:
                return result
            result = user_function(*args, **kwds)
            cache[key] = result
            return result
    
        def cache_enable():
            nonlocal cache_enabled
            cache_enabled = True
    
        def cache_disable():
            nonlocal cache_enabled
            cache_enabled = False
            cache.clear()
            
        def set_caching(enable):
            if enable == True:
                cache_enable()
            else:
                cache_disable()
        
        wrapper.cache_clear = cache.clear
        wrapper.cache_enable = cache_enable
        wrapper.cache_disable = cache_disable
        wrapper.set_caching = set_caching
        wrapper.cacheable = True
        return wrapper
    return decorator
