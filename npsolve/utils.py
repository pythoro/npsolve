# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 08:31:52 2019

@author: Reuben

This module is about logging extra variables during solving.

The recommended way is to use a List_Container.

"""
from collections import defaultdict

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