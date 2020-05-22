# -*- coding: utf-8 -*-
"""
Created on Fri May 22 11:55:42 2020

@author: Reuben
"""

import unittest

from npsolve import utils


class Test_Util_Containers(unittest.TestCase):
       
    def test_get_dict(self):
        d = utils.get_dict('test')
        self.assertTrue(isinstance(d, dict))
        self.assertTrue(d['a'] is None)
        
    def test_get_list(self):
        d = utils.get_list('test')
        self.assertTrue(isinstance(d, list))
        
    def test_get_set(self):
        d = utils.get_set('test')
        self.assertTrue(isinstance(d, set))

    def test_get_list_containert(self):
        d = utils.get_list_container('test')
        lst = d['a']
        self.assertTrue(isinstance(lst, list))

