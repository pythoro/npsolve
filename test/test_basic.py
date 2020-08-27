# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 11:44:00 2020

@author: Reuben
"""

import unittest
import numpy as np

from npsolve import basic


class Test_Basic(unittest.TestCase):
       
    def test_add_str(self):
        vs = basic.V_Set()
        vs.add('a')
        vs.add('b')
        self.assertDictEqual(vs._dct, {'a': 0, 'b': 1})
        return vs

    def test_add_str_list(self):
        vs = basic.V_Set()
        vs.add('a b')
        self.assertDictEqual(vs._dct, {'a': 0, 'b': 1})
        return vs

    def test_add_list(self):
        vs = basic.V_Set()
        vs.add(['a', 'b'])
        self.assertDictEqual(vs._dct, {'a': 0, 'b': 1})
        return vs

    def test_add_str_sizes(self):
        vs = basic.V_Set()
        vs.add('a', sizes=1)
        vs.add('b', sizes=3)
        self.assertDictEqual(vs._dct, {'a': 0, 'b': slice(1, 4, None)})
        return vs

    def test_add_str_list_sizes(self):
        vs = basic.V_Set()
        vs.add('a b', sizes=[1, 3])
        self.assertDictEqual(vs._dct, {'a': 0, 'b': slice(1, 4, None)})
        return vs
    
    def test_ind_scalars(self):
        vs = self.test_add_str()
        A, B = vs.ind()
        self.assertEqual(A, 0)
        self.assertEqual(B, 1)

    def test_ind_arrays(self):
        vs = self.test_add_str_sizes()
        A, B = vs.ind()
        self.assertEqual(A, 0)
        self.assertEqual(B, slice(1, 4, None))

    def test_array_scalars_kwargs(self):
        vs = self.test_add_str()
        arr = vs.array(a=3, b=5)
        self.assertTrue(np.array_equal(arr, np.array([3, 5])))
        
    def test_array_scalars_dcts(self):
        vs = self.test_add_str()
        arr = vs.array({'a': 3, 'b': 5})
        self.assertTrue(np.array_equal(arr, np.array([3, 5])))
        
    def test_array_vectors_kwargs(self):
        vs = self.test_add_str_sizes()
        arr = vs.array(a=3, b=[5, 7, 9])
        self.assertTrue(np.array_equal(arr, np.array([3, 5, 7, 9])))

    def test_array_vectors_ndarray(self):
        vs = self.test_add_str_sizes()
        arr = vs.array(a=3, b=np.array([5, 7, 9]))
        self.assertTrue(np.array_equal(arr, np.array([3, 5, 7, 9])))

    def test_lock_on_init(self):
        vs = basic.V_Set('a b')
        self.assertTrue(vs._locked)

    def test_lock(self):
        vs = self.test_add_str()
        vs.lock()
        self.assertTrue(vs._locked)
        return vs
    
    def test_lock_error(self):
        vs = self.test_lock()
        with self.assertRaises(RuntimeError) as context:
            vs.add('c')

    def test_unpack(self):
        vs = self.test_add_str_sizes()
        a, b = vs.unpack(np.array([3, 5, 7, 9]), 'a b')
        self.assertEqual(a, 3)
        self.assertTrue(np.array_equal(b, np.array([5, 7, 9])))

    def test_unpack_2(self):
        vs = self.test_add_str_sizes()
        b, a = vs.unpack(np.array([3, 5, 7, 9]), 'b a')
        self.assertEqual(a, 3)
        self.assertTrue(np.array_equal(b, np.array([5, 7, 9])))
        
