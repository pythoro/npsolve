# -*- coding: utf-8 -*-
"""
Created on Thu May 21 15:21:59 2020

@author: Reuben
"""


import unittest
import numpy as np

import npsolve.soft_functions as soft


class Test_lim_scalar(unittest.TestCase):
    def setUp(self):
        self.vals = [-1000, 2.5, 3.5, 1000]
        self.limit = 3.0
       
    def test_m1000_side1(self):
        val = soft.lim(self.vals[0], self.limit, side=1, scale=0.001)
        self.assertEqual(val, self.limit)
        
    def test_25_side1(self):
        val = soft.lim(self.vals[1], self.limit, side=1, scale=0.001)
        self.assertEqual(val, self.limit)

    def test_35_side1(self):
        val = soft.lim(self.vals[2], self.limit, side=1, scale=0.001)
        self.assertEqual(val, self.vals[2])

    def test_1000_side1(self):
        val = soft.lim(self.vals[3], self.limit, side=1, scale=0.001)
        self.assertEqual(val, self.vals[3])

    def test_m1000_sidem1(self):
        val = soft.lim(self.vals[0], self.limit, side=-1, scale=0.001)
        self.assertEqual(val, self.vals[0])
        
    def test_25_sidem1(self):
        val = soft.lim(self.vals[1], self.limit, side=-1, scale=0.001)
        self.assertEqual(val, self.vals[1])

    def test_35_sidem1(self):
        val = soft.lim(self.vals[2], self.limit, side=-1, scale=0.001)
        self.assertEqual(val, self.limit)

    def test_1000_sidem1(self):
        val = soft.lim(self.vals[3], self.limit, side=-1, scale=0.001)
        self.assertEqual(val, self.limit)


class Test_lim_numpy(Test_lim_scalar):
    def setUp(self):
        super().setUp()
        self.vals = np.array(self.vals)
        self.limit = 3.0
        

class Test_floor_scalar(unittest.TestCase):
    def setUp(self):
        self.vals = [-1000, 2.5, 3.5, 1000]
        self.limit = 3.0
       
    def test_m1000(self):
        val = soft.floor(self.vals[0], self.limit, scale=0.001)
        self.assertEqual(val, self.limit)
        
    def test_25(self):
        val = soft.floor(self.vals[1], self.limit, scale=0.001)
        self.assertEqual(val, self.limit)

    def test_35(self):
        val = soft.floor(self.vals[2], self.limit, scale=0.001)
        self.assertEqual(val, self.vals[2])

    def test_1000(self):
        val = soft.floor(self.vals[3], self.limit, scale=0.001)
        self.assertEqual(val, self.vals[3])

class Test_floor_numpy(Test_floor_scalar):
    def setUp(self):
        super().setUp()
        self.vals = np.array(self.vals)
        self.limit = 3.0
        

class Test_ceil_scalar(unittest.TestCase):
    def setUp(self):
        self.vals = [-1000, 2.5, 3.5, 1000]
        self.limit = 3.0
       
    def test_m1000(self):
        val = soft.ceil(self.vals[0], self.limit, scale=0.001)
        self.assertEqual(val, self.vals[0])
        
    def test_25(self):
        val = soft.ceil(self.vals[1], self.limit, scale=0.001)
        self.assertEqual(val, self.vals[1])

    def test_35(self):
        val = soft.ceil(self.vals[2], self.limit, scale=0.001)
        self.assertEqual(val, self.limit)

    def test_1000(self):
        val = soft.ceil(self.vals[3], self.limit, scale=0.001)
        self.assertEqual(val, self.limit)


class Test_ceil_numpy(Test_ceil_scalar):
    def setUp(self):
        super().setUp()
        self.vals = np.array(self.vals)
        self.limit = 3.0
        
        
class Test_clip_scalar(unittest.TestCase):
    def setUp(self):
        self.vals = [-1000, -1.0, 2.5, 3.5, 1000]
        self.lower = 1.5
        self.upper = 3.0
       
    def test_m1000(self):
        val = soft.clip(self.vals[0], self.lower, self.upper, scale=0.001)
        self.assertEqual(val, self.lower)

    def test_m1(self):
        val = soft.clip(self.vals[1], self.lower, self.upper, scale=0.001)
        self.assertEqual(val, self.lower)

    def test_25(self):
        val = soft.clip(self.vals[2], self.lower, self.upper, scale=0.001)
        self.assertEqual(val, self.vals[2])

    def test_35(self):
        val = soft.clip(self.vals[3], self.lower, self.upper, scale=0.001)
        self.assertEqual(val, self.upper)

    def test_1000(self):
        val = soft.clip(self.vals[4], self.lower, self.upper, scale=0.001)
        self.assertEqual(val, self.upper)


class Test_clip_numpy(Test_clip_scalar):
    def setUp(self):
        super().setUp()
        self.vals = np.array(self.vals)


class Test_excess_scalar(unittest.TestCase):
    def setUp(self):
        self.vals = [-1000, 2.5, 3.5, 1000]
        self.limit = 3.0
       
    def test_m1000(self):
        val = soft.excess(self.vals[0], self.limit, scale=0.001)
        self.assertEqual(val, 0)
        
    def test_25(self):
        val = soft.excess(self.vals[1], self.limit, scale=0.001)
        self.assertEqual(val, 0)

    def test_35(self):
        val = soft.excess(self.vals[2], self.limit, scale=0.001)
        self.assertEqual(val, self.vals[2] - self.limit)

    def test_1000(self):
        val = soft.excess(self.vals[3], self.limit, scale=0.001)
        self.assertEqual(val, self.vals[3] - self.limit)


class Test_excess_numpy(Test_excess_scalar):
    def setUp(self):
        super().setUp()
        self.vals = np.array(self.vals)
        self.limit = 3.0
        

class Test_shortfall_scalar(unittest.TestCase):
    def setUp(self):
        self.vals = [-1000, 2.5, 3.5, 1000]
        self.limit = 3.0
       
    def test_m1000(self):
        val = soft.shortfall(self.vals[0], self.limit, scale=0.001)
        self.assertEqual(val, self.vals[0] - self.limit)
        
    def test_25(self):
        val = soft.shortfall(self.vals[1], self.limit, scale=0.001)
        self.assertEqual(val, self.vals[1] - self.limit)

    def test_35(self):
        val = soft.shortfall(self.vals[2], self.limit, scale=0.001)
        self.assertEqual(val, 0)

    def test_1000(self):
        val = soft.shortfall(self.vals[3], self.limit, scale=0.001)
        self.assertEqual(val, 0)


class Test_shortfall_numpy(Test_shortfall_scalar):
    def setUp(self):
        super().setUp()
        self.vals = np.array(self.vals)
        self.limit = 3.0
        
        
class Test_step_scalar(unittest.TestCase):
    def setUp(self):
        self.vals = [-1000, 2.5, 3.5, 1000]
        self.limit = 3.0
       
    def test_m1000_side1(self):
        val = soft.step(self.vals[0], self.limit, side=1, scale=0.001)
        self.assertAlmostEqual(val, 0)
        
    def test_25_side1(self):
        val = soft.step(self.vals[1], self.limit, side=1, scale=0.001)
        self.assertAlmostEqual(val, 0)

    def test_35_side1(self):
        val = soft.step(self.vals[2], self.limit, side=1, scale=0.001)
        self.assertEqual(val, 1)

    def test_1000_side1(self):
        val = soft.step(self.vals[3], self.limit, side=1, scale=0.001)
        self.assertEqual(val, 1)
        
    def test_m1000_sidem1(self):
        val = soft.step(self.vals[0], self.limit, side=-1, scale=0.001)
        self.assertEqual(val, 1)
        
    def test_25_sidem1(self):
        val = soft.step(self.vals[1], self.limit, side=-1, scale=0.001)
        self.assertEqual(val, 1)

    def test_35_sidem1(self):
        val = soft.step(self.vals[2], self.limit, side=-1, scale=0.001)
        self.assertAlmostEqual(val, 0)

    def test_1000_sidem1(self):
        val = soft.step(self.vals[3], self.limit, side=-1, scale=0.001)
        self.assertAlmostEqual(val, 0)


class Test_step_numpy(Test_step_scalar):
    def setUp(self):
        super().setUp()
        self.vals = np.array(self.vals)
        self.limit = 3.0
        

class Test_above_scalar(unittest.TestCase):
    def setUp(self):
        self.vals = [-1000, 2.5, 3.5, 1000]
        self.limit = 3.0
       
    def test_m1000(self):
        val = soft.step(self.vals[0], self.limit, scale=0.001)
        self.assertAlmostEqual(val, 0)
        
    def test_25(self):
        val = soft.step(self.vals[1], self.limit, scale=0.001)
        self.assertAlmostEqual(val, 0)

    def test_35(self):
        val = soft.step(self.vals[2], self.limit, scale=0.001)
        self.assertEqual(val, 1)

    def test_1000(self):
        val = soft.step(self.vals[3], self.limit, scale=0.001)
        self.assertEqual(val, 1)
        

class Test_above_numpy(Test_above_scalar):
    def setUp(self):
        super().setUp()
        self.vals = np.array(self.vals)
        self.limit = 3.0
        
        
class Test_below_scalar(unittest.TestCase):
    def setUp(self):
        self.vals = [-1000, 2.5, 3.5, 1000]
        self.limit = 3.0
       
    def test_m1000(self):
        val = soft.below(self.vals[0], self.limit, scale=0.001)
        self.assertEqual(val, 1)
        
    def test_25(self):
        val = soft.below(self.vals[1], self.limit, scale=0.001)
        self.assertEqual(val, 1)

    def test_35(self):
        val = soft.below(self.vals[2], self.limit, scale=0.001)
        self.assertAlmostEqual(val, 0)

    def test_1000(self):
        val = soft.below(self.vals[3], self.limit, scale=0.001)
        self.assertAlmostEqual(val, 0)
        

class Test_below_numpy(Test_below_scalar):
    def setUp(self):
        super().setUp()
        self.vals = np.array(self.vals)
        self.limit = 3.0
        
        
class Test_within_scalar(unittest.TestCase):
    def setUp(self):
        self.vals = [-1000, -1.0, 2.5, 3.5, 1000]
        self.lower = 1.5
        self.upper = 3.0
       
    def test_m1000(self):
        val = soft.within(self.vals[0], self.lower, self.upper, scale=0.001)
        self.assertAlmostEqual(val, 0)

    def test_m1(self):
        val = soft.within(self.vals[1], self.lower, self.upper, scale=0.001)
        self.assertAlmostEqual(val, 0)

    def test_25(self):
        val = soft.within(self.vals[2], self.lower, self.upper, scale=0.001)
        self.assertEqual(val, 1)

    def test_35(self):
        val = soft.within(self.vals[3], self.lower, self.upper, scale=0.001)
        self.assertAlmostEqual(val, 0)

    def test_1000(self):
        val = soft.within(self.vals[4], self.lower, self.upper, scale=0.001)
        self.assertAlmostEqual(val, 0)


class Test_within_numpy(Test_within_scalar):
    def setUp(self):
        super().setUp()
        self.vals = np.array(self.vals)
        
        
class Test_outside_scalar(unittest.TestCase):
    def setUp(self):
        self.vals = [-1000, -1.0, 2.5, 3.5, 1000]
        self.lower = 1.5
        self.upper = 3.0
       
    def test_m1000(self):
        val = soft.outside(self.vals[0], self.lower, self.upper, scale=0.001)
        self.assertAlmostEqual(val, 1)

    def test_m1(self):
        val = soft.outside(self.vals[1], self.lower, self.upper, scale=0.001)
        self.assertAlmostEqual(val, 1)

    def test_25(self):
        val = soft.outside(self.vals[2], self.lower, self.upper, scale=0.001)
        self.assertAlmostEqual(val, 0)

    def test_35(self):
        val = soft.outside(self.vals[3], self.lower, self.upper, scale=0.001)
        self.assertAlmostEqual(val, 1)

    def test_1000(self):
        val = soft.outside(self.vals[4], self.lower, self.upper, scale=0.001)
        self.assertAlmostEqual(val, 1)


class Test_outside_numpy(Test_outside_scalar):
    def setUp(self):
        super().setUp()
        self.vals = np.array(self.vals)
        
        
class Test_sign_scalar(unittest.TestCase):
    def setUp(self):
        self.vals = [-1000, -0.5, 0.5, 1000]
        self.limit = 3.0
       
    def test_m1000(self):
        val = soft.sign(self.vals[0], scale=0.001)
        self.assertAlmostEqual(val, -1)

    def test_25(self):
        val = soft.sign(self.vals[1], scale=0.001)
        self.assertAlmostEqual(val, -1)

    def test_35(self):
        val = soft.sign(self.vals[2], scale=0.001)
        self.assertAlmostEqual(val, 1)

    def test_1000(self):
        val = soft.sign(self.vals[3], scale=0.001)
        self.assertAlmostEqual(val, 1)


class Test_sign_numpy(Test_sign_scalar):
    def setUp(self):
        super().setUp()
        self.vals = np.array(self.vals)