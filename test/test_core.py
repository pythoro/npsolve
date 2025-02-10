

import pytest
import timeit
from pytest import approx
import numpy as np


from npsolve.core import Slicer, Package



class MockObject:
    def __init__(self):
        self.states = []

    def step(self, state_dct, t, log):
        self.states.append(state_dct.copy())
    
    def get_derivs(self, state_dct, t, log):
        return state_dct.copy()


@pytest.fixture
def slicer1():
    slicer = Slicer()
    dct = {
        'test_key_1': np.array([2.0, 3.0]),
        'test_key_2': np.array([7.0]),
    }
    slicer.add_dict(dct)
    return slicer

class Test_Slicer:
    def test_add(self):
        slicer = Slicer()
        slicer.add('test_key_1', np.array([2.0, 3.0]))
        slicer.add('test_key_2', np.array([7.0]))
        assert slicer.length == 3
        slices = slicer.slices
        assert slices['test_key_1'] == slice(0, 2)
        assert slices['test_key_2'] == slice(2, 3)


    def test_add_dict(self, slicer1):
        slicer = slicer1
        assert slicer.length == 3
        slices = slicer.slices
        assert slices['test_key_1'] == slice(0, 2)
        assert slices['test_key_2'] == slice(2, 3)

    def test_get_slice(self, slicer1):
        slicer = slicer1
        slc = slicer.get_slice('test_key_1')
        assert slc == slice(0, 2)
        slc = slicer.get_slice('test_key_2')
        assert slc == slice(2, 3)

    def test_get_state_vec(self, slicer1):
        slicer = slicer1
        dct = {
            'test_key_1': np.array([2.0, 3.0]),
            'test_key_2': np.array([7.0]),
        }
        state_vec = slicer.get_state_vec(dct)
        assert state_vec == approx(np.array([2.0, 3.0, 7.0]))
    
    def test_get_state(self, slicer1):
        slicer = slicer1
        dct = {
            'test_key_1': np.array([2.0, 3.0]),
            'test_key_2': np.array([7.0]),
        }
        keys = ['test_key_1', 'test_key_2']
        state_vec = slicer.get_state_vec(dct)
        state = slicer.get_state(state_vec, keys)
        for key, val in state.items():
            assert val == approx(dct[key])
        assert state['test_key_1'].base is state_vec
        assert state['test_key_2'].base is state_vec
        assert not state['test_key_1'].flags.writeable
        assert not state['test_key_2'].flags.writeable

    def test_get_state_dct_writeable(self, slicer1):
        slicer = slicer1
        dct = {
            'test_key_1': np.array([2.0, 3.0]),
            'test_key_2': np.array([7.0]),
        }
        keys = ['test_key_1', 'test_key_2']
        state_vec = slicer.get_state_vec(dct)
        state = slicer.get_state(state_vec, keys, writeable=True)
        for key, val in state.items():
            assert val == approx(dct[key])
        assert state['test_key_1'].base is state_vec
        assert state['test_key_2'].base is state_vec
        assert state['test_key_1'].flags.writeable
        assert state['test_key_2'].flags.writeable


class Test_Package:
    def test_add_component(self):
        package = Package()
        mo = MockObject()
        package.add_component(mo, 'test_obj1', 'get_derivs')
        assert len(package._components) == 1
        
    def test_add_stage_call(self):
        package = Package()
        mo = MockObject()
        package.add_component(mo, 'test_obj1', 'get_derivs')
        package.add_stage_call('test_obj1', 'step')
        assert len(package._stage_calls) == 1
        assert package._stage_calls[0][0] == 'test_obj1'
        assert package._stage_calls[0][1] == mo.step
    
    def test_setup(self):
        inits = {'test_obj1_a': np.array([0.0, 0.1]),
                 'test_obj1_b': np.array([0.2])}
        package = Package()
        package.setup(inits)
        assert len(package.inits) == 2
        assert len(package._state) == 2
        assert len(package._ret) == 2
        assert package._state_vec == approx(np.array([0.0, 0.1, 0.2]))
        assert package._ret_vec == approx(np.zeros(3))

    def test_step(self):
        inits = {'test_obj1_a': np.array([0.0, 0.1]),
                 'test_obj1_b': np.array([0.2])}
        package = Package()
        mo = MockObject()
        package.add_component(mo, 'test_obj1', 'get_derivs')
        package.add_stage_call('test_obj1', 'step')
        vec = np.array([1.0, 2.0, 3.0])
        package.setup(inits)
        ret_vec = package.step(vec, 0, log=None)
        obj_states = package.components['test_obj1'].states[0]
        assert obj_states['test_obj1_a'] == approx(np.array([1.0, 2.0]))
        assert obj_states['test_obj1_b'] == approx(np.array([3.0]))
        assert ret_vec == approx(vec)
        

class MockObject2:    
    def step(self, state_dct, t, log):
        return {'test_obj1_a': state_dct['test_obj1_a'] + 1.0,
                'test_obj1_b': state_dct['test_obj1_b'] + 1.0,
                'test_obj1_c': state_dct['test_obj1_c'] + 1.0,
                'test_obj1_d': state_dct['test_obj1_d'] + 1.0}
        

npsolve_name = 'partial_1'


class Test_Performance():

    def test_steps(self):
        package = Package()
        mo = MockObject2()
        inits = {
            'test_obj1_a': np.linspace(0, 3, 3),
            'test_obj1_b': np.linspace(3, 6, 3),
            'test_obj1_c': np.linspace(6, 9, 3),
            'test_obj1_d': np.linspace(9, 12, 3),
        }
        package.add_component(mo, 'test_obj1', 'step')
        package.setup(inits)
        vec = package.init_vec
        
        globals_dct = {'package': package, 'vec': vec}

        time = timeit.timeit('package.step(vec, 0, log=None)',
                             globals=globals_dct,
                             number=100000)
        
        def step_baseline(vec):
            ret = np.zeros_like(vec)
            ret[0:3] = vec[0:3] + 1.0
            ret[3:6] = vec[3:6] + 1.0
            ret[6:9] = vec[6:9] + 1.0
            ret[9:12] = vec[9:12] + 1.0
            return ret
            
        globals_dct = {'step_baseline': step_baseline, 'vec': vec}
        baseline = timeit.timeit('step_baseline(vec)',
                                 globals=globals_dct,
                                 number=100000)
        relative_speed = time / baseline
        assert relative_speed <= 1.15
        
        




