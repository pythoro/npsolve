

import pytest
import timeit
from pytest import approx
import numpy as np


from npsolve.core import Variable, Component, Slicer, Package


class Test_Variable:
    def test_variable(self):
        var = Variable('test', np.zeros(3))
        assert var.name == 'test'
        assert var.init == approx(np.zeros(3))

    def test_set_init(self):
        var = Variable('test', np.zeros(3))
        var.set_init(np.array([1.0, 2.0, 3.0]))
        assert var.init == approx(np.array([1.0, 2.0, 3.0]))
    


class MockObject:
    def __init__(self):
        self.states = []

    def step(self, state_dct):
        self.states.append(state_dct.copy())
    
    def get_derivs(self, state_dct):
        ret = {k: np.zeros_like(v) for k, v in state_dct.items()}
        return ret


class Test_Component:
    def test_init(self):
        mo = MockObject()
        component = Component('test', mo)
        assert True
    
    def test_add_var(self):
        mo = MockObject()
        component = Component('test', mo)
        component.add_var('var1', np.array([0.1, 0.2]))
        var = Variable('var1', np.array([0.1, 0.2]))
        assert component._vars['var1'].name == var.name
        assert component._vars['var1'].init == approx(var.init)

    def test_set_init(self):
        mo = MockObject()
        component = Component('test', mo)
        component.add_var('var1', np.array([0.1, 0.2]))
        component.set_init('var1', np.array([0.3, 0.4]))
        variable = component.get_variable('var1')
        assert variable.name == 'var1'
        assert variable.init == approx(np.array([0.3, 0.4]))


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

    def test_get_state(self, slicer1):
        slicer = slicer1
        dct = {
            'test_key_1': np.array([2.0, 3.0]),
            'test_key_2': np.array([7.0]),
        }
        state = slicer.get_state(dct)
        assert state == approx(np.array([2.0, 3.0, 7.0]))
    
    def test_get_state_dct(self, slicer1):
        slicer = slicer1
        dct = {
            'test_key_1': np.array([2.0, 3.0]),
            'test_key_2': np.array([7.0]),
        }
        keys = ['test_key_1', 'test_key_2']
        state = slicer.get_state(dct)
        state_dct = slicer.get_state_dct(state, keys)
        for key, val in state_dct.items():
            assert val == approx(dct[key])
        assert state_dct['test_key_1'].base is state
        assert state_dct['test_key_2'].base is state
        assert not state_dct['test_key_1'].flags.writeable
        assert not state_dct['test_key_2'].flags.writeable

    def test_get_state_dct_writeable(self, slicer1):
        slicer = slicer1
        dct = {
            'test_key_1': np.array([2.0, 3.0]),
            'test_key_2': np.array([7.0]),
        }
        keys = ['test_key_1', 'test_key_2']
        state = slicer.get_state(dct)
        state_dct = slicer.get_state_dct(state, keys, writeable=True)
        for key, val in state_dct.items():
            assert val == approx(dct[key])
        assert state_dct['test_key_1'].base is state
        assert state_dct['test_key_2'].base is state
        assert state_dct['test_key_1'].flags.writeable
        assert state_dct['test_key_2'].flags.writeable


class Test_Package:
    def test_add_component(self):
        package = Package()
        mo = MockObject()
        component = Component('test_obj1', mo)
        package.add_component(component, 'get_derivs')
        assert len(package._components) == 1
    
    def test_next_stage(self):
        package = Package()
        package.next_stage()
        assert len(package._stages) == 2
    
    def test_add_stage_call(self):
        package = Package()
        mo = MockObject()
        component = Component('test_obj1', mo)
        package.add_component(component, 'get_derivs')
        package.add_stage_call('test_obj1', 'step')
        assert len(package._stages[0]) == 1
        assert package._stages[0][0][0] == 'test_obj1'
        assert package._stages[0][0][1] == mo.step
    
    def test_setup(self):
        inits = {'test_obj1_a': np.array([0.0, 0.1]),
                 'test_obj1_b': np.array([0.2])}
        package = Package()
        package.setup(inits)
        assert len(package._inits) == 2
        assert len(package._state_dct) == 2
        assert len(package._ret_dct) == 2
        assert package._state == approx(np.array([0.0, 0.1, 0.2]))
        assert package._ret == approx(np.zeros(3))

    def test_get_inits(self):
        package = Package()
        mo = MockObject()
        component = Component('test_obj1', mo)
        component.add_var('test_obj1_a', np.array([0.0, 0.1]))
        component.add_var('test_obj1_b', np.array([0.2]))
        package.add_component(component, 'get_derivs')
        package.add_stage_call('test_obj1', 'step')
        inits = package.get_inits()
        assert inits['test_obj1_a'] == approx(np.array([0.0, 0.1]))
        assert inits['test_obj1_b'] == approx(np.array([0.2]))

    def test_step(self):
        inits = {'test_obj1_a': np.array([0.0, 0.1]),
                 'test_obj1_b': np.array([0.2])}
        package = Package()
        mo = MockObject()
        component = Component('test_obj1', mo)
        component.add_var('test_obj1_a', np.array([0.0, 0.1]))
        component.add_var('test_obj1_b', np.array([0.2]))
        package.add_component(component, 'get_derivs')
        package.add_stage_call('test_obj1', 'step')
        vec = np.array([1.0, 2.0, 3.0])
        package.setup(inits)
        package.step(vec)
        obj_states = package.components['test_obj1'].obj.states[0]
        assert obj_states['test_obj1_a'] == approx(np.array([1.0, 2.0]))
        assert obj_states['test_obj1_b'] == approx(np.array([3.0]))
        

class MockObject2:    
    def step(self, state_dct, *args):
        return {'test_obj1_a': state_dct['test_obj1_a'] + 1.0,
                'test_obj1_b': state_dct['test_obj1_b'] + 1.0,
                'test_obj1_c': state_dct['test_obj1_c'] + 1.0,
                'test_obj1_d': state_dct['test_obj1_d'] + 1.0}
        

npsolve_name = 'partial_1'


class Test_Performance():

    def test_steps(self):
        package = Package()
        mo = MockObject2()
        component = Component('test_obj1', mo)
        component.add_var('test_obj1_a', np.linspace(0, 3, 3))
        component.add_var('test_obj1_b', np.linspace(3, 6, 3))
        component.add_var('test_obj1_c', np.linspace(6, 9, 3))
        component.add_var('test_obj1_d', np.linspace(9, 12, 3))
        package.add_component(component, 'step')
        inits = package.get_inits()
        package.setup(inits)
        vec = package._state
        
        globals_dct = {'package': package, 'vec': vec}

        time = timeit.timeit('package.step(vec)',
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
        
        




