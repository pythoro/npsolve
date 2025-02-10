
from tutorial_1 import get_package


import numpy as np
import npsolve
import matplotlib.pyplot as plt

# Unique variable names
COMP1_POS = 'position1'
COMP1_VEL = 'velocity1'
COMP2_VALUE = 'component2_value'
COMP2_FORCE = 'comp2_force'



def solve(package, t_end=10):
    ode_integrator = npsolve.solvers.ODEIntegrator()
    dct = ode_integrator.run(package, t_end)
    return dct


def run():
    package = get_package()
    inits = {COMP1_POS: 0.1,
             COMP1_VEL: 0.3,
             COMP2_VALUE: -0.1}
    package.setup(inits)
    dct = solve(package)
    return dct


def plot(dct):
    plt.figure(1)
    dct2 = dct.copy()
    t_vec = dct2.pop('time')
    for var_name, values in dct2.items():
        plt.plot(t_vec, values, label=var_name)
    plt.legend()
    plt.show()


def execute():
    dct = run()
    plot(dct)


if __name__ == '__main__':
    execute()