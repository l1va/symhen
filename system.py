from sympy import zeros, symbols
from lie_algebra import adf_g, lie_diff
from sympy.printing import sstr


def from_control_affine_form(state_evolution, input_field, states):
    return ControlSystem(state_evolution, input_field, states)


class ControlSystem:
    def __init__(self, f, g, x):
        self.f = f
        self.g = g
        self.x = x

    def controllability_matrix(self):
        n = len(self.f)
        C = zeros(n)
        for i in range(n):
            C[:, i] = adf_g(self.f, self.g, self.x, i)  # TODO: improve performance
        return C

    def find_dependency(self):
        n = len(self.f)
        c = zeros(n, 1)
        c[n - 1] = symbols('\phi')
        sol = (self.controllability_matrix().T).inv() * c
        j = None
        for i in range(n):
            if sol[i] != 0:
                j = i
        return self.x[j]

    def transformation(self):
        n = len(self.f)
        T = zeros(n, 1)
        T[0] = self.find_dependency()
        for i in range(1, n):
            T[i] = lie_diff(T[i - 1], self.f, self.x)
        return T

    def beta(self):
        T = self.transformation()
        Lg_Tn = lie_diff(T[-1], self.g, self.x)
        return 1 / Lg_Tn

    def alpha(self):
        T = self.transformation()
        Lf_Tn = lie_diff(T[-1], self.f, self.x)
        return self.beta() * Lf_Tn


    def __str__(self):
        return 'System(\n' + sstr(self.f) + ',\n' \
               + sstr(self.g) + ')'

    def __repr__(self):
        return sstr(self)
