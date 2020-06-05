from sympy import (symbols, Matrix, cos, sin, det, solve)
from system import from_control_affine_form


def test_System1():
    # Define symbols for parameters
    a, b = symbols('a, b')

    # Define vector for states
    x1, x2 = symbols('x1, x2')

    # Define state vector field: f(x)
    f = Matrix([- x1 + a * x2 + sin(x1),
                - x2 * cos(x1)])

    # Define control vector field: g(x)
    g = Matrix([0,
                cos(x1) + b])

    x = Matrix([x1, x2])

    cs = from_control_affine_form(f, g, x)
    assert cs.f == f

    cm = cs.controllability_matrix()
    print(cm)
    assert cm == Matrix(
        [[0, -a * (b + cos(x1))], [b + cos(x1), (b + cos(x1)) * cos(x1) - (a * x2 - x1 + sin(x1)) * sin(x1)]])
    print(det(cm))
    assert det(cm) == a * (b + cos(x1)) ** 2

    print(cs)
    assert cs.find_dependency() == x1
    tr = cs.transformation()
    print(tr)
    assert tr == Matrix([x1, a * x2 - x1 + sin(x1)])

    assert cs.beta() == 1 / (a * (b + cos(x1)))

    assert cs.alpha() == 1 / (a * (b + cos(x1))) * ( -a*x2*cos(x1)+ (cos(x1)-1)*(a*x2-x1+sin(x1)) )

def test_System2():
    I, b, k, m, L, grav = symbols('I, b, k, m, L, g')

    q = Matrix(symbols(r'theta_1, theta_2'))
    dq = Matrix(symbols(r'\dot{\theta}_1, \dot{\theta}_2'))
    x = Matrix([q, dq])

    f = Matrix([x[2],
            x[3],
            1 / I * (-b * x[2] - k * (x[0] - x[1])),
            1 / (m * L ** 2) * (k * (x[0] - x[1]) - m * grav * L * sin(x[1]))])

    g = Matrix([0,
                0,
                1 / I,
                0])

    cs = from_control_affine_form(f, g, x)
    assert det(cs.controllability_matrix()) == -k**2/(I**4*L**4*m**2)

    assert cs.beta() == I*m*L**2/k
