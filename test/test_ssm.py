from sympy import (symbols, Matrix, cos, sin, det)
from state_space_model import StateSpaceModel

def test_StateSpaceModel_create():
    a, b = symbols('a, b')

    A = Matrix([[2*a , a],
                [3*b, b]])

    B = Matrix([[0],
                [1]])

    C = Matrix([[2,4*b]])

    cs = StateSpaceModel(A,B,C)
    assert cs.A == A

    print(cs)
    print(repr(cs))

    cm = cs.controllability_matrix()
    print(cm)
    assert cm == Matrix([[0, a], [1 , b]])
    print(det(cm))
    assert det(cm) == -a

