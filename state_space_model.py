from sympy import (symbols, zeros, latex, ShapeError)
from sympy.printing import sstr


class StateSpaceModel:

    def __init__(self, A, B, C=None, D=None, x=None, u=None):
        if not (A.shape[0] == A.shape[1]):
            raise ShapeError("Shapes of A must be nxn")
        if C is None:
            C = zeros(1, A.shape[0])
        if D is None:
            D = zeros(C.shape[0], B.shape[1])
        if x is None:
            x = symbols("x1:" + str(A.shape[0]+1))
        if u is None:
            u = symbols("u1:" + str(B.shape[1]+1))
        if not ((A.shape[0] == A.shape[1]) and
                (A.shape[0] == B.shape[0]) and
                (A.shape[1] == C.shape[1]) and
                (B.shape[1] == D.shape[1]) and
                (C.shape[0] == D.shape[0]) and
                (len(x) == A.shape[0]) and
                (len(u) == B.shape[1])):
            print(A.shape[0] == A.shape[1])
            print(A.shape[0] == B.shape[0])
            print(A.shape[1] == C.shape[1])
            print(B.shape[1] == D.shape[1])
            print(C.shape[0] == D.shape[0])
            print(len(x), A.shape[0], len(x) == A.shape[0])
            print(len(u) == B.shape[1])
            raise ShapeError("Shapes of A,B,C,D,x,u must fit")
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.x = x
        self.u = u

    def __str__(self):
        return 'StateSpaceModel(\n' + sstr(self.A) + ',\n' \
               + sstr(self.B) + ',\n' \
               + sstr(self.C) + ',\n' \
               + sstr(self.D) + ')'

    def __repr__(self):
        return sstr(self)

    def controllability_matrix(self):
        res = self.B
        for i in range(len(self.x) - 1):
            res = res.row_join(self.A * res.col(i))
        return res

    def repr_latex(self):
        return '$' + latex([self.A, self.B, self.C, self.D]) + '$'
