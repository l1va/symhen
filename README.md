# symhen
Control with sympy

[Simeon's Colab Sources](https://colab.research.google.com/drive/1IbqQAi2Vsp-jGaP39JSP2Kl-MW8SGZQ_)


```python
from sympy import (symbols, Matrix, cos, sin)
from system import from_control_affine_form

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

print(cs.controllability_matrix())
# Matrix([[0, -a * (b + cos(x1))], [b + cos(x1), (b + cos(x1)) * cos(x1) - (a * x2 - x1 + sin(x1)) * sin(x1)]])
print(cs.find_dependency())
# x1
print(cs.transformation())
# Matrix([x1, a * x2 - x1 + sin(x1)])
print(cs.beta()) 
# 1 / (a * (b + cos(x1)))
print(cs.alpha())
# 1 / (a * (b + cos(x1))) * ( -a*x2*cos(x1)+ (cos(x1)-1)*(a*x2-x1+sin(x1)) )     

```