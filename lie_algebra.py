from sympy import Matrix

"""
>Let $h(\mathbf{x}) \in \mathbb{R}$ be a smooth function, and $\mathbf{f}(\mathbf{x}) \in \mathbb{R}^n$ be a smooth vector field, the **Lie derivative** of $h$ with respect to $\mathbf{f}$ is given by directional derivative:
>\begin{equation}
L_\mathbf{f} h = \nabla h^T\mathbf{f} = \sum_{i=1}^n \frac{\partial h}{\partial x_i} f_i
\end{equation}
"""


def lie_diff(h, f, x):
    grad = Matrix([h]).jacobian(x)
    lf_h = grad * f
    return lf_h[0]


"""
The $n$-th order Lie derivative is defined recursevely as:
\begin{equation}
L^k_\mathbf{f} h = L_\mathbf{f} (L_\mathbf{f}^{k-1}h), \quad \text{for }k=1,\dots, n
\end{equation}
With $L^0_\mathbf{f} h = h$
"""


def lie_diff_n(h, f, x, n):
    lf_h_k = h
    for k in range(n):
        lf_h_k = lie_diff(lf_h_k, f, x)
    return lf_h_k


"""
>The **Lie Bracket** of two vector fields $\mathbf{f}(\mathbf{x}),\mathbf{g}(\mathbf{x})$ is vector field defined as follows:
>
>\begin{equation}
[\mathbf{f},\mathbf{g}] = \frac{\partial \mathbf{g}}{\partial \mathbf{x}}\mathbf{f} - 
\frac{\partial \mathbf{f}}{\partial \mathbf{x}}\mathbf{g}
\end{equation}
"""


def lie_bracket(f, g, x):
    return g.jacobian(x) * f - f.jacobian(x) * g


"""
>We denote $ad_\mathbf{f}(\mathbf{g}) = [\mathbf{f},\mathbf{g}]$ ($ad$ - for *adjoint*) while $ad^k_\mathbf{f}(\mathbf{g})$  defined recursevely as:
\begin{equation}
ad^k_\mathbf{f}(\mathbf{g}) = [\mathbf{f},ad^{k-1}_\mathbf{f}(\mathbf{g})]
\end{equation}
with $ad^0_\mathbf{f}(\mathbf{g}) = \mathbf{g}$
"""


def adf_g(f, g, x, n):
    ad_k = g
    for k in range(n):
        ad_k = lie_bracket(f, ad_k, x)
    return ad_k
