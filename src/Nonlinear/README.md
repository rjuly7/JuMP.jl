# Nonlinear

This submodule contains data structures and functions for working with a
nonlinear program in the form of an expression tree.

## Standard form

[Nonlinear programs (NLPs)](https://en.wikipedia.org/wiki/Nonlinear_programming)
are a class of optimization problems in which some of the constraints or the
objective function are nonlinear:
```math
\begin{align}
    \min_{x \in \mathbb{R}^n} & f_0(x) \\
    \;\;\text{s.t.} & l_j \le f_j(x) \le u_j & j = 1 \ldots m
\end{align}
```
There may be additional constraints, as well as things like variable bounds
and integrality restrictions, but we do not consider them here because they are
best dealt with by other components of JuMP and MathOptInterface.

## API overview

The core element of the `Nonlinear` submodule is `NonlinearData`:
```julia
import JuMP: Nonlinear
data = Nonlinear.NonlinearData()
```
`NonlinearData` is a mutable struct that stores all of the nonlinear information
added to the model.

The input data-structure is a Julia `Expr`. The input expressions can
incorporate `MOI.VariableIndex`es, but these must be interpolated into the
expression with `$`:
```julia
import JuMP: MOI
x = MOI.VariableIndex(1)
input = :(1 + sin($x)^2)
```

Given an input expression, add an expression using `add_expression`:
```julia
expr = Nonlinear.add_expression(data, input)
```
The return value, `expr`, is a `Nonlinear.ExpressionIndex` that can then be
interpolated into other input expressions.

In addition to constant literals like `1` or `1.23`, you can create parameters.
Parameter are constants that you can change before passing the expression to the
solver. Create a parameter using `add_parameter`, which accepts a default value:
```julia
p = Nonlinear.add_parameter(data, 1.23)
```
The return value, `p`, is a `Nonlinear.ParameterIndex` that can then be
interpolated into other input expressions.

Update a parameter using `set_parameter`:
```julia
Nonlinear.set_parameter(data, p, 4.56)
```

Set a nonlinear objective using `set_objective`:
```julia
Nonlinear.set_objective(data, :($p + $expr + $x))
```

Add a constraint using `add_constraint`:
```julia
c = Nonlinear.add_constraint(data, :(1 + sqrt($x)), MOI.LessThan(2.0))
```
The return value, `c`, is a `Nonlinear.ConstraintIndex` that is a unique
identifier for the constraint.

Delete a constraint using `delete`:
```julia
Nonlinear.delete(data, c)
```

## User-defined functions

By default, `Nonlinear` supports a wide range of univariate and multivariate
functions. However, you can also define your own functions by _registering_
them.

### Univariate functions

Register a univariate user-defined function using `register_operator`:
```julia
f(x) = 1 + sin(x)^2
Nonlinear.register_operator(data, :my_f, 1, f)
```
Now, you can use `:my_f` in expressions:
```julia
new_expr = Nonlinear.add_expression(data, :(my_f($x + 1)))
```
By default, `Nonlinear` will compute first- and second-derivatives of the
registered operator using `ForwardDiff.jl`. Over-ride this by passing functions
which compute the respective derivative:
```julia
f(x) = 1 + sin(x)^2
f′(x) = 2 * sin(x) * cos(x)
Nonlinear.register_operator(data, :my_f, 1, f, f′)
```
or
```julia
f(x) = 1 + sin(x)^2
f′(x) = 2 * sin(x) * cos(x)
f′′(x) = 2 * (cos(x)^2 - sin(x)^2)
Nonlinear.register_operator(data, :my_f, 1, f, f′, f′′)
```

### Multivariate functions

Register a univariate user-defined function using `register_operator`:
```julia
f(x) = x[1]^2 + x[1] * x[2] + x[2]^2
Nonlinear.register_operator(data, :my_f, 2, f)
```
Now, you can use `:my_f` in expressions:
```julia
new_expr = Nonlinear.add_expression(data, :(my_f([$x + 1, $x])))
```
By default, `Nonlinear` will compute the gradient of the registered
operator using `ForwardDiff.jl`. (Hessian information is not supported.)
Over-ride this by passing a function to compute the gradient:
```julia
f(x) = x[1]^2 + x[1] * x[2] + x[2]^2
function ∇f(g, x)
    g[1] = 2 * x[1] + x[2]
    g[2] = x[1] + 2 * x[2]
    return
end
Nonlinear.register_operator(data, :my_f, 2, f, ∇f)
```
