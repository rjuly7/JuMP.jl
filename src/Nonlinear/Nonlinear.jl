module Nonlinear

import Base.Meta: isexpr
import ForwardDiff
import MathOptInterface

const MOI = MathOptInterface

include("univariate_expressions.jl")
include("operators.jl")
include("types.jl")
include("parse.jl")

"""
    set_objective(data::NonlinearData, obj)
"""
function set_objective(data::NonlinearData, obj)
    data.objective = parse_expression(data, obj)
    return
end

"""
    add_expression(data::NonlinearData, expr)
"""
function add_expression(data::NonlinearData, expr)
    push!(data.expressions, parse_expression(data, expr))
    return ExpressionIndex(length(data.expressions))
end

"""
    add_constraint(data::NonlinearData, expr::Expr)
"""
function add_constraint(data::NonlinearData, input::Expr)
    expr, set = _expr_to_constraint(input)
    f = parse_expression(data, expr)
    data.last_constraint_index += 1
    index = ConstraintIndex(data.last_constraint_index)
    data.constraints[index] = NonlinearConstraint(f, set)
    return index
end

"""
    delete(data::NonlinearData, c::ConstraintIndex)
"""
function delete(data::NonlinearData, c::ConstraintIndex)
    delete!(data.constraints, c)
    return
end

"""
    add_parameter(data::NonlinearData, value::Float64)
"""
function add_parameter(data::NonlinearData, value::Float64)
    push!(data.parameters, value)
    return ParameterIndex(length(data.parameters))
end

"""
    set_parameter(data::NonlinearData, p::ParameterIndex, value::Float64)
"""
function set_parameter(data::NonlinearData, p::ParameterIndex, value::Float64)
    data.parameters[p.value] = value
    return
end

"""
    register_operator(
        data::NonlinearData,
        op::Symbol,
        nargs::Int,
        f::Function,
    )

    register_operator(
        data::NonlinearData,
        op::Symbol,
        nargs::Int,
        f::Function,
        ∇f::Function,
    )

    register_operator(
        data::NonlinearData,
        op::Symbol,
        nargs::Int,
        f::Function,
        ∇f::Function,
        ∇²f::Function,
    )
"""
function register_operator(
    data::NonlinearData,
    op::Symbol,
    nargs::Int,
    f::Function...,
)
    return register_operator(data.operators, op, nargs, f...)
end

end  # module
