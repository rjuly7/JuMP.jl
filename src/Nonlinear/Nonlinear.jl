module Nonlinear

import Base.Meta: isexpr
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
    add_constraint(
        data::NonlinearData,
        expr::Expr,
        set::Union{
            MOI.LessThan{Float64},
            MOI.GreaterThan{Float64},
            MOI.EqualTo{Float64},
            MOI.Interval{Float64},
        },
    )
"""
function add_constraint(
    data::NonlinearData,
    expr::Expr,
    set::Union{
        MOI.LessThan{Float64},
        MOI.GreaterThan{Float64},
        MOI.EqualTo{Float64},
        MOI.Interval{Float64},
    },
)
    f = parse_expression(data, expr)
    data.last_constraint_index += 1
    data.constraints[data.last_constraint_index] = NonlinearConstraint(f, set)
    return ConstraintIndex(data.last_constraint_index)
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
    if nargs == 1
        _register_univariate_operator(data.operators, op, f...)
    else
        _register_multivariate_operator(data.operators, op, nargs, f...)
    end
    return
end

end  # module
