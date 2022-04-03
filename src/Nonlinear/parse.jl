#  Copyright 2017, Iain Dunning, Joey Huchette, Miles Lubin, and contributors
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at https://mozilla.org/MPL/2.0/.

function parse_expression(data::NonlinearData, input)
    expr = NonlinearExpression()
    _parse_expression(data, expr, input, -1)
    return expr
end

function _parse_expression(
    data::NonlinearData,
    expr::NonlinearExpression,
    x::Expr,
    parent_index::Union{Symbol,Int},
)
    if isexpr(x, :call)
        if length(x.args) == 2 && !isexpr(x.args[2], :...)
            return _parse_univariate_expression(data, expr, x, parent_index)
        else
            return _parse_multivariate_expression(data, expr, x, parent_index)
        end
    elseif isexpr(x, :comparison)
        return _parse_comparison_expression(data, expr, x, parent_index)
    elseif isexpr(x, :&&) || isexpr(x, :||)
        return _parse_logic_expression(data, expr, x, parent_index)
    end
    return _parse_expression(data, expr, x, parent_index)
end

function _parse_univariate_expression(
    data::NonlinearData,
    expr::NonlinearExpression,
    x::Expr,
    parent_index::Union{Symbol,Int},
)
    @assert isexpr(x, :call, 2)
    id = get(data.operators.univariate_operator_to_id, x.args[1], nothing)
    if id === nothing
        error()  # TODO: might be a multivariate operator with one argument.
    end
    push!(expr.nodes, Node(NODE_CALL_UNIVARIATE, id, parent_index))
    _parse_expression(data, expr, x.args[2], length(expr.nodes))
    return
end

function _parse_multivariate_expression(
    data::NonlinearData,
    expr::NonlinearExpression,
    x::Expr,
    parent_index::Union{Symbol,Int},
)
    @assert isexpr(x, :call)
    id = get(data.operators.multivariate_operator_to_id, x.args[1], nothing)
    if id === nothing
        error()
    end
    push!(expr.nodes, Node(NODE_CALL_MULTIVARIATE, id, parent_index))
    parent_var = length(expr.nodes)
    for i in 2:length(x.args)
        _parse_expression(data, expr, x.args[i], parent_var)
    end
    return
end

function _parse_comparison_expression(
    data::NonlinearData,
    expr::NonlinearExpression,
    x::Expr,
    parent_index::Union{Symbol,Int},
)
    for k in 2:2:length(x.args)-1
        @assert x.args[k] == x.args[2] # don't handle a <= b >= c
    end
    operator_id = data.operators.comparison_operator_to_id[x.args[2]]
    push!(expr.nodes, Node(NODE_COMPARISON, operator_id, parent_index))
    parent_var = length(expr.nodes)
    for i in 1:2:length(x.args)
        _parse_expression(data, expr, x.args[i], parent_var)
    end
    return
end

function _parse_logic_expression(
    data::NonlinearData,
    expr::NonlinearExpression,
    x::Expr,
    parent_index::Union{Symbol,Int},
)
    id = data.operators.logic_operator_to_id[x.head]
    push!(expr.nodes, Node(NODE_LOGIC, id, parent_index))
    parent_var = length(expr.nodes)
    _parse_expression(data, expr, x.args[1], parent_var)
    _parse_expression(data, expr, x.args[2], parent_var)
    return
end

function _parse_expression(
    ::NonlinearData,
    expr::NonlinearExpression,
    x::MOI.VariableIndex,
    parent_index::Int,
)
    push!(expr.nodes, Node(NODE_MOI_VARIABLE, x.value, parent_index))
    return
end

function _parse_expression(
    ::NonlinearData,
    expr::NonlinearExpression,
    x::Real,
    parent_index::Int,
)
    push!(expr.values, convert(Float64, x)::Float64)
    push!(expr.nodes, Node(NODE_VALUE, length(expr.values), parent_index))
    return
end

function _parse_expression(
    ::NonlinearData,
    expr::NonlinearExpression,
    x::ParameterIndex,
    parent_index::Int,
)
    push!(expr.nodes, Node(NODE_PARAMETER, x.value, parent_index))
    return
end

function _parse_expression(
    ::NonlinearData,
    expr::NonlinearExpression,
    x::ExpressionIndex,
    parent_index::Int,
)
    push!(expr.nodes, Node(NODE_SUBEXPRESSION, x.value, parent_index))
    return
end
