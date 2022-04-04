function _create_binary_switch(ids, exprs)
    if length(exprs) <= 3
        out = Expr(:if, Expr(:call, :(==), :id, ids[1]), exprs[1])
        if length(exprs) > 1
            push!(out.args, _create_binary_switch(ids[2:end], exprs[2:end]))
        end
        return out
    else
        mid = length(exprs) >>> 1
        return Expr(
            :if,
            Expr(:call, :(<=), :id, ids[mid]),
            _create_binary_switch(ids[1:mid], exprs[1:mid]),
            _create_binary_switch(ids[mid+1:end], exprs[mid+1:end]),
        )
    end
end

let exprs = map(SYMBOLIC_UNIVARIATE_EXPRESSIONS) do arg
        return :(return $(arg[1])(x), $(arg[2]))
    end
    @eval @inline function _eval_univariate(id, x::T) where {T}
        $(_create_binary_switch(1:length(exprs), exprs))
        return error("Invalid operator_id")
    end
end

let exprs = map(SYMBOLIC_UNIVARIATE_EXPRESSIONS) do arg
        if arg === :(nothing)  # f''(x) isn't defined
            :(error("Invalid operator_id"))
        else
            :(return $(arg[3]))
        end
    end
    @eval @inline function _eval_univariate_2nd_deriv(id, x::T) where {T}
        $(_create_binary_switch(1:length(exprs), exprs))
        return error("Invalid operator_id")
    end
end

struct UnivariateOperator{F,F′,F′′}
    f::F
    f′::F′
    f′′::F′′
end

function UnivariateOperator(f::Function)
    return UnivariateOperator(f, x -> ForwardDiff.derivative(f, x))
end

function UnivariateOperator(f::Function, f′::Function)
    return UnivariateOperator(f, f′, x -> ForwardDiff.derivative(f′, x))
end

struct MultivariateOperator{F,F′}
    N::Int
    f::F
    ∇f::F′
    function MultivariateOperator{N}(f::Function, ∇f::Function) where {N}
        return new{typeof(f),typeof(∇f)}(N, f, ∇f)
    end
end

function MultivariateOperator{N}(f::Function) where {N}
    ∇f = (g, x) -> ForwardDiff.gradient!(g, f, x)
    return MultivariateOperator{N}(f, ∇f)
end

struct OperatorRegistry
    # NODE_CALL_UNIVARIATE
    univariate_operators::Vector{Symbol}
    univariate_operator_to_id::Dict{Symbol,Int}
    univariate_user_operator_start::Int
    registered_univariate_operators::Vector{UnivariateOperator}
    # NODE_CALL_MULTIVARIATE
    multivariate_operators::Vector{Symbol}
    multivariate_operator_to_id::Dict{Symbol,Int}
    multivariate_user_operator_start::Int
    registered_multivariate_operators::Vector{MultivariateOperator}
    # NODE_LOGIC
    logic_operators::Vector{Symbol}
    logic_operator_to_id::Dict{Symbol,Int}
    # NODE_COMPARISON
    comparison_operators::Vector{Symbol}
    comparison_operator_to_id::Dict{Symbol,Int}
    function OperatorRegistry()
        univariate_operators = first.(SYMBOLIC_UNIVARIATE_EXPRESSIONS)
        multivariate_operators = [:+, :-, :*, :^, :/, :ifelse]
        logic_operators = [:&&, :||]
        comparison_operators = [:<=, :(==), :>=, :<, :>]
        return new(
            # NODE_CALL_UNIVARIATE
            univariate_operators,
            Dict{Symbol,Int}(
                op => i for (i, op) in enumerate(univariate_operators)
            ),
            length(univariate_operators),
            UnivariateOperator[],
            # NODE_CALL
            multivariate_operators,
            Dict{Symbol,Int}(
                op => i for (i, op) in enumerate(multivariate_operators)
            ),
            length(multivariate_operators),
            MultivariateOperator[],
            # NODE_LOGIC
            logic_operators,
            Dict{Symbol,Int}(op => i for (i, op) in enumerate(logic_operators)),
            # NODE_COMPARISON
            comparison_operators,
            Dict{Symbol,Int}(
                op => i for (i, op) in enumerate(comparison_operators)
            ),
        )
    end
end

function register_operator(
    registry::OperatorRegistry,
    op::Symbol,
    nargs::Int,
    f::Function...,
)
    if nargs == 1
        operator = UnivariateOperator(f...)
        push!(registry.univariate_operators, op)
        push!(registry.registered_univariate_operators, operator)
        registry.univariate_operator_to_id[op] =
            length(registry.univariate_operators)
    else
        operator = MultivariateOperator{nargs}(f...)
        push!(registry.multivariate_operators, op)
        push!(registry.registered_multivariate_operators, operator)
        registry.multivariate_operator_to_id[op] =
            length(registry.multivariate_operators)
    end
    return
end

function eval_univariate_function(
    registry::OperatorRegistry,
    op::Symbol,
    x::T,
)::T where {T}
    id = registry.univariate_operator_to_id[op]
    if id <= registry.univariate_user_operator_start
        f, _ = _eval_univariate(id, x)
        return f
    else
        offset = id - registry.univariate_user_operator_start
        operator = registry.registered_univariate_operators[offset]
        return operator.f(x)
    end
end

function eval_univariate_gradient(
    registry::OperatorRegistry,
    op::Symbol,
    x::T,
) where {T}
    id = registry.univariate_operator_to_id[op]
    if id <= registry.univariate_user_operator_start
        _, f′ = _eval_univariate(id, x)
        return f′
    else
        offset = id - registry.univariate_user_operator_start
        operator = registry.registered_univariate_operators[offset]
        return operator.f′(x)
    end
end

function eval_univariate_hessian(
    registry::OperatorRegistry,
    op::Symbol,
    x::T,
) where {T}
    id = registry.univariate_operator_to_id[op]
    if id <= registry.univariate_user_operator_start
        return _eval_univariate_2nd_deriv(id, x)
    else
        offset = id - registry.univariate_user_operator_start
        operator = registry.registered_univariate_operators[offset]
        return operator.f′′(x)
    end
end

function eval_multivariate_function(
    registry::OperatorRegistry,
    op::Symbol,
    x::AbstractVector{T},
)::T where {T}
    if op == :+
        return +(x...)
    elseif op == :-
        return -(x...)
    elseif op == :*
        return *(x...)
    elseif op == :^
        @assert length(x) == 2
        return x[1]^x[2]
    elseif op == :/
        @assert length(x) == 2
        return x[1] / x[2]
    elseif op == :ifelse
        @assert length(x) == 3
        return ifelse(Bool(x[1]), x[2], x[3])
    end
    id = registry.multivariate_operator_to_id[op]
    offset = id - registry.multivariate_user_operator_start
    operator = registry.registered_multivariate_operators[offset]
    @assert length(x) == operator.N
    return operator.f(x)
end

function eval_multivariate_gradient(
    registry::OperatorRegistry,
    op::Symbol,
    g::AbstractVector{T},
    x::AbstractVector{T},
) where {T}
    if op == :+
        fill!(g, 1.0)
    elseif op == :-
        fill!(g, -1.0)
        g[1] = 1.0
    elseif op == :*
        total = *(x...)
        for i in 1:length(x)
            g[i] = total / x[i]
        end
    elseif op == :^
        @assert length(x) == 2
        g[1] = x[2] * x[1]^(x[2] - 1)
        g[2] = x[1]^x[2] * log(x[1])
    elseif op == :/
        @assert length(x) == 2
        g[1] = 1 / x[2]
        g[2] = -x[1] / x[2]^2
    elseif op == :ifelse
        @assert length(x) == 3
        g[1] = NaN
        g[2] = x[1] == 1.0
        g[3] = x[1] == 0.0
    else
        id = registry.multivariate_operator_to_id[op]
        offset = id - registry.multivariate_user_operator_start
        operator = registry.registered_multivariate_operators[offset]
        @assert length(x) == operator.N
        operator.∇f(g, x)
    end
    return
end

function eval_multivariate_hessian(
    ::OperatorRegistry,
    ::Symbol,
    ::AbstractMatrix{T},
    ::AbstractVector{T},
) where {T}
    return error("Not implemented")
end

# These are not extendable!
function eval_logic_function(op::Symbol, lhs::T, rhs::T)::Bool where {T}
    if op == :&&
        return lhs && rhs
    else
        @assert op == :||
        return lhs || rhs
    end
end

# These are not extendable!
function eval_comparison_function(op::Symbol, lhs::T, rhs::T)::Bool where {T}
    if op == :<=
        return lhs <= rhs
    elseif op == :>=
        return lhs >= rhs
    elseif op == :(==)
        return lhs == rhs
    elseif op == :<
        return lhs < rhs
    else
        @assert op == :>
        return lhs > rhs
    end
end
