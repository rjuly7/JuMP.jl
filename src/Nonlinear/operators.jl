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

struct MultivariateOperator{N,F,F′,F′′}
    f::F
    ∇f::F′
    ∇²f::F′
    function MultivariateOperator{N}(f::Function, ∇f::Function) where {N}
        return new{N,typeof(f),typeof(∇f),Nothing}(f, ∇f, nothing)
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
    # NODE_CALL
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

function _register_univariate_operator(
    registry::OperatorRegistry,
    op::Symbol,
    f::Function...,
)
    operator = UnivariateOperator(f...)
    push!(registry.univariate_operators, operator)
    push!(registry.registered_univariate_operators, operator)
    registry.univariate_operator_to_id[op] =
        length(registry.univariate_operators)
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

function _register_multivariate_operator(
    registry::OperatorRegistry,
    op::Symbol,
    nargs::Int,
    f::Function...,
)
    operator = MultivariateOperator{nargs}(f...)
    push!(registry.multivariate_operators, operator)
    registry.multivariate_operator_to_id[op] =
        length(registry.multivariate_operators)
    return
end

function eval_multivariate_function(
    operator::MultivariateOperator{N},
    x::AbstractVector{T},
) where {T,N}
    @assert length(x) == N
    return operator.f(x)::T
end

function eval_multivariate_gradient(
    operator::MultivariateOperator{N},
    g::AbstractVector{T},
    x::AbstractVector{T},
) where {T,N}
    @assert length(x) == N
    operator.∇f(g, x)
    return
end

function eval_multivariate_hessian(
    operator::MultivariateOperator{N},
    H::AbstractMatrix{T},
    x::AbstractVector{T},
) where {T,N}
    @assert length(x) == N
    operator.∇²f(H, x)
    return
end
