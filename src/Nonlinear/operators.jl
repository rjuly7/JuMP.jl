function _create_binary_switch(ids, exprs)
    if length(exprs) <= 3
        out = Expr(:if, Expr(:call, :(==), :operator_id, ids[1]), exprs[1])
        if length(exprs) > 1
            push!(out.args, _create_binary_switch(ids[2:end], exprs[2:end]))
        end
        return out
    else
        mid = length(exprs) >>> 1
        return Expr(
            :if,
            Expr(:call, :(<=), :operator_id, ids[mid]),
            _create_binary_switch(ids[1:mid], exprs[1:mid]),
            _create_binary_switch(ids[mid+1:end], exprs[mid+1:end]),
        )
    end
end

let exprs = map(SYMBOLIC_UNIVARIATE_EXPRESSIONS) do arg
        return :(return $(arg[1])(x), $(arg[2]))
    end
    @eval @inline function _eval_univariate(operator_id, x::T) where {T}
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
    @eval @inline function _eval_univariate_2nd_deriv(
        operator_id,
        x::T,
        ::T,  # TODO: we could re-use the function evaluation
    ) where {T}
        $(_create_binary_switch(1:length(exprs), exprs))
        return error("Invalid operator_id")
    end
end

struct UnivariateOperator{F,F′,F′′}
    f::F
    f′::F′
    f′′::F′′
    function UnivariateOperator(f::Function)
        f′ = x -> ForwardDiff.derivative(f, x)
        f′′ = x -> ForwardDiff.derivative(f′, x)
        return new{typeof(f),typeof(f′),typeof(f′′)}(f, f′, f′′)
    end
    function UnivariateOperator(f::Function, f′::Function)
        f′′ = x -> ForwardDiff.derivative(f′, x)
        return new{typeof(f),typeof(f′),typeof(f′′)}(f, f′, f′′)
    end
    function UnivariateOperator(f::Function, f′::Function, f′′::Function)
        return new{typeof(f),typeof(f′),typeof(f′′)}(f, f′, f′′)
    end
end

struct MultivariateOperator end

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

function register_univariate_operator(
    registry::OperatorRegistry,
    op::Symbol,
    operator::UnivariateOperator,
)
    push!(registry.univariate_operators, operator)
    registry.univariate_operator_to_id[op] =
        length(registry.univariate_operators)
    return
end

function register_univariate_operator(
    registry::OperatorRegistry,
    op::Symbol,
    f::Function...,
)
    return register_univariate_operator(registry, op, UnivariateOperator(f...))
end

function evaluate_univariate(r::OperatorRegistry, id::Int, x::T)::T where {T}
    offset = id - r.univariate_user_operator_start
    if offset < 1
        f, _ = _eval_univariate(id, x)
        return f
    else
        operator = r.registered_univariate_operators[offset]
        return operator.f(x)
    end
end
