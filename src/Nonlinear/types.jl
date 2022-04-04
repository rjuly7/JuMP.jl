@enum(
    NodeType,
    # Index into multivariate operators
    NODE_CALL_MULTIVARIATE,
    # Index into univariate operators
    NODE_CALL_UNIVARIATE,
    # Index into logic operators
    NODE_LOGIC,
    # Index into comparison operators
    NODE_COMPARISON,
    # Index is the value of `MOI.VariableIndex`. This is from the original
    # model, and is not consecutive.
    NODE_MOI_VARIABLE,
    # Index of the internal, consecutive, and ordered `MOI.VariableIndex`.
    NODE_VARIABLE,
    # Index is into the list of constants
    NODE_VALUE,
    # Index is into the list of parameters
    NODE_PARAMETER,
    # Index is into the list of subexpressions
    NODE_SUBEXPRESSION,
)

struct Node
    type::NodeType
    index::Int
    parent::Int
end

struct NonlinearExpression
    nodes::Vector{Node}
    values::Vector{Float64}
    NonlinearExpression() = new(Node[], Float64[])
end

struct NonlinearConstraint
    expression::NonlinearExpression
    set::Union{
        MOI.LessThan{Float64},
        MOI.GreaterThan{Float64},
        MOI.EqualTo{Float64},
        MOI.Interval{Float64},
    }
end

struct ParameterIndex
    value::Int
end

struct ExpressionIndex
    value::Int
end

struct ConstraintIndex
    value::Int
end

mutable struct NonlinearData
    objective::Union{Nothing,NonlinearExpression}
    expressions::Vector{NonlinearExpression}
    constraints::Dict{ConstraintIndex,NonlinearConstraint}
    parameters::Vector{Float64}
    operators::OperatorRegistry
    last_constraint_index::Int64
    function NonlinearData()
        return new(
            nothing,
            NonlinearExpression[],
            Dict{ConstraintIndex,NonlinearConstraint}(),
            Float64[],
            OperatorRegistry(),
            0,
        )
    end
end

function Base.getindex(data::NonlinearData, index::ParameterIndex)
    return data.parameters[index.value]
end

function Base.getindex(data::NonlinearData, index::ExpressionIndex)
    return data.expressions[index.value]
end

function Base.getindex(data::NonlinearData, index::ConstraintIndex)
    return data.constraints[index.value]
end
