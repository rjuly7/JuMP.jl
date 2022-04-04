module TestNonlinear

using Test
import JuMP: MOI
import JuMP: Nonlinear

function runtests()
    for name in names(@__MODULE__; all = true)
        if startswith("$(name)", "test_")
            @testset "$(name)" begin
                getfield(@__MODULE__, name)()
            end
        end
    end
    return
end

function test_parse_sin()
    data = Nonlinear.NonlinearData()
    x = MOI.VariableIndex(1)
    expr = Nonlinear.parse_expression(data, :(sin($x)))
    # @test evaluate(data, ex, Dict(x => 1.5)) == sin(1.5)
    return
end

function test_moi_variable_parse()
    data = Nonlinear.NonlinearData()
    x = MOI.VariableIndex(1)
    expr = Nonlinear.parse_expression(data, :($x))
    @test expr.nodes == [Nonlinear.Node(Nonlinear.NODE_MOI_VARIABLE, 1, -1)]
    @test isempty(expr.values)
    return
end

function test_expression_parse()
    data = Nonlinear.NonlinearData()
    x = MOI.VariableIndex(1)
    ex = Nonlinear.add_expression(data, :(sin($x)^2))
    @test data[ex] isa Nonlinear.NonlinearExpression
    return
end

function test_parameter_parse()
    data = Nonlinear.NonlinearData()
    p = Nonlinear.add_parameter(data, 1.2)
    expr = Nonlinear.parse_expression(data, :($p))
    @test expr.nodes == [Nonlinear.Node(Nonlinear.NODE_PARAMETER, 1, -1)]
    @test isempty(expr.values)
    @test data.parameters == [1.2]
    return
end

function test_parameter_set()
    data = Nonlinear.NonlinearData()
    p = Nonlinear.add_parameter(data, 1.2)
    @test data.parameters == [1.2]
    @test data[p] == 1.2
    Nonlinear.set_parameter(data, p, 2.1)
    @test data.parameters == [2.1]
    @test data[p] == 2.1
    return
end

function test_eval_univariate_function()
    r = Nonlinear.OperatorRegistry()
    @test Nonlinear.eval_univariate_function(r, :+, 1.0) == 1.0
    @test Nonlinear.eval_univariate_function(r, :-, 1.0) == -1.0
    @test Nonlinear.eval_univariate_function(r, :abs, -1.1) == 1.1
    @test Nonlinear.eval_univariate_function(r, :abs, 1.1) == 1.1
    return
end

function test_eval_univariate_gradient()
    r = Nonlinear.OperatorRegistry()
    @test Nonlinear.eval_univariate_gradient(r, :+, 1.2) == 1.0
    @test Nonlinear.eval_univariate_gradient(r, :-, 1.2) == -1.0
    @test Nonlinear.eval_univariate_gradient(r, :abs, -1.1) == -1.0
    @test Nonlinear.eval_univariate_gradient(r, :abs, 1.1) == 1.0
    return
end

function test_eval_univariate_hessian()
    r = Nonlinear.OperatorRegistry()
    @test Nonlinear.eval_univariate_hessian(r, :+, 1.2) == 0.0
    @test Nonlinear.eval_univariate_hessian(r, :-, 1.2) == 0.0
    @test Nonlinear.eval_univariate_hessian(r, :abs, -1.1) == 0.0
    @test Nonlinear.eval_univariate_hessian(r, :abs, 1.1) == 0.0
    return
end

function test_eval_univariate_function_registered()
    r = Nonlinear.OperatorRegistry()
    f(x) = sin(x)^2
    f′(x) = 2 * sin(x) * cos(x)
    f′′(x) = 2 * (cos(x)^2 - sin(x)^2)
    Nonlinear.register_operator(r, :f, 1, f)
    x = 1.2
    @test Nonlinear.eval_univariate_function(r, :f, x) ≈ f(x)
    @test Nonlinear.eval_univariate_gradient(r, :f, x) ≈ f′(x)
    @test Nonlinear.eval_univariate_hessian(r, :f, x) ≈ f′′(x)
    return
end

function test_eval_multivariate_function()
    r = Nonlinear.OperatorRegistry()
    x = [1.1, 2.2]
    @test Nonlinear.eval_multivariate_function(r, :+, x) ≈ 3.3
    @test Nonlinear.eval_multivariate_function(r, :-, x) ≈ -1.1
    @test Nonlinear.eval_multivariate_function(r, :*, x) ≈ 1.1 * 2.2
    @test Nonlinear.eval_multivariate_function(r, :^, x) ≈ 1.1^2.2
    @test Nonlinear.eval_multivariate_function(r, :/, x) ≈ 1.1 / 2.2
    @test Nonlinear.eval_multivariate_function(r, :ifelse, [1; x]) == 1.1
    @test Nonlinear.eval_multivariate_function(r, :ifelse, [0; x]) == 2.2
    return
end

function test_eval_multivariate_gradient()
    r = Nonlinear.OperatorRegistry()
    x = [1.1, 2.2]
    g = zeros(2)
    Nonlinear.eval_multivariate_gradient(r, :+, g, x)
    @test g == [1.0, 1.0]
    Nonlinear.eval_multivariate_gradient(r, :-, g, x)
    @test g == [1.0, -1.0]
    Nonlinear.eval_multivariate_gradient(r, :*, g, x)
    @test g ≈ [2.2, 1.1]
    Nonlinear.eval_multivariate_gradient(r, :^, g, x)
    @test g ≈ [2.2 * 1.1^1.2, 1.1^2.2 * log(1.1)]
    Nonlinear.eval_multivariate_gradient(r, :/, g, x)
    @test g ≈ [1 / 2.2, -1.1 / 2.2^2]
    g = zeros(3)
    Nonlinear.eval_multivariate_gradient(r, :ifelse, g, [1; x])
    @test isnan(g[1])
    @test g[2:3] ≈ [1.0, 0.0]
    Nonlinear.eval_multivariate_gradient(r, :ifelse, g, [0; x])
    @test isnan(g[1])
    @test g[2:3] ≈ [0.0, 1.0]
    return
end

function test_eval_multivariate_hessian()
    r = Nonlinear.OperatorRegistry()
    x = [1.1, 2.2]
    H = zeros(2, 2)
    # TODO(odow): implement
    @test_throws(
        ErrorException,
        Nonlinear.eval_multivariate_hessian(r, :+, H, x),
    )
    return
end

function test_eval_multivariate_function_registered()
    r = Nonlinear.OperatorRegistry()
    f(x) = x[1]^2 + x[1] * x[2] + x[2]^2
    # f′(x) = 2 * sin(x) * cos(x)
    # f′′(x) = 2 * (cos(x)^2 - sin(x)^2)
    Nonlinear.register_operator(r, :f, 2, f)
    x = [1.1, 2.2]
    @test Nonlinear.eval_multivariate_function(r, :f, x) ≈ f(x)
    g = zeros(2)
    Nonlinear.eval_multivariate_gradient(r, :f, g, x)
    @test g ≈ [2 * x[1] + x[2], x[1] + 2 * x[2]]
    H = zeros(2, 2)
    # TODO(odow): implement
    @test_throws(
        ErrorException,
        Nonlinear.eval_multivariate_hessian(r, :f, H, x),
    )
    return
end

function test_eval_logic_function()
    for lhs in (true, false), rhs in (true, false)
        @test Nonlinear.eval_logic_function(:&&, lhs, rhs) == (lhs && rhs)
        @test Nonlinear.eval_logic_function(:||, lhs, rhs) == (lhs || rhs)
        @test_throws(
            AssertionError,
            Nonlinear.eval_logic_function(:⊻, lhs, rhs),
        )
    end
    return
end

function test_eval_comprison_function()
    for lhs in (true, false), rhs in (true, false)
        @test Nonlinear.eval_comparison_function(:<=, lhs, rhs) == (lhs <= rhs)
        @test Nonlinear.eval_comparison_function(:>=, lhs, rhs) == (lhs >= rhs)
        @test Nonlinear.eval_comparison_function(:(==), lhs, rhs) ==
              (lhs == rhs)
        @test Nonlinear.eval_comparison_function(:<, lhs, rhs) == (lhs < rhs)
        @test Nonlinear.eval_comparison_function(:>, lhs, rhs) == (lhs > rhs)
        @test_throws(
            AssertionError,
            Nonlinear.eval_comparison_function(:⊻, lhs, rhs),
        )
    end
    return
end

end

TestNonlinear.runtests()
