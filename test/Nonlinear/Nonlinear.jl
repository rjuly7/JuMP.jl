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
    @test expr.nodes == [
        Nonlinear.Node(Nonlinear.NODE_MOI_VARIABLE, 1, -1),
    ]
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
    @test expr.nodes == [
        Nonlinear.Node(Nonlinear.NODE_PARAMETER, 1, -1),
    ]
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

end

TestNonlinear.runtests()
