using CUDA
using CUDA.CUSPARSE
using Revise
using Printf
using FiniteDiff
using ForwardDiff
using BenchmarkTools
using KernelAbstractions
using LinearAlgebra
using Random
using SparseArrays
using Test
using TimerOutputs
using ExaPF
import ExaPF: PowerSystem, AD, Precondition, Iterative

const PS = PowerSystem

@testset "Polar formulation" begin
    datafile = joinpath(dirname(@__FILE__), "data", "case9.m")
    tolerance = 1e-8
    pf = PS.PowerNetwork(datafile, 1)

    if has_cuda_gpu()
        ITERATORS = zip([CPU(), CUDADevice()], [Array, CuArray])
    else
        ITERATORS = zip([CPU()], [Array])
    end

    @testset "Initiate polar formulation on device $device" for (device, M) in ITERATORS
        polar = PolarForm(pf, device)

        b = bounds(polar, State())
        b = bounds(polar, Control())

        # Test getters
        nᵤ = ExaPF.get(polar, NumberOfControl())
        nₓ = ExaPF.get(polar, NumberOfState())

        # Get initial position
        x0 = ExaPF.initial(polar, State())
        u0 = ExaPF.initial(polar, Control())
        p = ExaPF.initial(polar, Parameters())

        @test length(u0) == nᵤ
        @test length(x0) == nₓ
        for v in [x0, u0, p]
            @test isa(v, M)
        end

        cache  = ExaPF.NetworkState(polar)
        @testset "NetworkState cache" begin
            @test isa(cache.vmag, M)
            ExaPF.transfer!(polar, cache, x0, u0, p)
            Vm, Va, pbus, qbus = ExaPF.get_network_state(polar, x0, u0, p)
            @test isequal(Vm, cache.vmag)
            @test isequal(Va, cache.vang)
            @test isequal(pbus, cache.pinj)
            @test isequal(qbus, cache.qinj)
        end
        @testset "Polar model API" begin
            xₖ = copy(x0)
            # Init AD factory
            jx, ju = ExaPF.init_ad_factory(polar, cache)

            # Test powerflow with x, u, p signature
            conv = @time powerflow(polar, jx, cache, verbose_level=0, tol=tolerance)
            ExaPF.get!(polar, State(), xₖ, cache)

            # Bounds on state and control
            u_min, u_max = ExaPF.bounds(polar, Control())
            x_min, x_max = ExaPF.bounds(polar, State())
            @test isequal(u_min, [0.9, 0.1, 0.1, 0.9, 0.9])
            @test isequal(u_max, [1.1, 3.0, 2.7, 1.1, 1.1])
            @test isequal(x_min, [-Inf, -Inf, -Inf, -Inf, -Inf, -Inf, -Inf, -Inf, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9])
            @test isequal(x_max, [Inf, Inf, Inf, Inf, Inf, Inf, Inf, Inf, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1])
            # Test callbacks
            ## Power Balance
            ExaPF.power_balance!(polar, cache)
            g = cache.balance
            @test isa(g, M)
            # As we run powerflow before, the balance should be below tolerance
            @test norm(g, Inf) < tolerance
            ## Cost Production
            c = @time ExaPF.cost_production(polar, xₖ, u0, p)
            @test isa(c, Real)
            ## Inequality constraint
            for cons in [ExaPF.state_constraint, ExaPF.power_constraints]
                m = ExaPF.size_constraint(polar, cons)
                @test isa(m, Int)
                g = M{Float64, 1}(undef, m) # TODO: this signature is not great
                fill!(g, 0)
                cons(polar, g, xₖ, u0, p)

                g_min, g_max = ExaPF.bounds(polar, cons)
                @test length(g_min) == m
                @test length(g_max) == m
                # Are we on the correct device?
                @test isa(g_min, M)
                @test isa(g_max, M)
                # Test constraints are consistent
                @test isless(g_min, g_max)
            end
        end

    end
end
