using Printf
using FiniteDiff
using ForwardDiff
using KernelAbstractions
using LinearAlgebra
using Random
using SparseArrays
using Test
using TimerOutputs
using ExaPF
import ExaPF: PowerSystem, AutoDiff

const PS = PowerSystem

# Warning: currently works only on CPU, as depends on
# explicit evaluation of Hessian, using MATPOWER expressions
@testset "Compute reduced Hessian on CPU" begin
    @testset "Case $case" for case in ["case9.m", "case30.m"]
        ##################################################
        # Initialization
        ##################################################
        datafile = joinpath(dirname(@__FILE__), "..", "..", "data", case)
        tolerance = 1e-8
        pf = PS.PowerNetwork(datafile)
        polar = PolarForm(pf, CPU())
        pv = pf.pv ; npv = length(pv)
        pq = pf.pq ; npq = length(pq)
        ref = pf.ref ; nref = length(ref)
        nbus = pf.nbus
        ngen = get(polar, PS.NumberOfGenerators())

        pv2gen = polar.indexing.index_pv_to_gen
        ref2gen = polar.indexing.index_ref_to_gen
        gen2bus = polar.indexing.index_generators
        cache = ExaPF.get(polar, ExaPF.PhysicalState())
        ExaPF.init_buffer!(polar, cache)

        xk = ExaPF.initial(polar, State())
        u = ExaPF.initial(polar, Control())
        nx = length(xk) ; nu = length(u)

        jx, ju, ∂obj = ExaPF.init_autodiff_factory(polar, cache)

        ##################################################
        # Step 1: computation of first-order adjoint
        ##################################################
        conv = powerflow(polar, jx, cache, NewtonRaphson())
        ExaPF.update!(polar, PS.Generators(), PS.ActivePower(), cache)
        @test conv.has_converged
        # No need to recompute ∇gₓ
        ∇gₓ = jx.J
        ∇gᵤ = AutoDiff.jacobian!(polar, ju, cache)
        # test jacobian wrt x
        ∇gᵥ = AutoDiff.jacobian!(polar, jx, cache)
        @test isequal(∇gₓ, ∇gᵥ)

        # Fetch values found by Newton-Raphson algorithm
        vm = cache.vmag
        va = cache.vang
        pg = cache.pg
        # State & Control
        x = [va[pv] ; va[pq] ; vm[pq]]
        u = [vm[ref]; vm[pv]]
        # Test with Matpower's Jacobian
        V = vm .* exp.(im * va)
        Ybus = pf.Ybus
        Jₓ = ExaPF.matpower_jacobian(polar, State(), ExaPF.power_balance, V)
        @test isapprox(∇gₓ, Jₓ )
        # Hessian vector product
        ExaPF.∂cost(polar, ∂obj, cache)
        ∇fₓ = ∂obj.∇fₓ
        ∇fᵤ = ∂obj.∇fᵤ
        λ  = -(∇gₓ') \ ∇fₓ
        grad_adjoint = ∇fᵤ + ∇gᵤ' * λ

        ##################################################
        # Step 2: computation of Hessian of powerflow g
        ##################################################
        ## w.r.t. xx
        function jac_diff(x)
            vm_ = copy(vm)
            va_ = copy(va)
            va_[pv] = x[1:npv]
            va_[pq] = x[npv+1:npv+npq]
            vm_[pq] = x[npv+npq+1:end]
            V = vm_ .* exp.(im * va_)
            Jx = ExaPF.matpower_jacobian(polar, State(), ExaPF.power_balance, V)
            return Jx' * λ
        end

        # Evaluate Hessian-vector product (full ∇²gₓₓ is a 3rd dimension tensor)
        ∇²gλ = ExaPF.residual_hessian(polar, cache, λ)
        H_fd = FiniteDiff.finite_difference_jacobian(jac_diff, x)
        @test isapprox(∇²gλ.xx, H_fd, rtol=1e-6)
        ybus_re, ybus_im = ExaPF.Spmat{Vector{Int}, Vector{Float64}}(Ybus)
        pbus = real(pf.sbus)
        qbus = imag(pf.sbus)
        F = zeros(Float64, npv + 2*npq)
        nx = size(∇²gλ.xx, 1)
        nu = size(∇²gλ.uu, 1)

        # Hessian-vector product using forward over adjoint AD
        HessianAD = AutoDiff.Hessian(polar, ExaPF.power_balance)

        tgt = rand(nx + nu)
        # set tangents only for x direction
        tgt[nx+1:end] .= 0.0
        projxx = ExaPF.AutoDiff.tgt_adj_residual_hessian!(
            HessianAD, ExaPF.adj_residual_polar!, λ, tgt, vm, va,
            ybus_re, ybus_im, pbus, qbus, pf.pv, pf.pq, pf.ref, nbus)
        @test isapprox(projxx[1:nx], ∇²gλ.xx * tgt[1:nx])
        projp = AutoDiff.adj_hessian_prod!(polar, HessianAD, cache, λ, tgt)
        @test isapprox(projp[1:nx], ∇²gλ.xx * tgt[1:nx])


        tgt = rand(nx + nu)
        # set tangents only for u direction
        tgt[1:nx] .= 0.0
        projuu = AutoDiff.tgt_adj_residual_hessian!(
            HessianAD, ExaPF.adj_residual_polar!, λ, tgt, vm, va,
            ybus_re, ybus_im, pbus, qbus, pf.pv, pf.pq, pf.ref, nbus)
        @test isapprox(projuu[nx+1:end], ∇²gλ.uu * tgt[nx+1:end])
        projp = AutoDiff.adj_hessian_prod!(polar, HessianAD, cache, λ, tgt)
        @test isapprox(projp[nx+1:end], ∇²gλ.uu * tgt[nx+1:end])

        # check cross terms ux
        tgt = rand(nx + nu)
        tgt .= 1.0
        # Build full Hessian
        H = [
            ∇²gλ.xx ∇²gλ.xu';
            ∇²gλ.xu ∇²gλ.uu
        ]
        projxu = ExaPF.AutoDiff.tgt_adj_residual_hessian!(
            HessianAD, ExaPF.adj_residual_polar!, λ, tgt, vm, va,
            ybus_re, ybus_im, pbus, qbus, pf.pv, pf.pq, pf.ref, nbus)
        @test isapprox(projxu, H * tgt)
        projp = AutoDiff.adj_hessian_prod!(polar, HessianAD, cache, λ, tgt)
        @test isapprox(projp, H * tgt)

        ## w.r.t. uu
        function jac_u_diff(u)
            vm_ = copy(vm)
            va_ = copy(va)
            vm_[ref] = u[1:nref]
            vm_[pv] = u[nref+1:end]
            V = vm_ .* exp.(im * va_)
            Ju = ExaPF.matpower_jacobian(polar, Control(), ExaPF.power_balance, V)
            return Ju' * λ
        end

        Hᵤᵤ_fd = FiniteDiff.finite_difference_jacobian(jac_u_diff, u)

        if !iszero(∇²gλ.uu[1:nref+npv, 1:nref+npv])
            @test isapprox(∇²gλ.uu[1:nref+npv, 1:nref+npv], Hᵤᵤ_fd[1:nref+npv, :], rtol=1e-6)
        end

        ## w.r.t. xu
        function jac_xu_diff(x)
            vm_ = copy(vm)
            va_ = copy(va)
            va_[pv] = x[1:npv]
            va_[pq] = x[npv+1:npv+npq]
            vm_[pq] = x[npv+npq+1:end]
            V = vm_ .* exp.(im * va_)
            Ju = ExaPF.matpower_jacobian(polar, Control(), ExaPF.power_balance, V)[:, 1:nref+npv]
            return Ju' * λ
        end

        Hₓᵤ_fd = FiniteDiff.finite_difference_jacobian(jac_xu_diff, x)
        @test isapprox(∇²gλ.xu[1:nref+npv, :], Hₓᵤ_fd, rtol=1e-6)

        ##################################################
        # Step 3: computation of Hessian of objective f
        ##################################################

        # Finite difference routine
        function cost_x(z)
            x_ = z[1:nx]
            u_ = z[1+nx:end]
            # Transfer control
            ExaPF.transfer!(polar, cache, u_)
            # Transfer state (manually)
            cache.vang[pv] .= x_[1:npv]
            cache.vang[pq] .= x_[npv+1:npv+npq]
            cache.vmag[pq] .= x_[npv+npq+1:end]
            ExaPF.update!(polar, PS.Generators(), PS.ActivePower(), cache)
            return ExaPF.cost_production(polar, cache.pg)
        end

        # Update variables
        x = [va[pv] ; va[pq] ; vm[pq]]
        u = [vm[ref]; vm[pv]; pg[pv2gen]]

        H_ffd = FiniteDiff.finite_difference_hessian(cost_x, [x; u])

        # Hessians of objective
        ∇²f = ExaPF.hessian_cost(polar, cache)
        ∇²fₓₓ = ∇²f.xx
        ∇²fᵤᵤ = ∇²f.uu
        ∇²fₓᵤ = ∇²f.xu
        @test isapprox(∇²fₓₓ, H_ffd[1:nx, 1:nx], rtol=1e-6)
        index_u = nx+1:nx+nref+2*npv
        @test isapprox(∇²fₓᵤ, H_ffd[index_u, 1:nx], rtol=1e-6)
        @test isapprox(∇²fᵤᵤ, H_ffd[index_u, index_u], rtol=1e-6)

        ∇gaₓ = ∇²fₓₓ + ∇²gλ.xx

        # Computation of the reduced Hessian
        function reduced_hess(w)
            # Second-order adjoint
            z = -(∇gₓ ) \ (∇gᵤ * w)
            ψ = -(∇gₓ') \ (∇²fₓᵤ' * w + ∇²gλ.xu' * w +  ∇gaₓ * z)
            Hw = ∇²fᵤᵤ * w +  ∇²gλ.uu * w + ∇gᵤ' * ψ  + ∇²fₓᵤ * z + ∇²gλ.xu * z
            return Hw
        end

        w = zeros(nu)
        H = zeros(nu, nu)
        for i in 1:nu
            fill!(w, 0)
            w[i] = 1.0
            H[:, i] .= reduced_hess(w)
        end

        ##################################################
        # Step 4: include constraints in Hessian
        ##################################################
        # h1 (state)      : xl <= x <= xu
        # h2 (by-product) : yl <= y <= yu
        # Test sequential evaluation of Hessian

        μ = rand(ngen)
        ∂₂Q = ExaPF.hessian(polar, ExaPF.reactive_power_constraints, cache, μ)
        function jac_x(z)
            x_ = z[1:nx]
            u_ = z[1+nx:end]
            # Transfer control
            ExaPF.transfer!(polar, cache, u_)
            # Transfer state (manually)
            cache.vang[pv] .= x_[1:npv]
            cache.vang[pq] .= x_[npv+1:npv+npq]
            cache.vmag[pq] .= x_[npv+npq+1:end]
            ExaPF.update!(polar, PS.Generators(), PS.ActivePower(), cache)
            J = ExaPF.jacobian(polar, ExaPF.reactive_power_constraints, cache)
            return [J.x J.u]' * μ
        end

        H_fd = FiniteDiff.finite_difference_jacobian(jac_x, [x; u])
        @test isapprox(∂₂Q.uu, H_fd[nx+1:end, nx+1:end], rtol=1e-6)
        @test isapprox(∂₂Q.xx, H_fd[1:nx, 1:nx], rtol=1e-6)
        @test isapprox(∂₂Q.xu, H_fd[nx+1:end, 1:nx], rtol=1e-6)

        # Test with AutoDiff.Hessian
        hess_reactive = AutoDiff.Hessian(polar, ExaPF.reactive_power_constraints)

        # XX
        tgt = rand(nx + nu)
        tgt[nx+1:end] .= 0.0
        projp = AutoDiff.adj_hessian_prod!(polar, hess_reactive, cache, μ, tgt)
        @test isapprox(projp[1:nx], ∂₂Q.xx * tgt[1:nx])

        # UU
        tgt = rand(nx + nu)
        tgt[1:nx] .= 0.0
        projp = AutoDiff.adj_hessian_prod!(polar, hess_reactive, cache, μ, tgt)
        @test isapprox(projp[1+nx:end], ∂₂Q.uu * tgt[1+nx:end])

        # XU
        tgt = rand(nx + nu)
        projp = AutoDiff.adj_hessian_prod!(polar, hess_reactive, cache, μ, tgt)
        H = [
            ∂₂Q.xx ∂₂Q.xu' ;
            ∂₂Q.xu ∂₂Q.uu
        ]
        @test isapprox(projp, H * tgt)

        # Hessian w.r.t. Line-flow
        hess_lineflow = AutoDiff.Hessian(polar, ExaPF.flow_constraints)
        ncons = ExaPF.size_constraint(polar, ExaPF.flow_constraints)
        μ = rand(ncons)
        function flow_jac_x(z)
            x_ = z[1:nx]
            u_ = z[1+nx:end]
            # Transfer control
            ExaPF.transfer!(polar, cache, u_)
            # Transfer state (manually)
            cache.vang[pv] .= x_[1:npv]
            cache.vang[pq] .= x_[npv+1:npv+npq]
            cache.vmag[pq] .= x_[npv+npq+1:end]
            V = cache.vmag .* exp.(im .* cache.vang)
            Jx = ExaPF.matpower_jacobian(polar, State(), ExaPF.flow_constraints, V)
            Ju = ExaPF.matpower_jacobian(polar, Control(), ExaPF.flow_constraints, V)
            return [Jx Ju]' * μ
        end

        H_fd = FiniteDiff.finite_difference_jacobian(flow_jac_x, [x; u])
        tgt = rand(nx + nu)
        projp = AutoDiff.adj_hessian_prod!(polar, hess_lineflow, cache, μ, tgt)
        @test isapprox(projp, H_fd * tgt, rtol=1e-5)

        # Hessian w.r.t. Active-power generation
        hess_pg = AutoDiff.Hessian(polar, ExaPF.active_power_constraints)
        μ = rand(1)
        projp = AutoDiff.adj_hessian_prod!(polar, hess_pg, cache, μ, tgt)
        ∂₂P = ExaPF.hessian(polar, ExaPF.active_power_constraints, cache, μ)
        H = [
            ∂₂P.xx ∂₂P.xu' ;
            ∂₂P.xu ∂₂P.uu
        ]
        @test isapprox(projp, (H * tgt))
    end
end

