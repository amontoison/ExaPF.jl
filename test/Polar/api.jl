function test_polar_stack(polar, device, M)
    ngen = get(polar, PS.NumberOfGenerators())
    nbus = get(polar, PS.NumberOfBuses())
    nlines = get(polar, PS.NumberOfLines())

    mapx = ExaPF.mapping(polar, State()) |> M
    mapu = ExaPF.mapping(polar, Control()) |> M
    u0 = rand(length(mapu)) |> M

    stack = ExaPF.NetworkStack(polar)

    # Test display
    println(devnull, stack)
    # By defaut, values are equal to 0
    @test isa(stack.input, M)
    @test isa(stack.vmag, M)

    # Copty control u0 inside cache
    copyto!(stack, mapu, u0)
    @test stack.input[mapu] == u0

    # Get state
    nx = length(mapx)
    x = similar(u0, nx)
    copyto!(x, stack, mapx)
    @test stack.input[mapx] == x

    # Test that all attributes have valid length
    @test length(stack.vang) == length(stack.vmag) == nbus
    @test length(stack.pgen) == ngen
    @test length(stack.input) == ngen + 2 * nbus
    @test length(stack.ψ) == 2 * nlines + nbus

    # Bounds
    b♭, b♯ = ExaPF.bounds(polar, stack)
    @test myisless(b♭, b♯)

    empty!(stack)
    @test iszero(stack.input)

    return nothing
end

function test_block_stack(polar, device, M)
    nblocks = 2
    blk_polar = ExaPF.BlockPolarForm(polar, nblocks)
    ngen = get(polar, PS.NumberOfGenerators())
    nbus = get(polar, PS.NumberOfBuses())
    nlines = get(polar, PS.NumberOfLines())
    nu = ExaPF.number(polar, Control())
    nx = ExaPF.number(polar, State())

    mapx = ExaPF.mapping(polar, State(), nblocks) |> M
    @test length(mapx) == nx * nblocks
    mapu = ExaPF.mapping(polar, Control(), nblocks) |> M
    mapxu = ExaPF.mapping(polar, AllVariables(), nblocks) |> M
    u0 = rand(nu) |> M

    x0 = rand(nx * nblocks) |> M

    stack = ExaPF.NetworkStack(blk_polar)

    # Test display
    println(devnull, stack)
    # By defaut, values are equal to 0
    @test isa(stack.input, M)
    @test isa(stack.vmag, M)

    # Copy state x0 inside cache
    copyto!(stack, mapx, x0)
    @test stack.input[mapx] == x0
    # Copy control u0 inside cache
    ExaPF.blockcopy!(stack, mapu, u0)

    # Test that all attributes have valid length
    @test length(stack.vang) == length(stack.vmag) == nbus * nblocks
    @test length(stack.pgen) == ngen * nblocks
    @test length(stack.input) == (ngen + 2 * nbus) * nblocks
    @test length(stack.ψ) == (2 * nlines + nbus) * nblocks

    # Bounds
    b♭, b♯ = ExaPF.bounds(polar, stack)
    @test myisless(b♭, b♯)

    empty!(stack)
    @test iszero(stack.input)

    # Test constructor with scenarios
    pload = rand(nbus, nblocks)
    qload = rand(nbus, nblocks)
    stack = ExaPF.NetworkStack(blk_polar)
    ExaPF.set_params!(stack, pload, qload)
    return nothing
end

function test_polar_api(polar, device, M)
    pf = polar.network
    npv, npq, nref = length(pf.pv), length(pf.pq), length(pf.ref)
    ngen = pf.ngen

    tolerance = 1e-8
    nx = ExaPF.number(polar, State())
    nu = ExaPF.number(polar, Control())
    # Test mapping
    mapx = ExaPF.mapping(polar, State())
    mapu = ExaPF.mapping(polar, Control())
    @test length(mapx) == nx == npv + 2*npq
    @test length(mapu) == nu == npv + nref + ngen - 1

    stack = ExaPF.NetworkStack(polar)
    basis  = ExaPF.PolarBasis(polar)
    power_balance = ExaPF.PowerFlowBalance(polar) ∘ basis
    # Test that values are matching
    vbus = PS.voltage(polar.network)
    sbus = PS.power_balance(polar.network)
    @test myisapprox(vbus, stack.vmag .* exp.(im .* stack.vang))

    # Check that initial residual is correct
    mis = vbus .* conj.(pf.Ybus * vbus) .- sbus
    f_mat = [real(mis[[pf.pv; pf.pq]]); imag(mis[pf.pq])];

    cons = similar(stack.input, nx)
    power_balance(cons, stack)
    @test myisapprox(cons, f_mat)

    # Test powerflow with cache signature
    conv = ExaPF.run_pf(polar, stack)
    @test conv.has_converged

    # Test callbacks
    ## Power Balance
    power_balance(cons, stack)
    # As we run powerflow before, the balance should be below tolerance
    @test ExaPF.xnorm_inf(cons) < tolerance

    ## Cost Production
    cost_production = ExaPF.CostFunction(polar)
    c2 = CUDA.@allowscalar cost_production(stack)[1]
    @test isa(c2, Real)
    # Test display
    println(devnull, cost_production)
    println(devnull, basis)
    return nothing
end

function test_polar_constraints(polar, device, M)
    stack = ExaPF.NetworkStack(polar)
    basis  = ExaPF.PolarBasis(polar)

    io = devnull
    vpq = ExaPF.VoltageMagnitudeBounds(polar)
    pgb = ExaPF.PowerGenerationBounds(polar)
    lfw = ExaPF.LineFlows(polar)
    pfb = ExaPF.PowerFlowBalance(polar)
    mle = ExaPF.MultiExpressions([vpq, pgb, lfw, pfb])

    @testset "Expressions $expr" for expr in [
        vpq,
        pgb,
        lfw,
        pfb,
        mle,
    ]
        println(io, expr)
        # Instantiate
        constraints = expr ∘ basis
        m = length(constraints)
        @test isa(m, Int)
        g = M{Float64, 1}(undef, m) # TODO: this signature is not great
        fill!(g, 0)
        constraints(g, stack)

        g_min, g_max = ExaPF.bounds(polar, constraints)
        @test length(g_min) == m
        @test length(g_max) == m
        @test isa(g_min, M)
        @test isa(g_max, M)
        @test myisless(g_min, g_max)
    end
    return nothing
end

function test_polar_powerflow(polar, device, M)
    SMT = ExaPF.default_sparse_matrix(polar.device)
    # Init structures
    stack = ExaPF.NetworkStack(polar)
    pf_solver = NewtonRaphson(tol=1e-6)
    npartitions = 8

    basis = ExaPF.PolarBasis(polar)
    pflow = ExaPF.PowerFlowBalance(polar)
    n = length(pflow)

    # Init AD
    jx = ExaPF.Jacobian(polar, pflow ∘ basis, State())
    J = jx.J
    @test isa(J, SMT)
    J_host = SparseMatrixCSC(J)

    @test n == size(J, 1) == ExaPF.number(polar, State())

    # Build preconditioner
    precond = LS.BlockJacobiPreconditioner(J_host, npartitions, device)

    @testset "Powerflow solver $(LinSolver)" for LinSolver in ExaPF.list_solvers(device)
        algo = LinSolver(J; P=precond)
        ExaPF.init!(polar, stack)
        convergence = ExaPF.nlsolve!(
            pf_solver, jx, stack; linear_solver=algo)
        @test convergence.has_converged
        @test convergence.norm_residuals < pf_solver.tol
    end
end

function test_block_expressions(polar, device, M)
    nblocks = 2
    blk_polar = ExaPF.BlockPolarForm(polar, nblocks)
    stack = ExaPF.NetworkStack(polar)
    blk_stack = ExaPF.NetworkStack(blk_polar)

    # Test single expressions
    for expr in [
        ExaPF.PowerFlowBalance,
        ExaPF.PowerGenerationBounds,
        ExaPF.VoltageMagnitudeBounds,
        ExaPF.LineFlows,
        ExaPF.CostFunction,
    ]
        pf_blk = expr(blk_polar) ∘ ExaPF.PolarBasis(blk_polar)
        pf = expr(polar) ∘ ExaPF.PolarBasis(polar)
        m = length(pf)
        cons = zeros(m) |> M
        blk_cons = zeros(m * nblocks) |> M
        # One evaluation
        pf(cons, stack)
        # Block evaluation
        pf_blk(blk_cons, blk_stack)
        # Test that results match
        @test length(pf_blk) == m * nblocks
        @test blk_cons ≈ repeat(cons, nblocks)
    end
    # Test bounds
    for expr in [
        ExaPF.PowerFlowBalance,
        ExaPF.PowerGenerationBounds,
        ExaPF.VoltageMagnitudeBounds,
        ExaPF.LineFlows,
    ]
        pf_blk = expr(blk_polar) ∘ ExaPF.PolarBasis(blk_polar)
        lb, ub = ExaPF.bounds(blk_polar, pf_blk)
        @test length(lb) == length(ub) == length(pf_blk)
    end
    # Test multi-expressions
    constraints = [
        ExaPF.PowerFlowBalance(blk_polar),
        ExaPF.PowerGenerationBounds(blk_polar),
        ExaPF.VoltageMagnitudeBounds(blk_polar),
        ExaPF.LineFlows(blk_polar),
    ]
    multiblk = ExaPF.MultiExpressions(constraints) ∘ ExaPF.PolarBasis(blk_polar)
    lb, ub = ExaPF.bounds(blk_polar, multiblk)
    @test length(multiblk) == sum(length.(constraints))
    @test length(lb) == length(ub) == length(multiblk)
end

# NB: currently tested only with direct linear solver
function test_block_powerflow(polar, device, M)
    pf_solver = NewtonRaphson(tol=1e-10)
    nblocks = 10

    blk_polar = ExaPF.BlockPolarForm(polar, nblocks)
    blk_stack = ExaPF.NetworkStack(blk_polar)

    perturb = 0.01 .* rand(length(blk_stack.pload)) |> M
    blk_stack.pload .*= perturb

    pf = ExaPF.PowerFlowBalance(blk_polar) ∘ ExaPF.PolarBasis(blk_polar)

    blk_jac = ExaPF.ArrowheadJacobian(blk_polar, pf, State())
    ExaPF.set_params!(blk_jac, blk_stack)
    ExaPF.jacobian!(blk_jac, blk_stack)

    convergence = ExaPF.nlsolve!(
        pf_solver,
        blk_jac,
        blk_stack;
    )
    @test convergence.has_converged
    @test convergence.norm_residuals < pf_solver.tol
end

function test_contingency_powerflow(polar, device, M)
    nbus = get(polar, PS.NumberOfBuses())
    pf_solver = NewtonRaphson(tol=1e-10)

    contingencies = ExaPF.LineContingency[
        ExaPF.LineContingency(2),
        ExaPF.LineContingency(8),
    ]
    nblocks = length(contingencies) + 1

    blk_polar = ExaPF.BlockPolarForm(polar, nblocks)
    blk_stack = ExaPF.NetworkStack(blk_polar)

    pf = ExaPF.PowerFlowBalance(blk_polar, contingencies) ∘ ExaPF.PolarBasis(blk_polar)

    blk_jac = ExaPF.ArrowheadJacobian(blk_polar, pf, State())
    ExaPF.set_params!(blk_jac, blk_stack)
    ExaPF.jacobian!(blk_jac, blk_stack)

    convergence = ExaPF.nlsolve!(
        pf_solver,
        blk_jac,
        blk_stack;
    )
    @test convergence.has_converged
    @test convergence.norm_residuals < pf_solver.tol

    # Compare with references
    for (k, contingency) in enumerate(contingencies)
        line = contingency.line_id
        network = PS.PowerNetwork(polar.network)
        # Drop line manually inside model's data
        network.lines.Yff[line] = 0.0im
        network.lines.Yft[line] = 0.0im
        network.lines.Ytf[line] = 0.0im
        network.lines.Ytt[line] = 0.0im
        #
        post_model = ExaPF.PolarForm(network, device)

        stack = ExaPF.NetworkStack(post_model)
        conv = ExaPF.run_pf(post_model, stack)
        @test conv.has_converged

        vref = Array(stack.vmag)
        vmag = Array(blk_stack.vmag[k * nbus + 1:(k+1)*nbus])
        @test vref ≈ vmag
    end
end

