abstract type LinearIMEXAlgorithm{N} end

stages(::LinearIMEXAlgorithm{N}) where {N} = N

"""
	(::LinearIMEXAlgorithm{N})(res, uₙ, Δt, f!, du, u, p, t, stages, stage, workspace, RK, assume_p_const) where N

"""

@muladd begin
function (::LinearIMEXAlgorithm{N})(res, uₙ, tmp, Δt, f!, du, u, p, t, stages, u_stages,
                                    stage, RK, cache_jacobian, thread) where {N}
    if iszero(RK.a_im[stage, stage])
        Threads.@threads for i in eachindex(u)
            acc_r = uₙ[i]
            for j in 1:(stage - 1)
                acc_r = acc_r + (RK.a_ex[stage, j] * Δt) * stages[j][i]
            end
            u[i] = acc_r
            u_stages[stage][i] = acc_r
        end
    else
        inv_aii = inv(RK.a_im[stage, stage])
        inv_aii_dt = inv(RK.a_im[stage, stage] * Δt)
        Threads.@threads for i in eachindex(u)
            acc_r = uₙ[i]
            acc_tmp = zero(eltype(tmp))
            for j in 1:(stage - 1)
                acc_r = acc_r + (RK.a_ex[stage, j] * Δt) * stages[j][i]
                acc_tmp = acc_tmp + inv_aii * (RK.a_im[stage, j] - RK.a_ex[stage, j]) * u_stages[j][i]
            end
            tmp[i] = acc_tmp
            res[i] = (acc_r + acc_tmp) * inv_aii_dt
        end
        @trixi_timeit timer() "solve_jacobian!" solve_jacobian!(res, Δt, RK.a_im[2, 2], p,
                                                                p.equations, cache_jacobian,
                                                                cache_jacobian.u_vert)
        Threads.@threads for i in eachindex(u)
            u[i] = -tmp[i] + res[i]
            u_stages[stage][i] = u[i]
        end
    end

    f!(stages[stage], u, p, t + Δt)

    return if stage == N
        b = RK.b
        Threads.@threads for i in eachindex(u)
            acc = uₙ[i]
            for j in 1:N
                acc = acc + Δt * b[j] * stages[j][i]
            end
            u[i] = acc
        end
    end
end
end

mutable struct LinearIMEXOptions{Callback, TStops}
    callback::Callback # callbacks; used in Trixi.jl
    adaptive::Bool # whether the algorithm is adaptive; ignored
    dtmax::Float64 # ignored
    maxiters::Int # maximal number of time steps
    verbose::Int
    tstops::TStops # tstops from https://diffeq.sciml.ai/v6.8/basics/common_solver_opts/#Output-Control-1; ignored
end

function LinearIMEXOptions(callback, tspan; maxiters = typemax(Int), verbose = 0, kwargs...)
    tstops_internal = BinaryHeap{eltype(tspan)}(FasterForward())
    # We add last(tspan) to make sure that the time integration stops at the end time
    push!(tstops_internal, last(tspan))
    # We add 2 * last(tspan) because add_tstop!(integrator, t) is only called by DiffEqCallbacks.jl if tstops contains a time that is larger than t
    # (https://github.com/SciML/DiffEqCallbacks.jl/blob/025dfe99029bd0f30a2e027582744528eb92cd24/src/iterative_and_periodic.jl#L92)
    push!(tstops_internal, 2 * last(tspan))
    return LinearIMEXOptions{typeof(callback), typeof(tstops_internal)}(callback, false,
                                                                        Inf, maxiters,
                                                                        verbose,
                                                                        tstops_internal)
end

# This struct is needed to fake https://github.com/SciML/OrdinaryDiffEq.jl/blob/0c2048a502101647ac35faabd80da8a5645beac7/src/integrators/type.jl#L77
# This implements the interface components described at
# https://diffeq.sciml.ai/v6.8/basics/integrator/#Handing-Integrators-1
# which are used in Trixi.jl.
mutable struct LinearIMEX{RealT <: Real, uType, Params, Sol, F, M,
                          Alg <: LinearIMEXAlgorithm,
                          LinearIMEXOptions, RKTableau, JacobianCacheType, Thread} <:
               AbstractTimeIntegrator
    u::uType
    du::uType
    u_tmp::uType
    u_tmp2::uType
    stages::NTuple{M, uType}
    u_stages::NTuple{M, uType}
    res::uType
    t::RealT
    tdir::RealT # DIRection of time integration, i.e., if one marches forward or backward in time
    dt::RealT # current time step
    dtcache::RealT # ignored
    iter::Int # current number of time steps (iteration)
    p::Params # will be the semidiscretization from Trixi.jl
    sol::Sol # faked
    f::F # `rhs!` of the semidiscretization
    alg::Alg # LinearIMEXAlgorithm
    opts::LinearIMEXOptions
    finalstep::Bool # added for convenience
    RK::RKTableau
    const dtchangeable::Bool
    const force_stepfail::Bool
    cache_jacobian::JacobianCacheType
    thread::Thread
    fsal_du::uType
    fsal_valid::Bool
    jac_interval::Int
end


# To keep backwards compatibility with SciMLBase v2, see
# https://github.com/trixi-framework/Trixi.jl/pull/2918#issuecomment-4233720339
@static if isdefined(SciMLBase, :derivative_discontinuity!)
    import SciMLBase: derivative_discontinuity!
else
    const derivative_discontinuity! = SciMLBase.u_modified!
end

# some algorithms from DiffEq like FSAL-ones need to be informed when a callback has modified u
derivative_discontinuity!(integrator::LinearIMEX, ::Bool) = false

function init(ode::ODEProblem, alg::LinearIMEXAlgorithm{N};
              dt, callback::Union{CallbackSet, Nothing} = nothing,
              trees_per_dimension, semi_split, thread = Trixi.False,
              jac_interval = 1, operator_eltype = Float64, kwargs...,) where {N}
    u = copy(ode.u0)
    du = zero(u)
    res = zero(u)
    u_tmp = similar(u)
    u_tmp2 = similar(u)
    stages = ntuple(_ -> similar(u), Val(N))
    u_stages = ntuple(_ -> similar(u), Val(N))
    t = first(ode.tspan)
    iter = 0
    tdir = sign(ode.tspan[end] - ode.tspan[1])

    NumG = compute_number_of_vertical_lines(ode.p.solver, trees_per_dimension, ode.p.mesh)

    cache_jacobian = JacobianCache(trees_per_dimension[2], trees_per_dimension[1],
                                   NumG, ode.p, dt, ode.p.equations;
                                   semi_split = semi_split, operator_eltype = operator_eltype)

	rk_tableau = RKTableau(alg, eltype(u))
	cache_jacobian.inv_gamma_dt /= rk_tableau.a_im[2,2]
    fsal_du = zero(u)

    integrator = LinearIMEX(u, du, u_tmp, u_tmp2, stages, u_stages, res, t, tdir, dt, zero(dt), iter, ode.p,
                            (prob = ode,), ode.f, alg,
                            LinearIMEXOptions(callback, ode.tspan;
                                              kwargs...,), false, rk_tableau ,
                            false, false, cache_jacobian, thread, fsal_du, false, jac_interval)
    # initialize callbacks
    if callback isa CallbackSet
        foreach(callback.continuous_callbacks) do cb
            throw(ArgumentError("Continuous callbacks are unsupported with the implicit time integration methods."))
        end
        foreach(callback.discrete_callbacks) do cb
            cb.initialize(cb, integrator.u, integrator.t, integrator)
        end
    end

    return integrator
end

# Fakes `solve`: https://diffeq.sciml.ai/v6.8/basics/overview/#Solving-the-Problems-1
function solve(ode::ODEProblem, alg::LinearIMEXAlgorithm;
               dt, callback = nothing, trees_per_dimension, semi_split,
               jac_interval = 1, kwargs...,)
    integrator = init(ode, alg, dt = dt, callback = callback,
                      trees_per_dimension = trees_per_dimension,
                      semi_split = semi_split,
                      jac_interval = jac_interval; kwargs...)

    # Start actual solve
    return solve!(integrator)
end

function solve!(integrator::LinearIMEX)
    @unpack prob = integrator.sol

    integrator.finalstep = false

    while !integrator.finalstep
        step!(integrator)
    end # "main loop" timer

    finalize_callbacks(integrator)

    return TimeIntegratorSolution((first(prob.tspan), integrator.t),
                                  (prob.u0, integrator.u),
                                  integrator.sol.prob)
end

function stage!(integrator, alg::LinearIMEXAlgorithm)
    if integrator.iter % integrator.jac_interval == 0
        @trixi_timeit timer() "update_jacobian!" update_jacobian!(integrator.u, integrator.p,
                                                                  integrator.p.equations,
                                                                  integrator.cache_jacobian, integrator.RK.a_im[2,2])
    end

	   for stage in 1:stages(alg)
        @trixi_timeit timer() "linear imex" alg(integrator.res, integrator.u, integrator.u_tmp2, integrator.dt,
                                               integrator.f, integrator.du,
                                               integrator.u_tmp, integrator.p, integrator.t,
                                               integrator.stages, integrator.u_stages, stage, integrator.RK,
                                               integrator.cache_jacobian, integrator.thread)
        end

end

function step!(integrator::LinearIMEX)
    @unpack prob = integrator.sol
    @unpack alg = integrator
    t_end = last(prob.tspan)
    callbacks = integrator.opts.callback

    @assert !integrator.finalstep
    if isnan(integrator.dt)
        error("time step size `dt` is NaN")
    end

    # if the next iteration would push the simulation beyond the end time, set dt accordingly
    if integrator.t + integrator.dt > t_end ||
       isapprox(integrator.t + integrator.dt, t_end)
        integrator.dt = t_end - integrator.t
        terminate!(integrator)
    end

    # one time step
    # (no need to seed u_tmp: stage 1 of the algorithm overwrites it with uₙ)
    stage!(integrator, alg)

    @threaded for i in eachindex(integrator.u)
        integrator.u[i] = integrator.u_tmp[i]
    end

    integrator.iter += 1
    integrator.t += integrator.dt

    begin
        # handle callbacks
        if callbacks isa CallbackSet
            foreach(callbacks.discrete_callbacks) do cb
                if cb.condition(integrator.u, integrator.t, integrator)
                    cb.affect!(integrator)
                end
                return nothing
            end
        end
    end

    # respect maximum number of iterations
    return if integrator.iter >= integrator.opts.maxiters && !integrator.finalstep
        @warn "Interrupted. Larger maxiters is needed."
        terminate!(integrator)
    end
end
