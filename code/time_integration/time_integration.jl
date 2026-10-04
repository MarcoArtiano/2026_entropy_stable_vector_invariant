using Trixi: @unpack, @trixi_timeit, timer
using SparseArrays, LinearAlgebra
using RecipesBase
using FastBroadcast
using MuladdMacro

abstract type RKTableau end

struct TimeIntegratorSolution{tType, uType, P}
    t::tType
    u::uType
    prob::P
end

RecipesBase.@recipe function f(sol::TimeIntegratorSolution)
    # Redirect everything to the recipes below
    return sol.u[end], sol.prob.p
end

abstract type AbstractTimeIntegrator end

using DiffEqBase: DiffEqBase, get_tstops, get_tstops_array
import DiffEqBase: solve, CallbackSet, ODEProblem, SplitODEProblem
export solve, ODEProblem, SplitODEProblem
using DataStructures: BinaryHeap, FasterForward, extract_all!

# Interface required by DiffEqCallbacks.jl
function DiffEqBase.get_tstops(integrator::AbstractTimeIntegrator)
    return integrator.opts.tstops
end
function DiffEqBase.get_tstops_array(integrator::AbstractTimeIntegrator)
    return get_tstops(integrator).valtree
end
function DiffEqBase.get_tstops_max(integrator::AbstractTimeIntegrator)
    return maximum(get_tstops_array(integrator))
end

function finalize_callbacks(integrator::AbstractTimeIntegrator)
    callbacks = integrator.opts.callback

    return if callbacks isa CallbackSet
        foreach(callbacks.discrete_callbacks) do cb
            cb.finalize(cb, integrator.u, integrator.t, integrator)
        end
        foreach(callbacks.continuous_callbacks) do cb
            cb.finalize(cb, integrator.u, integrator.t, integrator)
        end
    end
end

import SciMLBase: get_du, get_tmp_cache, u_modified!,
                  init, step!, check_error,
                  get_proposed_dt, set_proposed_dt!,
                  terminate!, remake, add_tstop!, has_tstop, first_tstop

# get a cache where the RHS can be stored
get_du(integrator) = integrator.du
get_tmp_cache(integrator) = (integrator.u_tmp,)

include(joinpath(@__DIR__, "jacobian_cache.jl"))
include(joinpath(@__DIR__, "vertical_rhs.jl"))
include(joinpath(@__DIR__, "linear_imex.jl"))
include(joinpath(@__DIR__, "tableau.jl"))

# integrator glue (LinearIMEX only)
# LinearIMEX needs to inform FSAL-style callbacks that u was modified
u_modified!(integrator::LinearIMEX, ::Bool) = false

# used by adaptive timestepping algorithms in DiffEq
function set_proposed_dt!(integrator::LinearIMEX, dt)
    return integrator.dt = dt
end

# Required e.g. for `glm_speed_callback`
function get_proposed_dt(integrator::LinearIMEX)
    return integrator.dt
end

# stop the time integration
function terminate!(integrator::LinearIMEX)
    integrator.finalstep = true
    return nothing
end

function add_tstop!(integrator::LinearIMEX, t)
    # already finished
    if integrator.tdir * (integrator.sol.prob.tspan[2] - integrator.t) <=
       eps(eltype(integrator.t))
        return
    end
    # do not add past stops
    if integrator.tdir * (t - integrator.t) <= eps(eltype(integrator.t))
        return
    end
    if length(integrator.opts.tstops) > 1
        pop!(integrator.opts.tstops)
    end
    push!(integrator.opts.tstops, integrator.tdir * t)
end

# used for AMR
function Base.resize!(integrator::LinearIMEX, new_size)
    resize!(integrator.u, new_size)
    resize!(integrator.du, new_size)
    return resize!(integrator.u_tmp, new_size)
end

# Forward integrator.stats.naccept to integrator.iter (see GitHub PR#771)
function Base.getproperty(integrator::LinearIMEX, field::Symbol)
    if field === :stats
        return (naccept = getfield(integrator, :iter),)
    end
    # general fallback
    return getfield(integrator, field)
end
