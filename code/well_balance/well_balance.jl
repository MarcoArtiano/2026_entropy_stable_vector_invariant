using Trixi
using OrdinaryDiffEqLowStorageRK
using LinearAlgebra: norm
using Trixi: timer, @trixi_timeit

include(joinpath(@__DIR__, "..", "equations", "compressible_euler_vectorinvariant_2d.jl"))
include(joinpath(@__DIR__, "..", "solver", "noncons_kernel_2d.jl"))

using OrdinaryDiffEqSSPRK
using CairoMakie
using MultiFloats
MultiFloats.use_bigfloat_transcendentals()
using LaTeXStrings
using DelimitedFiles

struct TimeIntegratorSolution{tType, uType, P}
    t::tType
    u::uType
    prob::P
end

function initial_condition_isothermal(x, t, equations)
    RealT = eltype(x)
    c_p = equations.c_p
    g = equations.g
    c_v = equations.c_v
    # center of perturbation
    T0 = RealT(250.0)
    p0 = RealT(100_000)
    # perturbation in potential temperature
    R = c_p - c_v    # gas constant (dry air)
    delta = g / (R * T0)

    rho0 = p0 / (T0 * R)
    p = p0 * exp(-delta * x[2])
    rho = rho0 * exp(-delta * x[2])
    v1 = zero(RealT)
    v2 = zero(RealT)

    return prim2cons(SVector(rho, v1, v2, p, equations.g * x[2]), equations)
end

function initial_condition_isentropic(x, t, equations)
    RealT = eltype(x)
    c_p = equations.c_p
    g = equations.g
    c_v = equations.c_v

    potential_temperature = RealT(300)
    p0 = RealT(100_000)

    # Exner pressure, solves hydrostatic equation for x[2]
    exner = RealT(1) - g / (c_p * potential_temperature) * x[2]

    # pressure
    R = c_p - c_v    # gas constant (dry air)
    p = p0 * exner^(c_p / R)

    # temperature
    T = potential_temperature * exner
    # density
    rho = p / (R * T)
    v1 = zero(RealT)
    v2 = zero(RealT)

    return prim2cons(SVector(rho, v1, v2, p, equations.g * x[2]), equations)
end

function mapping(xi, eta)
    RealT = eltype(xi)
    x = xi + RealT(0.1) * sinpi(xi) * sinpi(eta)
    y = eta + RealT(0.1) * sinpi(xi) * sinpi(eta)
    return SVector(
        RealT(1000) * RealT(0.5) * (RealT(1) + x),
        RealT(1000) * RealT(0.5) * (RealT(1) + y),
    )
end

function get_max_vel(sol)
    semi = sol.prob.p
    u_wrap = Trixi.wrap_array(sol.u[end], semi)
    max_u = maximum(abs.(u_wrap[2,:,:,:]))
    max_v = maximum(abs.(u_wrap[3,:,:,:]))
    @show max_u, max_v
    return max_u, max_v
end

function run_isentropic(; days, dt, RealT, trees_per_dimension = (16,16), polydeg = 3, analysis_interval = 500000)
equations = CompressibleEulerVectorInvariantEquations2D(
    c_p = 1004,
    c_v = 717,
    gravity = RealT(9.81),
)
basis = LobattoLegendreBasis(RealT, polydeg)
surface_flux = flux_surface_combined_entropy_stable
volume_flux = flux_volume_combined_turbo_entropy_conservative
volume_integral = VolumeIntegralFluxDifferencing(volume_flux)
solver = DGSEM(basis, surface_flux, volume_integral)

mesh = P4estMesh(
    trees_per_dimension,
    polydeg = polydeg,
    mapping = mapping,
    periodicity = (false, false),
    initial_refinement_level = 0,
    RealT = RealT,
)

boundary_conditions = (
    x_pos = boundary_condition_slip_wall,
    x_neg = boundary_condition_slip_wall,
    y_pos = boundary_condition_slip_wall,
    y_neg = boundary_condition_slip_wall,
)

semi = SemidiscretizationHyperbolic(
    mesh,
    equations,
    initial_condition_isentropic,
    solver,
    boundary_conditions = boundary_conditions,
)

dt = RealT(dt)
T = RealT(24*3600*days)
tspan = (zero(RealT), T)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 500000

analysis_callback = AnalysisCallback(
    semi,
    interval = analysis_interval,
    extra_analysis_integrals = (well_balanced_v1, well_balanced_v2),
    save_analysis = true,
    output_directory = joinpath(@__DIR__, "..") * "/results/out",
    analysis_filename = "well_balancing_$(RealT)_isentropic_$(dt)_$(T).dat",
)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback)

sol = solve(
    ode,
    SSPRK43(thread = Trixi.Threaded());
    dt = dt,
    ode_default_options()...,
    callback = callbacks,
    adaptive = false, maxiters = 1e36
)

summary_callback();

    return sol
end

function run_isothermal(; days, dt, RealT, trees_per_dimension = (16,16), polydeg = 3, analysis_interval = 500000)
equations = CompressibleEulerVectorInvariantEquations2D(
    c_p = 1004,
    c_v = 717,
    gravity = RealT(9.81),
)
basis = LobattoLegendreBasis(RealT, polydeg)
surface_flux = flux_surface_combined_entropy_stable
volume_flux = flux_volume_combined_turbo_entropy_conservative
volume_integral = VolumeIntegralFluxDifferencing(volume_flux)
solver = DGSEM(basis, surface_flux, volume_integral)

mesh = P4estMesh(
    trees_per_dimension,
    polydeg = polydeg,
    mapping = mapping,
    periodicity = (false, false),
    initial_refinement_level = 0,
    RealT = RealT,
)

boundary_conditions = (
    x_pos = boundary_condition_slip_wall,
    x_neg = boundary_condition_slip_wall,
    y_pos = boundary_condition_slip_wall,
    y_neg = boundary_condition_slip_wall,
)

semi = SemidiscretizationHyperbolic(
    mesh,
    equations,
    initial_condition_isothermal,
    solver,
    boundary_conditions = boundary_conditions,
)

dt = RealT(dt)
T = RealT(24*3600*days)
tspan = (zero(RealT), T)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_callback = AnalysisCallback(
    semi,
    interval = analysis_interval,
    extra_analysis_integrals = (well_balanced_v1, well_balanced_v2),
    save_analysis = true,
    output_directory = joinpath(@__DIR__, "..") * "/results/out",
    analysis_filename = "well_balancing_$(RealT)_isothermal_$(dt)_$(T).dat",
)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback)


sol = solve(
    ode,
    SSPRK43(thread = Trixi.Threaded());
    dt = dt,
    ode_default_options()...,
    callback = callbacks,
    adaptive = false, maxiters = 1e36
)

summary_callback();

    return sol
end

sol = run_isentropic(days = 1, dt = 0.025, RealT = Float64x2)
sol = run_isothermal(days = 1, dt = 0.025, RealT = Float64x2)
sol = run_isentropic(days = 1, dt = 0.025, RealT = Float64)
sol = run_isothermal(days = 1, dt = 0.025, RealT = Float64)
sol = run_isentropic(days = 10, dt = 0.025, RealT = Float64)
sol = run_isothermal(days = 10, dt = 0.025, RealT = Float64)
sol = run_isentropic(days = 10, dt = 0.025, RealT = Float64x2)
sol = run_isothermal(days = 10, dt = 0.025, RealT = Float64x2)

include(joinpath(@__DIR__, "plot_wb_float_multifloat.jl"))
include(joinpath(@__DIR__, "mesh.jl"))