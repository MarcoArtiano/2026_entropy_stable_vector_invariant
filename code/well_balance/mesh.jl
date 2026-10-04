using Trixi
using OrdinaryDiffEqLowStorageRK
using LinearAlgebra: norm

include(joinpath(@__DIR__, "..", "equations", "compressible_euler_vectorinvariant_2d.jl"))
include(joinpath(@__DIR__, "..", "solver", "noncons_kernel_2d.jl"))

using OrdinaryDiffEqSSPRK
using CairoMakie
using DoubleFloats
using LaTeXStrings
using DelimitedFiles

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

@inline function flux_zero(u_ll, u_rr, normal_direction, equations)
    return zero(u_ll)
end

RealT = Float64
T = RealT(0.0)
equations = CompressibleEulerVectorInvariantEquations2D(
    c_p = 1004,
    c_v = 717,
    gravity = RealT(9.81),
)
polydeg = 3
basis = LobattoLegendreBasis(RealT, polydeg)
surface_flux = flux_surface_combined_entropy_stable
volume_flux = flux_volume_combined_turbo_entropy_conservative
volume_integral = VolumeIntegralFluxDifferencing(volume_flux)
solver = DGSEM(basis, surface_flux, volume_integral)

trees_per_dimension = (16, 16)

function mapping(xi, eta)
    x = xi + RealT(0.1) * sinpi(xi) * sinpi(eta)
    y = eta + RealT(0.1) * sinpi(xi) * sinpi(eta)
    return SVector(
        RealT(1000) * RealT(0.5) * (RealT(1) + x),
        RealT(1000) * RealT(0.5) * (RealT(1) + y),
    )
end

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

dt = RealT(0.01)
tspan = (zero(RealT), T)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 1000
analysis_callback = AnalysisCallback(
    semi,
    interval = analysis_interval,
    extra_analysis_integrals = (well_balanced_v1, well_balanced_v2),
    save_analysis = true,
    output_directory = joinpath(@__DIR__, "..") * "/results/out",
    analysis_filename = "well_balancing_$(RealT)_isothermal.dat",
)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback)

sol = solve(
    ode,
    SSPRK43();
    dt = dt,
    ode_default_options()...,
    callback = callbacks,
    adaptive = false,
)
include(joinpath(@__DIR__, "..", "contours.jl"))
x, y, data = ContourData(sol.u[end], semi, trees_per_dimension, equations)

h = Figure(size = (500, 500))
labelsize = 15
kwargs = (xlabel = L"$x$ [km]", xlabelsize = labelsize, ylabelsize = labelsize, limits = ((0, 1), (0, 1)), xticklabelsize = 13.0, yticklabelsize = 13.0)

Axis(h[1, 1]; kwargs..., ylabel = L"$z$ [km]", aspect = DataAspect())

for j in axes(y, 2)
    lines!(x[:, j] ./ 1e3, y[:, j] ./ 1e3, color = :gray, linewidth = 0.7)
end

for i in axes(x, 1)
    lines!(x[i, :] ./ 1e3, y[i, :] ./ 1e3, color = :gray, linewidth = 0.7)
end

resize_to_layout!(h)   # shrink the figure to the (square, DataAspect) content — no white borders
save(joinpath(@__DIR__, "..") * "/results/mesh_well_balance.pdf", h)