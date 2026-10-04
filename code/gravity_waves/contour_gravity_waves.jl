using OrdinaryDiffEqSSPRK
using CairoMakie
using Trixi
include(joinpath(@__DIR__, "..", "equations", "compressible_euler_vectorinvariant_3d.jl"))
include(joinpath(@__DIR__, "..", "equations", "compressible_euler_vectorinvariant_2d.jl"))
include(joinpath(@__DIR__, "..", "solver", "noncons_kernel_3d.jl"))
include(joinpath(@__DIR__, "..", "solver", "noncons_kernel_2d.jl"))
include(joinpath(@__DIR__, "..", "contours.jl"))

function initial_condition_gravity_wave(
    x,
    t,
    equations::CompressibleEulerVectorInvariantEquations2D,
)
    g = equations.g
    c_p = equations.c_p
    c_v = equations.c_v
    # center of perturbation
    x_c = 100_000.0
    a = 5_000
    H = 10_000
    # perturbation in potential temperature
    R = c_p - c_v    # gas constant (dry air)

    T0 = 250
    delta = 9.81 / (R * T0)
    DeltaT = 0.001
    Tb = DeltaT * sinpi(x[2] / H) * exp(-(x[1] - x_c)^2 / a^2)
    ps = 100_000.0  # reference pressure
    rhos = ps / (T0 * R)
    rho_b = rhos * (-Tb / T0)
    p = ps * exp(-delta * x[2])
    rho = rhos * exp(-delta * x[2]) + rho_b * exp(-0.5 * delta * x[2])
    v1 = 20.0
    v2 = 0.0

    return prim2cons(SVector(rho, v1, v2, p, equations.g * x[2]), equations)
end

equations = CompressibleEulerVectorInvariantEquations2D(c_p = 1004, c_v = 717, gravity = 9.81)
surface_flux = flux_surface_combined_entropy_stable
volume_flux = flux_volume_combined_turbo_entropy_conservative

polydeg = 3
solver = DGSEM(polydeg = polydeg, surface_flux = surface_flux, volume_integral = VolumeIntegralFluxDifferencing(volume_flux))

boundary_conditions = (
	y_neg = boundary_condition_slip_wall,
	y_pos = boundary_condition_slip_wall)

coordinates_min = (0.0, 0.0)
coordinates_max = (300_000.0, 10_000.0)
cells_per_dimension = (30*3, 4*3)
  mesh = P4estMesh(
        cells_per_dimension,
        polydeg = polydeg,
        initial_refinement_level = 0,
        coordinates_min = coordinates_min,
        coordinates_max = coordinates_max,
        periodicity = (true, false),
    )
initial_condition = initial_condition_gravity_wave
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver, source_terms = nothing,
	boundary_conditions = boundary_conditions)
tspan = (0.0, 1800.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 10000
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

stepsize_callback = StepsizeCallback(cfl = 1.0)

callbacks = CallbackSet(summary_callback,
	analysis_callback,
	alive_callback,
	stepsize_callback)

time_method = SSPRK43(thread = Trixi.Threaded())
sol = solve(ode,
	time_method,
	maxiters = 1.0e7,
	dt = 1e-1, # solve needs some value here but it will be overwritten by the stepsize_callback
	save_everystep = false, callback = callbacks, adaptive = false)

x, y, data = ContourData(sol.u[end], semi, cells_per_dimension, equations)

h = Figure(size = (800, 300))
labelsize = 15
kwargs = (xlabel = L"$x$ [km]", xlabelsize = labelsize, ylabelsize = labelsize, limits = ((0, 300), (0, 10)), xticklabelsize = 13.0, yticklabelsize = 13.0, ylabel = L"$z$ [km]")

Axis(h[1, 1]; kwargs...)

w_field = data[3, :, :]
w_abs = maximum(abs, w_field)
levels = range(-w_abs, w_abs, length = 15)
c = contourf!(x ./ 1e3, y ./ 1e3, w_field, levels = levels, colormap = :PuOr)
Colorbar(h[1, 2], c, ticklabelsize = 13)
Label(h[1, 2, Top()], L"w\;[\mathrm{m/s}]", fontsize = 18, padding = (0, 0, 6, 0))

save(joinpath(@__DIR__, "..") * "/results/contour_$(cells_per_dimension[1])x$(cells_per_dimension[2])_CFL1_polydeg$(polydeg).pdf", h)
