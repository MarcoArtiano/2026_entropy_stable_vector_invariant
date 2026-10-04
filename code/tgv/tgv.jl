
using DelimitedFiles
using CairoMakie
using LaTeXStrings
using OrdinaryDiffEqSSPRK

using Trixi
using OrdinaryDiffEqLowStorageRK
using LinearAlgebra: norm

include(joinpath(@__DIR__, "..", "equations", "compressible_euler_vectorinvariant_3d.jl"))
include(joinpath(@__DIR__, "..", "solver", "noncons_kernel_3d.jl"))

function energy_timederivative end
function Trixi.analyze(::typeof(energy_timederivative), du, u, t,
                 mesh::Trixi.P4estMesh{3},
                 equations, dg::DGSEM, cache)
    # Calculate ∫(∂S/∂u ⋅ ∂u/∂t)dΩ
    Trixi.integrate_via_indices(u, mesh, equations, dg, cache,
                          du) do u, i, j, k, element, equations, dg, du
        u_node = Trixi.get_node_vars(u, equations, dg, i, j, k, element)
        du_node = Trixi.get_node_vars(du, equations, dg, i, j, k, element)
        return Trixi.dot(cons2entropy_energy(u_node, equations), du_node)
    end
end

Trixi.pretty_form_utf(::typeof(energy_timederivative)) = "∑∂E/∂U ⋅ Uₜ"
Trixi.default_analysis_integrals(::CompressibleEulerVectorInvariantEquations3D) = (Trixi.entropy_timederivative, energy_timederivative, )

function initial_condition_taylor_green_vortex(
    x,
    t,
    equations::CompressibleEulerVectorInvariantEquations3D,
)

    A = 1.0 # magnitude of speed
    Ms = 0.1 # maximum Mach number

    rho = 1.0
    v1 = A * sin(x[1]) * cos(x[2]) * cos(x[3])
    v2 = -A * cos(x[1]) * sin(x[2]) * cos(x[3])
    v3 = 0.0
    p = (A / Ms)^2 * rho / equations.gamma # scaling to get Ms
    p = p +
        1.0 / 16.0 * A^2 * rho *
        (cos(2 * x[1]) * cos(2 * x[3]) + 2 * cos(2 * x[2]) + 2 * cos(2 * x[1]) +
         cos(2 * x[2]) * cos(2 * x[3]))
    return prim2cons(SVector(rho, v1, v2, v3, p, zero(eltype(x))), equations)
end

equations = CompressibleEulerVectorInvariantEquations3D(c_p = 1004, c_v = 717, gravity = 9.81)

surface_flux = flux_surface_combined_entropy_conservative
volume_flux = flux_volume_combined_turbo_entropy_conservative

function run_polydeg(polydeg)

solver = DGSEM(
    polydeg = polydeg,
    surface_flux = surface_flux,
    volume_integral = VolumeIntegralFluxDifferencing(volume_flux),
)

coordinates_min = (-1.0, -1.0, -1.0) .* pi
coordinates_max = (1.0, 1.0, 1.0) .*  pi
    cells_per_dimension = (4, 4, 4)
    mesh = P4estMesh(
        cells_per_dimension,
        polydeg = polydeg,
        initial_refinement_level = 0,
        coordinates_min = coordinates_min,
        coordinates_max = coordinates_max,
        periodicity = true,
    )


initial_condition = initial_condition_taylor_green_vortex

semi = SemidiscretizationHyperbolic(
    mesh,
    equations,
    initial_condition,
    solver,
    boundary_conditions = boundary_condition_periodic,
)
tspan = (0.0, 5.0)

ode = semidiscretize(semi, tspan)
analysis_interval = 100000
summary_callback = SummaryCallback()

analysis_callback = AnalysisCallback(
    semi,
    interval = analysis_interval,
    save_analysis = true,
    output_directory = joinpath(@__DIR__, "..") * "/results/",
    analysis_filename = "analysis_entropy_$(polydeg).dat",
    extra_analysis_integrals = (energy_total, entropy),
)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

stepsize_callback = StepsizeCallback(cfl = 0.01)

callbacks =
    CallbackSet(summary_callback, analysis_callback, alive_callback, stepsize_callback)

time_method = SSPRK43(thread = Trixi.Threaded())
    sol = solve(
        ode,
        time_method,
        dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
        save_everystep = false,
        callback = callbacks,
        adaptive = false)

end

polydegs = (2, 3, 4, 5)
for polydeg in polydegs
    run_polydeg(polydeg)
end

include(joinpath(@__DIR__, "entropy_table.jl"))
