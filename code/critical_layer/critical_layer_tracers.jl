using OrdinaryDiffEqSSPRK
using CSV, DataFrames
using JLD2
using Trixi
using Trixi: PassiveTracerEquations

include(joinpath(@__DIR__, "..", "equations", "compressible_euler_vectorinvariant_2d.jl"))
include(joinpath(@__DIR__, "..", "solver", "noncons_kernel_2d.jl"))
include(joinpath(@__DIR__, "..", "solver", "passive_tracers_2d.jl"))

const g = 9.81
const c_p = 1004.0
const c_v = 717.0
const R = c_p - c_v
const p_0 = 100_000.0
const theta_0 = 288.18
const u0 = 10.0
const Nf = 0.0097
const shear = 1.0e-3

const x0 = 10000.0
const L = 10000.0
const H = 20000.0
const a0 = 20.0

const z_B = 15000.0
const z_T = H
const alfa = 0.03

const n_tracers = 3
const tracer_shape = :layers
const tracer_centers_z = (8800.0, 9200.0, 9600.0)
const tracer_width_z = 250.0
const bubble_center_x = 0.0
const bubble_sigma_x = 1000.0

flow_equations =
    CompressibleEulerVectorInvariantEquations2D(c_p = c_p, c_v = c_v, gravity = g)
equations = PassiveTracerEquations(flow_equations; n_tracers = n_tracers)

@inline u_bg(z) = u0 - shear * z
@inline theta_bg(z) = theta_0 * exp(Nf^2 / g * z)

@inline function tracer_profiles(x, z)
    return SVector(
        ntuple(
            i -> begin
                chi = exp(-0.5 * ((z - tracer_centers_z[i]) / tracer_width_z)^2)
                if tracer_shape === :bubbles
                    chi *= exp(-0.5 * ((x - bubble_center_x) / bubble_sigma_x)^2)
                end
                chi
            end,
            Val(n_tracers),
        ),
    )
end

function initial_condition_turbulence(x, t, equations)
    exner = 1 + g^2 / (c_p * theta_0 * Nf^2) * (exp(-Nf^2 / g * x[2]) - 1)
    p = p_0 * exner^(c_p / R)
    theta = theta_bg(x[2])
    T = theta * exner
    rho = p / (R * T)
    v1 = u_bg(x[2])
    v2 = 0.0
    return prim2cons(SVector(rho, v1, v2, p, g * x[2]), equations)
end

function initial_condition_turbulence(x, t, equations::PassiveTracerEquations)
    u_flow = initial_condition_turbulence(x, t, equations.flow_equations)
    chi = tracer_profiles(x[1], x[2])
    return SVector(u_flow..., (u_flow[1] * chi)...)
end

@inline function sponge_v(z)
    z <= z_B ? 0.0 : -alfa * sinpi(0.5 * (z - z_B) / (z_T - z_B))^2
end

function source_terms_turbulence(u, x, t, equations::PassiveTracerEquations)
    rho, v1, v2, rho_theta, _ = u
    theta = rho_theta / rho
    S = sponge_v(x[2])

    du2 = (v1 - u_bg(x[2])) * S
    du3 = v2 * S
    du4 = rho * (theta - theta_bg(x[2])) * S

    return SVector(
        zero(eltype(u)),
        du2,
        du3,
        du4,
        zero(eltype(u)),
        ntuple(_ -> zero(eltype(u)), Val(n_tracers))...,
    )
end

function integrate_over_line(sol, semi, cells_per_dimension, polydeg)
    @unpack solver, mesh, cache = semi
    @unpack weights = solver.basis

    u = Trixi.wrap_array(sol.u[end], semi)
    u_init = Trixi.wrap_array(sol.u[1], semi)

    rho = u_init[1, :, :, :]
    up = u[2, :, :, :] .- u_init[2, :, :, :]
    wp = u[3, :, :, :] .- u_init[3, :, :, :]
    m = rho .* up .* wp

    integral = zeros(cells_per_dimension[2] * (polydeg + 1))
    z_coords = copy(integral)

    jstart = 1
    for z = 1:cells_per_dimension[2]*(polydeg+1)
        if (z % (polydeg + 1) == 1 && z != 1)
            jstart += 1
        end
        loc_integral = 0.0
        iterator = (1+cells_per_dimension[1]*(jstart-1)):(cells_per_dimension[1]*jstart)
        for element in iterator
            for i in eachnode(solver)
                j = z % (polydeg + 1)
                j == 0 && (j = polydeg + 1)
                jacobian = L / cells_per_dimension[1] / 2
                loc_integral += jacobian * weights[i] * m[i, j, element]
            end
        end
        j = z % (polydeg + 1)
        j == 0 && (j = polydeg + 1)
        z_coords[z] = cache.elements.node_coordinates[
            2,
            1,
            j,
            Int(iterator[end] - cells_per_dimension[1] / 2 + 1),
        ]
        integral[z] = loc_integral
    end

    rho_s = p_0 / (R * theta_0)
    k = 2 * pi / x0
    m0 = sqrt(Nf^2 / u0^2 - k^2)
    M_ref = -0.5 * rho_s * u0^2 * k * m0 * a0^2 * L
    integral ./= M_ref

    return integral, z_coords
end

polydeg = 3
basis = LobattoLegendreBasis(polydeg)
surface_flux = FluxTracerUpwindCombined(flux_surface_combined_entropy_stable)
volume_flux = flux_volume_combined_turbo_entropy_conservative
volume_integral = VolumeIntegralFluxDifferencing(volume_flux)
solver = DGSEM(basis, surface_flux, volume_integral)

zb(x) = a0 * cos(2π * x / x0)
f1(s) = SVector(-L / 2, zb(-L / 2) + (H - zb(-L / 2)) * 0.5 * (s + 1))
f2(s) = SVector(L / 2, zb(L / 2) + (H - zb(L / 2)) * 0.5 * (s + 1))
f3(s) = SVector(s * L / 2, zb(s * L / 2))
f4(s) = SVector(s * L / 2, H)

cells_per_dimension = (80, 180)

mesh = P4estMesh(
    cells_per_dimension,
    polydeg = polydeg,
    faces = (f1, f2, f3, f4),
    initial_refinement_level = 0,
    periodicity = (true, false),
)

boundary = BoundaryConditionDirichlet(initial_condition_turbulence)

boundary_conditions = (y_neg = boundary_condition_slip_wall, y_pos = boundary)

semi = SemidiscretizationHyperbolic(
    mesh,
    equations,
    initial_condition_turbulence,
    solver,
    source_terms = source_terms_turbulence,
    boundary_conditions = boundary_conditions,
)

T_hours = 12.0
saveat_interval = 1800.0
do_solve = true

if do_solve
    tspan = (0.0, T_hours * 3600.0)
    ode = semidiscretize(semi, tspan)

    summary_callback = SummaryCallback()
    analysis_callback = AnalysisCallback(semi, interval = 1000)
    alive_callback = AliveCallback(analysis_interval = 1000)
    callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback)
    tol = 1e-6
    sol = solve(
        ode,
        SSPRK43(thread = Trixi.Threaded());
        abstol = tol, reltol = tol,
        maxiters = 1.0e7,
        save_everystep = false,
        saveat = saveat_interval,
        callback = callbacks,
    )

    summary_callback()

    run_tag = "$(tracer_shape)_a$(a0)_$(T_hours)_$(cells_per_dimension[1])x$(cells_per_dimension[2])_$(polydeg)"
    mkpath(joinpath(@__DIR__, "..") * "/critical_layer/results")

    snapshot_path = joinpath(@__DIR__, "..") * "/critical_layer/results/snapshots_$(run_tag).jld2"
    jldsave(
        snapshot_path;
        t = collect(sol.t),
        u = [Vector(x) for x in sol.u],
        cells_per_dimension = collect(cells_per_dimension),
        polydeg = polydeg,
        a0 = a0,
        T_hours = T_hours,
        tracer_shape = String(tracer_shape),
        n_tracers = n_tracers,
        tracer_centers_z = collect(tracer_centers_z),
        tracer_width_z = tracer_width_z,
    )

    verticalmomentum, z_coords =
        integrate_over_line(sol, semi, cells_per_dimension, polydeg)
    data = DataFrame(z_coords = z_coords, verticalmomentum = verticalmomentum)
    CSV.write(
        joinpath(@__DIR__, "..") * "/critical_layer/results/turbulence_tracers_$(run_tag).csv",
        data)
end
