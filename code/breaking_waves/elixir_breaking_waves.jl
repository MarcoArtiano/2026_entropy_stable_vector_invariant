using OrdinaryDiffEqLowStorageRK
using Trixi

include(joinpath(@__DIR__, "..", "equations", "compressible_euler_vectorinvariant_2d.jl"))
include(joinpath(@__DIR__, "..", "solver", "noncons_kernel_2d.jl"))

struct BreakingWavesSetup
    # Physical constants
    g::Float64       # gravity
    c_p::Float64     # heat capacity for constant pressure (dry air)
    c_v::Float64     # heat capacity for constant volume (dry air)
    gamma::Float64   # heat capacity ratio (dry air)
    R::Float64       # gas constant (dry air)
    p_0::Float64     # reference pressure
    T_0::Float64     # isothermal background temperature
    Nf::Float64      # Brunt-Vaisala frequency
    rho_0::Float64   # surface density
    p_s::Float64     # surface pressure
    H_rho::Float64   # density scale height
    u0::Float64      # background wind
    # Sponge layers
    z_B::Float64
    z_T::Float64
    xr_B::Float64
    xr_T::Float64
    alfa::Float64
    # Orography and domain
    h_c::Float64
    a_c::Float64
    L::Float64
    H::Float64
    function BreakingWavesSetup(
        alfa,
        xr_B;
        g = 10.0,
        R = 287.0,
        gamma = 1.0696864111498257,
        c_p = R * gamma / (gamma - 1),
        c_v = c_p - R,
        p_0 = 100_000.0,
        Nf = 0.01,
        rho_0 = 1.0,
        u0 = 10.0,
        z_B = 40_000.0,
        h_c = 2 * pi * 100.0,
        a_c = 1_000.0,
        L = 120_000.0,
        H = 60_000.0,
    )
        T_0 = g^2 / (c_p * Nf^2)
        p_s = rho_0 * R * T_0
        H_rho = R * T_0 / g
        new(g, c_p, c_v, gamma, R, p_0, T_0, Nf, rho_0, p_s, H_rho, u0,
            z_B, H, xr_B, L / 2, alfa, h_c, a_c, L, H)
    end
end

@inline function (boundary_condition::Trixi.BoundaryConditionDirichlet)(
    u_inner,
    normal_direction::AbstractVector,
    x,
    t,
    surface_flux_function::typeof(flux_surface_combined_entropy_stable),
    equations,
)
    # get the external value of the solution
    u_boundary = boundary_condition.boundary_value_function(x, t, equations)

    # Calculate boundary flux
    flux, _ = surface_flux_function(u_inner, u_boundary, normal_direction, equations)
    return flux
end

@inline function rayleigh_damping(x, z_B, z_T, alfa, xr_B, xr_T)
    if x[2] <= z_B
        S_v = 0.0
    else
        S_v = -alfa * sinpi(0.5 * (x[2] - z_B) / (z_T - z_B))^2
    end
    if x[1] < xr_B
        S_h1 = 0.0
    else
        S_h1 = -alfa * sinpi(0.5 * (x[1] - xr_B) / (xr_T - xr_B))^2
    end

    if x[1] > -xr_B
        S_h2 = 0.0
    else
        S_h2 = -alfa * sinpi(0.5 * (x[1] + xr_B) / (-xr_T + xr_B))^2
    end
    return S_v, S_h1, S_h2
end

function (setup::BreakingWavesSetup)(
    u,
    x,
    t,
    equations::CompressibleEulerVectorInvariantEquations2D,
)
    @unpack g, c_p, R, p_0, T_0, p_s, H_rho, u0, z_B, z_T, xr_B, xr_T, alfa = setup

    rho, v1, v2, rho_theta, _ = u

    theta = rho_theta / rho

    S_v, S_h1, S_h2 = rayleigh_damping(x, z_B, z_T, alfa, xr_B, xr_T)

    p_bg = p_s * exp(-x[2] / H_rho)
    theta_bg = T_0 * (p_0 / p_bg)^(R / c_p)

    du2 = (v1 - u0) * (S_v + S_h1 + S_h2)
    du3 = v2 * (S_v + S_h1 + S_h2)
    du4 = rho * (theta - theta_bg) * (S_v + S_h1 + S_h2)

    return SVector(zero(eltype(u)), du2, du3, du4, zero(eltype(u)))
end

function (setup::BreakingWavesSetup)(
    x,
    t,
    equations::CompressibleEulerVectorInvariantEquations2D,
)
    @unpack g, p_s, H_rho, rho_0, u0 = setup

    # isothermal background, hydrostatic in x[2]
    p = p_s * exp(-x[2] / H_rho)
    rho = rho_0 * exp(-x[2] / H_rho)
    v1 = u0
    v2 = 0.0

    return prim2cons(SVector(rho, v1, v2, p, g * x[2]), equations)
end

function main(
    equations,
    surface_flux,
    volume_flux,
    T,
    cells_per_dimension,
    polydeg,
    alfa,
    xr_B,
)

    breaking_waves_setup = BreakingWavesSetup(alfa, xr_B)

    boundary = BoundaryConditionDirichlet(breaking_waves_setup)

    boundary_conditions = (y_neg = boundary_condition_slip_wall, y_pos = boundary)

    basis = LobattoLegendreBasis(polydeg)

    volume_integral = VolumeIntegralFluxDifferencing(volume_flux)

    solver = DGSEM(basis, surface_flux, volume_integral)

    @unpack h_c, a_c, L, H = breaking_waves_setup

    zb(x) = h_c / (1 + (x / a_c)^2)
    f1(s) = SVector(-L / 2, zb(-L / 2) + (H - zb(-L / 2)) * 0.5 * (s + 1))
    f2(s) = SVector(L / 2, zb(L / 2) + (H - zb(L / 2)) * 0.5 * (s + 1))
    f3(s) = SVector(s * L / 2, zb(s * L / 2))
    f4(s) = SVector(s * L / 2, H)

    mesh = P4estMesh(
        cells_per_dimension,
        polydeg = polydeg,
        faces = (f1, f2, f3, f4),
        initial_refinement_level = 0,
        periodicity = (true, false),
    )

    semi = SemidiscretizationHyperbolic(
        mesh,
        equations,
        breaking_waves_setup,
        solver,
        source_terms = breaking_waves_setup,
        boundary_conditions = boundary_conditions,
    )

    ###############################################################################
    # ODE solvers, callbacks etc.

    tspan = (0.0, T * 3600.0)
    ode = semidiscretize(semi, tspan)

    summary_callback = SummaryCallback()

    analysis_interval = 1000

    analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

    alive_callback = AliveCallback(analysis_interval = analysis_interval)

    callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback)

    ###############################################################################
    # run the simulation
    tol = 1e-6
    sol = solve(
        ode,
        RDPK3SpFSAL49(thread = Trixi.Threaded());
        abstol = tol,
        reltol = tol,
        maxiters = 1.0e7,
        save_everystep = false,
        saveat = 1800.0,
        callback = callbacks,
    )
    return sol, semi
end

function run_breaking_waves(T, cells_per_dimension, polydeg, alfa, xr_B)
    breaking_waves_setup = BreakingWavesSetup(alfa, xr_B)

    surface_flux = flux_surface_combined_entropy_stable
    equations = CompressibleEulerVectorInvariantEquations2D(
        c_p = breaking_waves_setup.c_p,
        c_v = breaking_waves_setup.c_v,
        gravity = breaking_waves_setup.g,
    )
    volume_flux = flux_volume_combined_turbo_entropy_conservative
    sol, semi = main(
        equations,
        surface_flux,
        volume_flux,
        T,
        cells_per_dimension,
        polydeg,
        alfa,
        xr_B,
    )
end

T = 3.0
cells_per_dimension = (50, 30)
polydeg = 4
sol, semi = run_breaking_waves(T, cells_per_dimension, polydeg, 1 / 600, 40_000.0)

# globals used by plot_theta.jl
equations = semi.equations
breaking_waves_setup = BreakingWavesSetup(1 / 600, 40_000.0)
L = breaking_waves_setup.L
H = breaking_waves_setup.H

include(joinpath(@__DIR__, "..", "breaking_waves", "plot_theta.jl"))