using OrdinaryDiffEqLowStorageRK
using CSV, DataFrames
using Trixi
using CairoMakie

include(joinpath(@__DIR__, "..", "equations", "compressible_euler_vectorinvariant_2d.jl"))
include(joinpath(@__DIR__, "..", "solver", "noncons_kernel_2d.jl"))
include(joinpath(@__DIR__, "..", "contours.jl"))

struct HydrostaticSetup
    # Physical constants
    g::Float64       # gravity of earth
    c_p::Float64     # heat capacity for constant pressure (dry air)
    c_v::Float64     # heat capacity for constant volume (dry air)
    gamma::Float64   # heat capacity ratio (dry air)
    p_0::Float64     # atmospheric pressure
    T_0::Float64     # 
    u0::Float64      #
    Nf::Float64      # 
    z_B::Float64     #
    z_T::Float64     #
    alfa::Float64    #
    xr_B::Float64
    form1::Bool
    form2::Bool
    function HydrostaticSetup(
        alfa,
        xr_B,
        form1,
        form2;
        g = 9.81,
        c_p = 1004.0,
        c_v = 717.0,
        gamma = c_p / c_v,
        p_0 = 100_000.0,
        T_0 = 250.0,
        u0 = 20.0,
        z_B = 15000.0,
        z_T = 30000.0,
    )
        Nf = g / sqrt(c_p * T_0)
        new(g, c_p, c_v, gamma, p_0, T_0, u0, Nf, z_B, z_T, alfa, xr_B, form1, form2)
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

@inline function rayleigh_damping(x, z_B, z_T, alfa, xr_B)
    xr_T = 120000.0

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

function (setup::HydrostaticSetup)(
    u,
    x,
    t,
    equations::CompressibleEulerVectorInvariantEquations2D,
)
    @unpack g, c_p, c_v, gamma, p_0, T_0, z_B, z_T, Nf, u0, alfa, xr_B = setup

    rho, v1, v2, rho_theta, _ = u

    R = c_p - c_v

    theta = rho_theta / rho

    S_v, S_h1, S_h2 = rayleigh_damping(x, z_B, z_T, alfa, xr_B)

    exner = exp(-Nf^2 / g * x[2])

    theta_0 = T_0 / exner

    du2 = (v1 - u0) * (S_v + S_h1 + S_h2)
    du3 = v2 * (S_v + S_h1 + S_h2)
    du4 = rho * (theta - theta_0) * (S_v + S_h1 + S_h2)

    return SVector(zero(eltype(u)), du2, du3, du4, zero(eltype(u)))
end

function (setup::HydrostaticSetup)(
    x,
    t,
    equations::CompressibleEulerVectorInvariantEquations2D,
)
    @unpack g, c_p, c_v, p_0, T_0, u0, Nf = setup

    # Exner pressure, solves hydrostatic equation for x[2]
    exner = exp(-Nf^2 / g * x[2])
    # pressure
    p_0 = 100_000.0  # reference pressure
    R = c_p - c_v    # gas constant (dry air)
    p = p_0 * exner^(c_p / R)

    # density
    rho = p / (R * T_0)
    v1 = u0
    v2 = 0.0

    return prim2cons(SVector(rho, v1, v2, p, g * x[2]), equations)
end

function integrate_over_line(sol, semi, cells_per_dimension, polydeg)

    @unpack solver, mesh, cache = semi
    @unpack weights = solver.basis

    u = Trixi.wrap_array(sol.u[end], semi)
    u0 = Trixi.wrap_array(sol.u[1], semi)
    m = similar(u[1, :, :, :])
    rho = similar(u[1, :, :, :])
    up = similar(u[1, :, :, :])
    wp = similar(u[1, :, :, :])

    rho .= u0[1, :, :, :]          # density
    up .= u[2, :, :, :] - u0[2, :, :, :]  # u perturbations
    wp .= u[3, :, :, :] - u0[3, :, :, :]  # w perturbations

    m .= rho .* up .* wp  # integrand

    # Define Surface reference values
    rhos_us_R = u0[2, 1, 1, Int(cells_per_dimension[1] / 2 + 1)]
    rhos_us_L = u0[2, end, 1, Int(cells_per_dimension[1] / 2)]
    rhos_us = 0.5f0 * (rhos_us_L + rhos_us_R)
    @show rhos_us
    R = 287
    p0 = 100000
    rhos_us = p0 / (R * 250) * 20
    N = 9.81 / sqrt(1004.5 * 250)
    @show rhos_us
    integral = zeros(cells_per_dimension[2] * (polydeg + 1))
    z_coords = copy(integral)
    ## Compute the integral
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
                if j == 0
                    j = polydeg + 1
                end
                jacobian = semi.cache.elements.jacobian_matrix[1, 1, i, j, element] # dx/deta
                jacobian = 240000 / cells_per_dimension[1] / 2
                loc_integral += jacobian * weights[i] * m[i, j, element]
            end


        end
        j = z % (polydeg + 1)
        if j == 0
            j = polydeg + 1
        end
        z_coords[z] = cache.elements.node_coordinates[
            2,
            1,
            j,
            Int(iterator[end] - cells_per_dimension[1] / 2 + 1),
        ]
        integral[z] = loc_integral
    end
    integral .= integral ./ (-pi / 4 * rhos_us * N)
    return integral, z_coords
end

function main(
    equations,
    surface_flux,
    volume_flux,
    T,
    filename,
    cells_per_dimension,
    polydeg,
    alfa,
    xr_B,
    form1,
)

    linear_hydrostatic_setup = HydrostaticSetup(alfa, xr_B, form1, !form1)

    boundary = BoundaryConditionDirichlet(linear_hydrostatic_setup)

    boundary_conditions = (
        x_neg = boundary,
        x_pos = boundary,
        y_neg = boundary_condition_slip_wall,
        y_pos = boundary,
    )

    basis = LobattoLegendreBasis(polydeg)

    volume_integral = VolumeIntegralFluxDifferencing(volume_flux)

    solver = DGSEM(basis, surface_flux, volume_integral)

    a = 10000.0
    L = 240000.0
    H = 30000.0
    peak = 1.0
    y_b = peak / (1 + (L / 2 / a)^2)
    alfa_b = (H - y_b) * 0.5

    f1(s) = SVector(-L / 2, y_b + alfa_b * (s + 1))
    f2(s) = SVector(L / 2, y_b + alfa_b * (s + 1))
    f3(s) = SVector((s + 1 - 1) * L / 2, peak / (1 + ((s + 1 - 1) * L / 2)^2 / a^2))
    f4(s) = SVector((s + 1 - 1) * L / 2, H)

    mesh = P4estMesh(
        cells_per_dimension,
        polydeg = polydeg,
        faces = (f1, f2, f3, f4),
        initial_refinement_level = 0,
        periodicity = (false, false),
    )

    semi = SemidiscretizationHyperbolic(
        mesh,
        equations,
        linear_hydrostatic_setup,
        solver,
        source_terms = linear_hydrostatic_setup,
        boundary_conditions = boundary_conditions,
    )

    ###############################################################################
    # ODE solvers, callbacks etc.

    tspan = (0.0, T * 3600.0)  # 1000 seconds final time
    ode = semidiscretize(semi, tspan)

    summary_callback = SummaryCallback()

    analysis_interval = 1000

    analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

    alive_callback = AliveCallback(analysis_interval = analysis_interval)

    callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback)

    ###############################################################################
    # run the simulation
    sol = solve(
        ode,
        #	SSPRK43(),
        RDPK3SpFSAL49(thread = Trixi.Threaded());
        maxiters = 1.0e7,
        dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
        save_everystep = false,
        callback = callbacks,
    )

    verticalmomentum, z_coords =
        integrate_over_line(sol, semi, cells_per_dimension, polydeg)
    data = DataFrame(z_coords = z_coords, verticalmomentum = verticalmomentum)
    CSV.write(
        joinpath(@__DIR__, "..") *
        "/results/" *
        filename *
        "_" *
        string(T) *
        "_" *
        string(cells_per_dimension[1]) *
        "x" *
        string(cells_per_dimension[2]) *
        "_$(polydeg)_$(alfa)_$(xr_B)_$(form1).csv",
        data,
    )
    return sol, semi
end

T =  12.5
cells_per_dimension = (100, 60)
 polydeg = 3
  alfa = 0.035
   xr_B = 60000
    form1 = false
    linear_hydrostatic_setup = HydrostaticSetup(alfa, xr_B, form1, !form1)

    surface_flux = flux_surface_combined_entropy_stable
    equations = CompressibleEulerVectorInvariantEquations2D(c_p = 1004, c_v = 717, gravity = 9.81)
    volume_flux = flux_volume_combined_turbo_entropy_conservative
    filename = "DataContour"
    sol, semi = main(
        equations,
        surface_flux,
        volume_flux,
        T,
        filename,
        cells_per_dimension,
        polydeg,
        alfa,
        xr_B,
        form1)

x, y, data = ContourData(sol.u[end], semi, cells_per_dimension, equations)

include(joinpath(@__DIR__, "..", "mountain_contour_utils.jl"))
limits = ((-40, 40), (0, 12))
base = joinpath(@__DIR__, "..") * "/results/contour_linear_hydrostatic_$(cells_per_dimension[1])x$(cells_per_dimension[2])_CFL1_polydeg$(polydeg)"
save_contour_fields(base * ".jld2", x, y, data;
                    cells_per_dimension = cells_per_dimension, polydeg = polydeg,
                    T = T, limits = limits, case = "linear_hydrostatic")
plot_uw_contour(x, y, data, limits, base * ".pdf")