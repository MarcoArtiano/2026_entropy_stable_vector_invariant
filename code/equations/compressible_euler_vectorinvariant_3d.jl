using Trixi
using Trixi: AbstractCompressibleEulerEquations, @muladd, norm
import Trixi:
    varnames,
    cons2cons,
    cons2prim,
    cons2entropy,
    entropy,
    FluxLMARS,
    boundary_condition_slip_wall,
    flux,
    max_abs_speeds,
    max_abs_speed,
    max_abs_speed_naive,
    have_nonconservative_terms,
    True,
    prim2cons,
    False
# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

struct CompressibleEulerVectorInvariantEquations3D{RealT<:Real} <:
       AbstractCompressibleEulerEquations{3,6}
    p_0::RealT # reference pressure in Pa
    c_p::RealT # specific heat at constant pressure in J/(kg K)
    c_v::RealT # specific heat at constant volume in J/(kg K)
    g::RealT # gravitational acceleration in m/s²
    R::RealT # gas constant in J/(kg K)
    gamma::RealT # ratio of specific heats
    inv_gamma_minus_one::RealT # = inv(gamma - 1); can be used to write slow divisions as fast multiplications
    K::RealT # = p_0 * (R / p_0)^gamma; scaling factor between pressure and weighted potential temperature
    stolarsky_factor::RealT # = (gamma - 1) / gamma; used in the stolarsky mean
    function CompressibleEulerVectorInvariantEquations3D(; c_p, c_v, gravity)
        c_p, c_v, g = promote(c_p, c_v, gravity)
        p_0 = 100_000
        R = c_p - c_v
        gamma = c_p / c_v
        inv_gamma_minus_one = inv(gamma - 1)
        K = p_0 * (R / p_0)^gamma
        stolarsky_factor = (gamma - 1) / gamma
        return new{typeof(c_p)}(
            p_0,
            c_p,
            c_v,
            g,
            R,
            gamma,
            inv_gamma_minus_one,
            K,
            stolarsky_factor,
        )
    end
end

function varnames(::typeof(cons2cons), ::CompressibleEulerVectorInvariantEquations3D)
    ("rho", "v1", "v2", "v3", "rho_theta", "phi")
end

have_nonconservative_terms(::CompressibleEulerVectorInvariantEquations3D) = True()

varnames(::typeof(cons2prim), ::CompressibleEulerVectorInvariantEquations3D) =
    ("rho", "v1", "v2", "v3", "p", "phi")


@inline function Trixi.boundary_condition_slip_wall(
    u_inner,
    normal_direction::AbstractVector,
    x,
    t,
    surface_flux_function,
    equations::CompressibleEulerVectorInvariantEquations3D,
)

    Trixi.boundary_condition_slip_wall(
        u_inner,
        normal_direction,
        x,
        t,
        surface_flux_function,
        equations,
        Trixi.combine_conservative_and_nonconservative_fluxes(
            surface_flux_function,
            equations,
        ),
    )
end

@inline function Trixi.boundary_condition_slip_wall(
    u_inner,
    normal_direction::AbstractVector,
    x,
    t,
    surface_flux_function,
    equations::CompressibleEulerVectorInvariantEquations3D,
    combine_conservative_and_nonconservative_fluxes::True,
)

    # normalize the outward pointing direction
    normal = normal_direction / norm(normal_direction)

    # compute the normal velocity
    u_normal = normal[1] * u_inner[2] + normal[2] * u_inner[3] + normal[3] * u_inner[4]

    # create the "external" boundary solution state
    u_boundary = SVector(
        u_inner[1],
        u_inner[2] - 2 * u_normal * normal[1],
        u_inner[3] - 2 * u_normal * normal[2],
        u_inner[4] - 2 * u_normal * normal[3],
        u_inner[5],
        u_inner[6],
    )

    #   calculate the boundary flux
    flux, _ = surface_flux_function(u_inner, u_boundary, normal_direction, equations)
    return flux
end

@inline function Trixi.boundary_condition_slip_wall(
    u_inner,
    normal_direction::AbstractVector,
    x,
    t,
    surface_flux_functions,
    equations::CompressibleEulerVectorInvariantEquations3D,
    combine_conservative_and_nonconservative_fluxes::False,
)
    surface_flux_function, nonconservative_flux_function = surface_flux_functions
    # normalize the outward pointing direction
    normal = normal_direction / norm(normal_direction)

    # compute the normal velocity
    u_normal = normal[1] * u_inner[2] + normal[2] * u_inner[3] + normal[3] * u_inner[4]

    # create the "external" boundary solution state
    u_boundary = SVector(
        u_inner[1],
        u_inner[2] - 2 * u_normal * normal[1],
        u_inner[3] - 2 * u_normal * normal[2],
        u_inner[4] - 2 * u_normal * normal[3],
        u_inner[5],
        u_inner[6],
    )

    #   calculate the boundary flux
    flux = surface_flux_function(u_inner, u_boundary, normal_direction, equations)
    noncons_flux =
        nonconservative_flux_function(u_inner, u_boundary, normal_direction, equations)

    return flux, noncons_flux
end


@inline function flux_surface_combined_entropy_stable(
    u_ll,
    u_rr,
    normal_direction::AbstractVector,
    equations::CompressibleEulerVectorInvariantEquations3D,
)
    rho_ll, v1_ll, v2_ll, v3_ll, rho_theta_ll, phi_ll = u_ll
    rho_rr, v1_rr, v2_rr, v3_rr, rho_theta_rr, phi_rr = u_rr

    _, _, _, _, exner_ll = cons2primexner(u_ll, equations)
    _, _, _, _, exner_rr = cons2primexner(u_rr, equations)

    theta_ll = rho_theta_ll / rho_ll
    theta_rr = rho_theta_rr / rho_rr

    v_dot_n_ll =
        v1_ll * normal_direction[1] +
        v2_ll * normal_direction[2] +
        v3_ll * normal_direction[3]
    v_dot_n_rr =
        v1_rr * normal_direction[1] +
        v2_rr * normal_direction[2] +
        v3_rr * normal_direction[3]

    rho_avg = 0.5f0 * (rho_ll + rho_rr)
    kin_avg =
        0.5f0 * (
            v1_ll * v1_ll +
            v2_ll * v2_ll +
            v3_ll * v3_ll +
            v1_rr * v1_rr +
            v2_rr * v2_rr +
            v3_rr * v3_rr
        )
    theta_avg = 0.5f0 * (theta_ll + theta_rr)
    v1_avg = 0.5f0 * (rho_ll * v1_ll + rho_rr * v1_rr) / rho_avg
    v2_avg = 0.5f0 * (rho_ll * v2_ll + rho_rr * v2_rr) / rho_avg
    v3_avg = 0.5f0 * (rho_ll * v3_ll + rho_rr * v3_rr) / rho_avg
    jump_v1 = v1_rr - v1_ll
    jump_v2 = v2_rr - v2_ll
    jump_v3 = v3_rr - v3_ll

    rho_v_ll =
        v1_ll * rho_ll * normal_direction[1] +
        v2_ll * rho_ll * normal_direction[2] +
        v3_ll * rho_ll * normal_direction[3]
    rho_v_rr =
        v1_rr * rho_rr * normal_direction[1] +
        v2_rr * rho_rr * normal_direction[2] +
        v3_rr * rho_rr * normal_direction[3]

    T_ll = theta_ll * exner_ll
    T_rr = theta_rr * exner_rr
    c_ll = sqrt(equations.gamma * equations.R * T_ll)
    c_rr = sqrt(equations.gamma * equations.R * T_rr)

    norm_ = norm(normal_direction)
    inv_norm_ = 1 / norm_
    c = 0.5f0 * (c_ll + c_rr)

    theta_mean = Trixi.inv_ln_mean(rho_ll / rho_theta_ll, rho_rr / rho_theta_rr)
    jump_theta = theta_rr - theta_ll
    test = 0.5f0 * min(theta_ll, theta_rr) * jump_theta / theta_avg
    theta_grad_exner = equations.c_p * theta_mean * (exner_rr - exner_ll)
    v_interface =
        0.5f0 * (v_dot_n_ll + v_dot_n_rr) - theta_grad_exner * norm_ / (2 * c) -
        (phi_rr - phi_ll) * norm_ / (2 * c)

    if v_interface > 0
        f1 = rho_ll * v_interface
        f5 = f1 * (theta_mean - test)
    else
        f1 = rho_rr * v_interface
        f5 = f1 * (theta_mean + test)
    end

    c_adv = 0.5f0 * abs(v_dot_n_ll + v_dot_n_rr) * inv_norm_
    diss1 =
        0.5f0 * c / rho_avg * (rho_v_rr - rho_v_ll) * normal_direction[1] * inv_norm_
    diss2 =
        0.5f0 * c / rho_avg * (rho_v_rr - rho_v_ll) * normal_direction[2] * inv_norm_
    diss3 =
        0.5f0 * c / rho_avg * (rho_v_rr - rho_v_ll) * normal_direction[3] * inv_norm_

    f2 =
        kin_avg * 0.5f0 * normal_direction[1] - diss1 -
        0.5f0 * c_adv / rho_avg * (rho_rr * v1_rr - rho_ll * v1_ll) * norm_
    f3 =
        kin_avg * 0.5f0 * normal_direction[2] - diss2 -
        0.5f0 * c_adv / rho_avg * (rho_rr * v2_rr - rho_ll * v2_ll) * norm_
    f4 =
        kin_avg * 0.5f0 * normal_direction[3] - diss3 -
        0.5f0 * c_adv / rho_avg * (rho_rr * v3_rr - rho_ll * v3_ll) * norm_

    vorticity_x =
        v2_avg * (jump_v1 * normal_direction[2] - jump_v2 * normal_direction[1]) +
        v3_avg * (jump_v1 * normal_direction[3] - jump_v3 * normal_direction[1])
    vorticity_y =
        v1_avg * (jump_v2 * normal_direction[1] - jump_v1 * normal_direction[2]) +
        v3_avg * (jump_v2 * normal_direction[3] - jump_v3 * normal_direction[2])
    vorticity_z =
        v1_avg * (jump_v3 * normal_direction[1] - jump_v1 * normal_direction[3]) +
        v2_avg * (jump_v3 * normal_direction[2] - jump_v2 * normal_direction[3])

    g2 = vorticity_x + theta_grad_exner * normal_direction[1]
    g3 = vorticity_y + theta_grad_exner * normal_direction[2]
    g4 = vorticity_z + theta_grad_exner * normal_direction[3]

    return SVector(
        f1,
        f2 + 0.5f0 * g2,
        f3 + 0.5f0 * g3,
        f4 + 0.5f0 * g4,
        f5,
        zero(eltype(u_ll)),
    ),
    SVector(
        f1,
        f2 - 0.5f0 * g2,
        f3 - 0.5f0 * g3,
        f4 - 0.5f0 * g4,
        f5,
        zero(eltype(u_ll)),
    )
end

@inline Trixi.combine_conservative_and_nonconservative_fluxes(
    ::typeof(flux_surface_combined_entropy_stable),
    equations::CompressibleEulerVectorInvariantEquations3D,
) = Trixi.True()

@inline function flux_surface_combined_entropy_conservative(
    u_ll,
    u_rr,
    normal_direction::AbstractVector,
    equations::CompressibleEulerVectorInvariantEquations3D,
)
    rho_ll, v1_ll, v2_ll, v3_ll, rho_theta_ll = u_ll
    rho_rr, v1_rr, v2_rr, v3_rr, rho_theta_rr = u_rr

    _, _, _, _, exner_ll = cons2primexner(u_ll, equations)
    _, _, _, _, exner_rr = cons2primexner(u_rr, equations)

    rho_avg = 0.5f0 * (rho_ll + rho_rr)
    kin_avg =
        0.5f0 * (
            v1_ll * v1_ll +
            v2_ll * v2_ll +
            v3_ll * v3_ll +
            v1_rr * v1_rr +
            v2_rr * v2_rr +
            v3_rr * v3_rr
        )
    v1_avg = 0.5f0 * (rho_ll * v1_ll + rho_rr * v1_rr) / rho_avg
    v2_avg = 0.5f0 * (rho_ll * v2_ll + rho_rr * v2_rr) / rho_avg
    v3_avg = 0.5f0 * (rho_ll * v3_ll + rho_rr * v3_rr) / rho_avg
    jump_v1 = v1_rr - v1_ll
    jump_v2 = v2_rr - v2_ll
    jump_v3 = v3_rr - v3_ll

    rho_v_ll =
        v1_ll * rho_ll * normal_direction[1] +
        v2_ll * rho_ll * normal_direction[2] +
        v3_ll * rho_ll * normal_direction[3]
    rho_v_rr =
        v1_rr * rho_rr * normal_direction[1] +
        v2_rr * rho_rr * normal_direction[2] +
        v3_rr * rho_rr * normal_direction[3]

    theta_mean = Trixi.inv_ln_mean(rho_ll / rho_theta_ll, rho_rr / rho_theta_rr)

    f1 = 0.5f0 * (rho_v_ll + rho_v_rr)
    f5 = f1 * theta_mean

    f2 = kin_avg * 0.5f0 * normal_direction[1]
    f3 = kin_avg * 0.5f0 * normal_direction[2]
    f4 = kin_avg * 0.5f0 * normal_direction[3]

    theta_grad_exner = equations.c_p * theta_mean * (exner_rr - exner_ll)

    vorticity_x =
        v2_avg * (jump_v1 * normal_direction[2] - jump_v2 * normal_direction[1]) +
        v3_avg * (jump_v1 * normal_direction[3] - jump_v3 * normal_direction[1])
    vorticity_y =
        v1_avg * (jump_v2 * normal_direction[1] - jump_v1 * normal_direction[2]) +
        v3_avg * (jump_v2 * normal_direction[3] - jump_v3 * normal_direction[2])
    vorticity_z =
        v1_avg * (jump_v3 * normal_direction[1] - jump_v1 * normal_direction[3]) +
        v2_avg * (jump_v3 * normal_direction[2] - jump_v2 * normal_direction[3])

    g2 = vorticity_x + theta_grad_exner * normal_direction[1]
    g3 = vorticity_y + theta_grad_exner * normal_direction[2]
    g4 = vorticity_z + theta_grad_exner * normal_direction[3]

    return SVector(
        f1,
        f2 + 0.5f0 * g2,
        f3 + 0.5f0 * g3,
        f4 + 0.5f0 * g4,
        f5,
        zero(eltype(u_ll)),
    ),
    SVector(
        f1,
        f2 - 0.5f0 * g2,
        f3 - 0.5f0 * g3,
        f4 - 0.5f0 * g4,
        f5,
        zero(eltype(u_ll)),
    )
end

@inline Trixi.combine_conservative_and_nonconservative_fluxes(
    ::typeof(flux_surface_combined_entropy_conservative),
    equations::CompressibleEulerVectorInvariantEquations3D,
) = Trixi.True()

@inline function flux_vorticity(
    u_ll,
    u_rr,
    normal_direction::AbstractVector,
    equations::CompressibleEulerVectorInvariantEquations3D,
)
    rho_ll, v1_ll, v2_ll, v3_ll, rho_theta_ll = u_ll
    rho_rr, v1_rr, v2_rr, v3_rr, rho_theta_rr = u_rr
		v1_avg = (v1_ll + v1_rr) * 0.5f0
		v2_avg = (v2_ll + v2_rr) * 0.5f0
		v3_avg = (v3_ll + v3_rr) * 0.5f0

	f2 = v3_avg * normal_direction[2] - v2_avg * normal_direction[3]
	f3 = v1_avg * normal_direction[3] - v3_avg * normal_direction[1]
	f4 = v2_avg * normal_direction[1] - v1_avg * normal_direction[2]
		f1 = zero(eltype(u_ll))
		f5 = f1


    return SVector(
        f1,
        f2,
        f3,
        f4,
        f5,
        zero(eltype(u_ll)),
    ),
    SVector(
        f1,
        f2,
        f3,
        f4,
        f5,
        zero(eltype(u_ll)),
    )
end

@inline Trixi.combine_conservative_and_nonconservative_fluxes(
    ::typeof(flux_vorticity),
    equations::CompressibleEulerVectorInvariantEquations3D,
) = Trixi.True()

@inline function flux_vorticity_rusanov(u_ll, u_rr, normal_direction::AbstractVector,
                                        equations::CompressibleEulerVectorInvariantEquations3D)
    f_ll, _ = flux_vorticity(u_ll, u_rr, normal_direction, equations)

    # wavespeed di Atum: |n⃗ · u⃗|
    v_ll = SVector(u_ll[2], u_ll[3], u_ll[4])
    v_rr = SVector(u_rr[2], u_rr[3], u_rr[4])
    λ = max(abs(dot(normal_direction, v_ll)), abs(dot(normal_direction, v_rr)))

    diss = -0.5f0 * λ * (u_rr - u_ll)
    f = f_ll + diss
    return f, f
end

@inline Trixi.combine_conservative_and_nonconservative_fluxes(
    ::typeof(flux_vorticity_rusanov),
    equations::CompressibleEulerVectorInvariantEquations3D) = Trixi.True()

@inline function max_abs_speed(
    u_ll,
    u_rr,
    orientation::Integer,
    equations::CompressibleEulerVectorInvariantEquations3D,
)
    rho_ll, v1_ll, v2_ll, v3_ll, p_ll = cons2prim(u_ll, equations)
    rho_rr, v1_rr, v2_rr, v3_rr, p_rr = cons2prim(u_rr, equations)

    # Get the velocity value in the appropriate direction
    if orientation == 1
        v_ll = v1_ll
        v_rr = v1_rr
    elseif orientation == 2
        v_ll = v2_ll
        v_rr = v2_rr
    else
        v_ll = v3_ll
        v_rr = v3_rr
    end
    # Calculate sound speeds
    c_ll = sqrt(equations.gamma * p_ll / rho_ll)
    c_rr = sqrt(equations.gamma * p_rr / rho_rr)

    return max(abs(v_ll) + c_ll, abs(v_rr) + c_rr)
end

# Convert conservative variables to primitive
@inline function cons2prim(u, equations::CompressibleEulerVectorInvariantEquations3D)
    rho, v1, v2, v3, rho_theta, phi = u

    p = equations.K * rho_theta^equations.gamma

    return SVector(rho, v1, v2, v3, p, phi)
end

# Convert primitive to conservative variables
@inline function prim2cons(prim, equations::CompressibleEulerVectorInvariantEquations3D)
    rho, v1, v2, v3, p, phi = prim
    rho_theta = (p / equations.p_0)^(1 / equations.gamma) * equations.p_0 / equations.R
    return SVector(rho, v1, v2, v3, rho_theta, phi)
end

@inline function density(u, equations::CompressibleEulerVectorInvariantEquations3D)
    rho = u[1]
    return rho
end

@inline function velocity(u, equations::CompressibleEulerVectorInvariantEquations3D)
    v1 = u[2]
    v2 = u[3]
    v3 = u[4]
    return SVector(v1, v2, v3)
end

@inline function pressure(u, equations::CompressibleEulerVectorInvariantEquations3D)
    rho, v1, v2, v3, rho_theta = u
    p = equations.K * rho_theta^equations.gamma
    return p
end

@inline function cons2primexner(
    u,
    equations::CompressibleEulerVectorInvariantEquations3D,
)

    rho, v1, v2, v3, rho_theta, phi = u

    exner = (rho_theta * equations.R / equations.p_0)^(equations.R / equations.c_v)
    return SVector(rho, v1, v2, v3, exner, phi)
end

@inline function exner_pressure(
    u,
    equations::CompressibleEulerVectorInvariantEquations3D,
)

    _, _, _, _, rho_theta = u

    exner = (rho_theta * equations.R / equations.p_0)^(equations.R / equations.c_v)
    return exner
end

@inline function cons2entropy(u, equations::CompressibleEulerVectorInvariantEquations3D)
    rho, v1, v2, v3, rho_theta = u

    w1 = -(log(equations.K * (rho_theta / rho)^equations.gamma) - equations.gamma) *
         equations.inv_gamma_minus_one
    w5 = -equations.gamma * rho / rho_theta * equations.inv_gamma_minus_one

    return SVector(w1, zero(eltype(u)), zero(eltype(u)), zero(eltype(u)), w5,
                   zero(eltype(u)))
end

@inline function cons2entropy_energy(u, equations::CompressibleEulerVectorInvariantEquations3D)
    rho, v1, v2, v3, rho_theta, phi = u

    w1 = 0.5f0 * (v1^2 + v2^2 + v3^2) + phi
    w2 = rho * v1
    w3 = rho * v2
    w4 = rho * v3
    w5 = equations.c_p * exner_pressure(u, equations)

    return SVector(w1, w2, w3, w4, w5, zero(eltype(u)))
end

@inline function Trixi.entropy(
    cons,
    equations::CompressibleEulerVectorInvariantEquations3D,
)
    p = equations.K * cons[5]^equations.gamma
    # Thermodynamic entropy
    s = log(p) - equations.gamma * log(cons[1])
    S = -s * cons[1] / (equations.gamma - 1)
    return S
end

@inline function Trixi.energy_total(
    cons,
    equations::CompressibleEulerVectorInvariantEquations3D,
)
    # Mathematical entropy
    p = equations.p_0 * (equations.R * cons[5] / equations.p_0)^equations.gamma

    U = (
        p / (equations.gamma - 1) +
        0.5f0 * (cons[2]^2 + cons[3]^2 + cons[4]^2) * cons[1]
    )

    return U
end

@inline function max_abs_speeds(
    u,
    equations::CompressibleEulerVectorInvariantEquations3D,
)
    rho, v1, v2, v3, p = cons2prim(u, equations)
    c = sqrt(equations.gamma * p / rho)

    return abs(v1) + c, abs(v2) + c, abs(v3) + c
end

end # @muladd
