using Trixi
using Trixi:
    True,
    False,
    PassiveTracerEquations,
    flow_variables,
    tracers,
    ntracers,
    @threaded,
    @turbo,
    PtrArray,
    StrideArray,
    StaticInt,
    indices

@inline Trixi.density(u, ::CompressibleEulerVectorInvariantEquations2D) = u[1]

@inline function Trixi.velocity(
    u,
    normal_direction::AbstractVector,
    ::CompressibleEulerVectorInvariantEquations2D,
)
    return u[2] * normal_direction[1] + u[3] * normal_direction[2]
end

@inline Trixi.velocity(u, ::CompressibleEulerVectorInvariantEquations2D) =
    SVector(u[2], u[3])

struct FluxTracerCentralCombined{FlowFlux}
    flow_flux::FlowFlux
end

@inline function (f::FluxTracerCentralCombined)(
    u_ll,
    u_rr,
    normal_direction::AbstractVector,
    equations::PassiveTracerEquations,
)
    flow_equations = equations.flow_equations
    u_flow_ll = flow_variables(u_ll, equations)
    u_flow_rr = flow_variables(u_rr, equations)

    flux_left, flux_right = f.flow_flux(u_flow_ll, u_flow_rr, normal_direction,
        flow_equations)

    f_rho = flux_left[1]
    chi_ll = tracers(u_ll, equations)
    chi_rr = tracers(u_rr, equations)
    f_tracer = f_rho * (0.5f0 * (chi_ll + chi_rr))

    return SVector(flux_left..., f_tracer...), SVector(flux_right..., f_tracer...)
end

struct FluxTracerUpwindCombined{FlowFlux}
    flow_flux::FlowFlux
end

@inline function (f::FluxTracerUpwindCombined)(
    u_ll,
    u_rr,
    normal_direction::AbstractVector,
    equations::PassiveTracerEquations,
)
    flow_equations = equations.flow_equations
    u_flow_ll = flow_variables(u_ll, equations)
    u_flow_rr = flow_variables(u_rr, equations)

    flux_left, flux_right = f.flow_flux(u_flow_ll, u_flow_rr, normal_direction,
        flow_equations)

    f_rho = flux_left[1]
    chi_ll = tracers(u_ll, equations)
    chi_rr = tracers(u_rr, equations)
    f_tracer = f_rho *
               (0.5f0 * (chi_ll + chi_rr) - 0.5f0 * sign(f_rho) * (chi_rr - chi_ll))

    return SVector(flux_left..., f_tracer...), SVector(flux_right..., f_tracer...)
end

@inline Trixi.combine_conservative_and_nonconservative_fluxes(
    ::FluxTracerCentralCombined,
    ::PassiveTracerEquations,
) = True()

@inline Trixi.combine_conservative_and_nonconservative_fluxes(
    ::FluxTracerUpwindCombined,
    ::PassiveTracerEquations,
) = True()

@inline function Trixi.boundary_condition_slip_wall(
    u_inner,
    normal_direction::AbstractVector,
    x,
    t,
    surface_flux_function::FluxTracerUpwindCombined,
    equations::PassiveTracerEquations,
)
    normal = normal_direction / Trixi.norm(normal_direction)
    u_normal = normal[1] * u_inner[2] + normal[2] * u_inner[3]

    u_boundary = SVector(
        u_inner[1],
        u_inner[2] - 2 * u_normal * normal[1],
        u_inner[3] - 2 * u_normal * normal[2],
        ntuple(@inline(v -> u_inner[3+v]), Val(nvariables(equations) - 3))...,
    )

    flux, _ = surface_flux_function(u_inner, u_boundary, normal_direction, equations)
    return flux
end

@inline function (boundary_condition::Trixi.BoundaryConditionDirichlet)(
    u_inner,
    normal_direction::AbstractVector,
    x,
    t,
    surface_flux_function::FluxTracerUpwindCombined,
    equations::PassiveTracerEquations,
)
    u_boundary = boundary_condition.boundary_value_function(x, t, equations)
    flux, _ = surface_flux_function(u_inner, u_boundary, normal_direction, equations)
    return flux
end

@inline function Trixi.flux_differencing_kernel!(
    _du::PtrArray,
    u_cons::PtrArray,
    element,
    mesh::Type{<:Union{StructuredMesh{2},UnstructuredMesh2D,P4estMesh{2}}},
    have_nonconservative_terms::True,
    combine_conservative_and_nonconservative_fluxes::False,
    equations::PassiveTracerEquations,
    volume_flux::typeof(flux_volume_combined_turbo_entropy_conservative),
    dg::DGSEM,
    cache,
    alpha,
)

    @unpack derivative_split = dg.basis
    @unpack contravariant_vectors = cache.elements
    flow_equations = equations.flow_equations
    n_tracers = ntracers(equations)

    du = StrideArray{eltype(u_cons)}(
        undef,
        (
            ntuple(_ -> StaticInt(nnodes(dg)), ndims(mesh))...,
            StaticInt(nvariables(equations)),
        ),
    )

    # Primitive variables: rho, v1, v2, theta, exner, phi, log(theta), chi_1, ..., chi_N.
    # The total count is nvariables(equations) + 2 = 7 + n_tracers.
    u_prim = StrideArray{eltype(u_cons)}(
        undef,
        (
            ntuple(_ -> StaticInt(nnodes(dg)), ndims(mesh))...,
            StaticInt(nvariables(equations) + 2),
        ),
    )

    @turbo for j in eachnode(dg), i in eachnode(dg)
        rho = u_cons[1, i, j, element]
        v1 = u_cons[2, i, j, element]
        v2 = u_cons[3, i, j, element]
        rho_theta = u_cons[4, i, j, element]
        phi = u_cons[5, i, j, element]

        exner =
            (rho_theta * flow_equations.R / flow_equations.p_0)^(flow_equations.R / flow_equations.c_v)
        theta = rho_theta / rho
        u_prim[i, j, 1] = rho
        u_prim[i, j, 2] = v1
        u_prim[i, j, 3] = v2
        u_prim[i, j, 4] = theta
        u_prim[i, j, 5] = exner
        u_prim[i, j, 6] = phi
        u_prim[i, j, 7] = log(theta)
    end

    @turbo for j in eachnode(dg), i in eachnode(dg)
        rho = u_cons[1, i, j, element]
        for t = 1:n_tracers
            u_prim[i, j, 7+t] = u_cons[5+t, i, j, element] / rho
        end
    end

    # x direction
    du_permuted = StrideArray{eltype(u_cons)}(
        undef,
        (StaticInt(nnodes(dg)), StaticInt(nnodes(dg)), StaticInt(nvariables(equations))),
    )

    u_prim_permuted = StrideArray{eltype(u_cons)}(
        undef,
        (
            StaticInt(nnodes(dg)),
            StaticInt(nnodes(dg)),
            StaticInt(nvariables(equations) + 2),
        ),
    )

    @turbo for v in indices(u_prim, 3), j in eachnode(dg), i in eachnode(dg)
        u_prim_permuted[j, i, v] = u_prim[i, j, v]
    end
    fill!(du_permuted, zero(eltype(du_permuted)))

    contravariant_vectors_x = StrideArray{eltype(contravariant_vectors)}(
        undef,
        (StaticInt(nnodes(dg)), StaticInt(nnodes(dg)), StaticInt(ndims(mesh))),
    )

    @turbo for j in eachnode(dg), i in eachnode(dg)
        contravariant_vectors_x[j, i, 1] = contravariant_vectors[1, 1, i, j, element]
        contravariant_vectors_x[j, i, 2] = contravariant_vectors[2, 1, i, j, element]
    end

    for i in eachnode(dg), ii ∈ (i+1):nnodes(dg)
        @turbo for j in eachnode(dg)
            rho_ll = u_prim_permuted[j, i, 1]
            v1_ll = u_prim_permuted[j, i, 2]
            v2_ll = u_prim_permuted[j, i, 3]
            theta_ll = u_prim_permuted[j, i, 4]
            exner_ll = u_prim_permuted[j, i, 5]
            phi_ll = u_prim_permuted[j, i, 6]
            log_theta_ll = u_prim_permuted[j, i, 7]

            rho_rr = u_prim_permuted[j, ii, 1]
            v1_rr = u_prim_permuted[j, ii, 2]
            v2_rr = u_prim_permuted[j, ii, 3]
            theta_rr = u_prim_permuted[j, ii, 4]
            exner_rr = u_prim_permuted[j, ii, 5]
            phi_rr = u_prim_permuted[j, ii, 6]
            log_theta_rr = u_prim_permuted[j, ii, 7]

            normal_direction_1 =
                0.5f0 *
                (contravariant_vectors_x[j, i, 1] + contravariant_vectors_x[j, ii, 1])
            normal_direction_2 =
                0.5f0 *
                (contravariant_vectors_x[j, i, 2] + contravariant_vectors_x[j, ii, 2])

            rho_v_dot_n_ll =
                rho_ll * (v1_ll * normal_direction_1 + v2_ll * normal_direction_2)
            rho_v_dot_n_rr =
                rho_rr * (v1_rr * normal_direction_1 + v2_rr * normal_direction_2)
            rho_avg = 0.5f0 * (rho_ll + rho_rr)
            v1_avg = 0.5f0 * (rho_ll * v1_ll + rho_rr * v1_rr) / rho_avg
            v2_avg = 0.5f0 * (rho_ll * v2_ll + rho_rr * v2_rr) / rho_avg

            kin_avg =
                0.5f0 * (v1_rr * v1_rr + v2_rr * v2_rr + v1_ll * v1_ll + v2_ll * v2_ll)

            # Algebraically equivalent to `inv_ln_mean(1 / theta_ll, 1 / theta_rr)`
            x2 = theta_rr
            log_x2 = log_theta_rr
            y2 = theta_ll
            log_y2 = log_theta_ll
            x2_plus_y2 = x2 + y2
            y2_minus_x2 = y2 - x2
            z2 = y2_minus_x2^2 / x2_plus_y2^2
            special_path2 = (2 + z2 * (2 / 3 + z2 * (2 / 5 + 2 / 7 * z2))) / x2_plus_y2
            regular_path2 = (log_y2 - log_x2) / y2_minus_x2
            theta_mean =
                theta_ll * theta_rr * ifelse(z2 < 1.0e-4, special_path2, regular_path2)

            f1 = 0.5f0 * (rho_v_dot_n_ll + rho_v_dot_n_rr)
            f2 = kin_avg * 0.5f0 * normal_direction_1
            f3 = kin_avg * 0.5f0 * normal_direction_2
            f4 = f1 * theta_mean

            jump_v1 = v1_rr - v1_ll
            jump_v2 = v2_rr - v2_ll
            gravity = phi_rr - phi_ll
            vorticity_x =
                v2_avg * jump_v1 * normal_direction_2 -
                v2_avg * jump_v2 * normal_direction_1
            vorticity_y =
                v1_avg * jump_v2 * normal_direction_1 -
                v1_avg * jump_v1 * normal_direction_2
            theta_grad_exner = flow_equations.c_p * theta_mean * (exner_rr - exner_ll)
            g2 = vorticity_x + (theta_grad_exner + gravity) * normal_direction_1
            g3 = vorticity_y + (theta_grad_exner + gravity) * normal_direction_2

            factor_i = alpha * derivative_split[i, ii]
            du_permuted[j, i, 1] += factor_i * f1
            du_permuted[j, i, 2] += factor_i * (f2 + 0.5f0 * g2)
            du_permuted[j, i, 3] += factor_i * (f3 + 0.5f0 * g3)
            du_permuted[j, i, 4] += factor_i * f4

            factor_ii = alpha * derivative_split[ii, i]
            du_permuted[j, ii, 1] += factor_ii * f1
            du_permuted[j, ii, 2] += factor_ii * (f2 - 0.5f0 * g2)
            du_permuted[j, ii, 3] += factor_ii * (f3 - 0.5f0 * g3)
            du_permuted[j, ii, 4] += factor_ii * f4

            for t = 1:n_tracers
                chi_avg = 0.5f0 * (u_prim_permuted[j, i, 7+t] + u_prim_permuted[j, ii, 7+t])
                f_tracer = f1 * chi_avg
                du_permuted[j, i, 5+t] += factor_i * f_tracer
                du_permuted[j, ii, 5+t] += factor_ii * f_tracer
            end
        end
    end

    @turbo for v in eachvariable(equations), j in eachnode(dg), i in eachnode(dg)
        du[i, j, v] = du_permuted[j, i, v]
    end

    # y direction
    contravariant_vectors_y = StrideArray{eltype(contravariant_vectors)}(
        undef,
        (StaticInt(nnodes(dg)), StaticInt(nnodes(dg)), StaticInt(ndims(mesh))),
    )

    @turbo for j in eachnode(dg), i in eachnode(dg)
        contravariant_vectors_y[i, j, 1] = contravariant_vectors[1, 2, i, j, element]
        contravariant_vectors_y[i, j, 2] = contravariant_vectors[2, 2, i, j, element]
    end

    for j in eachnode(dg), jj ∈ (j+1):nnodes(dg)
        @turbo for i in eachnode(dg)
            rho_ll = u_prim[i, j, 1]
            v1_ll = u_prim[i, j, 2]
            v2_ll = u_prim[i, j, 3]
            theta_ll = u_prim[i, j, 4]
            exner_ll = u_prim[i, j, 5]
            phi_ll = u_prim[i, j, 6]
            log_theta_ll = u_prim[i, j, 7]

            rho_rr = u_prim[i, jj, 1]
            v1_rr = u_prim[i, jj, 2]
            v2_rr = u_prim[i, jj, 3]
            theta_rr = u_prim[i, jj, 4]
            exner_rr = u_prim[i, jj, 5]
            phi_rr = u_prim[i, jj, 6]
            log_theta_rr = u_prim[i, jj, 7]

            normal_direction_1 =
                0.5f0 *
                (contravariant_vectors_y[i, j, 1] + contravariant_vectors_y[i, jj, 1])
            normal_direction_2 =
                0.5f0 *
                (contravariant_vectors_y[i, j, 2] + contravariant_vectors_y[i, jj, 2])

            rho_v_dot_n_ll =
                rho_ll * v1_ll * normal_direction_1 + rho_ll * v2_ll * normal_direction_2
            rho_v_dot_n_rr =
                rho_rr * v1_rr * normal_direction_1 + rho_rr * v2_rr * normal_direction_2
            rho_avg = 0.5f0 * (rho_ll + rho_rr)

            v1_avg = 0.5f0 * (rho_ll * v1_ll + rho_rr * v1_rr) / rho_avg
            v2_avg = 0.5f0 * (rho_ll * v2_ll + rho_rr * v2_rr) / rho_avg
            kin_avg =
                0.5f0 * (v1_rr * v1_rr + v2_rr * v2_rr + v1_ll * v1_ll + v2_ll * v2_ll)

            x2 = theta_rr
            log_x2 = log_theta_rr
            y2 = theta_ll
            log_y2 = log_theta_ll
            x2_plus_y2 = x2 + y2
            y2_minus_x2 = y2 - x2
            z2 = y2_minus_x2^2 / x2_plus_y2^2
            special_path2 = (2 + z2 * (2 / 3 + z2 * (2 / 5 + 2 / 7 * z2))) / x2_plus_y2
            regular_path2 = (log_y2 - log_x2) / y2_minus_x2
            theta_mean =
                theta_ll * theta_rr * ifelse(z2 < 1.0e-4, special_path2, regular_path2)

            f1 = 0.5f0 * (rho_v_dot_n_ll + rho_v_dot_n_rr)
            f2 = kin_avg * 0.5f0 * normal_direction_1
            f3 = kin_avg * 0.5f0 * normal_direction_2
            f4 = f1 * theta_mean

            jump_v1 = v1_rr - v1_ll
            jump_v2 = v2_rr - v2_ll
            gravity = phi_rr - phi_ll
            vorticity_x =
                v2_avg * jump_v1 * normal_direction_2 -
                v2_avg * jump_v2 * normal_direction_1
            vorticity_y =
                v1_avg * jump_v2 * normal_direction_1 -
                v1_avg * jump_v1 * normal_direction_2
            theta_grad_exner = flow_equations.c_p * theta_mean * (exner_rr - exner_ll)
            g2 = vorticity_x + (theta_grad_exner + gravity) * normal_direction_1
            g3 = vorticity_y + (theta_grad_exner + gravity) * normal_direction_2

            factor_j = alpha * derivative_split[j, jj]
            du[i, j, 1] += factor_j * f1
            du[i, j, 2] += factor_j * (f2 + 0.5f0 * g2)
            du[i, j, 3] += factor_j * (f3 + 0.5f0 * g3)
            du[i, j, 4] += factor_j * f4

            factor_jj = alpha * derivative_split[jj, j]
            du[i, jj, 1] += factor_jj * f1
            du[i, jj, 2] += factor_jj * (f2 - 0.5f0 * g2)
            du[i, jj, 3] += factor_jj * (f3 - 0.5f0 * g3)
            du[i, jj, 4] += factor_jj * f4

            for t = 1:n_tracers
                chi_avg = 0.5f0 * (u_prim[i, j, 7+t] + u_prim[i, jj, 7+t])
                f_tracer = f1 * chi_avg
                du[i, j, 5+t] += factor_j * f_tracer
                du[i, jj, 5+t] += factor_jj * f_tracer
            end
        end
    end

    @turbo for v in eachvariable(equations), j in eachnode(dg), i in eachnode(dg)
        _du[v, i, j, element] += du[i, j, v]
    end
end
