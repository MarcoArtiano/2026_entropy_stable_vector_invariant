using Trixi: eachnode

struct VerticalRhsCache{VI}
    vertical_interfaces::VI
end

# an interface is vertical iff its normal is nonzero at some face node
function build_vertical_rhs_cache(semi_split)
    (; cache_stiff, solver_stiff) = semi_split
    (; interfaces) = cache_stiff
    (; contravariant_vectors) = cache_stiff.elements
    index_range = eachnode(solver_stiff)
    vertical = Int[]
    for interface in Trixi.eachinterface(solver_stiff, cache_stiff)
        primary_element = interfaces.neighbor_ids[1, interface]
        primary_indices = interfaces.node_indices[1, interface]
        primary_direction = Trixi.indices2direction(primary_indices)
        i_start, i_si, i_sj = Trixi.index_to_start_step_3d(primary_indices[1],
                                                           index_range)
        j_start, j_si, j_sj = Trixi.index_to_start_step_3d(primary_indices[2],
                                                           index_range)
        k_start, k_si, k_sj = Trixi.index_to_start_step_3d(primary_indices[3],
                                                           index_range)
        i_node, j_node, k_node = i_start, j_start, k_start
        is_vertical = false
        for j in index_range
            for i in index_range
                normal = Trixi.get_normal_direction(primary_direction,
                                                    contravariant_vectors,
                                                    i_node, j_node, k_node,
                                                    primary_element)
                if normal[1]^2 + normal[2]^2 + normal[3]^2 > 0
                    is_vertical = true
                end
                i_node += i_si
                j_node += j_si
                k_node += k_si
            end
            i_node += i_sj
            j_node += j_sj
            k_node += k_sj
        end
        is_vertical && push!(vertical, interface)
    end
    return VerticalRhsCache(vertical)
end


function mask_contravariant_vertical!(semi_split)
    (; equations_stiff, cache_stiff, solver_stiff) = semi_split
    contravariant_vectors = cache_stiff.elements.contravariant_vectors
    node_coordinates = cache_stiff.elements.node_coordinates

    for element in eachelement(solver_stiff, cache_stiff)
        for k in eachnode(solver_stiff), j in eachnode(solver_stiff),
            i in eachnode(solver_stiff)

            x = Trixi.get_node_coords(node_coordinates, equations_stiff, solver_stiff,
                                      i, j, k, element)
            rn = sqrt(x[1]^2 + x[2]^2 + x[3]^2)
            r_hat = x / rn   # radial unit vector

            for d in 1:3
                Ja = Trixi.get_contravariant_vector(d, contravariant_vectors,
                                                    i, j, k, element)
                proj = Ja[1] * r_hat[1] + Ja[2] * r_hat[2] + Ja[3] * r_hat[3]
                mag2 = Ja[1]^2 + Ja[2]^2 + Ja[3]^2
                if proj^2 < 1e-16 * mag2
                    contravariant_vectors[1, d, i, j, k, element] = 0
                    contravariant_vectors[2, d, i, j, k, element] = 0
                    contravariant_vectors[3, d, i, j, k, element] = 0
                else
                    contravariant_vectors[1, d, i, j, k, element] = proj * r_hat[1]
                    contravariant_vectors[2, d, i, j, k, element] = proj * r_hat[2]
                    contravariant_vectors[3, d, i, j, k, element] = proj * r_hat[3]
                end
            end
        end
    end
    return nothing
end

@inline function flux_surface_combined_entropy_stable_vertical(
    u_ll,
    u_rr,
    normal_direction::AbstractVector,
    equations::CompressibleEulerVectorInvariantEquations3D,
)
    if iszero(normal_direction[1]) && iszero(normal_direction[2]) && iszero(normal_direction[3])
        z = zero(u_ll)
        return z, z
    end
    return flux_surface_combined_entropy_stable(u_ll, u_rr, normal_direction, equations)
end

@inline Trixi.combine_conservative_and_nonconservative_fluxes(
    ::typeof(flux_surface_combined_entropy_stable_vertical),
    equations::CompressibleEulerVectorInvariantEquations3D,
) = Trixi.True()


using Trixi: ForwardDiff

# 5 active variables (ϱ, v1, v2, v3, ϱθ); φ (slot 6) is a passive parameter.
@inline embed_active(w, phi) = SVector(w[1], w[2], w[3], w[4], w[5], phi)
@inline active_vars(u) = SVector(u[1], u[2], u[3], u[4], u[5])

@muladd @inline function flux_volume_ec_vertical(u_ll::SVector{6}, u_rr::SVector{6},
                                                 normal_direction::SVector{3},
                                                 equations::CompressibleEulerVectorInvariantEquations3D)
    rho_ll, v1_ll, v2_ll, v3_ll, rho_theta_ll = u_ll[1], u_ll[2], u_ll[3], u_ll[4], u_ll[5]
    rho_rr, v1_rr, v2_rr, v3_rr, rho_theta_rr = u_rr[1], u_rr[2], u_rr[3], u_rr[4], u_rr[5]
    theta_ll = rho_theta_ll / rho_ll
    theta_rr = rho_theta_rr / rho_rr
    exner_ll = (rho_theta_ll * equations.R / equations.p_0)^(equations.R / equations.c_v)
    exner_rr = (rho_theta_rr * equations.R / equations.p_0)^(equations.R / equations.c_v)
    phi_ll, phi_rr = u_ll[6], u_rr[6]

    rho_v_dot_n_ll = rho_ll * (v1_ll * normal_direction[1] + v2_ll * normal_direction[2] +
                               v3_ll * normal_direction[3])
    rho_v_dot_n_rr = rho_rr * (v1_rr * normal_direction[1] + v2_rr * normal_direction[2] +
                               v3_rr * normal_direction[3])
    rho_avg = 0.5f0 * (rho_ll + rho_rr)
    kin_avg = 0.5f0 * (v1_rr^2 + v2_rr^2 + v3_rr^2 + v1_ll^2 + v2_ll^2 + v3_ll^2)
    v1_avg = 0.5f0 * (rho_ll * v1_ll + rho_rr * v1_rr) / rho_avg
    v2_avg = 0.5f0 * (rho_ll * v2_ll + rho_rr * v2_rr) / rho_avg
    v3_avg = 0.5f0 * (rho_ll * v3_ll + rho_rr * v3_rr) / rho_avg

    x2 = theta_rr
    y2 = theta_ll
    x2_plus_y2 = x2 + y2
    y2_minus_x2 = y2 - x2
    z2 = y2_minus_x2^2 / x2_plus_y2^2
    special_path2 = (2 + z2 * (2 / 3 + z2 * (2 / 5 + 2 / 7 * z2))) / x2_plus_y2
    regular_path2 = (log(y2) - log(x2)) / y2_minus_x2
    theta_mean = theta_ll * theta_rr * ifelse(z2 < 1.0e-4, special_path2, regular_path2)

    jump_v1 = v1_rr - v1_ll
    jump_v2 = v2_rr - v2_ll
    jump_v3 = v3_rr - v3_ll
    gravity = phi_rr - phi_ll
    f1 = 0.5f0 * (rho_v_dot_n_ll + rho_v_dot_n_rr)
    f2 = kin_avg * 0.5f0 * normal_direction[1]
    f3 = kin_avg * 0.5f0 * normal_direction[2]
    f4 = kin_avg * 0.5f0 * normal_direction[3]
    f5 = f1 * theta_mean
    theta_grad_exner = equations.c_p * theta_mean * (exner_rr - exner_ll)

    vorticity_x = v2_avg * (jump_v1 * normal_direction[2] - jump_v2 * normal_direction[1]) +
                  v3_avg * (jump_v1 * normal_direction[3] - jump_v3 * normal_direction[1])
    vorticity_y = v1_avg * (jump_v2 * normal_direction[1] - jump_v1 * normal_direction[2]) +
                  v3_avg * (jump_v2 * normal_direction[3] - jump_v3 * normal_direction[2])
    vorticity_z = v1_avg * (jump_v3 * normal_direction[1] - jump_v1 * normal_direction[3]) +
                  v2_avg * (jump_v3 * normal_direction[2] - jump_v2 * normal_direction[3])

    g2 = vorticity_x + (theta_grad_exner + gravity) * normal_direction[1]
    g3 = vorticity_y + (theta_grad_exner + gravity) * normal_direction[2]
    g4 = vorticity_z + (theta_grad_exner + gravity) * normal_direction[3]

    return SVector(f1, f2 + 0.5 * g2, f3 + 0.5 * g3, f4 + 0.5 * g4, f5)
end

@inline function volume_flux_jacobian_ll(u_ll, u_rr, normal_direction, equations)
    ForwardDiff.jacobian(active_vars(u_ll)) do w
        flux_volume_ec_vertical(embed_active(w, u_ll[6]), u_rr, normal_direction, equations)
    end
end
@inline function volume_flux_jacobian_rr(u_ll, u_rr, normal_direction, equations)
    ForwardDiff.jacobian(active_vars(u_rr)) do w
        flux_volume_ec_vertical(u_ll, embed_active(w, u_rr[6]), normal_direction, equations)
    end
end

@inline function interface_flux_jacobian(surface_flux::F, u_ll, u_rr, normal_direction,
                                         equations, ::Val{SIDE}, ::Val{ARG}) where {F, SIDE, ARG}
    if ARG === :ll
        ForwardDiff.jacobian(active_vars(u_ll)) do w
            fl, fr = surface_flux(embed_active(w, u_ll[6]), u_rr, normal_direction, equations)
            active_vars(SIDE === :left ? fl : fr)
        end
    else
        ForwardDiff.jacobian(active_vars(u_rr)) do w
            fl, fr = surface_flux(u_ll, embed_active(w, u_rr[6]), normal_direction, equations)
            active_vars(SIDE === :left ? fl : fr)
        end
    end
end

@inline function boundary_flux_jacobian(surface_flux::F, u_inner, normal_direction, x, t, equations) where {F}
    ForwardDiff.jacobian(active_vars(u_inner)) do w
        f = Trixi.boundary_condition_slip_wall(embed_active(w, u_inner[6]), normal_direction,
                                               x, t, surface_flux, equations)
        active_vars(f)
    end
end

function assemble_volume_element!(diagonal_blocks, u_wrap, element, equations, dg,
                                  cache_stiff, vertical_line_map, vertical_layer_map, _nnodes)
    (; derivative_split) = dg.basis
    contravariant_vectors = cache_stiff.elements.contravariant_vectors
    inverse_jacobian = cache_stiff.elements.inverse_jacobian
    @inbounds for j in eachnode(dg), i in eachnode(dg)
        line = vertical_line_map[i, j, 1, element]
        layer = vertical_layer_map[i, j, 1, element]
        for k in eachnode(dg), kk in (k + 1):_nnodes
            u_ll = Trixi.get_node_vars(u_wrap, equations, dg, i, j, k, element)
            u_rr = Trixi.get_node_vars(u_wrap, equations, dg, i, j, kk, element)
            Ja3_k = Trixi.get_contravariant_vector(3, contravariant_vectors, i, j, k, element)
            Ja3_kk = Trixi.get_contravariant_vector(3, contravariant_vectors, i, j, kk, element)
            normal_direction = 0.5 * (Ja3_k + Ja3_kk)
            invJ_k = inverse_jacobian[i, j, k, element]
            invJ_kk = inverse_jacobian[i, j, kk, element]
            dsk = derivative_split[k, kk]
            dskk = derivative_split[kk, k]

            # rows at node k: contrib(u_ll, u_rr)
            Jll = volume_flux_jacobian_ll(u_ll, u_rr, normal_direction, equations)
            Jrr = volume_flux_jacobian_rr(u_ll, u_rr, normal_direction, equations)
            # rows at node kk: contrib(u_rr, u_ll) (arguments swapped)
            Jll2 = volume_flux_jacobian_ll(u_rr, u_ll, normal_direction, equations)
            Jrr2 = volume_flux_jacobian_rr(u_rr, u_ll, normal_direction, equations)

            r0k = 5 * (k - 1)
            r0kk = 5 * (kk - 1)
            RealT = eltype(diagonal_blocks)
            for c in 1:5, r in 1:5
                diagonal_blocks[r0k + r, r0k + c, layer, line] += RealT(invJ_k * dsk * Jll[r, c])
                diagonal_blocks[r0k + r, r0kk + c, layer, line] += RealT(invJ_k * dsk * Jrr[r, c])
                diagonal_blocks[r0kk + r, r0kk + c, layer, line] += RealT(invJ_kk * dskk * Jll2[r, c])
                diagonal_blocks[r0kk + r, r0k + c, layer, line] += RealT(invJ_kk * dskk * Jrr2[r, c])
            end
        end
    end
    return nothing
end

function assemble_interface!(diagonal_blocks, upper_blocks, lower_blocks, u_wrap,
                             interface, surface_flux::F, equations, dg, cache_stiff,
                             vertical_line_map, vertical_layer_map, _nnodes, w1) where {F}
    interfaces = cache_stiff.interfaces
    contravariant_vectors = cache_stiff.elements.contravariant_vectors
    inverse_jacobian = cache_stiff.elements.inverse_jacobian
    index_range = eachnode(dg)
    RealT = eltype(diagonal_blocks)

    e_p = interfaces.neighbor_ids[1, interface]
    e_s = interfaces.neighbor_ids[2, interface]
    idx_p = interfaces.node_indices[1, interface]
    idx_s = interfaces.node_indices[2, interface]
    dir_p = Trixi.indices2direction(idx_p)

    ip0, ipi, ipj = Trixi.index_to_start_step_3d(idx_p[1], index_range)
    jp0, jpi, jpj = Trixi.index_to_start_step_3d(idx_p[2], index_range)
    kp0, kpi, kpj = Trixi.index_to_start_step_3d(idx_p[3], index_range)
    is0, isi, isj = Trixi.index_to_start_step_3d(idx_s[1], index_range)
    js0, jsi, jsj = Trixi.index_to_start_step_3d(idx_s[2], index_range)
    ks0, ksi, ksj = Trixi.index_to_start_step_3d(idx_s[3], index_range)

    i_p, j_p, k_p = ip0, jp0, kp0
    i_s, j_s, k_s = is0, js0, ks0
    @inbounds for q in index_range
        for pq in index_range
            line = vertical_line_map[i_p, j_p, k_p, e_p]
            layer_primary = vertical_layer_map[i_p, j_p, k_p, e_p]
            layer_secondary = vertical_layer_map[i_s, j_s, k_s, e_s]

            u_ll = Trixi.get_node_vars(u_wrap, equations, dg, i_p, j_p, k_p, e_p)
            u_rr = Trixi.get_node_vars(u_wrap, equations, dg, i_s, j_s, k_s, e_s)
            normal_direction = Trixi.get_normal_direction(dir_p, contravariant_vectors,
                                                          i_p, j_p, k_p, e_p)
            invJ_p = inverse_jacobian[i_p, j_p, k_p, e_p]
            invJ_s = inverse_jacobian[i_s, j_s, k_s, e_s]

            JL_ll = interface_flux_jacobian(surface_flux, u_ll, u_rr, normal_direction, equations, Val(:left), Val(:ll))
            JL_rr = interface_flux_jacobian(surface_flux, u_ll, u_rr, normal_direction, equations, Val(:left), Val(:rr))
            JR_ll = interface_flux_jacobian(surface_flux, u_ll, u_rr, normal_direction, equations, Val(:right), Val(:ll))
            JR_rr = interface_flux_jacobian(surface_flux, u_ll, u_rr, normal_direction, equations, Val(:right), Val(:rr))

            rp = 5 * (k_p - 1)
            rs = 5 * (k_s - 1)
            for c in 1:5, r in 1:5
                # primary rows: du_pre += +w1 * flux_left(u_ll, u_rr)
                diagonal_blocks[rp + r, rp + c, layer_primary, line] += RealT(invJ_p * w1 * JL_ll[r, c])
                # secondary rows: du_pre += -w1 * flux_right(u_ll, u_rr)
                diagonal_blocks[rs + r, rs + c, layer_secondary, line] += RealT(-invJ_s * w1 * JR_rr[r, c])
            end
            if layer_secondary == layer_primary + 1        # secondary above: U/C of layer_primary
                for c in 1:5, r in 1:5
                    upper_blocks[rp + r, c, layer_primary, line] += RealT(invJ_p * w1 * JL_rr[r, c])
                    lower_blocks[r, c, layer_primary, line] += RealT(-invJ_s * w1 * JR_ll[r, c])
                end
            else                                           # secondary below: U/C of layer_secondary
                for c in 1:5, r in 1:5
                    upper_blocks[rs + r, c, layer_secondary, line] += RealT(-invJ_s * w1 * JR_ll[r, c])
                    lower_blocks[r, c, layer_secondary, line] += RealT(invJ_p * w1 * JL_rr[r, c])
                end
            end

            i_p += ipi
            j_p += jpi
            k_p += kpi
            i_s += isi
            j_s += jsi
            k_s += ksi
        end
        i_p += ipj
        j_p += jpj
        k_p += kpj
        i_s += isj
        j_s += jsj
        k_s += ksj
    end
    return nothing
end

function assemble_boundary!(diagonal_blocks, u_wrap, boundary, surface_flux::F, equations,
                            dg, cache_stiff, vertical_line_map, vertical_layer_map, _nnodes, w1) where {F}
    boundaries = cache_stiff.boundaries
    contravariant_vectors = cache_stiff.elements.contravariant_vectors
    inverse_jacobian = cache_stiff.elements.inverse_jacobian
    node_coordinates = cache_stiff.elements.node_coordinates
    index_range = eachnode(dg)
    RealT = eltype(diagonal_blocks)

    element = boundaries.neighbor_ids[boundary]
    node_idx = boundaries.node_indices[boundary]
    direction = Trixi.indices2direction(node_idx)
    i0, ii, ij = Trixi.index_to_start_step_3d(node_idx[1], index_range)
    j0, ji, jj = Trixi.index_to_start_step_3d(node_idx[2], index_range)
    k0, ki, kj = Trixi.index_to_start_step_3d(node_idx[3], index_range)

    i, j, k = i0, j0, k0
    @inbounds for q in index_range
        for pq in index_range
            line = vertical_line_map[i, j, k, element]
            layer = vertical_layer_map[i, j, k, element]
            u_inner = Trixi.get_node_vars(u_wrap, equations, dg, i, j, k, element)
            normal_direction = Trixi.get_normal_direction(direction, contravariant_vectors,
                                                          i, j, k, element)
            x = Trixi.get_node_coords(node_coordinates, equations, dg, i, j, k, element)
            invJ = inverse_jacobian[i, j, k, element]
            Jbnd = boundary_flux_jacobian(surface_flux, u_inner, normal_direction, x,
                                          zero(eltype(u_wrap)), equations)
            r0 = 5 * (k - 1)
            for c in 1:5, r in 1:5
                diagonal_blocks[r0 + r, r0 + c, layer, line] += RealT(invJ * w1 * Jbnd[r, c])
            end
            i += ii
            j += ji
            k += ki
        end
        i += ij
        j += jj
        k += kj
    end
    return nothing
end

function assemble_jacobian_analytic!(u, semi, equations, cache::JacobianCache)
    (; nlayers, inv_gamma_dt, semi_split, vertical_interfaces,
       diagonal_blocks, upper_blocks, lower_blocks, nvertical_lines,
       vertical_line_map, vertical_layer_map) = cache
    dg = semi_split.solver_stiff
    _nnodes = Trixi.nnodes(dg)
    block_size = 5 * _nnodes
    cache_stiff = semi_split.cache_stiff
    surface_flux = dg.surface_integral.surface_flux
    w1 = dg.basis.inverse_weights[1]
    u_wrap = Trixi.wrap_array(u, semi)

    @threaded for line in 1:nvertical_lines
        init_column!(diagonal_blocks, upper_blocks, lower_blocks, inv_gamma_dt,
                     block_size, nlayers, line)
    end

    @trixi_timeit timer() "analytic volume" @threaded for element in eachelement(dg, cache_stiff)
        assemble_volume_element!(diagonal_blocks, u_wrap, element, equations, dg,
                                 cache_stiff, vertical_line_map, vertical_layer_map, _nnodes)
    end
    @trixi_timeit timer() "analytic interfaces" @threaded for idx in eachindex(vertical_interfaces)
        assemble_interface!(diagonal_blocks, upper_blocks, lower_blocks, u_wrap,
                            vertical_interfaces[idx], surface_flux, equations, dg,
                            cache_stiff, vertical_line_map, vertical_layer_map, _nnodes, w1)
    end
    @trixi_timeit timer() "analytic boundaries" @threaded for boundary in Trixi.eachboundary(dg, cache_stiff)
        assemble_boundary!(diagonal_blocks, u_wrap, boundary, surface_flux, equations,
                           dg, cache_stiff, vertical_line_map, vertical_layer_map, _nnodes, w1)
    end
    return nothing
end
