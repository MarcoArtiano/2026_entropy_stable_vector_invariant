
using LinearAlgebra: BlasInt, LAPACK

mutable struct JacobianCache{RealT, SplitSemi, VIType, DType, WType, CType, UVertType,
                             IDMapType, IZMapType, ABT, PivT, RhsT}
    nlayers::Int
    nhorizontal::Int
    nvertical_lines::Int
    inv_gamma_dt::RealT          
    semi_split::SplitSemi       
    vertical_interfaces::VIType 
    diagonal_blocks::DType       # (5·(p+1), 5·(p+1), nlayers, nvertical_lines)
    upper_blocks::WType          # (5·(p+1), 5, nlayers-1, nvertical_lines) raw U (rows at node p+1)
    lower_blocks::CType          # (5, 5, nlayers-1, nvertical_lines)
    u_vert::UVertType            # (p+1, nlayers, nvertical_lines, 5)
    vertical_line_map::IDMapType
    vertical_layer_map::IZMapType
    kl::Int
    ku::Int
    AB::ABT                      # (2kl+ku+1, 5·(p+1)·nlayers, nvertical_lines)
    ipiv_band::PivT              # (5·(p+1)·nlayers, nvertical_lines)
    rhs_band::RhsT               # (5·(p+1)·nlayers, nvertical_lines) per-column rhs/solution
end

function JacobianCache(nlayers, nhorizontal, nvertical_lines, semi, dt,
                       equations::CompressibleEulerVectorInvariantEquations3D;
                       semi_split, operator_eltype = Float64)
    RealT = operator_eltype
    _nnodes = Trixi.nnodes(semi.solver)
    inv_gamma_dt = RealT(inv(dt))
    block_size = 5 * _nnodes
    diagonal_blocks = zeros(RealT, block_size, block_size, nlayers, nvertical_lines)
    upper_blocks = zeros(RealT, block_size, 5, nlayers - 1, nvertical_lines)
    lower_blocks = zeros(RealT, 5, 5, nlayers - 1, nvertical_lines)
    u_vert = zeros(RealT, _nnodes, nlayers, nvertical_lines, 5)
    vertical_line_map, vertical_layer_map = build_vertical_mapping(semi, nhorizontal, nlayers, equations)
    vertical_interfaces = build_vertical_rhs_cache(semi_split).vertical_interfaces

    kl = block_size - 1
    ku = block_size - 1
    n = block_size * nlayers
    ldab = 2 * kl + ku + 1
    AB = zeros(RealT, ldab, n, nvertical_lines)
    ipiv_band = zeros(BlasInt, n, nvertical_lines)
    rhs_band = zeros(RealT, n, nvertical_lines)

    return JacobianCache{RealT, typeof(semi_split), typeof(vertical_interfaces),
                         typeof(diagonal_blocks), typeof(upper_blocks),
                         typeof(lower_blocks), typeof(u_vert), typeof(vertical_line_map),
                         typeof(vertical_layer_map), typeof(AB), typeof(ipiv_band),
                         typeof(rhs_band)}(
        nlayers, nhorizontal, nvertical_lines, inv_gamma_dt, semi_split,
        vertical_interfaces, diagonal_blocks, upper_blocks, lower_blocks, u_vert,
        vertical_line_map, vertical_layer_map, kl, ku, AB, ipiv_band, rhs_band)
end

function gather_columns!(u_vert, u_wrap, cache::JacobianCache, semi)
    (; vertical_line_map, vertical_layer_map) = cache
    (; solver) = semi
    @threaded for element in eachelement(solver, semi.cache)
        for k in eachnode(solver), j in eachnode(solver), i in eachnode(solver)
            line = vertical_line_map[i, j, k, element]
            layer = vertical_layer_map[i, j, k, element]
            for v in 1:5
                u_vert[k, layer, line, v] = u_wrap[v, i, j, k, element]
            end
        end
    end
    return nothing
end

function init_column!(diagonal_blocks, upper_blocks, lower_blocks, inv_gamma_dt,
                      block_size, nlayers, line)
    RealT = eltype(diagonal_blocks)
    @inbounds begin
        for layer in 1:nlayers
            fill!(view(diagonal_blocks, :, :, layer, line), zero(RealT))
            for r in 1:block_size
                diagonal_blocks[r, r, layer, line] = RealT(inv_gamma_dt)
            end
        end
        for layer in 1:(nlayers - 1)
            fill!(view(upper_blocks, :, :, layer, line), zero(RealT))
            fill!(view(lower_blocks, :, :, layer, line), zero(RealT))
        end
    end
    return nothing
end

function build_vertical_mapping(semi, Kh, Kv,
                                equations::CompressibleEulerVectorInvariantEquations3D)
    (; solver) = semi
    _nnodes = Trixi.nnodes(solver)
    n_elements = Trixi.nelements(solver, semi.cache)

    vertical_layer_map = zeros(Int, _nnodes, _nnodes, _nnodes, n_elements)
    vertical_line_map = zeros(Int, _nnodes, _nnodes, _nnodes, n_elements)

    for element in eachelement(solver, semi.cache)
        el_0 = element - 1
        h_elem = (el_0 % Kh^2) + 1
        layer = ((el_0 ÷ Kh^2) % Kv) + 1
        block = (el_0 ÷ (Kh^2 * Kv)) + 1

        for k in eachnode(solver), j in eachnode(solver), i in eachnode(solver)
            line_0 = (block - 1) * _nnodes^2 * Kh^2 +
                     (h_elem - 1) * _nnodes^2 +
                     (j - 1) * _nnodes +
                     i - 1
            vertical_line_map[i, j, k, element] = line_0 + 1
            vertical_layer_map[i, j, k, element] = layer
        end
    end
    return vertical_line_map, vertical_layer_map
end

function compute_number_of_vertical_lines(solver, trees_per_dimension,
                                          mesh::Trixi.P4estMesh{3})
    @warn "Only the cubed sphere is supported."
    nvertical_lines = (Trixi.polydeg(solver) + 1)^2 * trees_per_dimension[1]^2 * 6
end

function pack_factorize_band_column!(AB, ipiv_band, diagonal_blocks, upper_blocks,
                                     lower_blocks, kl, ku, _nnodes, nlayers, line)
    RealT = eltype(AB)
    block_size = 5 * _nnodes
    rowc = kl + ku + 1
    @inbounds begin
        fill!(view(AB, :, :, line), zero(RealT))
        # dense element blocks
        for layer in 1:nlayers
            off = block_size * (layer - 1)
            for cl in 1:block_size
                c = off + cl
                for rl in 1:block_size
                    AB[rowc + rl - cl, c, line] = diagonal_blocks[rl, cl, layer, line]
                end
            end
        end
        for layer in 1:(nlayers - 1)
            off = block_size * (layer - 1)
            # U block: node-(p+1) rows of layer <- node-1 cols of layer+1
            for vc in 1:5
                c = off + block_size + vc
                for k in 1:5
                    r = off + 5 * (_nnodes - 1) + k
                    AB[rowc + r - c, c, line] = upper_blocks[5 * (_nnodes - 1) + k, vc, layer, line]
                end
            end
            # C block: node-1 rows of layer+1 <- node-(p+1) cols of layer
            for k in 1:5
                c = off + 5 * (_nnodes - 1) + k
                for vr in 1:5
                    r = off + block_size + vr
                    AB[rowc + r - c, c, line] = lower_blocks[vr, k, layer, line]
                end
            end
        end
        n = block_size * nlayers
        _, piv = LAPACK.gbtrf!(kl, ku, n, view(AB, :, :, line))
        copyto!(view(ipiv_band, :, line), piv)
    end
    return nothing
end

function solve_band_column!(AB, ipiv_band, rhs_band, u_vert, kl, ku, _nnodes, nlayers, line)
    RealT = eltype(AB)
    block_size = 5 * _nnodes
    y = view(rhs_band, :, line)
    @inbounds for layer in 1:nlayers, i in 1:_nnodes, v in 1:5
        y[block_size * (layer - 1) + 5 * (i - 1) + v] = RealT(u_vert[i, layer, line, v])
    end
    LAPACK.gbtrs!('N', kl, ku, block_size * nlayers, view(AB, :, :, line),
                  view(ipiv_band, :, line), y)
    @inbounds for layer in 1:nlayers, i in 1:_nnodes, v in 1:5
        u_vert[i, layer, line, v] = y[block_size * (layer - 1) + 5 * (i - 1) + v]
    end
    return nothing
end

function update_jacobian!(u, semi, equations, cache::JacobianCache, gamma)
    (; nlayers, diagonal_blocks, upper_blocks, lower_blocks,
       nvertical_lines, AB, ipiv_band, kl, ku) = cache
    _nnodes = Trixi.nnodes(semi.solver)
    @trixi_timeit timer() "analytic assembly" assemble_jacobian_analytic!(u, semi,
                                                                          equations,
                                                                          cache)
    @trixi_timeit timer() "band pack + gbtrf" @threaded for line in 1:nvertical_lines
        pack_factorize_band_column!(AB, ipiv_band, diagonal_blocks, upper_blocks,
                                    lower_blocks, kl, ku, _nnodes, nlayers, line)
    end
    return nothing
end

function solve_jacobian!(b, dt, gamma, semi, equations, cache::JacobianCache,
                         b_vertical_wrap)
    (; nlayers, u_vert, nvertical_lines, vertical_line_map, vertical_layer_map,
       AB, ipiv_band, rhs_band, kl, ku) = cache
    _nnodes = Trixi.nnodes(semi.solver)
    b_wrap = Trixi.wrap_array(b, semi)
    @trixi_timeit timer() "gather" gather_columns!(u_vert, b_wrap, cache, semi)

    @trixi_timeit timer() "band gbtrs solve" @threaded for line in 1:nvertical_lines
        solve_band_column!(AB, ipiv_band, rhs_band, u_vert, kl, ku, _nnodes, nlayers, line)
    end

    solver = semi.solver
    @trixi_timeit timer() "scatter" @threaded for element in eachelement(solver,
                                                                         semi.cache)
        for k in eachnode(solver), j in eachnode(solver), i in eachnode(solver)
            line = vertical_line_map[i, j, k, element]
            layer = vertical_layer_map[i, j, k, element]
            for v in 1:5
                b_wrap[v, i, j, k, element] = u_vert[k, layer, line, v]
            end
        end
    end
    return nothing
end
