using CairoMakie
using DelaunayTriangulation
using JLD2
using LinearAlgebra: dot, norm

function cons2temperature(u, equations::CompressibleEulerVectorInvariantEquations3D)
    rho, v1, v2, v3, rho_theta = u

    p = equations.K * rho_theta^equations.gamma
    return p / (rho * equations.R)
end

function cons2pressure(u, equations::CompressibleEulerVectorInvariantEquations3D)
    rho, v1, v2, v3, rho_theta = u

    p = equations.K * rho_theta^equations.gamma
    return p / 1e2
end

function cart2sphere(x, y, z)
    r = sqrt.(x .^ 2 + y .^ 2 + z .^ 2)
    lat = asin.(z ./ r) .* (180 / π)
    lon = atan.(y, x) .* (180 / π)
    return vec(lon), vec(lat)
end

function barycentric_eval(ξs, w, f, x)
    num = zero(eltype(f))
    den = zero(eltype(f))
    for i in eachindex(ξs)
        d = x - ξs[i]
        d == 0 && return f[i]
        t = w[i] / d
        num += t * f[i]
        den += t
    end
    return num / den
end

function find_pressure_root(ξs, w, pe, plevel)
    ga = pe[1] - plevel
    gb = pe[end] - plevel
    ga * gb > 0 && return nothing
    a, b = -1.0, 1.0
    for _ = 1:60
        m = 0.5 * (a + b)
        gm = barycentric_eval(ξs, w, pe, m) - plevel
        if ga * gm <= 0
            b = m
        else
            a = m
            ga = gm
        end
    end
    return 0.5 * (a + b)
end

function plotting_interpolation_matrix_no_boundary(
    dg::DGSEM;
    nvisnodes = length(dg.basis.nodes),
)

    dξ = 2 / nvisnodes
    interia = [-1 + (j - 1 / 2) * dξ for j ∈ 1:nvisnodes]
    Vp1D = Trixi.polynomial_interpolation_matrix(dg.basis.nodes, interia)
    return Trixi.kron(Vp1D, Vp1D)
end

function compute_vorticity(sol, semi, equations)
    du = similar(sol.u[end])
    u_ode = sol.u[end]

    polydeg = Trixi.polydeg(semi.solver)
    solver = DGSEM(
        polydeg = polydeg,
        surface_flux = flux_vorticity_rusanov,
        volume_integral = VolumeIntegralFluxDifferencing(flux_vorticity),
    )

    GC.@preserve du u_ode begin
        u_wrap  = Trixi.wrap_array(u_ode, semi)
        du_wrap = Trixi.wrap_array(du, semi)
        Trixi.rhs!(du_wrap, u_wrap, 0, semi.mesh, equations,
                   semi.boundary_conditions, nothing, solver, semi.cache)
        return Array(du_wrap)
    end
end

function fields_850(Kh, Kv, semi, sol, vorticity, equations;
                    plevel = 850.0, nvis = Trixi.nnodes(semi.solver))
    data = Trixi.wrap_array(sol.u[end], semi)
    xyz  = semi.cache.elements.node_coordinates
    nq   = Trixi.nnodes(semi.solver)
    nelem = 6 * Kv * Kh^2

    P = zeros(nq, nq, nq, nelem); T = similar(P); W = similar(P)
    for e = 1:nelem, k = 1:nq, j = 1:nq, i = 1:nq
        u = SVector(ntuple(v -> data[v, i, j, k, e], 6))
        P[i, j, k, e] = cons2pressure(u, equations)      # hPa
        T[i, j, k, e] = cons2temperature(u, equations)
        x_vec = SVector(xyz[1,i,j,k,e], xyz[2,i,j,k,e], xyz[3,i,j,k,e])
        w_vec = SVector(vorticity[2,i,j,k,e], vorticity[3,i,j,k,e], vorticity[4,i,j,k,e])
        W[i, j, k, e] = -dot(x_vec, w_vec) / norm(x_vec) / 1e-5
    end

    Vh = plotting_interpolation_matrix_no_boundary(semi.solver; nvisnodes = nvis)
    nh = size(Vh, 1)
    refine_h(F) = reshape(Vh * reshape(F, nq^2, :), nh, nq, nelem)
    Pr, Tr, Wr = refine_h(P), refine_h(T), refine_h(W)

    ξ = semi.solver.basis.nodes
    wbary = Trixi.barycentric_weights(ξ)

    nelem_h = 6 * Kh^2
    psurf = zeros(nh, nelem_h)
    T850  = fill(NaN, nh, nelem_h)
    W850  = fill(NaN, nh, nelem_h)

    for block = 1:6, eh = 1:Kh^2
        e_h = (block - 1) * Kh^2 + eh
        for a = 1:nh
            psurf[a, e_h] = Pr[a, 1, (block - 1) * Kv * Kh^2 + eh]
            for ev = 1:Kv
                e = (block - 1) * Kv * Kh^2 + (ev - 1) * Kh^2 + eh
                pe = @view Pr[a, :, e]
                pmin, pmax = extrema(pe)
                pmin <= plevel <= pmax || continue
                ξr = find_pressure_root(ξ, wbary, pe, plevel)
                ξr === nothing && continue
                T850[a, e_h] = barycentric_eval(ξ, wbary, @view(Tr[a, :, e]), ξr)
                W850[a, e_h] = barycentric_eval(ξ, wbary, @view(Wr[a, :, e]), ξr)
            end
        end
    end

    ebot = [(block - 1) * Kv * Kh^2 + eh for block = 1:6 for eh = 1:Kh^2]
    X = Vh * reshape(xyz[1, :, :, 1, ebot], nq^2, nelem_h)
    Y = Vh * reshape(xyz[2, :, :, 1, ebot], nq^2, nelem_h)
    Z = Vh * reshape(xyz[3, :, :, 1, ebot], nq^2, nelem_h)

    return vec(psurf), vec(T850), vec(W850), vec(X), vec(Y), vec(Z)
end

function save_baroclinic_fields(sol, semi, (Kh, Kv), polydeg, equations, T_final, time_method, surface_flux;
                                prefix = "baroclinic")
    nvisnodes = polydeg + 1
    vorticity = compute_vorticity(sol, semi, equations)
    p_interp, T_interp, vor_interp, x, y, z =
        fields_850(Kh, Kv, semi, sol, vorticity, equations; nvis = nvisnodes)
    lon, lat = cart2sphere(x, y, z)
    fname = "$(prefix)_euler_$(Kh)x$(Kv)_p$(polydeg)_$(T_final)_$(time_method)_$(surface_flux).jld2"
    JLD2.jldsave(fname;
        p_interp = p_interp, T_interp = T_interp, vor_interp = vor_interp,
        lon = lon, lat = lat, Kh = Kh, Kv = Kv, polydeg = polydeg,
        T_final = T_final)
    return fname
end

function nice_levels(vmin, vmax, nlevels)
    if isapprox(vmin, vmax)   
        delta = max(one(vmin), abs(vmin))
        vmin, vmax = vmin - delta, vmax + delta
    end
    raw = (vmax - vmin) / max(nlevels, 1)
    mag = 10.0^floor(log10(raw))
    step = 10.0 * mag
    for st in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0)
        if st * mag >= raw
            step = st * mag
            break
        end
    end
    lo = floor(vmin / step) * step
    hi = ceil(vmax / step) * step
    return collect(lo:step:hi)
end

function twoslope_band_colors(levels, cmap, vmin, vcenter, vmax)
    cg = CairoMakie.cgrad(cmap)
    mids = 0.5 .* (levels[1:(end - 1)] .+ levels[2:end])
    ts(x) = x <= vcenter ? 0.5 * (x - vmin) / (vcenter - vmin) :
                           0.5 + 0.5 * (x - vcenter) / (vmax - vcenter)
    return [cg[clamp(ts(m), 0.0, 1.0)] for m in mids]
end

function linear_band_colors(levels, cmap)
    cg = CairoMakie.cgrad(cmap)
    n = length(levels) - 1
    return [cg[(k - 0.5) / n] for k in 1:n]
end

function iso_segments(tri, lon, lat, vals, levels)
    seg_pos = CairoMakie.Point2f[]
    seg_neg = CairoMakie.Point2f[]
    isempty(levels) && return seg_pos, seg_neg
    for T in DelaunayTriangulation.each_solid_triangle(tri)
        a, b, c = DelaunayTriangulation.triangle_vertices(T)
        va, vb, vc = vals[a], vals[b], vals[c]
        tmin = min(va, vb, vc)
        tmax = max(va, vb, vc)
        for lev in levels
            (lev < tmin || lev > tmax) && continue
            npts = 0
            p1 = CairoMakie.Point2f(0, 0)
            p2 = CairoMakie.Point2f(0, 0)
            for (i, j, vi, vj) in ((a, b, va, vb), (b, c, vb, vc), (c, a, vc, va))
                if (vi < lev) != (vj < lev) && vi != vj
                    t = (lev - vi) / (vj - vi)
                    pt = CairoMakie.Point2f(lon[i] + t * (lon[j] - lon[i]),
                                            lat[i] + t * (lat[j] - lat[i]))
                    npts += 1
                    npts == 1 ? (p1 = pt) : (p2 = pt)
                end
            end
            if npts == 2
                dest = lev < 0 ? seg_neg : seg_pos
                push!(dest, p1)
                push!(dest, p2)
            end
        end
    end
    return seg_pos, seg_neg
end

function tricontourf_triangles(tri)
    ntri = DelaunayTriangulation.num_solid_triangles(tri)
    triangles = Matrix{Int32}(undef, 3, ntri)
    for (k, T) in enumerate(DelaunayTriangulation.each_solid_triangle(tri))
        a, b, c = DelaunayTriangulation.triangle_vertices(T)
        triangles[1, k] = a
        triangles[2, k] = b
        triangles[3, k] = c
    end
    return triangles
end

function baroclinic_masked_fields(p_interp, T_interp, vor_interp, lon, lat; lonshift = 60)
    lon = @. mod(lon + 180 - lonshift, 360) - 180
    mask = (lat .>= 0) .& (lon .>= -lonshift)
    lon_m = lon[mask]
    lat_m = lat[mask]
    tri = DelaunayTriangulation.triangulate(collect(zip(lon_m, lat_m)))
    faces = tricontourf_triangles(tri)
    return lon_m, lat_m, tri, faces, p_interp[mask], T_interp[mask], vor_interp[mask]
end

function draw_baroclinic_panel!(gl, lon, lat, vals, tri, faces, levels, colormap,
                                descr, day, unit;
                                extendlow = nothing, extendhigh = nothing,
                                cbar_ticks = CairoMakie.Makie.automatic,
                                titlesize = 58, ticklabelsize = 50,
                                cbar_ticklabelsize = 44, axwidth = 1400, axheight = 525)
    CairoMakie.Label(gl[1, 1], descr; fontsize = titlesize, font = :regular,
                     halign = :left, tellwidth = false)
    CairoMakie.Label(gl[1, 1], "Day $(day)"; fontsize = titlesize, font = :regular,
                     halign = :center, tellwidth = false)
    CairoMakie.Label(gl[1, 1], unit; fontsize = titlesize, font = :regular,
                     halign = :right, tellwidth = false)
    ax = CairoMakie.Axis(
        gl[2, 1],
        xticks = ([-60, -30, 0, 30, 60, 90, 120, 150, 180],
                  ["0", "30E", "60E", "90E", "120E", "150E", "180", "150W", "120W"]),
        yticks = ([0, 30, 60, 90], ["0", "30N", "60N", "90N"]),
        xticklabelsize = ticklabelsize,
        yticklabelsize = ticklabelsize,
        limits = ((-60, 180), (0, 90)),
        aspect = CairoMakie.DataAspect(),
        width = axwidth,
        height = axheight,
    )
    CairoMakie.tricontourf!(ax, lon, lat, vals; levels = levels, colormap = colormap,
                            extendlow = extendlow, extendhigh = extendhigh,
                            triangulation = faces)
    seg_pos, seg_neg = iso_segments(tri, lon, lat, vals, levels)
    isempty(seg_pos) ||
        CairoMakie.linesegments!(ax, seg_pos, color = :black, linewidth = 1.2)
    isempty(seg_neg) ||
        CairoMakie.linesegments!(ax, seg_neg, color = :black, linewidth = 1.2,
                                 linestyle = :dash)
    CairoMakie.Colorbar(gl[3, 1]; colormap = colormap,
                        limits = (first(levels), last(levels)),
                        vertical = false, flipaxis = false,
                        ticklabelsize = cbar_ticklabelsize, height = 42,
                        width = CairoMakie.Relative(0.75), ticks = cbar_ticks)
    CairoMakie.rowgap!(gl, 6)
    return ax
end

function contour_baroclinic_compare_days_cairomakie(trees_per_cube_face, polydeg,
                                                    time_method = "";
                                                    day1 = 8, day2 = 10,
                                                    prefix = "baroclinic",
                                                    fname_out = nothing)
    Kh, Kv = trees_per_cube_face
    function find_jld2(day)
        for tstr in unique((string(day), string(float(day)), string(round(Int, day))))
            f = "$(prefix)_euler_$(Kh)x$(Kv)_p$(polydeg)_$(tstr)_$(time_method).jld2"
            isfile(f) && return f
        end
        error("JLD2 not found for day $day (looked for " *
              "$(prefix)_euler_$(Kh)x$(Kv)_p$(polydeg)_{$(day)|$(float(day))}.jld2)")
    end
    days = (day1, day2)
    files = (find_jld2(day1), find_jld2(day2))
    plasma = CairoMakie.cgrad(:plasma)
    cg_vor = CairoMakie.cgrad(:RdBu; rev = true)

    CairoMakie.with_theme(CairoMakie.theme_latexfonts()) do
        fig = CairoMakie.Figure(figure_padding = (10, 40, 10, 10))
        for pc in 1:2
            day = days[pc]
            d = JLD2.load(files[pc])
            lon_m, lat_m, tri, faces, p_m, T_m, vor_m =
                baroclinic_masked_fields(d["p_interp"], d["T_interp"],
                                         d["vor_interp"], d["lon"], d["lat"])

            plevels, pticks = if isapprox(day, 8; atol = 1)
                lv = Float64.(vcat([955], [960 + 5i for i = 0:11], [1020]))
                (lv, lv[3:2:(end - 1)])
            elseif isapprox(day, 10; atol = 1)
                lv = Float64.(vcat([920], [930 + 10i for i = 0:9], [1030]))
                (lv, lv[3:2:(end - 1)])
            else
                lv = nice_levels(extrema(p_m)..., 11)
                (lv, lv[1:2:end])
            end
            gl1 = fig[1, pc] = CairoMakie.GridLayout()
            draw_baroclinic_panel!(gl1, lon_m, lat_m, p_m, tri, faces, plevels,
                CairoMakie.cgrad(linear_band_colors(plevels, :plasma); categorical = true),
                "Surface Pressure", day, "hPa";
                extendlow = plasma[0.0], extendhigh = plasma[1.0], cbar_ticks = pticks)

            Tlevels = Float64.(vcat([220], [230 + 10i for i = 0:7], [310]))
            gl2 = fig[2, pc] = CairoMakie.GridLayout()
            draw_baroclinic_panel!(gl2, lon_m, lat_m, T_m, tri, faces, Tlevels,
                CairoMakie.cgrad(linear_band_colors(Tlevels, :plasma); categorical = true),
                "850 hPa Temperature", day, "K";
                extendlow = plasma[0.0], extendhigh = plasma[1.0],
                cbar_ticks = Tlevels[2:(end - 1)])

            vorlevels = Float64.(vcat([-10], [-5 + 5i for i = 0:6], [30]))
            band_colors = twoslope_band_colors(vorlevels, CairoMakie.cgrad(:RdBu; rev = true), -10.0, 0.0, 30.0)
            gl3 = fig[3, pc] = CairoMakie.GridLayout()
            draw_baroclinic_panel!(gl3, lon_m, lat_m, vor_m, tri, faces, vorlevels,
                CairoMakie.cgrad(band_colors; categorical = true),
                "850 hPa Vorticity", day, "1e-5/s";
                extendlow = cg_vor[0.0], extendhigh = cg_vor[1.0],
                cbar_ticks = vorlevels[3:2:(end - 1)])
        end
        CairoMakie.colgap!(fig.layout, 110)
        CairoMakie.rowgap!(fig.layout, 75)
        CairoMakie.resize_to_layout!(fig)
        out = fname_out === nothing ?
              "contour_compare_euler_$(Kh)x$(Kv)_p$(polydeg)_$(time_method)_days$(day1)-$(day2)_cairomakie.png" :
              fname_out
        CairoMakie.save(out, fig; px_per_unit = 2)
    end
    return nothing
end
