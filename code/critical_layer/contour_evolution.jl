using CairoMakie
using LaTeXStrings

include(joinpath(@__DIR__, "..", "contours.jl"))

function evolution_figure(sol, semi, cells_per_dimension, polydeg, equations;
                          times_hours, wmax = 0.25, nlevels = 13, zlims = (0, 15))
    snap_idx = [findmin(abs.(sol.t .- th * 3600))[2] for th in times_hours]
    ncols = min(3, length(snap_idx))
    nrows = ceil(Int, length(snap_idx) / ncols)

    wlevels = range(-wmax, wmax, length = nlevels)

    h = Figure(size = (400 * ncols + 120, 380 * nrows))
    c = nothing
    for (n, i) in enumerate(snap_idx)
        row = div(n - 1, ncols) + 1
        col = mod(n - 1, ncols) + 1
        x, y, data = ContourData(sol.u[i], semi, cells_per_dimension, equations)
        theta = data[4, :, :] ./ data[1, :, :]
        Axis(
            h[row, col];
            title = L"$t = %$(round(sol.t[i] / 3600, digits = 2))$ h",
            xlabel = L"$x$ [km]",
            ylabel = col == 1 ? L"$z$ [km]" : "",
            xlabelsize = 23,
            ylabelsize = 23,
            titlesize = 25,
            xticklabelsize = 20.0,
            yticklabelsize = 20.0,
            limits = ((-5, 5), zlims),
        )
        dw = step(wlevels)
        w = clamp.(data[3, :, :], -wmax + dw / 2, wmax - dw / 2)
        c = CairoMakie.contourf!(
            x ./ 1e3,
            y ./ 1e3,
            w,
            levels = wlevels,
            colormap = :PuOr,
        )
        CairoMakie.contour!(
            x ./ 1e3,
            y ./ 1e3,
            theta,
            levels = 20,
            color = (:gray30, 0.8),
            linewidth = 0.7,
        )
    end
    Colorbar(h[:, ncols+1], c, ticklabelsize = 20,
             ticks = range(-wmax, wmax, length = 5),
             tickformat = vs -> [replace(string(round(v, digits = 3)), r"\.?0+$" => "") for v in vs])
    Label(h[:, ncols+1, Top()], L"w\;[\mathrm{m/s}]", fontsize = 28,
          padding = (0, 0, 18, 0))

    outdir = joinpath(@__DIR__, "..") * "/results"
    mkpath(outdir)
    save(
        outdir *
        "/critical_layer_w_$(cells_per_dimension[1])x$(cells_per_dimension[2])_p$(polydeg).pdf",
        h,
    )
    return h
end