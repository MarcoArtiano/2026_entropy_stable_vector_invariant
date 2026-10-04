using CairoMakie
using LaTeXStrings

include(joinpath(@__DIR__, "..", "contours.jl"))

cons2theta(u, z, equations::Trixi.PassiveTracerEquations) =
    cons2theta(u, z, equations.flow_equations)

function tracer_evolution_figure(sol, semi, cells_per_dimension, polydeg, equations;
                                 times_hours, tracer_index = 1, nlevels = 11,
                                 zlims = (7, 11))
    n_flow = Trixi.nvariables(equations.flow_equations)
    snap_idx = [findmin(abs.(sol.t .- th * 3600))[2] for th in times_hours]
    ncols = min(3, length(snap_idx))
    nrows = ceil(Int, length(snap_idx) / ncols)

    chi_levels = range(0.0, 1.0, length = nlevels)

    outdir = joinpath(@__DIR__, "..") * "/results"
    mkpath(outdir)

    t = tracer_index
    h = Figure(size = (400 * ncols + 120, 380 * nrows))
    c = nothing
    for (n, i) in enumerate(snap_idx)
        row = div(n - 1, ncols) + 1
        col = mod(n - 1, ncols) + 1
        x, y, data = ContourData(sol.u[i], semi, cells_per_dimension, equations)
        chi = data[n_flow+t, :, :] ./ data[1, :, :]
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
        dc = step(chi_levels)
        chi = clamp.(chi, dc / 2, 1 - dc / 2)
        c = CairoMakie.contourf!(
            x ./ 1e3,
            y ./ 1e3,
            chi,
            levels = chi_levels,
            colormap = :Purples,
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
    Colorbar(h[:, ncols+1], c, ticklabelsize = 20, ticks = 0:0.25:1)
    Label(h[:, ncols+1, Top()], L"\chi", fontsize = 28, padding = (0, 0, 6, 0))

    save(
        outdir *
        "/critical_layer_chi_$(cells_per_dimension[1])x$(cells_per_dimension[2])_p$(polydeg).pdf",
        h,
    )
    return h
end