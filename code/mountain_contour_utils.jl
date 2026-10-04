using CairoMakie
using LaTeXStrings
using JLD2

function save_contour_fields(out_jld2, x, y, data; kwargs...)
    JLD2.jldsave(out_jld2; x = x, y = y, data = data, kwargs...)
    return nothing
end

function plot_uw_contour(x, y, data, limits, out_pdf; levels = 13)
    h = Figure(size = (850, 400))
    labelsize = 15
    kwargs = (xlabel = L"$x$ [km]", xlabelsize = labelsize, ylabelsize = labelsize,
              limits = limits, xticklabelsize = 13.0, yticklabelsize = 13.0)

    Axis(h[1, 1]; kwargs..., ylabel = L"$z$ [km]")
    c = contourf!(x ./ 1e3, y ./ 1e3, data[2, :, :], levels = levels, colormap = :PuOr)
    Colorbar(h[1, 2], c, ticklabelsize = 13)
    Label(h[1, 2, Top()], L"u\;[\mathrm{m/s}]", fontsize = 18, padding = (0, 0, 6, 0))

    Axis(h[1, 3]; kwargs...)
    c = contourf!(x ./ 1e3, y ./ 1e3, data[3, :, :], levels = levels, colormap = :PuOr)
    Colorbar(h[1, 4], c, ticklabelsize = 13)
    Label(h[1, 4, Top()], L"w\;[\mathrm{m/s}]", fontsize = 18, padding = (0, 0, 6, 0))

    save(out_pdf, h)
    println("saved figure -> ", out_pdf)
    return nothing
end

function replot_from_jld2(in_jld2; out_pdf = replace(in_jld2, ".jld2" => ".pdf"),
                          limits = nothing, levels = 13)
    d = JLD2.load(in_jld2)
    lim = limits === nothing ? d["limits"] : limits
    plot_uw_contour(d["x"], d["y"], d["data"], lim, out_pdf; levels = levels)
end