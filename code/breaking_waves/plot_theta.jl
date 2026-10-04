using CairoMakie
using LaTeXStrings

include(joinpath(@__DIR__, "..", "contours.jl"))

times_hours = [2.5, 3.0]             

mkpath(joinpath(@__DIR__, "..", "results"))
for th in times_hours
    i = findmin(abs.(sol.t .- th * 3600))[2]
    x, y, data = ContourData(sol.u[i], semi, cells_per_dimension, equations)
    theta = data[4, :, :] ./ data[1, :, :]

    h = Figure(size = (900, 500))
    Axis(
        h[1, 1];
        title = "t = $(round(sol.t[i] / 3600, digits = 2)) h",
        xlabel = L"$x$ [km]",
        ylabel = L"$z$ [km]",
        xlabelsize = 20,
        ylabelsize = 20,
        limits = ((-L / 2e3, L / 2e3), (0, H / 1e3)),
    )
    CairoMakie.contour!(
        x ./ 1e3,
        y ./ 1e3,
        theta,
        levels = 230:5:430,
        color = :black,
        linewidth = 0.7,
    )
    save(
        joinpath(
            @__DIR__, "..", "results",
            "breaking_theta_$(round(sol.t[i] / 3600, digits = 2))h_" *
            "$(cells_per_dimension[1])x$(cells_per_dimension[2])_p$(polydeg).pdf",
        ),
        h,
    )
end