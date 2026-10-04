using CairoMakie
using DataFrames
using CSV

function retrieve_data_momentum(name, cells_per_dimension, polydeg, alfa, xr_B, form)
    folder_path = joinpath(@__DIR__, "..") * "/results/"

    t1 = CSV.read(
        folder_path *
        "$(name)_2.5_$(cells_per_dimension[1])x$(cells_per_dimension[2])_$(polydeg)_$(alfa)_$(xr_B)_$(form).csv",
        DataFrame,
    )
    t2 = CSV.read(
        folder_path *
        "$(name)_5_$(cells_per_dimension[1])x$(cells_per_dimension[2])_$(polydeg)_$(alfa)_$(xr_B)_$(form).csv",
        DataFrame,
    )
    t3 = CSV.read(
        folder_path *
        "$(name)_7.5_$(cells_per_dimension[1])x$(cells_per_dimension[2])_$(polydeg)_$(alfa)_$(xr_B)_$(form).csv",
        DataFrame,
    )
    t4 = CSV.read(
        folder_path *
        "$(name)_10_$(cells_per_dimension[1])x$(cells_per_dimension[2])_$(polydeg)_$(alfa)_$(xr_B)_$(form).csv",
        DataFrame,
    )
    t5 = CSV.read(
        folder_path *
        "$(name)_12.5_$(cells_per_dimension[1])x$(cells_per_dimension[2])_$(polydeg)_$(alfa)_$(xr_B)_$(form).csv",
        DataFrame,
    )
    z = t1[:, 1] ./ 1e3
    return t1, t2, t3, t4, t5, z
end

function plot_time_momentum(;
    cells_per_dimension = (100, 60),
    polydeg = 3,
    alfa = 0.035,
    name = "VectorInvariant_hydrostatic",
    form = "false",
    xr_B = 60000,
)

    t1, t2, t3, t4, t5, z =
        retrieve_data_momentum(name, cells_per_dimension, polydeg, alfa, xr_B, form)

    fig = CairoMakie.Figure(size = (700, 600))
    kwargs = (
        xlabel = L"\overline{m}(z)",
        xlabelsize = 19,
        limits = ((0.5, 1.2), (0, 12)),
        xticklabelsize = 17.0,
        yticklabelsize = 17.0,
        titlesize = 20,
    )

    ax1 = Axis(fig[2, 1]; kwargs..., ylabel = L"$z$ [km]", ylabelsize = 19)

    kwargs_plot = (linewidth = 2.6,)

    lines!(ax1, t1[:, 2], z; label = L"$2.5$ h", kwargs_plot..., color = colors[1])
    lines!(ax1, t2[:, 2], z; label = L"$5$ h", kwargs_plot..., color = colors[2])
    lines!(ax1, t3[:, 2], z; label = L"$7.5$ h", kwargs_plot..., color = colors[3])
    lines!(ax1, t4[:, 2], z; label = L"$10$ h", kwargs_plot..., color = colors[4])
    lines!(ax1, t5[:, 2], z; label = L"$12.5$ h", kwargs_plot..., color = colors[5])

    leg = Legend(fig[1, 1], ax1, nothing, orientation = :horizontal, labelsize = 18.0)

    save(joinpath(@__DIR__, "..") * "/results/linearhydrostatic_momentum.pdf", fig)

    return nothing
end

colors = Makie.wong_colors()

plot_time_momentum(alfa = 0.035)