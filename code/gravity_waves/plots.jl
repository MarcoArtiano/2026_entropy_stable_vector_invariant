using CSV, DataFrames, CairoMakie, LaTeXStrings

function convergence_plots_warped!(vars, fluxes, yboundss, title, filename)
    var = vars[1]
    flux = fluxes[1]
    ybounds = yboundss[1]
    fig = CairoMakie.Figure(size = (1600, 1100))
    CFL = 0.1
    var1 = "up"
    var2 = "wp"
    var3 = "Tp"
    var4 = "pp"
    flux1 = flux_volume_combined_turbo_entropy_conservative
    df1 = CSV.read(
        joinpath(@__DIR__, "..") * "/results/warped_" * var1 * "_error_CFL_$(CFL)PT_NC_$(flux1)___.csv",
        DataFrame,
    )
    df2 = CSV.read(
        joinpath(@__DIR__, "..") * "/results/warped_" * var2 * "_error_CFL_$(CFL)PT_NC_$(flux1)___.csv",
        DataFrame,
    )
    df3 = CSV.read(
        joinpath(@__DIR__, "..") * "/results/warped_" * var3 * "_error_CFL_$(CFL)PT_NC_$(flux1)___.csv",
        DataFrame,
    )
    df4 = CSV.read(
        joinpath(@__DIR__, "..") * "/results/warped_" * var4 * "_error_CFL_$(CFL)PT_NC_$(flux1)___.csv",
        DataFrame,
    )

    ymin1, ymax1 = (10^-10, 10^-3)
    ymin2, ymax2 = (10^-10, 10^1)
    ymin3, ymax3 = (10^-10, 10^2)
    ymin4, ymax4 = (10^-10, 10^1)
    xlabel1 = L"$\Delta x$ [km]"
    title1 = L"L_2 \text{ Error } u"
    title2 = L"L_2 \text{ Error } w"
    title3 = L"L_2 \text{ Error } T"
    title4 = L"L_2 \text{ Error } p"
    ymin = ybounds[1]
    ymax = ybounds[2]
    xlabel1 = L"$\Delta x$ [km]"
    if var == "wp"
        ylabel1 = L"||w||_2"
        titlelegend = L"L_2 \text{ Error } w"
    elseif var == "up"
        ylabel1 = L"||u||_2"
        titlelegend = L"L_2 \text{ Error } u"
    elseif var == "pp"
        ylabel1 = L"||p||_2"
        titlelegend = L"L_2 \text{ Error } p"
    elseif var == "Tp"
        ylabel1 = L"||T||_2"
        titlelegend = L"L_2 \text{ Error } T"
    elseif var == "thetap"
        ylabel1 = L"||\theta||_2"
        titlelegend = L"L_2 \text{ Error } \theta"
    end
    titlesize = 32
    xlabelsize = 26.0
    ylabelsize = 32
    xticklabelsize = 25.0
    yticklabelsize = 25.0
    xticks1 = ([1.25, 2.5, 5.0, 10.0])
    yticks1 = LogTicks(WilkinsonTicks(6, k_min = 4))

    ax1 = Axis(
        fig[2, 1],
        title = title1,
        limits = (nothing, (ymin, ymax)),
        xticks = xticks1,
        yticks = yticks1,
        xlabel = xlabel1,
        xscale = log10,
        yscale = log10,
        xlabelsize = xlabelsize,
        ylabelsize = ylabelsize,
        xticklabelsize = xticklabelsize,
        yticklabelsize = yticklabelsize,
        titlesize = titlesize,
    )
    ax2 = Axis(
        fig[2, 2],
        title = title2,
        limits = (nothing, (ymin, ymax)),
        xticks = xticks1,
        yticks = yticks1,
        xlabel = xlabel1,
        xscale = log10,
        yscale = log10,
        xlabelsize = xlabelsize,
        ylabelsize = ylabelsize,
        xticklabelsize = xticklabelsize,
        yticklabelsize = yticklabelsize,
        titlesize = titlesize,
    )
    ax3 = Axis(
        fig[3, 1],
        title = title3,
        limits = (nothing, (ymin, ymax)),
        xticks = xticks1,
        yticks = yticks1,
        xlabel = xlabel1,
        xscale = log10,
        yscale = log10,
        xlabelsize = xlabelsize,
        ylabelsize = ylabelsize,
        xticklabelsize = xticklabelsize,
        yticklabelsize = yticklabelsize,
        titlesize = titlesize,
    )
    ax4 = Axis(
        fig[3, 2],
        title = title4,
        limits = (nothing, (ymin, ymax)),
        xticks = xticks1,
        yticks = yticks1,
        xlabel = xlabel1,
        xscale = log10,
        yscale = log10,
        xlabelsize = xlabelsize,
        ylabelsize = ylabelsize,
        xticklabelsize = xticklabelsize,
        yticklabelsize = yticklabelsize,
        titlesize = titlesize,
    )

    plot_data!(ax1, df1)
    plot_data!(ax2, df2)
    plot_data!(ax3, df3)
    plot_data!(ax4, df4)
    titlelegend = nothing
    leg = Legend(
        fig[1, 1:2],
        ax3,
        titlelegend,
        titlesize = 30,
        orientation = :horizontal,
        labelsize = 28.0,
    )

    save(joinpath(@__DIR__, "..") * "/results/warped_" * filename * "_" * vars[1] * vars[2] * ".pdf", fig)
end

function convergence_plots!(
    vars,
    fluxes,
    ybounds,
    title,
    filename;
    CFL = 0.1,
    flux_noncons = nothing,
)
    var1 = vars[1]
    var2 = vars[2]

    flux1 = fluxes[1]
    flux2 = fluxes[2]

    ybounds1 = ybounds[1]
    ybounds2 = ybounds[2]
    titlesizeleg = 30
    titlesize = 27
    fig = CairoMakie.Figure(size = (1600, 1100))
    flux1 = flux_volume_combined_turbo_entropy_conservative

    var1 = "up"
    var2 = "wp"
    var3 = "Tp"
    var4 = "pp"

    df1 = CSV.read(
        joinpath(@__DIR__, "..") * "/results/" * var1 * "_error_CFL_$(CFL)PT_NC_$(flux1)___.csv",
        DataFrame,
    )
    df2 = CSV.read(
        joinpath(@__DIR__, "..") * "/results/" * var2 * "_error_CFL_$(CFL)PT_NC_$(flux1)___.csv",
        DataFrame,
    )
    df3 = CSV.read(
        joinpath(@__DIR__, "..") * "/results/" * var3 * "_error_CFL_$(CFL)PT_NC_$(flux1)___.csv",
        DataFrame,
    )
    df4 = CSV.read(
        joinpath(@__DIR__, "..") * "/results/" * var4 * "_error_CFL_$(CFL)PT_NC_$(flux1)___.csv",
        DataFrame,
    )
    ymin1, ymax1 = (10^-10, 10^-3)
    ymin2, ymax2 = (10^-10, 10^0)
    ymin3, ymax3 = (10^-10, 10^3)
    ymin4, ymax4 = (10^-10, 10^1)
    ymin = ybounds1[1]
    ymax = ybounds1[2]
    xlabel1 = L"$\Delta x$ [km]"
    title1 = L"L_2 \text{ Error } u"
    title2 = L"L_2 \text{ Error } w"
    title3 = L"L_2 \text{ Error } T"
    title4 = L"L_2 \text{ Error } p"
    if var1 == "wp"
        ylabel1 = L"||w||_2"
        titlelegend = L"L_2 \text{ Error } w"
    elseif var1 == "up"
        ylabel1 = L"||u||_2"
        titlelegend = L"L_2 \text{ Error } u"
    elseif var1 == "pp"
        ylabel1 = L"||p||_2"
        titlelegend = L"L_2 \text{ Error } p"
    elseif var1 == "Tp"
        ylabel1 = L"||T||_2"
        titlelegend = L"L_2 \text{ Error } T"
    elseif var1 == "thetap"
        ylabel1 = L"||\theta||_2"
        titlelegend = L"L_2 \text{ Error } \theta"
    end
 titlesize = 32
    xlabelsize = 26.0
    ylabelsize = 32
    xticklabelsize = 25.0
    yticklabelsize = 25.0

    xticks1 = ([1.25, 2.5, 5.0, 10.0])
    yticks1 = LogTicks(WilkinsonTicks(6, k_min = 4))

    ax1 = Axis(
        fig[2, 1],
        title = title1,
        limits = (nothing, (ymin1, ymax1)),
        xticks = xticks1,
        yticks = yticks1,
        xlabel = xlabel1,
        xscale = log10,
        yscale = log10,
        xlabelsize = xlabelsize,
        ylabelsize = ylabelsize,
        xticklabelsize = xticklabelsize,
        yticklabelsize = yticklabelsize,
        titlesize = titlesize,
    )
    ax2 = Axis(
        fig[2, 2],
        title = title2,
        limits = (nothing, (ymin2, ymax2)),
        xticks = xticks1,
        yticks = yticks1,
        xlabel = xlabel1,
        xscale = log10,
        yscale = log10,
        xlabelsize = xlabelsize,
        ylabelsize = ylabelsize,
        xticklabelsize = xticklabelsize,
        yticklabelsize = yticklabelsize,
        titlesize = titlesize,
    )
    ax3 = Axis(
        fig[3, 1],
        title = title3,
        limits = (nothing, (ymin3, ymax3)),
        xticks = xticks1,
        yticks = yticks1,
        xlabel = xlabel1,
        xscale = log10,
        yscale = log10,
        xlabelsize = xlabelsize,
        ylabelsize = ylabelsize,
        xticklabelsize = xticklabelsize,
        yticklabelsize = yticklabelsize,
        titlesize = titlesize,
    )
    ax4 = Axis(
        fig[3, 2],
        title = title4,
        limits = (nothing, (ymin4, ymax4)),
        xticks = xticks1,
        yticks = yticks1,
        xlabel = xlabel1,
        xscale = log10,
        yscale = log10,
        xlabelsize = xlabelsize,
        ylabelsize = ylabelsize,
        xticklabelsize = xticklabelsize,
        yticklabelsize = yticklabelsize,
        titlesize = titlesize,
    )

    plot_data!(ax1, df1)
    plot_data!(ax2, df2)
    plot_data!(ax3, df3)
    plot_data!(ax4, df4)

    titlelegend = nothing
    leg = Legend(
        fig[1, 1:2],
        ax4,
        titlelegend,
        titlesize = titlesizeleg,
        orientation = :horizontal,
        labelsize = 28.0,
    )

    save(joinpath(@__DIR__, "..") * "/results/" * filename * "_" * var1 * var2 * ".pdf", fig)
end

function plot_data!(ax, df)

    unique_degrees = sort(unique(df.polydeg))
    for (i, deg) in enumerate(unique_degrees)
        subdf = filter(row -> row.polydeg == deg, df)
        scatterlines!(
            ax,
            subdf.dx ./ 1000,
            subdf.l2_error,
            color = colors[i],
            label = L"$p = %$(deg)$",
            markersize = 17,
            marker = markers[i],
        )
        x_ref = [maximum(df.dx), minimum(df.dx)]
        if deg == 4 #&& df.problem_name[1] == "EUNC"
            y_ref = subdf.l2_error[1] * ((x_ref) ./ subdf.dx[1]) .^ (deg + 1)
        else
            y_ref = subdf.l2_error[end] * ((x_ref) ./ subdf.dx[end]) .^ (deg + 1)
        end
        order = deg + 1
        label = latexstring("\$\\mathcal{O}(\\Delta x^{ $order })\$")
        #label = L"Order  %$(order)$"
        lines!(
            ax,
            x_ref ./ 1000,
            y_ref,
            color = colors[i],
            linestyle = linestyles[i],
            label = label,
            linewidth = 3,
        )
    end
end

colors = Makie.wong_colors()
markers = (:circle, :star5, :rect)
linestyles = (:dash, :dashdot, :dashdotdot)
convergence_plots!(
    ("up", "wp"),
    ("flux_theta", "flux_theta"),
    ((10^-10, 10^-3), (10^-10, 10^0)),
    "L2 error w",
    "Convergencefluxtheta",
)

convergence_plots_warped!(
    ("wp", "Tp"),
    ("flux_theta", "flux_theta"),
    ((10^-10, 10^1), (10^-10, 10^2)),
    "Convergence of w",
    "Convergencefluxtheta",
)