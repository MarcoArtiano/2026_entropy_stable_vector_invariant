using Trixi, JLD2, CSV, DataFrames
using CairoMakie, LaTeXStrings

const plot_times = [4.5, 6.0, 7.5, 9.0, 10.5, 12.0]
const tracer_index = 1

if !@isdefined(snapshot_file)
    snapshot_file = joinpath(@__DIR__, "..") *
                    "/critical_layer/results/snapshots_layers_a20.0_12.0_80x180_3.jld2"
end

snap = load(snapshot_file)
ts = snap["t"]
us = snap["u"]

trixi_include(
    joinpath(@__DIR__, "critical_layer_tracers.jl");
    do_solve = false,
    cells_per_dimension = Tuple(snap["cells_per_dimension"]),
    polydeg = snap["polydeg"],
    T_hours = snap["T_hours"],
)
@assert snap["a0"] == a0 "a0 of the snapshot ($(snap["a0"])) differs from the elixir const ($a0)"
@assert snap["tracer_shape"] == String(tracer_shape) "tracer_shape mismatch with the elixir"
@assert length(snap["tracer_centers_z"]) == n_tracers "the snapshot stores $(length(snap["tracer_centers_z"])) tracers, the elixir defines $n_tracers"
@assert tracer_index <= n_tracers "tracer_index = $tracer_index exceeds the $n_tracers available tracers"

run_tag = "$(snap["tracer_shape"])_a$(snap["a0"])_$(snap["T_hours"])_$(cells_per_dimension[1])x$(cells_per_dimension[2])_$(polydeg)"
res_tag = "$(cells_per_dimension[1])x$(cells_per_dimension[2])_p$(polydeg)"
figdir = joinpath(@__DIR__, "..") * "/results"
outdir = joinpath(@__DIR__, "..") * "/critical_layer/results"
mkpath(figdir)
mkpath(outdir)

n_flow = Trixi.nvariables(equations.flow_equations)
sol = (u = us, t = ts)

include(joinpath(@__DIR__, "contour_evolution.jl"))
include(joinpath(@__DIR__, "contour_tracers_evolution.jl"))

evolution_figure(sol, semi, cells_per_dimension, polydeg, equations;
                 times_hours = plot_times, wmax = 0.25)

tracer_evolution_figure(sol, semi, cells_per_dimension, polydeg, equations;
                        times_hours = plot_times, tracer_index = tracer_index)

function profile_over_line(m, semi, cells_per_dimension, polydeg)
    @unpack solver, cache = semi
    @unpack weights = solver.basis

    integral = zeros(cells_per_dimension[2] * (polydeg + 1))
    z_coords = copy(integral)

    jstart = 1
    for z = 1:cells_per_dimension[2]*(polydeg+1)
        if (z % (polydeg + 1) == 1 && z != 1)
            jstart += 1
        end
        loc_integral = 0.0
        iterator = (1+cells_per_dimension[1]*(jstart-1)):(cells_per_dimension[1]*jstart)
        for element in iterator
            for i in eachnode(solver)
                j = z % (polydeg + 1)
                j == 0 && (j = polydeg + 1)
                jacobian = L / cells_per_dimension[1] / 2
                loc_integral += jacobian * weights[i] * m[i, j, element]
            end
        end
        j = z % (polydeg + 1)
        j == 0 && (j = polydeg + 1)
        z_coords[z] = cache.elements.node_coordinates[
            2,
            1,
            j,
            Int(iterator[end] - cells_per_dimension[1] / 2 + 1),
        ]
        integral[z] = loc_integral
    end

    return integral, z_coords
end

nz = cells_per_dimension[2] * (polydeg + 1)
chi_profiles = zeros(nz, length(ts))
z_line = zeros(nz)
for (k, u_ode) in enumerate(us)
    uw = Trixi.wrap_array(u_ode, semi)
    num, zc = profile_over_line(uw[n_flow+tracer_index, :, :, :], semi,
                                cells_per_dimension, polydeg)
    den, _ = profile_over_line(uw[1, :, :, :], semi, cells_per_dimension, polydeg)
    chi_profiles[:, k] = num ./ den
    global z_line = zc
end

df_prof = DataFrame(z_coords = z_line)
for (k, t) in enumerate(ts)
    df_prof[!, "chi_t$(round(t / 3600, digits = 2))h"] = chi_profiles[:, k]
end
CSV.write(outdir * "/tracer_mean_profiles_$(run_tag).csv", df_prof)

colors = Makie.wong_colors()
fig = Figure(size = (700, 600))
ax = Axis(fig[2, 1]; xlabel = L"\langle \chi \rangle - \langle \chi \rangle_0",
          ylabel = L"$z$ [km]", xlabelsize = 19, ylabelsize = 19,
          xticklabelsize = 17.0, yticklabelsize = 17.0, limits = (nothing, (7, 11)))
for (n, th) in enumerate(plot_times)
    k = findmin(abs.(ts .- th * 3600))[2]
    lines!(ax, chi_profiles[:, k] .- chi_profiles[:, 1], z_line ./ 1e3;
           label = L"$%$(round(ts[k] / 3600, digits = 1))$ h", linewidth = 2.6,
           color = colors[n])
end
Legend(fig[1, 1], ax, nothing, orientation = :horizontal, labelsize = 18.0)
save(figdir * "/critical_layer_mixing_$(res_tag).pdf", fig)