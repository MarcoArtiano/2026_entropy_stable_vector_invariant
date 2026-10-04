using DelimitedFiles
using CairoMakie
using LaTeXStrings

const OUTDIR = joinpath(@__DIR__, "..", "results", "out")
load(f) = readdlm(joinpath(OUTDIR, f); comments = true, comment_char = '#')

# data files (Float64 / MultiFloat{Float64,2}, isothermal / isentropic), T = 864000 s
f64_iso = "well_balancing_Float64_isothermal_0.025_864000.0.dat"
f64_ise = "well_balancing_Float64_isentropic_0.025_864000.0.dat"
mfd = "0.025000000000000001387778780781445675529539585113525390625"
mf_iso = "well_balancing_MultiFloat{Float64, 2}_isothermal_$(mfd)_864000.0.dat"
mf_ise = "well_balancing_MultiFloat{Float64, 2}_isentropic_$(mfd)_864000.0.dat"

d = Dict(:f64_iso => load(f64_iso), :f64_ise => load(f64_ise),
         :mf_iso => load(mf_iso), :mf_ise => load(mf_ise))

# columns: time = 2, ||u||_2 = l2_v1 = 5, ||w||_2 = l2_v2 = 6.
const TCOL, UCOL, WCOL = 2, 5, 6
const SECONDS_PER_DAY = 86400.0
c_iso = Makie.RGBf(0/255, 114/255, 178/255)    # blue  — T = const (isothermal)
c_ise = Makie.RGBf(230/255, 159/255, 0/255)    # orange — θ = const (isentropic)

tdays(data) = data[2:end, TCOL] ./ SECONDS_PER_DAY

function draw_panel!(ax, data_iso, data_ise)
    lines!(ax, tdays(data_iso), abs.(data_iso[2:end, UCOL]);
           color = c_iso, linewidth = 2.5)
    lines!(ax, tdays(data_iso), abs.(data_iso[2:end, WCOL]);
           color = c_iso, linewidth = 4.0, linestyle = :dash)
    lines!(ax, tdays(data_ise), abs.(data_ise[2:end, UCOL]);
           color = c_ise, linewidth = 2.5)
    lines!(ax, tdays(data_ise), abs.(data_ise[2:end, WCOL]);
           color = c_ise, linewidth = 4.0, linestyle = :dash)
    return nothing
end

fig = Figure(size = (1200, 470))
kw = (xlabel = L"t\;[\mathrm{days}]", xlabelsize = 24, xticklabelsize = 17,
      yticklabelsize = 17, titlesize = 20, yscale = log10,
      xticks = 0:2:10, limits = ((0, 10), nothing))

axL = Axis(fig[2, 1]; kw..., title = L"\mathrm{MultiFloat\{Float64,2\}}")
axR = Axis(fig[2, 2]; kw..., title = L"\mathrm{Float64}")
draw_panel!(axL, d[:mf_iso], d[:mf_ise])
draw_panel!(axR, d[:f64_iso], d[:f64_ise])

legend_elems = [
    LineElement(color = c_iso, linewidth = 2.5),
    LineElement(color = c_iso, linewidth = 4.0, linestyle = :dash),
    LineElement(color = c_ise, linewidth = 2.5),
    LineElement(color = c_ise, linewidth = 4.0, linestyle = :dash),
]
legend_labels = [
    L"\|u\|_2\ (T=\mathrm{const})",
    L"\|w\|_2\ (T=\mathrm{const})",
    L"\|u\|_2\ (\theta=\mathrm{const})",
    L"\|w\|_2\ (\theta=\mathrm{const})",
]
Legend(fig[1, 1:2], legend_elems, legend_labels;
       orientation = :horizontal, nbanks = 1, labelsize = 20, patchsize = (45, 18),
       framevisible = true)

outpdf = joinpath(@__DIR__, "..", "results", "balance_float_multifloat.pdf")
save(outpdf, fig)