using DelimitedFiles
using CairoMakie
using LaTeXStrings

files = [("out/analysis_30x8_p3_ARK2GKC()_90.0_20.dat", L"p = 3"),
         ("out/analysis_15x4_p7_ARK2GKC()_50.0_20.dat", L"p = 7"),
         ("out/analysis_8x4_p5_ARK2GKC()_150.0_20.dat", L"p = 7")]
c_p3 = Makie.RGBf(0 / 255, 114 / 255, 178 / 255)
c_p7 = Makie.RGBf(230 / 255, 159 / 255, 0 / 255)
colors = [c_p3, c_p7]
const SECONDS_PER_DAY = 86400.0

set_theme!(theme_latexfonts())
fig = Figure(size = (1700, 620))
kw = (xlabel = L"t\;[\mathrm{days}]", xlabelsize = 31, xticklabelsize = 24,
      yticklabelsize = 24, ylabelsize = 28,
      xticks = 0:4:20, limits = ((0, 20), nothing))

ax1 = Axis(fig[2, 1]; kw...,
           ylabel = L"(U_{\varrho s}(t) - U_{\varrho s}(0))\, /\, |U_{\varrho s}(0)|")
ax2 = Axis(fig[2, 2]; kw..., ylabel = L"\mathrm{Minimum\;Surface\;Pressure\;[hPa]}")
ax3 = Axis(fig[2, 3]; kw..., ylabel = L"\mathrm{Maximum\;Horizontal\;Wind\;Speed\;[m/s]}")

for ((file, lab), colr) in zip(files, colors)
    header = split(replace(readline(file), "#" => ""))
    data = readdlm(file, skipstart = 1)
    col = Dict(name => idx for (idx, name) in enumerate(header))
    day = data[:, col["time"]] ./ SECONDS_PER_DAY
    S = data[:, col["entropy"]]
    lines!(ax1, day, (S .- S[1]) ./ abs(S[1]); color = colr, linewidth = 3.5)
    lines!(ax2, day, data[:, col["min_psurf"]]; color = colr, linewidth = 3.5)
    lines!(ax3, day, data[:, col["max_vh"]]; color = colr, linewidth = 3.5)
end

legend_elems = [LineElement(color = c, linewidth = 3.5) for c in colors]
Legend(fig[1, 1:3], legend_elems, [lab for (_, lab) in files];
       orientation = :horizontal, nbanks = 1, labelsize = 28, patchsize = (60, 24),
       framevisible = true)

outpdf = joinpath(@__DIR__, "..", "results", "baroclinic_tseries_p3_p7.pdf")
save(outpdf, fig)