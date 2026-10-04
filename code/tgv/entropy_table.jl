using Printf

const RESULTS_DIR = joinpath(@__DIR__, "..", "results")
const OUT_TEX = joinpath(@__DIR__, "entropy_table.tex")
const DEGREES = 2:5
const COL_RES_S = 16   # dsdu_ut               (entropy residual)
const COL_RES_E = 17   # energy_timederivative (energy residual)
const COL_ETOT  = 18   # e_total
const COL_ENTR  = 19   # entropy

function conservation_metrics(path)
    S0 = En0 = NaN
    maxdS = maxdE = 0.0
    maxrs = maxre = 0.0
    for line in eachline(path)
        (isempty(line) || startswith(strip(line), "#")) && continue
        c = split(strip(line))
        isempty(c) && continue
        s   = parse(Float64, c[COL_ENTR])
        en  = parse(Float64, c[COL_ETOT])
        rs  = parse(Float64, c[COL_RES_S])
        re  = parse(Float64, c[COL_RES_E])
        isnan(S0) && (S0 = s; En0 = en)
        maxdS = max(maxdS, abs(s - S0))
        maxdE = max(maxdE, abs(en - En0))
        maxrs = max(maxrs, abs(rs))
        maxre = max(maxre, abs(re))
    end
    return maxdS, maxdE, maxrs, maxre
end

fmt(x) = x == 0 ? "\\num{0}" : "\\num{" * @sprintf("%.2e", x) * "}"

rows = String[]
for p in DEGREES
    f = joinpath(RESULTS_DIR, "analysis_entropy_$(p).dat")
    isfile(f) || (@warn("missing file, skipped", file = f); continue)
    dS, dE, rs, re = conservation_metrics(f)
    push!(rows,
          "        $(p) & $(fmt(dS)) & $(fmt(rs)) & $(fmt(dE)) & $(fmt(re)) \\\\")
end

table = """
\\begin{table}[h!]
    \\centering
    \\caption{Maximum entropy error \$| \\int U_{\\varrho s} - \\int U_{\\varrho s,0}|\$ and
    entropy residual \$\\vec{\\omega}_{\\varrho s}^T \\vec{M} \\partial_t \\vec{u}\$, and total-energy
    error \$|\\int U_{\\varrho E} - \\int U_{\\varrho E,0}|\$ and total-energy residual
    \$\\vec{\\omega}_{\\varrho E}^T \\vec{M} \\partial_t \\vec{u}\$, for the Taylor-Green vortex for
    different polynomial degrees \$p\$ up to the final time \$T = 5\$.}
    \\label{tab:tgv_conservation}
    \\begin{tabular}{ccccc}
        \\toprule
        & \\multicolumn{2}{c}{Thermodynamic entropy} & \\multicolumn{2}{c}{Total energy} \\\\
        \\cmidrule(lr){2-3} \\cmidrule(lr){4-5}
        \$p\$ & \$|\\int U_{\\varrho s} - \\int U_{\\varrho s,0}|\$ & \$\\vec{\\omega}_{\\varrho s}^T \\vec{M} \\partial_t \\vec{u}\$ & \$|\\int U_{\\varrho E} - \\int U_{\\varrho E,0}|\$ & \$\\vec{\\omega}_{\\varrho E}^T \\vec{M} \\partial_t \\vec{u}\$ \\\\
        \\midrule
$(join(rows, "\n"))
        \\bottomrule
    \\end{tabular}
\\end{table}
"""

print(table)
write(OUT_TEX, table)
println("\n# written to ", OUT_TEX)
