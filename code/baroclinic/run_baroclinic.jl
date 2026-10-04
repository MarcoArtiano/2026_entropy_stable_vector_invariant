using Trixi

trixi_include(joinpath(@__DIR__, "elixir_imex_baroclinic.jl"), trees_per_cube_face = (8, 4), polydeg = 5, dt = 180.0, T = 8, analysis_interval = 1000, jac_interval = 20)
trixi_include(joinpath(@__DIR__, "elixir_imex_baroclinic.jl"), trees_per_cube_face = (8, 4), polydeg = 5, dt = 180.0, T = 10, analysis_interval = 1000, jac_interval = 20)

contour_baroclinic_compare_days_cairomakie(trees_per_cube_face, polydeg, time_method)

trixi_include(joinpath(@__DIR__, "elixir_imex_baroclinic.jl"), trees_per_cube_face = (30, 8), polydeg = 3, dt = 90.0, T = 8, analysis_interval = 1000, jac_interval = 20)
trixi_include(joinpath(@__DIR__, "elixir_imex_baroclinic.jl"), trees_per_cube_face = (30, 8), polydeg = 3, dt = 90.0, T = 10, analysis_interval = 1000, jac_interval = 20)

contour_baroclinic_compare_days_cairomakie(trees_per_cube_face, polydeg, time_method)

trixi_include(joinpath(@__DIR__, "elixir_imex_baroclinic.jl"), trees_per_cube_face = (15, 4), polydeg = 7, dt = 50.0, T = 8, analysis_interval = 1000, jac_interval = 20)
trixi_include(joinpath(@__DIR__, "elixir_imex_baroclinic.jl"), trees_per_cube_face = (15, 4), polydeg = 7, dt = 50.0, T = 10, analysis_interval = 1000, jac_interval = 20)

contour_baroclinic_compare_days_cairomakie(trees_per_cube_face, polydeg, time_method)

trixi_include(joinpath(@__DIR__, "elixir_imex_baroclinic.jl"), trees_per_cube_face = (15, 4), polydeg = 7, dt = 50.0, T = 20, analysis_interval = 20, jac_interval = 20)
trixi_include(joinpath(@__DIR__, "elixir_imex_baroclinic.jl"), trees_per_cube_face = (30, 8), polydeg = 3, dt = 90.0, T = 20, analysis_interval = 10, jac_interval = 20)

include(joinpath(@__DIR__, "plot_time_series.jl"))