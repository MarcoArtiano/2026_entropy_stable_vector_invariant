# Numerical experiments
The test cases have been run with Julia v1.10.6. It is strongly recommended to start julia with multiple threads, as some simulations take hours to be completed. To do that, start julia with the following command:
```bash
julia --project=. --threads=num
```
where `num` is the number of threads that you want to use for that session. Alternatively, you can also set `--threads=auto` to use a reasonable default number of threads for your system.

The equations are defined inside the folder `equations/`,  as well as other utility functions.

Before running the simulations, please activate the project and instantiate, i.e., start Julia as described above and run the following code in the Julia REPL:
```julia
julia> using Pkg
julia> Pkg.activate(".")
julia> Pkg.instantiate()
```

## Test case 1: Taylor-Green Vortex

Run the following command to reproduce the results
```julia
julia> include("tgv/tgv.jl")
```

## Test case 2: Well-balancedness
Run the following commands to reproduce the results
```julia
julia> include("well_balance/well_balance.jl")
```
to reproduce the results about the well-balancedness on curved mesh.

## Test case 3: Inertia gravity waves
Run the following command to reproduce the results
```julia
julia> include("gravity_waves/convergence_analysis.jl")
```

## Test case 4: Linear hydrostatic mountain
Run the following command to reproduce the results
```julia
julia> include("linear_hydrostatic/linear_hydrostatic.jl")
```

## Test case 5: Linear nonhydrostatic mountain
Run the following command to reproduce the results
```julia
julia> include("linear_nonhydrostatic/linear_nonhydrostatic.jl")
```

## Test case 6: Breaking waves
Run the following command to reproduce the results
```julia
julia> include("breaking_waves/elixir_breaking_waves.jl")
```

## Test case 7: Critical layer
Run the following commands to reproduce the results
```julia
julia> include("critical_layer/critical_layer_tracers.jl")
julia> include("critical_layer/postprocess_tracers.jl")
```

## Test case 8: Baroclinic instability
Run the following command to reproduce the results
```julia
julia> include("baroclinic/run_baroclinic.jl")
```
