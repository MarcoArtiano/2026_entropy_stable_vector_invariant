function ContourData(sol, semi, cells_per_dimension, equations)

    node_coordinates = semi.cache.elements.node_coordinates
    polydeg = Trixi.nnodes(semi.solver.basis) - 1
    Nx = polydeg * cells_per_dimension[1] + 1
    Ny = polydeg * cells_per_dimension[2] + 1
    u = Trixi.wrap_array(sol, semi)

    x = zeros(Ny, Nx)
    y = copy(x)
    # u = sol.u
    nvars = size(u, 1)
    @show Ny, Nx
    data = zeros(Float64, nvars + 2, Ny, Nx)

    linear_indices = LinearIndices(cells_per_dimension)
    for cell_x = 1:cells_per_dimension[1]
        for cell_y = 1:cells_per_dimension[2]
            element = linear_indices[cell_x, cell_y]

            for jlocal = 1:polydeg
                for ilocal = 1:polydeg
                    i, j = compute_global_index(
                        element,
                        ilocal,
                        jlocal,
                        polydeg,
                        cell_x,
                        cell_y,
                    )
                    x[j, i] = node_coordinates[1, ilocal, jlocal, element]
                    y[j, i] = node_coordinates[2, ilocal, jlocal, element]
                    data[1:nvars, j, i] = u[:, ilocal, jlocal, element]
                    data[end-1, j, i] = cons2theta(
                        u[:, ilocal, jlocal, element],
                        node_coordinates[2, ilocal, jlocal, element],
                        equations,
                    )
                    data[end, j, i] = element
                end
            end

            if cell_x == cells_per_dimension[1]
                ilocal = polydeg + 1
                for jlocal = 1:polydeg
                    i, j = compute_global_index(
                        element,
                        ilocal,
                        jlocal,
                        polydeg,
                        cell_x,
                        cell_y,
                    )
                    x[j, i] = node_coordinates[1, ilocal, jlocal, element]
                    y[j, i] = node_coordinates[2, ilocal, jlocal, element]
                    data[1:nvars, j, i] = u[:, ilocal, jlocal, element]
                    data[end-1, j, i] = cons2theta(
                        u[:, ilocal, jlocal, element],
                        node_coordinates[2, ilocal, jlocal, element],
                        equations,
                    )
                    data[end, j, i] = element
                end
            end

            if cell_y == cells_per_dimension[2]
                jlocal = polydeg + 1
                for ilocal = 1:polydeg
                    i, j = compute_global_index(
                        element,
                        ilocal,
                        jlocal,
                        polydeg,
                        cell_x,
                        cell_y,
                    )
                    x[j, i] = node_coordinates[1, ilocal, jlocal, element]
                    y[j, i] = node_coordinates[2, ilocal, jlocal, element]
                    data[1:nvars, j, i] = u[:, ilocal, jlocal, element]
                    data[end-1, j, i] = cons2theta(
                        u[:, ilocal, jlocal, element],
                        node_coordinates[2, ilocal, jlocal, element],
                        equations,
                    )
                    data[end, j, i] = element
                end
            end

            if cell_y == cells_per_dimension[2] && cell_x == cells_per_dimension[1]
                ilocal = polydeg + 1
                jlocal = polydeg + 1
                i, j =
                    compute_global_index(element, ilocal, jlocal, polydeg, cell_x, cell_y)
                x[j, i] = node_coordinates[1, ilocal, jlocal, element]
                y[j, i] = node_coordinates[2, ilocal, jlocal, element]
                data[1:nvars, j, i] = u[:, ilocal, jlocal, element]
                data[end-1, j, i] = cons2theta(
                    u[:, ilocal, jlocal, element],
                    node_coordinates[2, ilocal, jlocal, element],
                    equations,
                )
                data[end, j, i] = element
            end
        end
    end
    return x, y, data
end

function cons2theta(u, z, equations::CompressibleEulerVectorInvariantEquations2D)
    rho, rho_v1, rho_v2, rho_theta = u

    theta = rho_theta / rho - 288.18 * exp(0.0097^2 / equations.g * z)

    return theta
end

function compute_global_index(element, ilocal, jlocal, polydeg, cell_x, cell_y)

    i = polydeg * (cell_x - 1) + ilocal
    j = polydeg * (cell_y - 1) + jlocal

    return i, j
end
