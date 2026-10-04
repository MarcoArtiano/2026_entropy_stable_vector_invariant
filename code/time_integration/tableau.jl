struct LinearIMEXButcher{T1 <: AbstractArray, T2 <: AbstractArray} <: RKTableau
    a_ex::T1
    a_im::T1
    b::T2
    c::T2
end

struct ARK2GKC <: LinearIMEXAlgorithm{3} end

function RKTableau(::ARK2GKC, RealT)
    two = RealT(2)
    gamma = 1 - 1 / sqrt(two)
    delta = 1 / (2 * sqrt(two))
    a32 = RealT(1) / 2
    a_ex = [0 0 0; 2*gamma 0 0; 1-a32 a32 0]
    a_im = [0 0 0; gamma gamma 0; delta delta gamma]
    b = [delta, delta, gamma]
    c = [0, 2 * gamma, 1]
    return LinearIMEXButcher{Matrix{RealT}, Vector{RealT}}(RealT.(a_ex), RealT.(a_im),
                                                    RealT.(b), RealT.(c))
end
