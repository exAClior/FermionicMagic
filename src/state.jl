
# covariance matrix 
# this is updated by R Γ R^T where R ∈ O(2n) and Γ is the covariance matrix
# U_{R} represents the unitary evolution
struct CovMatrix
    data::Array{Complex{Float64},2}
end

abstract type AbstractQCState end # could include classical state

abstract type AbstractGaussianState <: AbstractQCState end # includes Gaussian State

struct GaussianState{T<:AbstractFloat} <: AbstractGaussianState
    covMatrix::AbstractMatrix{T}
    # reference state
    x::BitArray
    # overlap with reference state
    r::Complex{T}
end

# create a Fock basis state in GaussianState notation
macro G_str(a)
    quote
        T = Float64
        x = BitArray(map(x -> (x == '1'), collect($a)))
        Γ = directsum([ xi ? [0 -1; 1 0] : [0 1; -1 0] for xi in x])
        GaussianState{T}(Γ, x,Complex{T}(1.0))
    end
end

# functionality in BlockDiagonals
function directsum(as::AbstractVector{MT}) where {T <: Number, MT <: AbstractMatrix{T}}
    r_dim = mapreduce(x -> size(x,1), +, as)
    c_dim = mapreduce(x -> size(x,2), +, as)

    n_rows = size.(as,1)
    n_cols = size.(as,2)

    cum_rows = cumsum(n_rows) .- n_rows .+ 1
    cum_cols = cumsum(n_cols) .- n_cols .+ 1

    res = zeros(Complex{T},r_dim,c_dim)

    for ii in eachindex(as)
        block_rows = cum_rows[ii]:cum_rows[ii]+n_rows[ii]-1
        block_cols = cum_cols[ii]:cum_cols[ii]+n_cols[ii]-1
        res[block_rows,block_cols] .= as[ii]
    end
    return res
end








function overlap(a::GaussianState, b::GaussianState)
    return 0.0
end