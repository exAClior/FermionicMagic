

# covariance matrix 
# this is updated by R Γ R^T where R ∈ O(2n) and Γ is the covariance matrix
# U_{R} represents the unitary evolution
struct CovMatrix
    data::Array{Complex{Float64},2}
end

abstract type AbstractQCState end # could include classical state

abstract type AbstractGaussianState <: AbstractQCState end # includes Gaussian State


struct GaussianState{T<: CovMatrix} <: AbstractGaussianState
    mean::Array{Complex{Float64},1}
    cov::T
end