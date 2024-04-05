
# covariance matrix 
# this is updated by R Γ R^T where R ∈ O(2n) and Γ is the covariance matrix
# U_{R} represents the unitary evolution
# struct CovMatrix
#     data::Array{Complex{Float64},2}
# end

abstract type AbstractQCState end # could include classical state

abstract type AbstractGaussianState <: AbstractQCState end # includes Gaussian State


# TODO: given a covariance matrix, how can I compute the r easily?
struct GaussianState{T<:AbstractFloat} <: AbstractGaussianState
    # covariance matrix
    # The order is diagm([Γ_1 Γ_2 ... Γ_n])
    #  Γ = ⊕_i = 1 ^ n ( 0  (-1)^xi; - (-1)^xi  0)  
    Γ::AbstractMatrix{T}
    # reference state
    # x = x1 x2 ... xn
    x::BitArray
    # overlap with reference state
    r::Complex{T}
end

# describe function
function GaussianState(Γ::AbstractMatrix{T}) where {T<:AbstractFloat}
    n = size(Γ,1) ÷ 2
    x = BitArray([Γ[2*i-1,2*i] == -1 for i in 1:n])
    r = Complex{T}(1.0)
    return GaussianState(Γ,x,r)
end

# create a Fock basis state in GaussianState notation
macro G_str(a)
    quote
        T = Float64
        matched = match(r"(^[01]+)", $a)
        matched === nothing && error("Input should be a string of 0s and 1s")
        x = BitArray(map(x -> (x == '1'), collect(matched[1])))
        Γ = directsum([ xi ? [0 -1; 1 0] : [0 1; -1 0] for xi in x])
        GaussianState{T}(Γ, x,Complex{T}(1.0))
    end
end

# same functionality could be found in BlockDiagonals
function directsum(as::AbstractVector{MT}) where {T <: Number, MT <: AbstractMatrix{T}}
    r_dim = mapreduce(x -> size(x,1), +, as)
    c_dim = mapreduce(x -> size(x,2), +, as)

    n_rows = size.(as,1)
    n_cols = size.(as,2)

    cum_rows = cumsum(n_rows) .- n_rows .+ 1
    cum_cols = cumsum(n_cols) .- n_cols .+ 1

    res = zeros(T,r_dim,c_dim)

    for ii in eachindex(as)
        block_rows = cum_rows[ii]:cum_rows[ii]+n_rows[ii]-1
        block_cols = cum_cols[ii]:cum_cols[ii]+n_cols[ii]-1
        res[block_rows,block_cols] .= as[ii]
    end
    return res
end

# find the Fock basis state with largest overlap with Gaussian state
# represented by covariance matrix Γ
function findsupport(Γ::AbstractMatrix{T}) where {T<:AbstractFloat}
    Γ = copy(Γ)
    n = size(Γ,1) ÷ 2
    res = BitArray(undef,n)

    for jj in 1:n
        # probability of measuring the jj-th fermion in the |0> state
        p_jj = 0.5*(1 + Γ[2*jj-1,2*jj])
        res[jj] = p_jj < 0.5
        # change to most probable state probability
        p_jj  = p_jj < 0.5 ? (1- p_jj) : p_jj
        Γ_nxt = zeros(T,2*n,2*n)
        Γ_nxt[2*jj,2*jj-1] = (-1)^res[jj] 

        for ll in 1:2*n-1, kk in ll+1:2*n
            (kk == 2*jj && ll == 2*jj-1) && continue
            Γ_nxt[kk,ll] = Γ[kk,ll] - (-1)^res[jj] /(2*p_jj)*(Γ[2*jj-1,ll]*Γ[2*jj,kk] - Γ[2*jj-1,kk]*Γ[2*jj,ll])
        end
        Γ = Γ_nxt - transpose(Γ_nxt)
        @show Γ
    end
    return res
end

function overlap(a::GaussianState, b::GaussianState)
    return 0.0
end

function evolve(a::GaussianState,R::AbstractMatrix)

    return copy(a)
end

# j: fermion index , s:: fermion occupation 
function measureprob(a::GaussianState,j::Integer,s::Bool)

end

function postmeasure(a::GaussianState,j::Integer,s::Bool,p::Real)

end
