
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
    x::BitVector
    # overlap with reference state
    r::Complex{T}
end

# TODO: not general, only for basis states
function GaussianState(Γ::AbstractMatrix{T}) where {T<:AbstractFloat}
    n = size(Γ, 1) ÷ 2
    x = BitVector([Γ[2 * i - 1, 2 * i] == -1 for i in 1:n])
    r = Complex{T}(1.0)
    return GaussianState(Γ, x, r)
end

ref_state(a::GaussianState{T}) where {T} = a.x

cov_mtx(a::GaussianState{T}) where {T} = a.Γ
cov_mtx(::Type{T}, x::BitVector) where {T<:AbstractFloat} = directsum([xi ? T[0 -1; 1 0] : T[0 1; -1 0] for xi in x])
cov_mtx(x::BitVector) = cov_mtx(Float64, x)

phase(a::GaussianState{T}) where {T} = a.r

# create a Fock basis state in GaussianState notation
macro G_str(a)
    quote
        T = Float64
        matched = match(r"(^[01]+)", $a)
        matched === nothing && error("Input should be a string of 0s and 1s")
        x = BitVector(map(x -> (x == '1'), collect(matched[1])))
        Γ = cov_mtx(x)
        GaussianState{T}(Γ, x, Complex{T}(1.0))
    end
end

# same functionality could be found in BlockDiagonals
function directsum(as::AbstractVector{MT}) where {T<:Number,MT<:AbstractMatrix{T}}
    r_dim = mapreduce(x -> size(x, 1), +, as)
    c_dim = mapreduce(x -> size(x, 2), +, as)

    n_rows = size.(as, 1)
    n_cols = size.(as, 2)

    cum_rows = cumsum(n_rows) .- n_rows .+ 1
    cum_cols = cumsum(n_cols) .- n_cols .+ 1

    res = zeros(T, r_dim, c_dim)

    for ii in eachindex(as)
        block_rows = cum_rows[ii]:(cum_rows[ii] + n_rows[ii] - 1)
        block_cols = cum_cols[ii]:(cum_cols[ii] + n_cols[ii] - 1)
        res[block_rows, block_cols] .= as[ii]
    end
    return res
end

# find the Fock basis state with largest overlap with Gaussian state
# represented by covariance matrix Γ
function findsupport(Γ::AbstractMatrix{T}) where {T<:AbstractFloat}
    Γ = copy(Γ)
    n = size(Γ, 1) ÷ 2
    res = BitVector(undef, n)

    for jj in 1:n
        # probability of measuring the jj-th fermion in the |0> state
        p_jj = 0.5 * (1 + Γ[2 * jj - 1, 2 * jj])
        res[jj] = p_jj < 0.5
        # change to most probable state probability
        p_jj = p_jj < 0.5 ? (1 - p_jj) : p_jj
        Γ_nxt = zeros(T, 2 * n, 2 * n)
        Γ_nxt[2 * jj, 2 * jj - 1] = (-1)^res[jj]

        for ll in 1:(2 * n - 1), kk in (ll + 1):(2 * n)
            (kk == 2 * jj && ll == 2 * jj - 1) && continue
            Γ_nxt[kk, ll] =
                Γ[kk, ll] -
                (-1)^res[jj] / (2 * p_jj) *
                (Γ[2 * jj - 1, ll] * Γ[2 * jj, kk] - Γ[2 * jj - 1, kk] * Γ[2 * jj, ll])
        end
        Γ = Γ_nxt - transpose(Γ_nxt)
    end
    return res
end

function relatebasiselements(::Type{T}, x::BitVector, y::BitVector) where {T<:AbstractFloat}
    length(x) == length(y) || throw(ArgumentError("x and y should have the same length"))
    N = length(x)
    α = BitVector([
        isodd(i) ? (x[i ÷ 2 + 1] ⊻ y[i ÷ 2 + 1]) : zero(eltype(x)) for i in 1:(2 * N)
    ])
    ν = zero(T)
    η_j = zero(T)
    # α0 + α^†0 don't contribute to the overlap
    @inbounds for j in 2:N
        η_j += x[j] ? one(T) : zero(T)
        ν += x[j] ⊻ y[j] ? one(T) : zero(T)
    end
    ν *= π
    x_y_mod = count(x .⊻ y)
    ν += π / 4 * x_y_mod * (x_y_mod - 1)
    return (α, ν)
end

relatebasiselements(x::BitVector, y::BitVector) = relatebasiselements(Float64, x, y)

function overlaptriple(
    Γ0::AbstractMatrix{T},
    Γ1::AbstractMatrix{T},
    Γ2::AbstractMatrix{T},
    α::BitVector,
    u::Complex{T},
    v::Complex{T},
) where {T<:AbstractFloat}
    @show pfaffian(Γ0), pfaffian(Γ1), pfaffian(Γ2)
    parity = pfaffian(Γ0)
    (parity == pfaffian(Γ1) == pfaffian(Γ2)) ||
        throw(ArgumentError("Γ0, Γ1, and Γ2 should have the same Pfaffian"))
    !iszero(u) || throw(ArgumentError("u should be non-zero"))
    !iszero(v) || throw(ArgumentError("v should be non-zero"))

    n = size(Γ0, 1) ÷ 2
    mag_α = count(α)

    D_α = Diagonal([α[i] ? zero(T) : one(T) for i in 1:(2 * n)])
    J_α = zeros(mag_α, 2 * n)
    jj = 0
    for ii in 1:(2 * n)
        if α[ii]
            jj += 1
            J_α[jj, ii] = one(T)
        end
    end
    # TODO: need to verify this part mathematically
    R_α = zeros(Complex{T},6 * n + mag_α, 6 * n + mag_α)
    R_α[1:(2 * n), 1:(2 * n)] = im .* Γ0
    R_α[1:(2 * n), (2 * n + 1):(4 * n)] = -one(T) * I(2 * n)
    R_α[(2 * n + 1):(4 * n), 1:(2 * n)] = one(T) * I(2 * n)
    R_α[(2 * n + 1):(4 * n), (2 * n + 1):(4 * n)] = im .* Γ1
    R_α[(2 * n + 1):(4 * n), (4 * n + 1):(6 * n)] = -one(T) * I(2 * n)
    R_α[(4 * n + 1):(6 * n), 1:(2 * n)] = -one(T) * I(2 * n)
    R_α[(4 * n + 1):(6 * n), (2 * n + 1):(4n)] = one(T) * I(2 * n)
    R_α[(4 * n + 1):(6 * n), (4 * n + 1):(6 * n)] = im .* D_α * Γ2 * D_α
    R_α[(4 * n + 1):(6 * n), (6 * n + 1):(6 * n + mag_α)] =
        transpose(J_α) .+ im .* D_α * Γ2 * transpose(J_α)
    R_α[(6 * n + 1):(6 * n + mag_α), (4 * n + 1):(6 * n)] = -J_α .+ im .* J_α * Γ2 * D_α
    R_α[(6 * n + 1):(6 * n + mag_α), (6 * n + 1):(6 * n + mag_α)] =
        im .* J_α * Γ2 * transpose(J_α)

    @show findall(x->!isapprox(abs(x),0.0),R_α .+ transpose(R_α))
    @show size(R_α), n , mag_α

    return parity * im^(n + mag_α * (mag_α - 1) / 2) * pfaffian(R_α) / u / v / 4^(n)
end

function convert(d::GaussianState{T}, y::BitVector) where {T}
    # TODO: test overlap btw y and Ψ_d is nonzero
    α, ν = relatebasiselements(y, ref_state(d))
    Γ0 = cov_mtx(d)
    Γ1 = cov_mtx(ref_state(d))
    Γ2 = cov_mtx(y)
    u = phase(d)'
    v = exp(im * ν)
    w = overlaptriple(Γ0, Γ1, Γ2, α, u, v)
    return GaussianState(Γ0, y, w)
end

function overlap(d1::GaussianState{T}, d2::GaussianState{T}) where {T}
    return nothing
end

function evolve(a::GaussianState, R::AbstractMatrix)
    return copy(a)
end

# j: fermion index , s:: fermion occupation 
function measureprob(a::GaussianState, j::Integer, s::Bool) end

function postmeasure(a::GaussianState, j::Integer, s::Bool, p::Real) end
