abstract type AbstractQCState end # could include classical state

abstract type AbstractGaussianState <: AbstractQCState end # includes Gaussian State

"""
    struct GaussianState{T<:AbstractFloat} <: AbstractGaussianState

A struct representing a Gaussian state.

# Fields
- `Γ::AbstractMatrix{T}`: The covariance matrix of the Gaussian state. 
   The order is diagm([Γ_1 Γ_2 ... Γ_n])
   Γ = ⊕ Γ_i = 1 ^ n ( 0  (-1)^xi; - (-1)^xi  0)  
- `ref_state::BitVector`: A number state acting as the reference of the Gaussian state for phase.
- `overlap::Complex{T}`: The overlap of the Gaussian state with reference state.
"""
struct GaussianState{T<:AbstractFloat} <: AbstractGaussianState
    Γ::AbstractMatrix{T}
    ref_state::BitVector
    overlap::Complex{T}
end

function GaussianState(Γ::AbstractMatrix{T}) where {T<:AbstractFloat}
    # TODO: check if Γ is a valid covariance matrix for pure gaussian state
    # TODO: check why are some pfaffian of cov_mtx + Γ zero
    n = size(Γ, 1) ÷ 2
    x = findsupport(Γ)
    σ = pfaffian(Γ)
    @show σ, pfaffian(cov_mtx(x))
    r = Complex{T}(sqrt(σ * pfaffian(cov_mtx(x) .+ Γ) / 2.0^n))
    return GaussianState(Γ, x, r)
end

ref_state(a::GaussianState{T}) where {T} = a.ref_state
overlap(a::GaussianState{T}) where {T} = a.overlap

cov_mtx(a::GaussianState{T}) where {T} = a.Γ

function cov_mtx(::Type{T}, x::BitVector) where {T<:AbstractFloat}
    return directsum([xi ? T[0 -1; 1 0] : T[0 1; -1 0] for xi in x])
end

cov_mtx(x::BitVector) = cov_mtx(Float64, x)



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


"""
    findsupport(Γ::AbstractMatrix{T}) where {T<:AbstractFloat}

Given a matrix Γ representing a fermionic state ψ, this function finds the number state with greatest overlap with ψ.

# Arguments
- `Γ::AbstractMatrix{T}`: The matrix representing the fermionic state.

# Returns
- `res::BitVector`: The support of the number state, where `res[jj]` is `true` if the jj-th fermion is in the |0⟩ state, and `false` otherwise.
"""
function findsupport(Γ::AbstractMatrix{T}) where {T<:AbstractFloat}
    Γ = copy(Γ)
    n = size(Γ, 1) ÷ 2
    res = BitVector(undef, n)
    prob = one(T) 
    for jj in 1:n
        # probability of measuring the jj-th fermion in the |0⟩ state
        p_jj = 0.5 * (1 + Γ[2 * jj - 1, 2 * jj])
        res[jj] = p_jj < 0.5
        # change to most probable state probability
        p_jj = res[jj] ? (1 - p_jj) : p_jj
        prob *= p_jj
        Γ_nxt = zeros(T, 2 * n, 2 * n)
        Γ_nxt[2 * jj, 2 * jj - 1] = -(-1)^res[jj]
        Γ_nxt[2 * jj-1, 2 * jj] = (-1)^res[jj]

        for ll in 1:(2 * n - 1), kk in (ll + 1):(2 * n)
            (kk == 2 * jj && ll == 2 * jj - 1) && continue
            Γ_nxt[kk, ll] =
                Γ[kk, ll] -
                (-1)^res[jj] / (2 * p_jj) *
                (Γ[2 * jj - 1, ll] * Γ[2 * jj, kk] - Γ[2 * jj - 1, kk] * Γ[2 * jj, ll])
        end
        Γ = Γ_nxt .- transpose(Γ_nxt)
    end
    @show prob
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
    R_α = zeros(Complex{T}, 6 * n + mag_α, 6 * n + mag_α)
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

    @show findall(x -> !isapprox(abs(x), 0.0), R_α .+ transpose(R_α))
    @show size(R_α), n, mag_α

    return parity * im^(n + mag_α * (mag_α - 1) / 2) * pfaffian(R_α) / u / v / 4^(n)
end

function convert(d::GaussianState{T}, y::BitVector) where {T}
    # TODO: test overlap btw y and Ψ_d is nonzero
    α, ν = relatebasiselements(y, ref_state(d))
    Γ0 = cov_mtx(d)
    Γ1 = cov_mtx(ref_state(d))
    Γ2 = cov_mtx(y)
    u = overlap(d)'
    v = exp(im * ν)
    w = overlaptriple(Γ0, Γ1, Γ2, α, u, v)
    return GaussianState(Γ0, y, w)
end

function overlap(d1::GaussianState{T}, d2::GaussianState{T}) where {T}
    σ1, σ2 = pfaffian(cov_mtx(d1)), pfaffian(cov_mtx(d2))
    isapprox(σ1, σ2; atol=1e-10) ||
        throw(ArgumentError("The Pfaffians of the covariance matrices should be the same"))
    α, ν = relatebasiselements(ref_state(d2), ref_state(d1))
    Γ0_p = cov_mtx(d1)
    Γ1_p = cov_mtx(ref_state(d1))
    Γ2_p = cov_mtx(d2)

    u = overlap(d1)'
    v = exp(im * ν) * overlap(d2)
    w = overlaptriple(Γ0_p, Γ1_p, Γ2_p, α, u, v)
    return w'
end

function evolve(R::AbstractMatrix{T}, a::GaussianState{T}) where {T}
    Γ0 = R * cov_mtx(a) * transpose(R)
    y = findsupport(Γ0)

    z, s = rot_fock_basis(R, ref_state(a))

    α, ν = relatebasiselements(y, z)
    Γ1 = R * cov_mtx(ref_state(a)) * transpose(R)
    Γ2 = cov_mtx(y)
    u = overlap(a)'
    v = exp(im * ν) * s'
    w = overlaptriple(Γ0, Γ1, Γ2, α, u, v)
    return GaussianState(Γ0, y, w)
end

β_k(x::BitVector, k::Int) = count(x[1:(k - 1)]) + (x[k] - 1 / 2) * (k + 1)

function rot_fock_basis(R::AbstractMatrix{T}, x::BitVector) where {T}
    if isapprox(det(R), one(T))
        j = findfirst(x -> !isone(x), diag(R))
        k = findfirst(x -> !iszero(x), R[(j + 1):end, j])
        ν = atan(R[k, j] / R[j, j])
        if cos(ν / 2)^2 >= 1 / 2
            z = x
            s = cos(ν / 2)
        else
            z = x .⊻ BitVector([(j == i || k == i) for i in 1:length(x)])
            β = β_k[x, j] + β_k[x, k]
            s = exp(im * π * β) * sin(ν / 2)
        end
    elseif isapprox(det(R), -one(T))
        j, k = findfirst(x -> isapprox(x, one(T)), R)
        j != k || throw(ArgumentError("R should be a reflection matrix"))
        z = x .⊻ BitVector([j == i for i in 1:length(x)])
        s = exp(im * β_k[x, j])
    else
        throw(ArgumentError("R should be a rotation matrix"))
    end
    return z, s
end

# j: fermion index , s:: fermion occupation 
measureprob(a::GaussianState{T}, j::Int, s::Bool) where {T} = (1 + (-1)^s * cov_mtx(a)[2 * j - 1, 2 * j]) / 2

function postmeasure(a::GaussianState{T}, j::Int, s::Bool, p::Real) where {T}
    Γ_a = cov_mtx(a)
    n = size(Γ_a, 1) ÷ 2
    Γ_p = zeros(T, size(Γ_a))
    Γ_p[2 * j, 2 * j - 1] = (-one(T))^s
    for ll in 1:(n - 1), kk in (ll + 1):n
        (kk, ll) == (2 * j, 2 * j - 1) && continue
        Γ_p[kk, ll] =
            Γ_a[kk, ll] -
            (-one(T))^s / (2 * p) *
            (Γ_a[2 * j - 1, ll] * Γ_a[2 * j, kk] - Γ_a[2 * j - 1, kk] * Γ_a[2 * j, ll])
    end
    Γ_p = Γ_p .- transpose(Γ_p)
    y = findsupport(Γ_p)
    α, ν = relatebasiselements(y, ref_state(a))
    Γ_0 = Γ_a
    Γ_1 = cov_mtx(ref_state(a))
    Γ_2 = cov_mtx(y)
    u = r'
    v = exp(im * ν)
    w = overlaptriple(Γ_0, Γ_1, Γ_2, α, u, v)
    return GaussianState(Γ_p, y, w / sqrt(p))
end

function samplestate(y::BitVector, p::AbstractVector{T}) where {T}
    n = length(p)
    R_π = sparse(p, collect(1:n), ones(T, n))
    Γ = R_π * cov_mtx(y) * R_π'
    return describe(Γ)
end
