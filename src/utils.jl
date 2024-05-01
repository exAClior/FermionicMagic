function rand_Orth_mtx(::Type{T}, n) where {T}
    Q, R = qr(randn(T, n, n))
    return Matrix{T}(Q * Diagonal(sign.(diag(R))))
end

rand_Orth_mtx(n) = rand_Orth_mtx(Float64, n)

reflection(::Type{T}, n::Int, i::Int) where {T} = Diagonal([ii == i ? one(T) : -one(T) for ii in 1:2*n])
reflection(n::Int, i::Int) = reflection(Float64, n, i)


function rand_cov_mtx(::Type{T}, n; even_parity::Bool=true) where {T}
    x = cov_mtx(BitVector(fill(false, n)))
    R = rand_Orth_mtx(T, 2 * n)
    if xor(even_parity, sign(det(R)) > zero(T))
        R *= reflection(T, n, 1)
    end
    return R * x * transpose(R)
end

rand_cov_mtx(n; even_parity::Bool=true) = rand_cov_mtx(Float64, n; even_parity)

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
        block_rows = cum_rows[ii]:(cum_rows[ii]+n_rows[ii]-1)
        block_cols = cum_cols[ii]:(cum_cols[ii]+n_cols[ii]-1)
        res[block_rows, block_cols] .= as[ii]
    end
    return res
end

# https://arxiv.org/pdf/1102.3440.pdf
# TODO: need to reimplement
# direct copy of https://github.com/KskAdch/TopologicalNumbers.jl/blob/main/src/pfaffian.jl
function pfaffian(A::AbstractMatrix{T}; overwrite_a=false) where {T<:Number}
    # Check if matrix is square
    @assert size(A, 1) == size(A, 2) > 0
    # Check if it's skew-symmetric
    @assert maximum(abs.(A .+ transpose(A))) < 1e-14

    n = size(A, 1)

    if n % 2 == 1
        return zero(T)
    end

    if !overwrite_a
        A = copy(A)
    end

    tau = Array{T}(undef, n - 2)

    pfaffian_val = one(T)

    @inbounds for k in 1:2:(n-1)
        tau0 = @view tau[k:end]

        # First, find the largest entry in A[k+1:end, k] and permute it to A[k+1, k]
        @views kp = k + findmax(abs.(A[(k+1):end, k]))[2]

        # Check if we need to pivot
        if kp != k + 1
            # Interchange rows k+1 and kp
            @inbounds @simd for l in k:n
                t = A[k+1, l]
                A[k+1, l] = A[kp, l]
                A[kp, l] = t
            end

            # Then interchange columns k+1 and kp
            @inbounds @simd for l in k:n
                t = A[l, k+1]
                A[l, k+1] = A[l, kp]
                A[l, kp] = t
            end

            # Every interchange corresponds to a "-" in det(P)
            pfaffian_val *= -one(T)
        end

        # Now form the Gauss vector
        @inbounds if A[k+1, k] != zero(T)
            @inbounds @views tau0 .= A[k, (k+2):end] ./ A[k, k+1]

            pfaffian_val *= @inbounds A[k, k+1]

            if k + 2 <= n
                # Update the matrix block A[k+2:end, k+2:end]
                @inbounds for l1 in eachindex(tau0)
                    @simd for l2 in eachindex(tau0)
                        @fastmath A[k+1+l2, k+1+l1] +=
                            tau0[l2] * A[k+1+l1, k+1] -
                            tau0[l1] * A[k+1+l2, k+1]
                    end
                end
            end
        else
            # If we encounter a zero on the super/subdiagonal, the Pfaffian is 0
            return zero(T)
        end
    end

    return pfaffian_val
end
