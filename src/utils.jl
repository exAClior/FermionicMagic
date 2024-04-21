# https://arxiv.org/pdf/1102.3440.pdf
# TODO: need to reimplement
# direct copy of https://github.com/KskAdch/TopologicalNumbers.jl/blob/main/src/pfaffian.jl
function pfaffian(A::AbstractMatrix{T}; overwrite_a=false) where {T<:Number}
    # Check if matrix is square
    @assert size(A, 1) == size(A, 2) > 0
    # Check if it's skew-symmetric
    @assert maximum(abs.(A .+ transpose(A))) < 1e-14

    n, m = size(A)
    # if !(eltype(A) <: Complex)
    #     A = convert(Array{Float64,2}, A)
    # end

    # Quick return if possible
    if n % 2 == 1
        return zero(T)
    end

    if !overwrite_a
        A = copy(A)
    end

    tau = Array{T}(undef, n - 2)

    pfaffian_val = one(T)

    @inbounds for k in 1:2:(n - 1)
        tau0 = @view tau[k:end]

        # First, find the largest entry in A[k+1:end, k] and permute it to A[k+1, k]
        @views kp = k + findmax(abs.(A[(k + 1):end, k]))[2]

        # Check if we need to pivot
        if kp != k + 1
            # Interchange rows k+1 and kp
            @inbounds @simd for l in k:n
                t = A[k + 1, l]
                A[k + 1, l] = A[kp, l]
                A[kp, l] = t
            end

            # Then interchange columns k+1 and kp
            @inbounds @simd for l in k:n
                t = A[l, k + 1]
                A[l, k + 1] = A[l, kp]
                A[l, kp] = t
            end

            # Every interchange corresponds to a "-" in det(P)
            pfaffian_val *= -1
        end

        # Now form the Gauss vector
        @inbounds if A[k + 1, k] != zero(T)
            @inbounds @views tau0 .= A[k, (k + 2):end] ./ A[k, k + 1]

            pfaffian_val *= @inbounds A[k, k + 1]

            if k + 2 <= n
                # Update the matrix block A[k+2:end, k+2:end]
                @inbounds for l1 in eachindex(tau0)
                    @simd for l2 in eachindex(tau0)
                        @fastmath A[k + 1 + l2, k + 1 + l1] +=
                            tau0[l2] * A[k + 1 + l1, k + 1] -
                            tau0[l1] * A[k + 1 + l2, k + 1]
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

# function rand_SOn(n)
#     nothing
# end
