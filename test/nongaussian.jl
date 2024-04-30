using Test, FermionicMagic, LinearAlgebra
using Random
using FermionicMagic: Χ_norm, Χmeasureprob, Χevolve

@testset "Χmeasureprob" begin
    n = 5
    Χ = 3
    T = Float64
    weights = rand(Complex{T}, Χ)
    weights = weights ./ sqrt(sum(abs.(weights) .^ 2))
    norm(weights)
    bit_strs = [BitVector(Bool.(digits(ii | 2, base=2, pad=n))) for ii in shuffle(0:4:2^n-1)[1:Χ]]
    @show bit_strs
    Set(bit_strs)
    ψs = GaussianState.(cov_mtx.(bit_strs))
    mixedGaussianState = GaussianMixture([(w, ψ) for (w, ψ) in zip(weights, ψs)])

    Χ_norm(mixedGaussianState)

    ii = 2
    prob, mixedGaussianState3 = Χmeasureprob(mixedGaussianState, ii, true)
    @show prob
end

@testset "Mixed Gaussian State and Naive Norm" begin
    n = 5
    Χ = 10
    T = Float64
    weights = rand(Complex{T}, Χ)
    weights = weights ./ sqrt(sum(abs.(weights) .^ 2))
    bit_strs = [BitVector(Bool.(digits(ii, base=2, pad=n))) for ii in shuffle(0:2^n-1)[1:Χ]]
    ψs = GaussianState.(cov_mtx.(bit_strs))
    mixedGaussianState = GaussianMixture([(w, ψ) for (w, ψ) in zip(weights, ψs)])

    @test Χ_norm(mixedGaussianState) ≈ one(Float64)
end

@testset "Χevolve" begin
    n = 5
    Χ = 10
    T = Float64
    weights = rand(Complex{T}, Χ)
    weights = weights ./ sqrt(sum(abs.(weights) .^ 2))
    bit_strs = [BitVector(Bool.(digits(ii, base=2, pad=n))) for ii in shuffle(0:2^n-1)[1:Χ]]
    ψs = GaussianState.(cov_mtx.(bit_strs))
    mixedGaussianState = GaussianMixture([(w, ψ) for (w, ψ) in zip(weights, ψs)])

    angle = rand() .* 2 * π
    ii, jj = shuffle(1:2*n)[1:2]
    R = LinearAlgebra.Givens(ii, jj, cos(angle), sin(angle)) * Diagonal(ones(2 * n))
    mixedGaussianState2 = Χevolve(R, mixedGaussianState)
end

@testset "Χ_norm" begin

end
