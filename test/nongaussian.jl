using Test, FermionicMagic, LinearAlgebra
using Random

@testset "Mixed Gaussian State and Naive Norm" begin
    n = 5
    Χ = 10
    T = Float64
    weights = rand(Complex{T},Χ)
    weights = weights ./ sqrt(sum(abs.(weights).^2))
    bit_strs = [BitVector(Bool.(digits(ii,base=2,pad=n))) for ii in shuffle(0:2^n-1)[1:Χ]]
    ψs = GaussianState.(cov_mtx.(bit_strs))
    mixedGaussianState = GaussianMixture([(w,ψ) for (w,ψ) in zip(weights,ψs)]) 

    @test Χ_norm(mixedGaussianState) ≈ one(Float64) 
end

@testset "Χ_norm" begin

end
