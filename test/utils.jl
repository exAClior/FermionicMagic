using Test, LinearAlgebra, FermionicMagic
using FermionicMagic: rand_cov_mtx, pfaffian, rand_Orth_mtx

@testset "Random Matrix Generation" begin
    n = 10
    r = rand_Orth_mtx(n)
    @test r * transpose(r) ≈ I(n)

    x = rand_cov_mtx(n)
    @test isapprox(x,-transpose(x), atol=1e-14)
    @test x * transpose(x) ≈ I(2*n) 

end

@testset "Pfaffian" begin
    n= 10
    Γ = rand_cov_mtx(n)
    @test maximum(abs.(Γ .+ transpose(Γ))) < 1.0e-14
    @test isapprox(pfaffian(Γ)^2 ,det(Γ),atol=1.0e-14)
end

@testset "Covariance matrix" begin
   n = 5 
   bits = BitVector([false, true, false, true , true])
   Γ = cov_mtx(bits)
   @show pfaffian(Γ)
#    bits = BitVector(rand(Bool, n))

end