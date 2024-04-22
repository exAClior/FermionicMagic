using Test, LinearAlgebra, FermionicMagic
using FermionicMagic: rand_cov_mtx, pfaffian, rand_Orth_mtx, givens_product, reflection

@testset "Random Matrix Generation" begin
    n = 10
    r = rand_Orth_mtx(n)
    @test r * transpose(r) ≈ I(n)

    R1 = reflection(n, 1)
    @test det(R1) == - one(eltype(R1))

    angles = rand(n*(2*n-1)).*(2*π)
    R = givens_product(n, angles)

    @test R*transpose(R) ≈ I(2*n)
    @test det(R) ≈ 1.0


    x = rand_cov_mtx(n)
    @test isapprox(x,-transpose(x), atol=1e-14)
    @test x * transpose(x) ≈ I(2*n) 
    @test det(x) ≈ one(eltype(x)) 

end

@testset "Pfaffian" begin
    n= 10
    Γ = rand_cov_mtx(n)
    @test maximum(abs.(Γ .+ transpose(Γ))) < 1.0e-14
    @test isapprox(pfaffian(Γ)^2 ,det(Γ),atol=1.0e-14)
end

@testset "Direct Sum" begin
    A = rand(2,2)
    B = rand(2,2)
    as = rand([A,B],3)
    res = directsum(as)
    @test all([res[2*(ii-1)+1:2*(ii-1)+2, 2*(ii-1)+1:2*(ii-1)+2] ≈ as[ii]  for ii in eachindex(as)])
end

# @testset "Covariance matrix" begin
#    n = 5 
#    bits = BitVector([false, true, false, true , true])
#    Γ = cov_mtx(bits)

# end