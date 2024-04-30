using Test, LinearAlgebra, FermionicMagic
using FermionicMagic: rand_cov_mtx, pfaffian, rand_Orth_mtx, reflection

@testset "Random Matrix Generation" begin
    n = 500
    r = rand_Orth_mtx(n)
    @test r * transpose(r) ≈ I(n)

    R1 = reflection(n, 1)
    @test det(R1) == -one(eltype(R1))
    @test R1 * R1 ≈ I(2 * n)

    for parity in (true, false)
        x = rand_cov_mtx(n, even_parity=parity)
        @test isapprox(x + transpose(x), zeros(eltype(x), size(x)), atol=1e-14)
        @test x * transpose(x) ≈ I(2 * n)
        @test pfaffian(x) ≈ (-one(eltype(x)))^parity * (-one(eltype(x)))
    end
end

@testset "Pfaffian" begin
    n = 20
    Γ = rand_cov_mtx(n)
    @test maximum(abs.(Γ .+ transpose(Γ))) < 1.0e-14
    @test isapprox(pfaffian(Γ)^2, det(Γ), atol=1.0e-14)
end

@testset "Direct Sum" begin
    n = 100
    A = rand(2, 2)
    B = rand(2, 2)
    as = rand([A, B], n)
    res = directsum(as)
    @test all([res[2*(ii-1)+1:2*(ii-1)+2, 2*(ii-1)+1:2*(ii-1)+2] ≈ as[ii] for ii in eachindex(as)])
end
