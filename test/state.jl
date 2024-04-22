using Test, FermionicMagic, LinearAlgebra
using FermionicMagic: rand_cov_mtx, pfaffian
using Random

@testset "Gaussian State" begin
    Random.seed!(1234)
    n = 5
    rand_bits = BitVector(rand(Bool,n))
    Γ = directsum([x ? Float64[0 -1; 1 0] : Float64[0 1; -1 0] for x in rand_bits])

    ψ_num = GaussianState(Γ)

    @test ref_state(ψ_num) == rand_bits
    @test abs(overlap(ψ_num)) ≈ 1.0

    Random.seed!(1234)
    n = 5
    Γ = rand_cov_mtx(n)
    ψ = GaussianState(Γ)

    @show abs(overlap(ψ))^2 , overlap(ψ)
    @test abs(overlap(ψ))^2 >= 2.0^(-n)
    @test isapprox(abs(overlap(ψ))^4, det(cov_mtx(ref_state(ψ)) .+ cov_mtx(ψ)) / 2^(2*n),atol=1e-10)

    vac_state = G"01"

    @test cov_mtx(vac_state) ≈ Float64[0 1 0 0; -1 0 0 0; 0 0 0 -1; 0 0 1 0]
    @test ref_state(vac_state) == [false,true]
    @test overlap(vac_state) == 1.0 + 0.0im

    @test_throws ErrorException vac_state = G"21" # should throw an error
end

@testset "Find support" begin
    n = 5

    rand_bits = BitVector(rand(Bool,n))
    Γ = directsum([x ? Float64[0 -1; 1 0] : Float64[0 1; -1 0] for x in rand_bits])

    @test (pfaffian(Γ) > 0) == iseven(count(rand_bits))
    @test findsupport(Γ) == rand_bits 

    rand_bits1 = BitVector(rand(Bool,n))
    rand_bits2 = BitVector(rand(Bool,n))
    Γ = FermionicMagic.cov_mtx(rand_bits1)./√3 .+ FermionicMagic.cov_mtx(rand_bits2).*(√2/√3)

    @test (pfaffian(Γ) > 0) == iseven(count(rand_bits2))
    @test findsupport(Γ) == rand_bits2

    pfaffian(cov_mtx(BitVector([true ,false])))
    pfaffian(cov_mtx(BitVector([false, true])))
    Γ_rnd = rand_cov_mtx(n)
    x = findsupport(Γ_rnd) 
    @show pfaffian(Γ_rnd), pfaffian(cov_mtx(x))
    @test sign(pfaffian(Γ_rnd)) == sign(pfaffian(cov_mtx(x)))
end


@testset "relatebasiselements" begin
    x = BitArray([true]) 
    y = BitArray([false]) 

    @test relatebasiselements(x,x) == (BitArray([false, false]), 0.0)
    @test relatebasiselements(x,y) == (BitArray([true, false]), 0.0)

    # TODO: need more rigorous tests
    x = BitArray([true, false, true])
    y = BitArray([false, true, true])
    @test relatebasiselements(x,y) == (BitArray([true, false, true, false, false, false]), π*3/2)
end

@testset "Overlap triple" begin
   # TODO: need to implement 
end

@testset "Convert" begin
    n = 5
    rand_bits1 = BitVector(rand(Bool,n))
    Γ = FermionicMagic.cov_mtx(rand_bits1)
    Q,_ = qr(rand(2*n,2*n))
    Γ = Q*Γ*Q'

    # support_bits = findsupport(Γ)

    # d = GaussianState(Γ, support_bits,)
    # d_c = FermionicMagic.convert(d, rand_bits1)
    # @test FermionicMagic.convert(d, rand_bits1) == GaussianState(Γ, rand_bits1, Complex(prob_amp1))
end
