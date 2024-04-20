using Test, FermionicMagic

@testset "Direct Sum" begin
    A = rand(2,2)
    B = rand(2,2)
    as = rand([A,B],3)
    res = directsum(as)
    @test all([res[2*(ii-1)+1:2*(ii-1)+2, 2*(ii-1)+1:2*(ii-1)+2] ≈ as[ii]  for ii in eachindex(as)])


    vac_state = G"01"

    @test vac_state.Γ ≈ Float64[0 1 0 0; -1 0 0 0; 0 0 0 -1; 0 0 1 0]
    @test vac_state.x == [false,true]
    @test vac_state.r == 1.0 + 0.0im

    @test_throws ErrorException vac_state = G"21" # should throw an error

    rand_bits = rand(Bool,5)
    Γ = directsum([x ? Float64[0 -1; 1 0] : Float64[0 1; -1 0] for x in rand_bits])

    @test findsupport(Γ) == rand_bits 

    rand_bits1 = BitVector(rand(Bool,5))
    rand_bits2 = BitVector(rand(Bool,5))

    Γ = FermionicMagic.cov_mtx(rand_bits1)./√3 .+ FermionicMagic.cov_mtx(rand_bits2).*(√2/√3)

    @test findsupport(Γ) == rand_bits2
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
    rand_bits1 = BitVector(rand(Bool,5))
    rand_bits2 = BitVector(rand(Bool,5))
    prob_amp1 = rand(0:1/√2)
    prob_amp2 = sqrt(1- prob_amp1^2)
    Γ = FermionicMagic.cov_mtx(rand_bits1).*prob_amp1 .+ FermionicMagic.cov_mtx(rand_bits2).*prob_amp2

    d = GaussianState(Γ, rand_bits2, Complex(prob_amp2))
    @test FermionicMagic.convert(d, rand_bits1) == GaussianState(Γ, rand_bits1, Complex(prob_amp1))
end
