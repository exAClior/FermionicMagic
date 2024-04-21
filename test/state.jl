using Test, FermionicMagic, LinearAlgebra

function rand_On_mtx(n) 
    Q,_ = qr(rand(2*n,2*n))
    return Matrix(Q)
end

function rand_cov_mtx(n)
    x = Diagonal(rand([-im,im],2*n))
    Q = rand_On_mtx(n)
    return Q*x*Q'
end
@testset "utils here" begin
    n = 10
    x = rand_cov_mtx(n)
    @test isapprox(x,-x', atol=1e-10)
    @test x * x' ≈ I(2*n) 
end

@testset "Gaussian State" begin


    vac_state = G"01"

    @test cov_mtx(vac_state) ≈ Float64[0 1 0 0; -1 0 0 0; 0 0 0 -1; 0 0 1 0]
    @test ref_state(vac_state) == [false,true]
    @test overlap(vac_state) == 1.0 + 0.0im
end

@testset "Direct Sum" begin
    A = rand(2,2)
    B = rand(2,2)
    as = rand([A,B],3)
    res = directsum(as)
    @test all([res[2*(ii-1)+1:2*(ii-1)+2, 2*(ii-1)+1:2*(ii-1)+2] ≈ as[ii]  for ii in eachindex(as)])



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
