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