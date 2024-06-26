using Test, FermionicMagic, LinearAlgebra, Random, SparseArrays
using FermionicMagic: rand_cov_mtx, pfaffian, J_x, reflection, convert, β_k, rot_fock_basis
using BenchmarkTools, Profile, ProfileView


@testset "measure" begin
    n = 10

    Γ = rand_cov_mtx(n)
    ψ = GaussianState(Γ)
    x = ref_state(ψ)
    Γ_post = cov_mtx(x)
    ψ_fin = GaussianState(Γ)
    for ii in 1:n
        p_ii = measureprob(ψ_fin, ii, x[ii])
        ψ_fin = postmeasure(ψ_fin, ii, x[ii], p_ii)
    end

    p_ii = measureprob(ψ_fin, 1, x[1])
    @code_warntype postmeasure(ψ_fin, 1, x[1], p_ii)
    @btime postmeasure($ψ_fin, 1, $x[1], $p_ii)
    @benchmark postmeasure(ψ_fin, 1, x[1], p_ii)


    @test isapprox(cov_mtx(ψ_fin), Γ_post, atol=1e-14)

    x = BitVector(rand(Bool, n))
    Γ = cov_mtx(x)
    ψ = GaussianState(Γ)
    for ii in 1:n
        @test measureprob(ψ, ii, x[ii]) ≈ 1.0
    end

end

@testset "Evolve" begin
    n = 5
    bit_str = BitVector(fill(false, n))

    Γ = cov_mtx(bit_str)
    ψ = GaussianState(Γ)
    angle = rand() .* 2 * π
    ii, jj = shuffle(1:2*n)[1:2]

    R = LinearAlgebra.Givens(ii, jj, cos(angle), sin(angle)) * Diagonal(ones(2 * n))

    using FermionicMagic: decompose_rotation, decompose_reflection
    @code_warntype decompose_rotation(R, bit_str)
    @code_warntype decompose_reflection(R, bit_str)
    @code_warntype rot_fock_basis(R, bit_str)

    ψ2 = evolve(R, ψ)

    @test cov_mtx(ψ2) ≈ R * Γ * transpose(R)
    @test abs(overlap(ψ2))^2 >= 1.0 / 2^n
end

@testset "Two state overlap" begin
    n = 5
    Γ = rand_cov_mtx(n)
    ψ = GaussianState(Γ)

    @test abs(overlap(ψ, ψ)) ≈ one(eltype(Γ))

    Γ_op = reflection(n, 1) * Γ * transpose(reflection(n, 1))
    ψ2 = GaussianState(Γ_op)
    @test abs(overlap(ψ, ψ2)) ≈ zero(eltype(Γ))
end

@testset "Convert" begin
    y = BitVector([true, true])
    x = BitVector([false, false])
    Γ = cov_mtx(BitVector([false, false]))
    θ = π / 3
    alpha = cos(θ)
    beta = sin(θ)
    R = LinearAlgebra.Givens(1, 3, cos(θ), sin(θ))
    Γ = R * Γ * R'

    ψ1 = GaussianState(Γ)
    @test cov_mtx(ψ1) ≈ Γ
    @test ref_state(ψ1) == x
    @test overlap(ψ1) ≈ ComplexF64(beta)

    ψ2 = convert(ψ1, y)
    @test cov_mtx(ψ2) ≈ Γ
    @test ref_state(ψ2) == y
    @test overlap(ψ2) ≈ ComplexF64(alpha)
end

@testset "Overlap triple" begin
    n = 3
    α = BitVector([true, false, true])
    J_α = J_x(ComplexF64, α)
    @test J_α == ComplexF64[1.0 0.0 0.0; 0.0 0.0 1.0]


    n = 20
    bit_str = rand(Bool, n)
    x0 = BitVector(bit_str)
    α = BitVector(fill(false, 2 * n))
    Ψ0 = GaussianState(cov_mtx(x0))
    Ψ1 = GaussianState(cov_mtx(x0))
    Ψ2 = GaussianState(cov_mtx(x0))
    @test overlaptriple(cov_mtx(Ψ0), cov_mtx(Ψ1), cov_mtx(Ψ2), α, ComplexF64(1.0), ComplexF64(1.0)) ≈ ComplexF64(1.0)

    bit_str2 = copy(bit_str)
    bit_str2[1] ⊻= true
    x1 = BitVector(bit_str2)
    Ψ3 = GaussianState(cov_mtx(x1))
    @test overlaptriple(cov_mtx(Ψ0), cov_mtx(Ψ1), cov_mtx(Ψ3), α, ComplexF64(1.0), ComplexF64(0.0)) ≈ ComplexF64(0.0)

end

@testset "Gaussian State" begin
    Random.seed!(1234)
    n = 10
    rand_bits = BitVector(rand(Bool, n))
    Γ = cov_mtx(rand_bits)

    ψ_num = GaussianState(Γ)

    @test ref_state(ψ_num) == rand_bits
    @test abs(overlap(ψ_num)) ≈ 1.0

    n = 5
    Γ = rand_cov_mtx(n)
    ψ = GaussianState(Γ)

    @test abs(overlap(ψ))^2 >= 2.0^(-n)
    @test isapprox(abs(overlap(ψ))^4, det(cov_mtx(ref_state(ψ)) .+ cov_mtx(ψ)) / 2^(2 * n), atol=1e-14)

    vac_state = G"01"

    @test cov_mtx(vac_state) ≈ Float64[0 1 0 0; -1 0 0 0; 0 0 0 -1; 0 0 1 0]
    @test ref_state(vac_state) == [false, true]
    @test overlap(vac_state) == 1.0 + 0.0im

    @test_throws ErrorException vac_state = G"21" # should throw an error
end

@testset "Find support" begin
    n = 3
    rand_bits = BitVector(rand(Bool, n))
    Γ = directsum([x ? Float64[0 -1; 1 0] : Float64[0 1; -1 0] for x in rand_bits])

    rand_bits1 = BitVector(rand(Bool, n))
    rand_bits2 = BitVector(rand(Bool, n))
    Γ = FermionicMagic.cov_mtx(rand_bits1) ./ 3 .+ FermionicMagic.cov_mtx(rand_bits2) .* (2 / 3)

    @test (pfaffian(Γ) > 0) == iseven(count(rand_bits2))
    @test findsupport(Γ) == rand_bits2

    Γ_rnd = rand_cov_mtx(n)
    x = findsupport(Γ_rnd)
    @test sign(pfaffian(Γ_rnd)) == sign(pfaffian(cov_mtx(x)))

    Γ_p = [2.7473131173941344e-18 -0.5491779180444556 0.7003638282055721 -0.18219904815758856 -0.4042335033995079 -0.10626807636078786; 0.5491779180444556 7.103776214785612e-19 -0.18302771782112415 0.36269090891334005 -0.3062937833865339 -0.6629810643539726; -0.7003638282055722 0.18302771782112418 1.993096873782267e-17 -0.16360718573842614 0.00128506630073143 -0.6702405538534518; 0.18219904815758856 -0.36269090891334005 0.16360718573842614 -8.075827175686925e-18 0.8518592717065105 -0.2877972922466259; 0.4042335033995079 0.3062937833865339 -0.0012850663007314281 -0.8518592717065105 -1.417321766106611e-17 -0.13081866380749932; 0.10626807636078783 0.6629810643539726 0.6702405538534519 0.287797292246626 0.13081866380749932 -2.895659638696641e-18]

    x = findsupport(Γ_p)
    @test sign(pfaffian(Γ_p)) == sign(pfaffian(cov_mtx(x)))
end

@testset "relatebasiselements" begin
    x = BitArray([true])
    y = BitArray([false])

    @test relatebasiselements(x, x) == (BitArray([false, false]), 0.0)
    @test relatebasiselements(x, y) == (BitArray([true, false]), 0.0)

    # TODO: need more rigorous tests
    x = BitArray([true, false, true])
    y = BitArray([false, true, true])
    @test relatebasiselements(x, y) == (BitArray([true, false, true, false, false, false]), π * 3 / 2)
end
