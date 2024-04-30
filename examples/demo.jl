using Pkg;
Pkg.activate(dirname(@__FILE__));

using Random, LinearAlgebra, Test
using FermionicMagic
using FermionicMagic: rand_cov_mtx


# State Creation API
## Number States
ψ = G"001101"
@show ψ
display(ψ)

## Random Gaussian State
n = 20
Γ = rand_cov_mtx(n)
ψ = GaussianState(Γ)

## Non-Gaussian State 
Χ = 10
T = Float64
weights = rand(Complex{T}, Χ)
weights = weights ./ sqrt(sum(abs.(weights) .^ 2))
bit_strs = [BitVector(Bool.(digits(ii, base=2, pad=n))) for ii in shuffle(0:2^n-1)[1:Χ]]
ψs = GaussianState.(cov_mtx.(bit_strs))
mixedGaussianState = GaussianMixture([(w, ψ) for (w, ψ) in zip(weights, ψs)])

# State Evolution
angle = rand() .* 2 * π
ii, jj = shuffle(1:2*n)[1:2]
R = LinearAlgebra.Givens(ii, jj, cos(angle), sin(angle)) * Diagonal(ones(2 * n))
ψ2 = evolve(R, ψ)

@test cov_mtx(ψ2) ≈ R * Γ * transpose(R)
@test abs(overlap(ψ2))^2 >= 1.0 / 2^n


## Non Gaussian State Evolution
Χevolve(R, mixedGaussianState)


# State Measurement
x = ref_state(ψ)
Γ_post = cov_mtx(x)
ψ_fin = GaussianState(Γ)
for ii in 1:n
    p_ii = measureprob(ψ_fin, ii, x[ii])
    ψ_fin = postmeasure(ψ_fin, ii, x[ii], p_ii)
end


@test isapprox(cov_mtx(ψ_fin), Γ_post, atol=1e-14)

n = 5
Χ = 3
T = Float64
weights = rand(Complex{T}, Χ)
weights = weights ./ sqrt(sum(abs.(weights) .^ 2))
bit_strs = [BitVector(Bool.(digits(ii | 2, base=2, pad=n))) for ii in shuffle(0:4:2^n-1)[1:Χ]]
ψs = GaussianState.(cov_mtx.(bit_strs))
mixedGaussianState = GaussianMixture([(w, ψ) for (w, ψ) in zip(weights, ψs)])

ii = 2
prob, mixedGaussianState3 = Χmeasureprob(mixedGaussianState, ii, true)

