module FermionicMagic

using LinearAlgebra, Random, LuxurySparse
# Write your package code here.
export directsum, findsupport
export relatebasiselements, overlaptriple, convert
export overlap, evolve, measureprob, postmeasure
export @G_str, cov_mtx, overlap, ref_state, GaussianState

export GaussianMixture, Χevolve, Χmeasureprob, Χ_norm
export rand_cov_mtx

include("state.jl")
include("utils.jl")
include("nongaussian.jl")
end
