module FermionicMagic

using LinearAlgebra
# Write your package code here.
export directsum, findsupport
export relatebasiselements, overlaptriple, convert
export overlap, evolve, measureprob, postmeasure

export @G_str, cov_mtx, overlap, ref_state, GaussianState
include("state.jl")
include("utils.jl")
end
