module FermionicMagic

# TODO: reimplement pfaffian for complex matrix
using TopologicalNumbers
using LinearAlgebra
# Write your package code here.
export directsum, @G_str, findsupport, GaussianState
export relatebasiselements, overlaptriple, convert
export overlap, evolve, measureprob, postmeasure

include("state.jl")
include("utils.jl")
end
