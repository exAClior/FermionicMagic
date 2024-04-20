module FermionicMagic

using SkewLinearAlgebra
using LinearAlgebra
# Write your package code here.
export directsum, @G_str, findsupport, GaussianState
export relatebasiselements, overlaptriple, convert
export overlap, evolve, measureprob, postmeasure

include("state.jl")
end
