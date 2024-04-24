using FermionicMagic
using Test

@testset "FermionicMagic.jl" begin
    include("state.jl")
    include("utils.jl")
    include("nongaussian.jl")
end

