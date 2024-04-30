mutable struct GaussianMixture{T<:AbstractFloat} <: AbstractQCState
    states::Vector{Tuple{Complex{T},GaussianState{T}}}
end

function Χevolve(R::AbstractMatrix{T}, Χ::GaussianMixture{T}) where {T}
    return GaussianMixture([(c, evolve(R, ψ)) for (c, ψ) in Χ.states])
end

function Χmeasureprob(Χ::GaussianMixture{T}, j::Int, s::Bool) where {T}
    probs = [measureprob(ψ, j, s) for (_, ψ) in Χ.states]
    @show probs
    new_state = GaussianMixture([(c * sqrt(p), postmeasure(ψ, j, s, p)) for (p, (c, ψ)) in zip(probs, Χ.states)])
    return Χ_norm(new_state), new_state
end

# naive Χ norm
function Χ_norm(a::GaussianMixture{T}) where {T}
    Χ = length(a.states)
    state_norm = zero(T)
    for ii in 1:Χ, jj in 1:Χ
        state_norm += a.states[ii][1]' * a.states[jj][1] * abs(overlap(a.states[ii][2], a.states[jj][2]))^2
    end
    return state_norm
end
