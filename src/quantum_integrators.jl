module QuantumIntegrators

export KetIntegrator
export UnitaryIntegrator
export DensityMatrixIntegrator
export AdjointUnitaryIntegrator

using LinearAlgebra
using NamedTrajectories
using DirectTrajOpt
using PiccoloQuantumObjects

const ⊗ = kron

function KetIntegrator(
    sys::QuantumSystem,
    traj::NamedTrajectory, 
    ψ̃::Symbol, 
    a::Symbol 
) 
    return BilinearIntegrator(sys.G, traj, ψ̃, a)
end

function UnitaryIntegrator(
    sys::QuantumSystem,
    traj::NamedTrajectory, 
    Ũ⃗::Symbol, 
    a::Symbol
) 
    Ĝ = a_ -> I(sys.levels) ⊗ sys.G(a_)
    return BilinearIntegrator(Ĝ, traj, Ũ⃗, a)
end

function DensityMatrixIntegrator(
    sys::OpenQuantumSystem,
    traj::NamedTrajectory, 
    ρ̃::Symbol, 
    a::Symbol
) 
    return BilinearIntegrator(sys.𝒢, traj, ρ̃, a)
end


function AdjointUnitaryIntegrator(
    sys::ParameterizedQuantumSystem,
    traj::NamedTrajectory, 
    Ũ⃗::Symbol, 
    Ũ⃗ₐ::Vector{Symbol},
    a::Symbol
) 
    n_sys = length(sys.Gₐ)
    
    G = a_ -> I(sys.levels) ⊗ sys.G(a_)

    Gai = (i,a_) -> I(sys.levels) ⊗ sys.Gₐ[i](a_)

    Ĝ = a_ ->  vcat(reduce(vcat,[[zeros(size(G(a_))[1],(i-1) * size(G(a_))[1]) G(a_) zeros(size(G(a_))[1],(n_sys-i) * size(G(a_))[1]) Gai(i,a_)] for i in 1:length(sys.Gₐ)]),[zeros(size(G(a_))[1],size(G(a_))[1]*n_sys) G(a_)])

    return AdjointBilinearIntegrator(Ĝ, traj, Ũ⃗, Ũ⃗ₐ, a)
end


end