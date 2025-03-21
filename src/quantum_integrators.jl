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
    Ũ⃗ₐ::Symbol,
    a::Symbol
) 
    Ĝ = a_ ->  [I(sys.levels) ⊗ sys.G(a_) I(sys.levels) ⊗ sys.Gₐ(a_) ; I(sys.levels) ⊗ sys.G(a_)*0 I(sys.levels) ⊗ sys.G(a_) ]
    return AdjointBilinearIntegrator(Ĝ, traj, Ũ⃗, Ũ⃗ₐ, a)
end


end