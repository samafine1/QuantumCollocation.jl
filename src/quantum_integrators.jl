module QuantumIntegrators

export KetIntegrator
export UnitaryIntegrator
export DensityMatrixIntegrator
export AdjointUnitaryIntegrator

using LinearAlgebra
using NamedTrajectories
using DirectTrajOpt
using PiccoloQuantumObjects
using SparseArrays

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

    function Ĝ(a_)
        G_eval = G(a_)
        dim = size(G_eval)[1]    
        Gx_index, Gy_index, G_val = findnz(G_eval)
        G_full = spzeros((n_sys+1).*size(G_eval))
    
        for i ∈ 0:n_sys
            G_full +=    sparse((i*dim) .+ Gx_index, (i*dim) .+ Gy_index, G_val, size(G_full)...)
            if(i<n_sys)
                Ga_x_index, Ga_y_index, Ga_val = findnz(Gai(i+1,a_))
                G_full +=    sparse((i*dim) .+ Ga_x_index, (n_sys*dim) .+ Ga_y_index, Ga_val, size(G_full)...)
            end
        end 
        return G_full
    end
    
    return AdjointBilinearIntegrator(Ĝ, traj, Ũ⃗, Ũ⃗ₐ, a)
end


end