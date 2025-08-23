module QuantumObjectives

export KetInfidelityObjective
export UnitaryInfidelityObjective
export DensityMatrixPureStateInfidelityObjective
export UnitarySensitivityObjective
export FirstOrderObjective
export UnitaryFreePhaseInfidelityObjective
export LeakageObjective

using LinearAlgebra
using NamedTrajectories
using PiccoloQuantumObjects
using DirectTrajOpt
using TestItems
using TrajectoryIndexingUtils
using ForwardDiff
using SparseArrays
# using Zygote
# --------------------------------------------------------- 
#                       Kets
# ---------------------------------------------------------

function ket_fidelity_loss(
    ψ̃::AbstractVector, 
    ψ_goal::AbstractVector{<:Complex{Float64}}
)
    ψ = iso_to_ket(ψ̃)
    return abs2(ψ_goal' * ψ)
end 

function KetInfidelityObjective(
    ψ̃_name::Symbol,
    traj::NamedTrajectory;
    Q=100.0
)
    ψ_goal = iso_to_ket(traj.goal[ψ̃_name])
    ℓ = ψ̃ -> abs(1 - ket_fidelity_loss(ψ̃, ψ_goal))
    return TerminalObjective(ℓ, ψ̃_name, traj; Q=Q)
end


# ---------------------------------------------------------
#                       Unitaries
# ---------------------------------------------------------

function unitary_fidelity_loss(
    Ũ⃗::AbstractVector{<:Real},
    U_goal::AbstractMatrix{<:Complex{<:Real}}
)
    U = iso_vec_to_operator(Ũ⃗)
    n = size(U, 1)
    return abs2(tr(U_goal' * U)) / n^2
end

function unitary_fidelity_loss(
    Ũ⃗::AbstractVector{<:Real},
    op::EmbeddedOperator
)
    U_goal = unembed(op)
    U = iso_vec_to_operator(Ũ⃗)[op.subspace, op.subspace]
    n = length(op.subspace)
    M = U_goal'U
    return 1 / (n * (n + 1)) * (abs(tr(M'M)) + abs2(tr(M))) 
end

function UnitaryInfidelityObjective(
    U_goal::AbstractPiccoloOperator,
    Ũ⃗_name::Symbol,
    traj::NamedTrajectory;
    Q=100.0
)
    ℓ = Ũ⃗ -> abs(1 - unitary_fidelity_loss(Ũ⃗, U_goal))
    return TerminalObjective(ℓ, Ũ⃗_name, traj; Q=Q)
end

function UnitaryFreePhaseInfidelityObjective(
    U_goal::Function,
    Ũ⃗_name::Symbol,
    θ_names::AbstractVector{Symbol},
    traj::NamedTrajectory;
    Q=100.0
)
    d = sum(traj.global_dims[n] for n in θ_names)
    function ℓ(z)
        Ũ⃗, θ = z[1:end-d], z[end-d+1:end]
        return abs(1 - QuantumObjectives.unitary_fidelity_loss(Ũ⃗, U_goal(θ)))
    end
    return TerminalObjective(ℓ, Ũ⃗_name, θ_names, traj; Q=Q)
end

function UnitaryFreePhaseInfidelityObjective(
    U_goal::Function,
    Ũ⃗_name::Symbol,
    θ_name::Symbol,
    traj::NamedTrajectory;
    kwargs...
)
    return UnitaryFreePhaseInfidelityObjective(U_goal, Ũ⃗_name, [θ_name], traj; kwargs...)
end

# ---------------------------------------------------------
#                       Density Matrices
# ---------------------------------------------------------

function density_matrix_pure_state_infidelity_loss(
    ρ̃::AbstractVector, 
    ψ::AbstractVector{<:Complex{Float64}}
)
    ρ = iso_vec_to_density(ρ̃)
    ℱ = real(ψ' * ρ * ψ)
    return abs(1 - ℱ)
end

function DensityMatrixPureStateInfidelityObjective(
    ρ̃_name::Symbol,
    ψ_goal::AbstractVector{<:Complex{Float64}},
    traj::NamedTrajectory;
    Q=100.0
)
    ℓ = ρ̃ -> density_matrix_pure_state_infidelity_loss(ρ̃, ψ_goal)
    return TerminalObjective(ℓ, ρ̃_name, traj; Q=Q)
end

# ---------------------------------------------------------
#                       Sensitivity
# ---------------------------------------------------------

function unitary_fidelity_loss(
    Ũ⃗::AbstractVector{<:Real}
)
    U = iso_vec_to_operator(Ũ⃗)
    n = size(U, 1)
    return abs2(tr(U' * U)) / n^2
end

function UnitarySensitivityObjective(
    name::Symbol,
    traj::NamedTrajectory,
    times::AbstractVector{Int};
    Qs::AbstractVector{<:Float64}=fill(1.0, length(times)),
    scale::Float64=1.0,
)
    ℓ = Ũ⃗ -> scale^4 * unitary_fidelity_loss(Ũ⃗)

    return KnotPointObjective(
        ℓ,
        name,
        traj;
        Qs=Qs,
        times=times
    )
end

function FirstOrderObjective(
    H_err::Function,
    traj::NamedTrajectory;
    Q_t::Float64=1.0
)
    a_indices  = [collect(slice(k, traj.components.a, traj.dim)) for k in 1:traj.T]
    Ũ⃗_indices  = [collect(slice(k, traj.components.Ũ⃗, traj.dim)) for k in 1:traj.T]
    a_ref = ones(length(traj.components.a))
    H_scale = norm(H_err(a_ref), 2)

    @views function toggle(Z::AbstractVector, a_idx::AbstractVector{<:Int}, U_idx::AbstractVector{<:Int})
        a  = Z[a_idx]
        U  = iso_vec_to_operator(Z[U_idx])
        He = H_err(a)
        return U' * He * U
    end

    function ℓ(Z::AbstractVector{<:Real})
        sum_terms = sum(toggle(Z, a_idx, U_idx) for (a_idx, U_idx) in zip(a_indices, Ũ⃗_indices))
        return Q_t * real(norm(tr(sum_terms' * sum_terms), 2)) / real(traj.T^2 * H_scale)
    end

    ∇ℓ = Z -> Q_t * ForwardDiff.gradient(ℓ, Z)

    function ∂²ℓ_structure()
        Z_dim = traj.dim * traj.T + traj.global_dim
        structure = spzeros(Z_dim, Z_dim)
        all_Ũ⃗_indices = vcat(Ũ⃗_indices...)
        
        for i in all_Ũ⃗_indices
            for j in all_Ũ⃗_indices
                structure[i, j] = 1.0
            end
        end
        
        structure_pairs = collect(zip(findnz(structure)[1:2]...))
        return structure_pairs
    end

    function ∂²ℓ(Z::AbstractVector{<:Real})
        structure_pairs = ∂²ℓ_structure()
        H_full = ForwardDiff.hessian(ℓ, Z)
        ∂²ℓ_values = [Q_t * H_full[i, j] for (i, j) in structure_pairs]
        
        return ∂²ℓ_values
    end

    return Objective(ℓ, ∇ℓ, ∂²ℓ, ∂²ℓ_structure)
    end

# ---------------------------------------------------------
#                       Leakage
# ---------------------------------------------------------

"""
    LeakageObjective(indices, name, traj::NamedTrajectory)

Construct a `KnotPointObjective` that penalizes leakage of `name` at the knot points specified by `times` at any `indices` that are outside the computational subspace.

"""
function LeakageObjective(
    indices::AbstractVector{Int},
    name::Symbol,
    traj::NamedTrajectory;
    times=1:traj.T,
    Qs::AbstractVector{<:Float64}=fill(1.0, length(times)),
)
    leakage_objective(x) = sum(abs2, x[indices]) / length(indices)

    return KnotPointObjective(
        leakage_objective,
        name,
        traj;
        Qs=Qs,
        times=times,
    )
end


end