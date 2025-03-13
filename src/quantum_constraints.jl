module QuantumConstraints

using ..QuantumObjectives

using NamedTrajectories
using PiccoloQuantumObjects
using DirectTrajOpt

export FinalKetFidelityConstraint
export FinalUnitaryFidelityConstraint


# ---------------------------------------------------------
#                        Kets
# ---------------------------------------------------------

function FinalKetFidelityConstraint(
    ψ_goal::AbstractVector{<:Complex{Float64}},
    ψ̃_name::Symbol,
    final_fidelity::Float64,
    traj::NamedTrajectory
)
    terminal_constraint = ψ̃ -> [
        ket_infidelity_loss(ψ̃, ψ_goal) - abs(1 - final_fidelity)
    ]

    return NonlinearKnotPointConstraint(
        terminal_constraint,
        ψ̃_name,
        traj,
        equality=final_fidelity == 1.0,
        times=[traj.T]
    )
end

# ---------------------------------------------------------
#                        Unitaries
# ---------------------------------------------------------

function FinalUnitaryFidelityConstraint(
    U_goal::AbstractMatrix{<:Complex{Float64}},
    Ũ⃗_name::Symbol,
    final_fidelity::Float64,
    traj::NamedTrajectory
)
    terminal_constraint = Ũ⃗ -> [
        unitary_infidelity_loss(Ũ⃗, U_goal) - abs(1 - final_fidelity)
    ]

    return NonlinearKnotPointConstraint(
        terminal_constraint,
        Ũ⃗_name,
        traj,
        equality=final_fidelity == 1.0,
        times=[traj.T]
    )
end

function FinalUnitaryFidelityConstraint(
    op::EmbeddedOperator,
    Ũ⃗_name::Symbol,
    final_fidelity::Float64,
    traj::NamedTrajectory
)

    U_goal = unembed(op)
    terminal_constraint = Ũ⃗ -> [
        unitary_subspace_infidelity_loss(Ũ⃗, U_goal, op.subspace) - abs(1 - final_fidelity)
    ]

    return NonlinearKnotPointConstraint(
        terminal_constraint,
        Ũ⃗_name,
        traj,
        equality=final_fidelity == 1.0,
        times=[traj.T]
    )
end

end