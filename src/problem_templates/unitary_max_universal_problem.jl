export UnitaryMaxUniversalProblem


@doc raw"""
    UnitaryMaxUniversalProblem(
        trajectory::NamedTrajectory,
        goal::AbstractPiccoloOperator,
        objective::Objective,
        dynamics::TrajectoryDynamics,
        constraints::AbstractVector{<:AbstractConstraint};
        H_err::Function,
        times = 1:trajectory.T,
        Q_t::Float64 = 1.0,
        unitary_name::Symbol = :Ũ⃗,
        final_fidelity::Float64 = 1.0,
        piccolo_options::PiccoloOptions = PiccoloOptions()
    )

Create a maximum-robustness problem for unitary control.

```math
\begin{aligned}
\underset{\vec{\tilde{U}}, a, \dot{a}, \ddot{a}, \Delta t}{\text{minimize}} & \quad
J(\vec{\tilde{U}}, a, \dot{a}, \ddot{a}) + D \sum_t \Delta t_t \\
\text{ subject to } & \quad \vb{P}^{(n)}\qty(\vec{\tilde{U}}_{t+1}, \vec{\tilde{U}}_t, a_t, \Delta t_t) = 0 \\
& c(\vec{\tilde{U}}, a, \dot{a}, \ddot{a}) = 0 \\
& \quad \Delta t_{\text{min}} \leq \Delta t_t \leq \Delta t_{\text{max}} \\
\end{aligned}
```

# Keyword Arguments
- `piccolo_options::PiccoloOptions=PiccoloOptions()`: The Piccolo options.
- `unitary_name::Symbol=:Ũ⃗`: The name of the unitary for the goal.
- `final_fidelity::Float64=1.0`: The final fidelity constraint.
- `D::Float64=1.0`: The scaling factor for the minimum-time objective.
- `H_err::Function`: Error term of hamiltonian (as a function of controls)
"""
function UnitaryMaxUniversalProblem end

function UnitaryMaxUniversalProblem(
    trajectory::NamedTrajectory,
    goal::AbstractPiccoloOperator,
    objective::Objective,
    dynamics::TrajectoryDynamics,
    constraints::AbstractVector{<:AbstractConstraint};
    Q_t::Float64 = 1.0,
    unitary_name::Symbol = :Ũ⃗,
    final_fidelity::Float64 = 1.0,
    piccolo_options::PiccoloOptions = PiccoloOptions(),
)
    if piccolo_options.verbose
        println("    constructing UnitaryMaxUniversalProblem (maximize robustness)...")
        println("\tfinal fidelity: $(final_fidelity), Q_t = $(Q_t)")
    end

    objective += TurboUniversalObjective(trajectory; Q_t=Q_t)

    fidelity_constraint = FinalUnitaryFidelityConstraint(
        goal, unitary_name, final_fidelity, trajectory
    )
    constraints = push!(constraints, fidelity_constraint)

    return DirectTrajOptProblem(trajectory, objective, dynamics, constraints)
end


function UnitaryMaxUniversalProblem(
    prob::DirectTrajOptProblem,
    goal::AbstractPiccoloOperator;
    objective::Objective = NullObjective(prob.trajectory),
    constraints::AbstractVector{<:AbstractConstraint} = deepcopy(prob.constraints),
    Q_t::Float64 = 1.0,
    unitary_name::Symbol = :Ũ⃗,
    control_name::Symbol = :a,
    R=1e-2,
    R_a::Union{Float64, Vector{Float64}}=R,
    R_da::Union{Float64, Vector{Float64}}=R,
    R_dda::Union{Float64, Vector{Float64}}=R,
    final_fidelity::Float64 = 1.0,
    piccolo_options::PiccoloOptions = PiccoloOptions(),
)
    traj = prob.trajectory
    J = objective
    # Q = 1/Q_t
    # J = UnitaryInfidelityObjective(goal, unitary_name, traj; Q=Q)
    control_names = [
            name for name ∈ traj.names
                if endswith(string(name), string(control_name))
        ]

    J += QuadraticRegularizer(control_names[1], traj, R_a)
    J += QuadraticRegularizer(control_names[2], traj, R_da)
    J += QuadraticRegularizer(control_names[3], traj, R_dda)

    # Optional Piccolo constraints and objectives
    J += apply_piccolo_options!(
        piccolo_options, constraints, traj; 
        state_names=unitary_name,
        state_leakage_indices=goal isa EmbeddedOperator ? 
            get_iso_vec_leakage_indices(goal) : 
            nothing
    )


    return UnitaryMaxUniversalProblem(
        deepcopy(prob.trajectory),
        goal,
        J,
        prob.dynamics,
        constraints;
        Q_t=Q_t,
        unitary_name=unitary_name,
        final_fidelity=final_fidelity,
        piccolo_options=piccolo_options,
    )
end

# function UnitaryMaxUniversalProblem(
#     trajectory::NamedTrajectory,
#     goal::AbstractPiccoloOperator,
#     objective::Objective,
#     dynamics::TrajectoryDynamics,
#     constraints::AbstractVector{<:AbstractConstraint},
#     H_err::Function;
#     Q_t::Float64 = 1.0,
#     unitary_name::Symbol = :Ũ⃗,
#     final_fidelity::Float64 = 1.0,
#     piccolo_options::PiccoloOptions = PiccoloOptions(),
# )
#     if piccolo_options.verbose
#         println("    constructing UnitaryMaxUniversalProblem (maximize robustness)...")
#         println("\tfinal fidelity: $(final_fidelity)")
#     end

#     objective += TurboUniversalObjective(trajectory; Q_t=Q_t)

#     fidelity_constraint = FinalUnitaryFidelityConstraint(
#         goal, unitary_name, final_fidelity, trajectory
#     )
#     constraints = push!(constraints, fidelity_constraint)

#     return DirectTrajOptProblem(
#         trajectory,
#         objective,
#         dynamics,
#         constraints
#     )
# end

# --------------------------------------------------------------------------- #
# Free phases
# --------------------------------------------------------------------------- #

function UnitaryMaxUniversalProblem(
    trajectory::NamedTrajectory,
    goal::Function,
    objective::Objective,
    dynamics::TrajectoryDynamics,
    constraints::AbstractVector{<:AbstractConstraint};
    unitary_name::Symbol=:Ũ⃗,
    phase_name::Symbol=:θ,
    final_fidelity::Float64=1.0,
    D::Float64=100.0,
    piccolo_options::PiccoloOptions=PiccoloOptions(),
)
    # Collect phase names
    phase_names = [n for n in trajectory.global_names if endswith(string(n), string(phase_name))]

    if piccolo_options.verbose
        println("    constructing UnitaryMaxUniversalProblem...")
        println("\tfinal fidelity: $(final_fidelity)")
        println("\tphase names: $(phase_names)")
    end

    objective += F(trajectory, D=D)
    # timesteps_all_equal=piccolo_options.timesteps_all_equal

    fidelity_constraint = FinalUnitaryFidelityConstraint(
        goal, unitary_name, phase_names, final_fidelity, trajectory
    )

    constraints = push!(constraints, fidelity_constraint)

    return DirectTrajOptProblem(
        trajectory,
        objective,
        dynamics,
        constraints
    )
end

function UnitaryMaxUniversalProblem(
    prob::DirectTrajOptProblem,
    goal::Function,
    H_err::Function;
    objective::Objective=prob.objective,
    constraints::AbstractVector{<:AbstractConstraint}=deepcopy(prob.constraints),
    kwargs...
)
    return UnitaryMaxUniversalProblem(
        deepcopy(prob.trajectory),
        goal,
        objective,
        deepcopy(prob.dynamics),
        constraints;
        kwargs...
    )
end

# *************************************************************************** #

@testitem "Maximum Toggle Hadamard gate" begin
    using NamedTrajectories
    using PiccoloQuantumObjects 

    H_drift = 0.1PAULIS[:Z]
    H_drives = [PAULIS[:X], PAULIS[:Y]]
    U_goal = GATES[:H]
    T = 51
    Δt = 0.2

    sys = QuantumSystem(H_drift, H_drives)

    prob = UnitarySmoothPulseProblem(
        sys, U_goal, T, Δt, Δt_min=Δt * 0.01,
        piccolo_options=PiccoloOptions(verbose=false)
    )

    before = unitary_rollout_fidelity(prob.trajectory, sys)
    solve!(prob; max_iter=150, verbose=false, print_level=1)
    after = unitary_rollout_fidelity(prob.trajectory, sys)
    @test after > before

    # soft fidelity constraint
    min_prob = UnitaryMaxUniversalProblem(
        prob, U_goal;
        piccolo_options=PiccoloOptions(verbose=false)
    )
    solve!(min_prob; max_iter=150, verbose=false, print_level=1)

    # test fidelity has stayed above the constraint
    constraint_tol = 0.95
    final_fidelity = minimum([0.99, after])
    @test unitary_rollout_fidelity(min_prob.trajectory, sys) ≥ constraint_tol * final_fidelity
    duration_after = sum(get_timesteps(min_prob.trajectory))
    duration_before = sum(get_timesteps(prob.trajectory))
    @test duration_after <= duration_before
end

@testitem "Test relaxed final_fidelity constraint" begin
    final_fidelity = 0.95
    @test_broken false
end
