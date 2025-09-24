export UnitaryMaxVariationalProblem


@doc raw"""
    UnitaryMaxVariationalProblem(
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
function UnitaryMaxVariationalProblem end

function UnitaryMaxVariationalProblem(
    system::VariationalQuantumSystem,
    trajectory::NamedTrajectory,
    goal::AbstractPiccoloOperator;
    robust_times::AbstractVector{<:AbstractVector{Int}}=[Int[] for s ∈ system.G_vars],
    sensitive_times::AbstractVector{<:AbstractVector{Int}}=[Int[] for s ∈ system.G_vars],
    variational_integrator=VariationalUnitaryIntegrator,
    variational_scales::AbstractVector{<:Float64}=fill(1.0, length(system.G_vars)),
    state_name::Symbol = :Ũ⃗,
    variational_state_name::Symbol = :Ũ⃗ᵥ,
    control_name::Symbol = :a,
    a_bound::Float64=1.0,
    a_bounds=fill(a_bound, system.n_drives),
    da_bound::Float64=Inf,
    da_bounds=fill(da_bound, system.n_drives),
    dda_bound::Float64=1.0,
    dda_bounds=fill(dda_bound, system.n_drives),
    Q_s::Float64=1e-2,
    Q_r::Float64=100.0,
    R=1e-2,
    final_fidelity::Float64 = 1.0,
    R_a::Union{Float64, Vector{Float64}}=R,
    R_da::Union{Float64, Vector{Float64}}=R,
    R_dda::Union{Float64, Vector{Float64}}=R,
    constraints::Vector{<:AbstractConstraint}=AbstractConstraint[],
    piccolo_options::PiccoloOptions=PiccoloOptions(),
)
    if piccolo_options.verbose
        println("    constructing UnitaryVariationalProblem...")
        println("\tusing integrator: $(typeof(variational_integrator))")
        println("\ttotal variational parameters: $(length(system.G_vars))")
        if !isempty(robust_times)
            println("\trobust knot points: $(robust_times)")
        end
        if !isempty(sensitive_times)
            println("\tsensitive knot points: $(sensitive_times)")
        end
    end

    # Trajectory
    traj = deepcopy(trajectory)

    _, Ũ⃗_vars =  variational_unitary_rollout(
        traj,
        system;
        unitary_name=state_name,
        drive_name=control_name
    )

    # Add variational components to the trajectory
    var_state_names = Tuple(
        Symbol(string(variational_state_name) * "$(i)") for i in eachindex(system.G_vars)
    )
    var_comps_inits = NamedTuple{var_state_names}(
        Ũ⃗_v[:, 1] / scale for (scale, Ũ⃗_v) in zip(variational_scales, Ũ⃗_vars)
    )
    var_comps_data = NamedTuple{var_state_names}(
        Ũ⃗_v / scale for (scale, Ũ⃗_v) in zip(variational_scales, Ũ⃗_vars)
    )
    traj = add_components(
        traj, 
        var_comps_data; 
        type=:state,
        initial=merge(traj.initial, var_comps_inits)
    )

    control_names = [
        name for name ∈ traj.names
            if endswith(string(name), string(control_name))
    ]

    # Objective
    J = NullObjective(traj)
    J += QuadraticRegularizer(control_names[1], traj, R_a)
    J += QuadraticRegularizer(control_names[2], traj, R_da)
    J += QuadraticRegularizer(control_names[3], traj, R_dda)

    # sensitivity
    for (name, scale, s, r) ∈ zip(
        var_state_names, 
        variational_scales, 
        sensitive_times, 
        robust_times
    )
        @assert isdisjoint(s, r)
        J += UnitarySensitivityObjective(
            name, 
            traj, 
            [s; r]; 
            Qs=[fill(-Q_s, length(s)); fill(Q_r, length(r))], 
            scale=scale
        )
    end
    
    # Optional Piccolo constraints and objectives
    J += apply_piccolo_options!(
        piccolo_options, constraints, traj;
        state_names=state_name,
        state_leakage_indices=goal isa EmbeddedOperator ? 
            get_iso_vec_leakage_indices(goal) :
            nothing
    )

    integrators = [
        variational_integrator(
            system, traj, state_name, [var_state_names...], control_name, 
            scales=variational_scales
        ),
        DerivativeIntegrator(traj, control_name, control_names[2]),
        DerivativeIntegrator(traj, control_names[2], control_names[3]),
    ]

    fidelity_constraint = FinalUnitaryFidelityConstraint(
        goal, unitary_name, final_fidelity, traj
    )
    constraints = push!(constraints, fidelity_constraint)

    return DirectTrajOptProblem(
        traj,
        J,
        integrators;
        constraints=constraints
    )
end


# function UnitaryMaxVariationalProblem(
#     system::VariationalQuantumSystem,
#     trajectory::NamedTrajectory,
#     goal::AbstractPiccoloOperator,
#     objective::Objective,
#     dynamics::TrajectoryDynamics,
#     constraints::AbstractVector{<:AbstractConstraint};
#     robust_times::AbstractVector{<:AbstractVector{Int}}=[Int[] for s ∈ system.G_vars],
#     sensitive_times::AbstractVector{<:AbstractVector{Int}}=[Int[] for s ∈ system.G_vars],
#     variational_integrator=VariationalUnitaryIntegrator,
#     variational_scales::AbstractVector{<:Float64}=fill(1.0, length(system.G_vars)),
#     state_name::Symbol = :Ũ⃗,
#     variational_state_name::Symbol = :Ũ⃗ᵥ,
#     Q_s::Float64=1e-2,
#     Q_r::Float64=100.0,
#     unitary_name::Symbol = :Ũ⃗,
#     final_fidelity::Float64 = 1.0,
#     piccolo_options::PiccoloOptions = PiccoloOptions(),
# )
#     if piccolo_options.verbose
#         println("    constructing UnitaryMaxVariationalProblem (maximize robustness)...")
#         println("\tfinal fidelity: $(final_fidelity), Q_r = $(Q_r)")
#     end


#     # Add variational components to the trajectory
#     var_state_names = Tuple(
#         Symbol(string(variational_state_name) * "$(i)") for i in eachindex(system.G_vars)
#     )

#     # sensitivity
#     for (name, scale, s, r) ∈ zip(
#         var_state_names, 
#         variational_scales, 
#         sensitive_times, 
#         robust_times
#     )
#         @assert isdisjoint(s, r)
#         objective += UnitarySensitivityObjective(
#             name, 
#             trajectory, 
#             [s; r]; 
#             Qs=[fill(-Q_s, length(s)); fill(Q_r, length(r))], 
#             scale=scale
#         )
#     end

#     fidelity_constraint = FinalUnitaryFidelityConstraint(
#         goal, unitary_name, final_fidelity, trajectory
#     )
#     constraints = push!(constraints, fidelity_constraint)

#     return DirectTrajOptProblem(trajectory, objective, dynamics, constraints)
# end


function UnitaryMaxVariationalProblem(
    system::VariationalQuantumSystem,
    prob::DirectTrajOptProblem,
    goal::AbstractPiccoloOperator;
    objective::Objective = NullObjective(prob.trajectory),
    constraints::AbstractVector{<:AbstractConstraint} = deepcopy(prob.constraints),
    robust_times::AbstractVector{<:AbstractVector{Int}}=[Int[] for s ∈ system.G_vars],
    Q_r::Float64 = 100.0,
    state_name::Symbol = :Ũ⃗,
    control_name::Symbol = :a,
    variational_state_name::Symbol = :Ũ⃗ᵥ,
    variational_scales::AbstractVector{<:Float64}=fill(1.0, length(system.G_vars)),
    R::Float64 =1e-2,
    R_a::Union{Float64, Vector{Float64}}=R,
    R_da::Union{Float64, Vector{Float64}}=R,
    R_dda::Union{Float64, Vector{Float64}}=R,
    final_fidelity::Float64 = 1.0,
    piccolo_options::PiccoloOptions = PiccoloOptions(),
)
    
        

    traj = prob.trajectory

    _, Ũ⃗_vars =  variational_unitary_rollout(
        traj, 
        system;
        unitary_name=state_name,
        drive_name=control_name
    )
    # Add variational components to the trajectory
    var_state_names = Tuple(
        Symbol(string(variational_state_name) * "$(i)") for i in eachindex(system.G_vars)
    )
    var_comps_inits = NamedTuple{var_state_names}(
        Ũ⃗_v[:, 1] / scale for (scale, Ũ⃗_v) in zip(variational_scales, Ũ⃗_vars)
    )
    var_comps_data = NamedTuple{var_state_names}(
        Ũ⃗_v / scale for (scale, Ũ⃗_v) in zip(variational_scales, Ũ⃗_vars)
    )
    traj = add_components(
        traj, 
        var_comps_data; 
        type=:state,
        initial=merge(traj.initial, var_comps_inits)
    )

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
        state_names=state_name,
        state_leakage_indices=goal isa EmbeddedOperator ? 
            get_iso_vec_leakage_indices(goal) : 
            nothing
    )

    # robust_times = [[traj.T] for i in 1:system.n_drives]


    return UnitaryMaxVariationalProblem(
        system,
        traj,
        goal;
        objective=J,
        constraints=constraints,
        robust_times,
        Q_r=Q_r,
        state_name=state_name,
        final_fidelity=final_fidelity,
        piccolo_options=piccolo_options
    )
end

# function UnitaryMaxVariationalProblem(
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
#         println("    constructing UnitaryMaxVariationalProblem (maximize robustness)...")
#         println("\tfinal fidelity: $(final_fidelity)")
#     end

#     objective += FirstOrderObjective(H_err, trajectory; Q_t=Q_t)

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

function UnitaryMaxVariationalProblem(
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
        println("    constructing UnitaryMaxVariationalProblem...")
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

# function UnitaryMaxVariationalProblem(
#     prob::DirectTrajOptProblem,
#     goal::Function,
#     H_err::Function;
#     objective::Objective=prob.objective,
#     constraints::AbstractVector{<:AbstractConstraint}=deepcopy(prob.constraints),
#     kwargs...
# )
#     return UnitaryMaxVariationalProblem(
#         deepcopy(prob.trajectory),
#         goal,
#         objective,
#         deepcopy(prob.dynamics),
#         constraints,
#         H_err;
#         kwargs...
#     )
# end

# *************************************************************************** #

# @testitem "Maximum Toggle Hadamard gate" begin
#     using NamedTrajectories
#     using PiccoloQuantumObjects 

#     H_drift = 0.1PAULIS[:Z]
#     H_drives = [PAULIS[:X], PAULIS[:Y]]
#     U_goal = GATES[:H]
#     T = 51
#     Δt = 0.2

#     sys = QuantumSystem(H_drift, H_drives)

#     prob = UnitarySmoothPulseProblem(
#         sys, U_goal, T, Δt, Δt_min=Δt * 0.01,
#         piccolo_options=PiccoloOptions(verbose=false)
#     )

#     before = unitary_rollout_fidelity(prob.trajectory, sys)
#     solve!(prob; max_iter=150, verbose=false, print_level=1)
#     after = unitary_rollout_fidelity(prob.trajectory, sys)
#     @test after > before

#     # soft fidelity constraint
#     min_prob = UnitaryMaxVariationalProblem(
#         prob, U_goal,
#         piccolo_options=PiccoloOptions(verbose=false)
#     )
#     solve!(min_prob; max_iter=150, verbose=false, print_level=1)

#     # test fidelity has stayed above the constraint
#     constraint_tol = 0.95
#     final_fidelity = minimum([0.99, after])
#     @test unitary_rollout_fidelity(min_prob.trajectory, sys) ≥ constraint_tol * final_fidelity
#     duration_after = sum(get_timesteps(min_prob.trajectory))
#     duration_before = sum(get_timesteps(prob.trajectory))
#     @test duration_after <= duration_before
# end

# @testitem "Test relaxed final_fidelity constraint" begin
#     final_fidelity = 0.95
#     @test_broken false
# end
