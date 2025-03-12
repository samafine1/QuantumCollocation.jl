export UnitarySmoothPulseProblem


@doc raw"""
    UnitarySmoothPulseProblem(system::AbstractQuantumSystem, operator, T, Δt; kwargs...)
    UnitarySmoothPulseProblem(H_drift, H_drives, operator, T, Δt; kwargs...)

Construct a `DirectTrajOptProblem` for a free-time unitary gate problem with smooth control pulses enforced by constraining the second derivative of the pulse trajectory, i.e.,

```math
\begin{aligned}
\underset{\vec{\tilde{U}}, a, \dot{a}, \ddot{a}, \Delta t}{\text{minimize}} & \quad
Q \cdot \ell\qty(\vec{\tilde{U}}_T, \vec{\tilde{U}}_{\text{goal}}) + \frac{1}{2} \sum_t \qty(R_a a_t^2 + R_{\dot{a}} \dot{a}_t^2 + R_{\ddot{a}} \ddot{a}_t^2) \\
\text{ subject to } & \quad \vb{P}^{(n)}\qty(\vec{\tilde{U}}_{t+1}, \vec{\tilde{U}}_t, a_t, \Delta t_t) = 0 \\
& \quad a_{t+1} - a_t - \dot{a}_t \Delta t_t = 0 \\
& \quad \dot{a}_{t+1} - \dot{a}_t - \ddot{a}_t \Delta t_t = 0 \\
& \quad |a_t| \leq a_{\text{bound}} \\
& \quad |\ddot{a}_t| \leq \ddot{a}_{\text{bound}} \\
& \quad \Delta t_{\text{min}} \leq \Delta t_t \leq \Delta t_{\text{max}} \\
\end{aligned}
```

where, for $U \in SU(N)$,

```math
\ell\qty(\vec{\tilde{U}}_T, \vec{\tilde{U}}_{\text{goal}}) =
\abs{1 - \frac{1}{N} \abs{ \tr \qty(U_{\text{goal}}, U_T)} }
```

is the *infidelity* objective function, $Q$ is a weight, $R_a$, $R_{\dot{a}}$, and $R_{\ddot{a}}$ are weights on the regularization terms, and $\vb{P}^{(n)}$ is the $n$th-order Pade integrator.

# Arguments

- `system::AbstractQuantumSystem`: the system to be controlled
or
- `H_drift::AbstractMatrix{<:Number}`: the drift hamiltonian
- `H_drives::Vector{<:AbstractMatrix{<:Number}}`: the control hamiltonians
with
- `operator::AbstractPiccoloOperator`: the target unitary, either in the form of an `EmbeddedOperator` or a `Matrix{ComplexF64}
- `T::Int`: the number of timesteps
- `Δt::Float64`: the (initial) time step size

# Keyword Arguments
- `ipopt_options::IpoptOptions=IpoptOptions()`: the options for the Ipopt solver
- `piccolo_options::PiccoloOptions=PiccoloOptions()`: the options for the Piccolo solver
- `state_name::Symbol = :Ũ⃗`: the name of the state
- `control_name::Symbol = :a`: the name of the control
- `timestep_name::Symbol = :Δt`: the name of the timestep
- `init_trajectory::Union{NamedTrajectory, Nothing}=nothing`: an initial trajectory to use
- `a_bound::Float64=1.0`: the bound on the control pulse
- `a_bounds::Vector{Float64}=fill(a_bound, length(system.G_drives))`: the bounds on the control pulses, one for each drive
- `a_guess::Union{Matrix{Float64}, Nothing}=nothing`: an initial guess for the control pulses
- `da_bound::Float64=Inf`: the bound on the control pulse derivative
- `da_bounds::Vector{Float64}=fill(da_bound, length(system.G_drives))`: the bounds on the control pulse derivatives, one for each drive
- `zero_initial_and_final_derivative::Bool=false`: whether to enforce zero initial and final control pulse derivatives
- `dda_bound::Float64=1.0`: the bound on the control pulse second derivative
- `dda_bounds::Vector{Float64}=fill(dda_bound, length(system.G_drives))`: the bounds on the control pulse second derivatives, one for each drive
- `Δt_min::Float64=Δt isa Float64 ? 0.5 * Δt : 0.5 * mean(Δt)`: the minimum time step size
- `Δt_max::Float64=Δt isa Float64 ? 1.5 * Δt : 1.5 * mean(Δt)`: the maximum time step size
- `Q::Float64=100.0`: the weight on the infidelity objective
- `R=1e-2`: the weight on the regularization terms
- `R_a::Union{Float64, Vector{Float64}}=R`: the weight on the regularization term for the control pulses
- `R_da::Union{Float64, Vector{Float64}}=R`: the weight on the regularization term for the control pulse derivatives
- `R_dda::Union{Float64, Vector{Float64}}=R`: the weight on the regularization term for the control pulse second derivatives
- `phase_name::Symbol=:ϕ`: the name of the phase
- `phase_operators::Union{AbstractVector{<:AbstractMatrix}, Nothing}=nothing`: the phase operators for free phase corrections
- `constraints::Vector{<:AbstractConstraint}=AbstractConstraint[]`: the constraints to enforce

"""
function UnitarySmoothPulseProblem(
    system::AbstractQuantumSystem,
    operator::AbstractPiccoloOperator,
    T::Int,
    Δt::Union{Float64, <:AbstractVector{Float64}};
    unitary_integrator=UnitaryIntegrator,
    piccolo_options::PiccoloOptions=PiccoloOptions(),
    state_name::Symbol = :Ũ⃗,
    control_name::Symbol = :a,
    timestep_name::Symbol = :Δt,
    init_trajectory::Union{NamedTrajectory, Nothing}=nothing,
    a_bound::Float64=1.0,
    a_bounds=fill(a_bound, system.n_drives),
    a_guess::Union{Matrix{Float64}, Nothing}=nothing,
    da_bound::Float64=Inf,
    da_bounds::Vector{Float64}=fill(da_bound, system.n_drives),
    zero_initial_and_final_derivative::Bool=false,
    dda_bound::Float64=1.0,
    dda_bounds::Vector{Float64}=fill(dda_bound, system.n_drives),
    Δt_min::Float64=Δt isa Float64 ? 0.5 * Δt : 0.5 * mean(Δt),
    Δt_max::Float64=Δt isa Float64 ? 1.5 * Δt : 1.5 * mean(Δt),
    Q::Float64=100.0,
    R=1e-2,
    R_a::Union{Float64, Vector{Float64}}=R,
    R_da::Union{Float64, Vector{Float64}}=R,
    R_dda::Union{Float64, Vector{Float64}}=R,
    phase_name::Symbol=:ϕ,
    phase_operators::Union{AbstractVector{<:AbstractMatrix}, Nothing}=nothing,
    constraints::Vector{<:AbstractConstraint}=AbstractConstraint[],
    kwargs...
)

    # Trajectory
    if !isnothing(init_trajectory)
        traj = init_trajectory
    else
        traj = initialize_trajectory(
            operator,
            T,
            Δt,
            system.n_drives,
            (a_bounds, da_bounds, dda_bounds);
            state_name=state_name,
            control_name=control_name,
            timestep_name=timestep_name,
            free_time=piccolo_options.free_time,
            Δt_bounds=(Δt_min, Δt_max),
            zero_initial_and_final_derivative=zero_initial_and_final_derivative,
            geodesic=piccolo_options.geodesic,
            bound_state=piccolo_options.bound_state,
            a_guess=a_guess,
            system=system,
            rollout_integrator=piccolo_options.rollout_integrator,
            phase_name=phase_name,
            phase_operators=phase_operators
        )
    end

    # Objective
    J = UnitaryInfidelityLoss(operator, state_name, traj; Q=Q)

    control_names = [
        name for name ∈ traj.names
            if endswith(string(name), string(control_name))
    ]

    J += QuadraticRegularizer(control_names[1], traj, R_a)
    J += QuadraticRegularizer(control_names[2], traj, R_da)
    J += QuadraticRegularizer(control_names[3], traj, R_dda)

    # Optional Piccolo constraints and objectives
    apply_piccolo_options!(
        J, constraints, piccolo_options, traj, state_name, timestep_name;
        state_leakage_indices=operator isa EmbeddedOperator ? get_leakage_indices(operator) : nothing
    )

    integrators = [
        unitary_integrator(system, traj, state_name, control_name),
        DerivativeIntegrator(traj, control_name, control_names[2]),
        DerivativeIntegrator(traj, control_names[2], control_names[3]),
    ]

    return DirectTrajOptProblem(
        traj,
        J,
        integrators;
        constraints=constraints
    )
end

function UnitarySmoothPulseProblem(
    H_drift::AbstractMatrix{<:Number},
    H_drives::Vector{<:AbstractMatrix{<:Number}},
    args...;
    kwargs...
)
    system = QuantumSystem(H_drift, H_drives)
    return UnitarySmoothPulseProblem(system, args...; kwargs...)
end

# *************************************************************************** #

@testitem "Hadamard gate" begin
    sys = QuantumSystem(GATES[:Z], [GATES[:X], GATES[:Y]])
    U_goal = GATES[:H]
    T = 51
    Δt = 0.2

    prob = UnitarySmoothPulseProblem(
        sys, U_goal, T, Δt;
        da_bound=1.0,
        piccolo_options=PiccoloOptions(verbose=false)
    )


    ipopt_options=IpoptOptions(print_level=1)

    initial = unitary_rollout_fidelity(prob.trajectory, sys)
    solve!(prob, max_iter=50, options=ipopt_options)
    final = unitary_rollout_fidelity(prob.trajectory, sys)
    println(final)
    @test final > initial
end

@testitem "Hadamard gate with exponential integrator, bounded states, and control norm constraint" begin
    sys = QuantumSystem(GATES[:Z], [GATES[:X], GATES[:Y]])
    U_goal = GATES[:H]
    T = 51
    Δt = 0.2

    piccolo_options = PiccoloOptions(
        verbose=false,
        integrator=:exponential,
        # jacobian_structure=false,
        bound_state=true,
        complex_control_norm_constraint_name=:a
    )

    prob = UnitarySmoothPulseProblem(
        sys, U_goal, T, Δt,
        piccolo_options=piccolo_options
    )

    ipopt_options=IpoptOptions(print_level=1)

    initial = unitary_rollout_fidelity(prob.trajectory, sys)
    solve!(prob, max_iter=50, options=ipopt_options)
    final = unitary_rollout_fidelity(prob.trajectory, sys)
    @test final > initial
end



@testitem "EmbeddedOperator Hadamard gate" begin
    a = annihilate(3)
    sys = QuantumSystem([(a + a')/2, (a - a')/(2im)])
    U_goal = EmbeddedOperator(GATES[:H], sys)
    T = 51
    Δt = 0.2

    print_level = 1 # 5 is normal

    # Test embedded operator
    # ----------------------
    prob = UnitarySmoothPulseProblem(
        sys, U_goal, T, Δt,
        piccolo_options=PiccoloOptions(verbose=false)
    )

    initial = unitary_rollout_fidelity(prob.trajectory, sys, subspace=U_goal.subspace)
    solve!(prob, max_iter=50, options=IpoptOptions(print_level=print_level))
    final = unitary_rollout_fidelity(prob.trajectory, sys, subspace=U_goal.subspace)
    @test final > initial

    # Test leakage suppression
    # ------------------------
    a = annihilate(4)
    sys = QuantumSystem([(a + a')/2, (a - a')/(2im)])
    U_goal = EmbeddedOperator(GATES[:H], sys)
    T = 50
    Δt = 0.2

    prob = UnitarySmoothPulseProblem(
        sys, U_goal, T, Δt,
        leakage_suppression=true, R_leakage=1e-1,
        piccolo_options=PiccoloOptions(verbose=false)
    )

    initial = unitary_rollout_fidelity(prob.trajectory, sys, subspace=U_goal.subspace)
    solve!(prob, max_iter=50, options=IpoptOptions(print_level=print_level))
    final = unitary_rollout_fidelity(prob.trajectory, sys, subspace=U_goal.subspace)
    @test final > initial
end

# @testitem "Additional Objective" begin
#     H_drift = GATES[:Z]
#     H_drives = [GATES[:X], GATES[:Y]]
#     U_goal = GATES[:H]
#     T = 50
#     Δt = 0.2

#     prob_vanilla = UnitarySmoothPulseProblem(
#         H_drift, H_drives, U_goal, T, Δt,
#         ipopt_options=IpoptOptions(print_level=1),
#         piccolo_options=PiccoloOptions(verbose=false),
#     )

#     J_additional = QuadraticRegularizer(:dda, prob_vanilla.trajectory, 10.0)

#     prob_additional = UnitarySmoothPulseProblem(
#         H_drift, H_drives, U_goal, T, Δt,
#         ipopt_options=IpoptOptions(print_level=1),
#         piccolo_options=PiccoloOptions(verbose=false),
#         additional_objective=J_extra,
#     )

#     J_prob_vanilla = Problems.get_objective(prob_vanilla)

#     J_additional = Problems.get_objective(prob_additional)

#     Z = prob_vanilla.trajectory
#     Z⃗ = vec(prob_vanilla.trajectory)

#     @test J_prob_vanilla.L(Z⃗, Z) + J_extra.L(Z⃗, Z) ≈ J_additional.L(Z⃗, Z)
# end
@testitem "Free phase Y gate using X" begin
    using Random
    # Random.seed!(1234)
    phase_name = :ϕ
    phase_operators = [PAULIS[:Z]]
    sys = QuantumSystem([PAULIS[:X]])
    prob = UnitarySmoothPulseProblem(
        sys,
        GATES[:Y],
        51,
        0.2;
        phase_operators=phase_operators,
        phase_name=phase_name,
        ipopt_options=IpoptOptions(print_level=1),
        piccolo_options=PiccoloOptions(verbose=false, free_time=false)
    )

    before = prob.trajectory.global_data[phase_name]
    solve!(prob, max_iter=50)
    after = prob.trajectory.global_data[phase_name]

    @test before ≠ after

    @test unitary_rollout_fidelity(
        prob.trajectory,
        sys;
        phases=prob.trajectory.global_data[phase_name],
        phase_operators=phase_operators
    ) > 0.9

    @test unitary_rollout_fidelity(prob.trajectory, sys) < 0.9
end
