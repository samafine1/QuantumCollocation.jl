export UnitaryVariationalProblem


function UnitaryVariationalProblem(
    system::VariationalQuantumSystem,
    goal::AbstractPiccoloOperator,
    T::Int,
    Δt::Union{Float64, <:AbstractVector{Float64}};
    robust_times::AbstractVector{<:Union{AbstractVector{Int}, Nothing}}=[nothing for s ∈ system.G_vars],
    sensitive_times::AbstractVector{<:Union{AbstractVector{Int}, Nothing}}=[nothing for s ∈ system.G_vars],
    unitary_integrator=VariationalUnitaryIntegrator,
    state_name::Symbol = :Ũ⃗,
    variational_state_name::Symbol = :Ũ⃗ₐ,
    control_name::Symbol = :a,
    timestep_name::Symbol = :Δt,
    init_trajectory::Union{NamedTrajectory, Nothing}=nothing,
    a_bound::Float64=1.0,
    a_bounds=fill(a_bound, system.n_drives),
    da_bound::Float64=Inf,
    da_bounds::Vector{Float64}=fill(da_bound, system.n_drives),
    dda_bound::Float64=1.0,
    dda_bounds::Vector{Float64}=fill(dda_bound, system.n_drives),
    Δt_min::Float64=Δt isa Float64 ? 0.5 * Δt : 0.5 * mean(Δt),
    Δt_max::Float64=Δt isa Float64 ? 1.5 * Δt : 1.5 * mean(Δt),
    Q::Float64 = 100.0,
    Q_v::Float64=1.0,  
    R=1e-2,
    R_a::Union{Float64, Vector{Float64}}=R,
    R_da::Union{Float64, Vector{Float64}}=R,
    R_dda::Union{Float64, Vector{Float64}}=R,
    constraints::Vector{<:AbstractConstraint}=AbstractConstraint[],
    piccolo_options::PiccoloOptions=PiccoloOptions(),
    final_fidelity::Float64 = 1.0,
)
    if piccolo_options.verbose
        println("    constructing UnitaryVariationalProblem...")
        println("\tusing integrator: $(typeof(unitary_integrator))")
        println("\ttotal variational parameters: $(length(system.G_vars))")
    end

    # Trajectory
    if !isnothing(init_trajectory)
        traj = deepcopy(init_trajectory)
    else
        traj = initialize_trajectory(
            goal,
            T,
            Δt,
            system.n_drives,
            (a_bounds, da_bounds, dda_bounds);
            state_name=state_name,
            control_name=control_name,
            timestep_name=timestep_name,
            Δt_bounds=(Δt_min, Δt_max),
            zero_initial_and_final_derivative=piccolo_options.zero_initial_and_final_derivative,
            geodesic=piccolo_options.geodesic,
            bound_state=piccolo_options.bound_state,
            rollout_integrator=piccolo_options.rollout_integrator,
            verbose=piccolo_options.verbose
        )
    end

    _, Ũ⃗_vars =  variational_unitary_rollout(
        traj, 
        system;
        unitary_name=state_name,
        drive_name=control_name
    )

    variational_state_names = [
        Symbol(string(variational_state_name) * "$(i)") for i in eachindex(system.G_vars)
    ]

    for (name, var) in zip(variational_state_names, Ũ⃗_vars)
        add_component!(traj, name, var; type=:state)
        traj.initial = merge(traj.initial, (name => var[:, 1], ))
    end

    control_names = [
        name for name ∈ traj.names
            if endswith(string(name), string(control_name))
    ]

    # fidelity_constraint = FinalUnitaryFidelityConstraint(
    #     goal, state_name, final_fidelity, traj
    # )

    # # constraints
    # constraints = push!(constraints, fidelity_constraint)

    # objective
    J = UnitaryInfidelityObjective(goal, state_name, traj; Q=Q)

    J += QuadraticRegularizer(control_names[1], traj, R_a)
    J += QuadraticRegularizer(control_names[2], traj, R_da)
    J += QuadraticRegularizer(control_names[3], traj, R_dda)

    # enhance sensitivity
    for (name, times) ∈ zip(variational_state_names, sensitive_times)
        if !isnothing(times)
            J += UnitaryNormLoss(name, traj, times; Q=Q_v)
        end
    end

    # suppress sensitivity
    for (name, times) ∈ zip(variational_state_names, robust_times)
        if !isnothing(times)
            J += UnitarySensitivityObjective(name, traj, times; Q=Q_v, robust=true)
        end
    end
    
    # Optional Piccolo constraints and objectives
    apply_piccolo_options!(
        J, constraints, piccolo_options, traj, state_name, timestep_name;
        state_leakage_indices=goal isa EmbeddedOperator ? get_leakage_indices(goal) : nothing
    )

    integrators = [
        unitary_integrator(system, traj, state_name, variational_state_names, control_name),
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