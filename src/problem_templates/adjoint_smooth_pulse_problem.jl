export AdjointUnitarySmoothPulseProblem
function AdjointUnitarySmoothPulseProblem(
    system::ParameterizedQuantumSystem,
    goal::AbstractPiccoloOperator,
    T::Int,
    Δt::Union{Float64, <:AbstractVector{Float64}};
    times::Vector{<:Union{AbstractVector,Nothing}}=[nothing for s∈system.Gₐ],
    down_times::Vector{<:Union{AbstractVector,Nothing}}=[nothing for s∈system.Gₐ],
    unitary_integrator=AdjointUnitaryIntegrator,
    state_name::Symbol = :Ũ⃗,
    state_adjoint_name::Symbol = :Ũ⃗ₐ,
    control_name::Symbol = :a,
    timestep_name::Symbol = :Δt,
    init_trajectory::Union{NamedTrajectory, Nothing}=nothing,
    a_guess::Union{Matrix{Float64}, Nothing}=nothing,
    a_bound::Float64=1.0,
    a_bounds=fill(a_bound, system.n_drives),
    da_bound::Float64=Inf,
    da_bounds::Vector{Float64}=fill(da_bound, system.n_drives),
    dda_bound::Float64=1.0,
    dda_bounds::Vector{Float64}=fill(dda_bound, system.n_drives),
    Δt_min::Float64=Δt isa Float64 ? 0.5 * Δt : 0.5 * mean(Δt),
    Δt_max::Float64=Δt isa Float64 ? 1.5 * Δt : 1.5 * mean(Δt),
    Q = 1.0,
    R=1e-2,
    R_a::Union{Float64, Vector{Float64}}=R,
    R_da::Union{Float64, Vector{Float64}}=R,
    R_dda::Union{Float64, Vector{Float64}}=R,
    constraints::Vector{<:AbstractConstraint}=AbstractConstraint[],
    piccolo_options::PiccoloOptions=PiccoloOptions(),
    final_fidelity::Float64 = 1.0,
)
    if piccolo_options.verbose
        println("    constructing AdjointUnitarySmoothPulseProblem...")
        println("\tusing integrator: $(typeof(unitary_integrator))")
    end

    # Trajectory
    if !isnothing(init_trajectory)
        traj = copy(init_trajectory)
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
            a_guess=a_guess,
            rollout_integrator=piccolo_options.rollout_integrator,
            verbose=piccolo_options.verbose
        )
    end



    adj = [traj[state_name] for s ∈ system.Gₐ]
    

    # for a∈adj
    #     a[:,1] *= 0 
    # end


    full_rollout =  adjoint_unitary_rollout(traj, system ;unitary_name=state_name,drive_name = control_name)[2]

    for i in 1:length(system.Gₐ)
        adj[i] = full_rollout[i]
    end


    state_adjoint_names = [add_suffix(state_adjoint_name,string(i)) for i in 1:length(system.Gₐ)]


    for (name,a) in zip(state_adjoint_names,adj)
        add_component!(traj,name,a;type=:state)
        traj.initial = merge(traj.initial, (name=> a[:,1], ))
        
    end

    fidelity_constraint = FinalUnitaryFidelityConstraint(
        goal, state_name, final_fidelity, traj
    )

    constraints = push!(constraints, fidelity_constraint)

    control_names = [
        name for name ∈ traj.names
            if endswith(string(name), string(control_name))
    ]

    J = QuadraticRegularizer(control_names[1], traj, R_a)
    J += QuadraticRegularizer(control_names[2], traj, R_da)
    J += QuadraticRegularizer(control_names[3], traj, R_dda)

    for (name,t) ∈ zip(state_adjoint_names,times)
        if(!isnothing(t))
            J += UnitaryNormLoss(name, traj, t; Q=Q)
        end
    end

    for (name,down_t) ∈ zip(state_adjoint_names,down_times)
        if(!isnothing(down_t))
            J += UnitaryNormLoss(name, traj, down_t; Q=Q, rep = false)
        end
    end

    # Optional Piccolo constraints and objectives
    apply_piccolo_options!(
        J, constraints, piccolo_options, traj, state_name, timestep_name;
        state_leakage_indices=goal isa EmbeddedOperator ? get_leakage_indices(goal) : nothing
    )

    integrators = [
        unitary_integrator(system, traj, state_name,state_adjoint_names,control_name),
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