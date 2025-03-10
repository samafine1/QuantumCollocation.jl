module Options 

export PiccoloOptions

using ExponentialAction

"""
    PiccoloOptions

Options for the Piccolo quantum optimal control library.

# Fields
- `verbose::Bool = true`: Print verbose output
- `verbose_evaluator::Bool = false`: Print verbose output from the evaluator
- `free_time::Bool = true`: Allow free time optimization
- `timesteps_all_equal::Bool = true`: Use equal timesteps
- `integrator::Symbol = :pade`: Integrator to use
- `pade_order::Int = 4`: Order of the Pade approximation
- `rollout_integrator::Function = expv`: Integrator to use for rollout
- `eval_hessian::Bool = false`: Evaluate the Hessian
- `geodesic = true`: Use the geodesic to initialize the optimization.
- `blas_multithreading::Bool = true`: Use BLAS multithreading.
- `build_trajectory_constraints::Bool = true`: Build trajectory constraints.
- `complex_control_norm_constraint_name::Union{Nothing, Symbol} = nothing`: Name of the complex control norm constraint.
- `complex_control_norm_constraint_radius::Float64 = 1.0`: Radius of the complex control norm constraint.
- `bound_state::Bool = false`: Bound the state.
- `leakage_suppression::Bool = false`: Suppress leakage.
- `R_leakage::Float64 = 1.0`: Leakage suppression parameter.
- `free_phase_infidelity::Bool = false`: Free phase infidelity.
- `phase_operators::Union{Nothing, AbstractVector{<:AbstractMatrix{<:Complex}}} = nothing`: Phase operators.
- `phase_name::Symbol = :ϕ`: Name of the phase.
"""
@kwdef mutable struct PiccoloOptions
    verbose::Bool = true
    verbose_evaluator::Bool = false
    free_time::Bool = true
    timesteps_all_equal::Bool = true
    integrator::Symbol = :pade
    pade_order::Int = 4
    rollout_integrator::Function = expv
    eval_hessian::Bool = false
    geodesic = true
    blas_multithreading::Bool = true
    build_trajectory_constraints::Bool = true
    complex_control_norm_constraint_name::Union{Nothing, Symbol} = nothing
    complex_control_norm_constraint_radius::Float64 = 1.0
    bound_state::Bool = integrator == :exponential
    leakage_suppression::Bool = false
    R_leakage::Float64 = 1.0
    free_phase_infidelity::Bool = false
    phase_operators::Union{Nothing, AbstractVector{<:AbstractMatrix{<:Complex}}} = nothing
    phase_name::Symbol = :ϕ
end

end