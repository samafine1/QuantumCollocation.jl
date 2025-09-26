module QuantumObjectives

export KetInfidelityObjective
export UnitaryInfidelityObjective
export DensityMatrixPureStateInfidelityObjective
export UnitarySensitivityObjective
export FirstOrderObjective
export FirstOrderObjective
export UnitaryFreePhaseInfidelityObjective
export LeakageObjective
export UniversalObjective
export FastUniversalObjective
export TurboUniversalObjective
export FastToggleObjective
export UltraUniversalObjective

using LinearAlgebra
using NamedTrajectories
using PiccoloQuantumObjects
using DirectTrajOpt
using TestItems
using TrajectoryIndexingUtils
using ForwardDiff
using SparseArrays
# using Zygote
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
        He_vec = H_err(a)
        return [U' * He * U for He in He_vec]

    end

    function ℓ(Z::AbstractVector{<:Real})
        terms = []
        for j in 1:length(toggle(Z, a_indices[1], Ũ⃗_indices[1]))
            sum_terms = sum(toggle(Z, a_idx, U_idx)[j] for (a_idx, U_idx) in zip(a_indices, Ũ⃗_indices))
            push!(terms, sum_terms)
        end
        FO_obj = sum(real(norm(tr(term' * term), 2)) / real(traj.T^2 * H_scale^2) for term in terms) 
        return Q_t * FO_obj
    end

    # function ℓ(Z::AbstractVector{<:Real})
    #     sum_terms = sum(toggle(Z, a_idx, U_idx) for (a_idx, U_idx) in zip(a_indices, Ũ⃗_indices))
    #     return Q_t * real(norm(tr(sum_terms' * sum_terms), 2)) / real(traj.T^2 * H_scale)
    # end

    ∇ℓ = Z ->  ForwardDiff.gradient(ℓ, Z)

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
        ∂²ℓ_values = [H_full[i, j] for (i, j) in structure_pairs]
        
        return ∂²ℓ_values
    end

    return Objective(ℓ, ∇ℓ, ∂²ℓ, ∂²ℓ_structure)
end
    
function FastToggleObjective(
    H_err::Function,
    traj::NamedTrajectory;
    Q_t::Float64=1.0
)
    a_indices   = [collect(slice(k, traj.components.a, traj.dim)) for k in 1:traj.T]
    Ũ⃗_indices  = [collect(slice(k, traj.components.Ũ⃗, traj.dim)) for k in 1:traj.T]
    a_ref       = ones(length(traj.components.a))
    H_scale     = norm(H_err(a_ref), 2)

    packZ(Z) = vcat((@views Z[r] for r in Ũ⃗_indices)...)  # length L = m*T

    # Build V (m×T) from the packed vector z̃ = [vec(U₁); vec(U₂); … ; vec(U_T)]
    build_V_from_packed(z̃) = begin
        T = length(Ũ⃗_indices)
        m = length(Ũ⃗_indices[1])
        V = Matrix{complex(eltype(z̃))}(undef, m, T)
        @inbounds for k in 1:T
            V[:, k] = @view z̃[(k-1)*m+1 : k*m]
        end
        V
    end


    @views function toggle(Z::AbstractVector, a_idx::AbstractVector{<:Int}, U_idx::AbstractVector{<:Int})
        a  = Z[a_idx]
        U  = iso_vec_to_operator(Z[U_idx])
        He_vec = H_err(a)
        # compute once per trajectory step
        return [U' * He * U for He in He_vec]
    end
    norm_val = Q_t / (traj.T^2 * H_scale)

    function ℓ(Z::AbstractVector{<:Real})
        toggled_sets = [toggle(Z, a_idx, U_idx) for (a_idx, U_idx) in zip(a_indices, Ũ⃗_indices)]
        n_terms = length(first(toggled_sets))  # number of He operators
        dims = size(toggled_sets[1][1])
        obj = 0

        @inbounds for j in 1:n_terms
            # accumulate over time steps
            sum_term = zeros(eltype(toggled_sets[1][j]), dims)
            for set in toggled_sets
                sum_term .+= set[j]
            end
            trval = tr(sum_term' * sum_term)
            obj  += norm_val * real(norm(trval, 2))
        end

        return obj
    end
    
    ∇ℓ = Z -> ForwardDiff.gradient(ℓ, Z)

    function ∂²ℓ_structure()
        all_idx = vcat(Ũ⃗_indices...)
        return [(i, j) for i in all_idx for j in all_idx]
    end

    function ∂²ℓ(Z::AbstractVector{<:Real})
        z̃ = packZ(Z)
        H̃ = ForwardDiff.hessian(z -> begin
            V = build_V_from_packed(z)
            G = V' * V
            norm_val * sum(abs2, G)
        end, z̃)

        L = length(z̃)
        vals = Vector{eltype(H̃)}(undef, L*L)
        t = 1
        @inbounds for i in 1:L, j in 1:L
            vals[t] = H̃[i, j]
            t += 1
        end
        return vals
    end

    return Objective(ℓ, ∇ℓ, ∂²ℓ, ∂²ℓ_structure)
end
    
function UltraUniversalObjective(
        traj::NamedTrajectory;
        basis::Vector{Matrix{ComplexF64}}=[Matrix{ComplexF64}(undef, 0, 0)],
        Qu::Float64=1.0,
        Qb::Vector{Float64} = fill(1.0, length(basis)),
        toggle::Bool=false,
)
    ⊗ = kron
    Ũ⃗_indices  = [collect(slice(k, traj.components.Ũ⃗, traj.dim)) for k in 1:traj.T]
    function ℓ(Z::AbstractVector{<:Real})    
        ϵ = sum(iso_vec_to_operator(Z[r])⊗conj(iso_vec_to_operator(Z[r]))/traj.T for r in Ũ⃗_indices)
        d = size(ϵ)[1]

        if !toggle
            s = Qu * norm(tr(ϵ'ϵ))/d
        else
            s = 0
            for (i, b) in enumerate(basis)
                s += Qb[i]*norm(vec(b)'*ϵ'ϵ*vec(b))
            end
        end
        return s
    end

    ∇ℓ = Z ->  ForwardDiff.gradient(ℓ, Z)

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
        ∂²ℓ_values = [H_full[i, j] for (i, j) in structure_pairs]
        
        return ∂²ℓ_values
    end

    return Objective(ℓ, ∇ℓ, ∂²ℓ, ∂²ℓ_structure)
    end

function UniversalObjective(
    traj::NamedTrajectory;
    Q_t::Float64 = 1.0,
)

    T = traj.T
    U = ones(length(traj.components.Ũ⃗))
    U_scale = norm(U, 2)
    Ũ⃗_indices  = [collect(slice(k, traj.components.Ũ⃗, traj.dim)) for k in 1:traj.T]
    # one = Ũ⃗_indices[1]
    # U_scale  = norm(iso_vec_to_operator(Z[one]), 2)
    # Ũ⃗_indices  = [collect(slice(k, traj.components.Ũ⃗, traj.dim)) for k in 1:traj.T]
    # a_ref = ones(length(traj.components.a))
    # H_scale = norm(H_err(a_ref), 2)

    @views function trace(Z::AbstractVector, k_idx::AbstractVector{<:Int}, l_idx::AbstractVector{<:Int})
        Uₗ = iso_vec_to_operator(Z[l_idx])
        Uₖ = iso_vec_to_operator(Z[k_idx])
        return tr(Uₖ * Uₗ')
    end

    # ---- Objective ----
    function ℓ(Z::AbstractVector{<:Real})
        # Double sum over (k, ℓ) of |tr(U_{kℓ})|^2
        s = 0
        for k in 1:T
            for l in 1:T
                τ = trace(Z, Ũ⃗_indices[k], Ũ⃗_indices[l])
                s += abs2(τ)
            end
        end
        return Q_t * (s / (U_scale * T^2) - 1.0)
    end

    ∇ℓ = Z -> ForwardDiff.gradient(ℓ, Z)

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
        ∂²ℓ_values = [H_full[i, j] for (i, j) in structure_pairs]
        
        return ∂²ℓ_values
    end

    return Objective(ℓ, ∇ℓ, ∂²ℓ, ∂²ℓ_structure)
end

function FastUniversalObjective(
    traj::NamedTrajectory;
    Q_t::Float64 = 1.0,
)

    T = traj.T
    U = ones(length(traj.components.Ũ⃗))
    U_scale = norm(U, 2)
    Ũ⃗_indices  = [collect(slice(k, traj.components.Ũ⃗, traj.dim)) for k in 1:traj.T]
    normalization = Q_t / (U_scale * T^2)
    @views function trace(Z::AbstractVector, k_idx::AbstractVector{<:Int}, l_idx::AbstractVector{<:Int})
        Uₗ = iso_vec_to_operator(Z[l_idx])
        Uₖ = iso_vec_to_operator(Z[k_idx])
        return tr(Uₖ * Uₗ'), tr(Uₗ * Uₖ')
    end

    # ---- Objective ----
    function ℓ(Z::AbstractVector{<:Real})
        # Double sum over (k, ℓ) of |tr(U_{kℓ})|^2
        s = zero(eltype(Z))
        for k in 1:T
            for l in 1:k
                τ1, τ2 = trace(Z, Ũ⃗_indices[k], Ũ⃗_indices[l])
                s += abs2(τ1) + abs2(τ2)
            end
        end
        return normalization * s - Q_t
    end

    ∇ℓ = Z -> ForwardDiff.gradient(ℓ, Z)

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
        ∂²ℓ_values = [H_full[i, j] for (i, j) in structure_pairs]
        
        return ∂²ℓ_values
    end

    return Objective(ℓ, ∇ℓ, ∂²ℓ, ∂²ℓ_structure)
end

function TurboUniversalObjective(
    traj::NamedTrajectory;
    Q_t::Float64 = 1.0,
)

    T = traj.T
    U = ones(length(traj.components.Ũ⃗))
    U_scale = norm(U, 2)
    Ũ⃗_indices  = [collect(slice(k, traj.components.Ũ⃗, traj.dim)) for k in 1:traj.T]
    normalization = Q_t / (U_scale * T^2)
    # Build a column-stacked matrix V whose k-th column is the vectorized U(t_k,0).
    # Ũ⃗_indices[k] are the index ranges/slices inside Z for the k-th unitary's vectorization.
    
    packZ(Z) = vcat((@views Z[r] for r in Ũ⃗_indices)...)  # length L = m*T

    # Build V (m×T) from the packed vector z̃ = [vec(U₁); vec(U₂); … ; vec(U_T)]
    build_V_from_packed(z̃) = begin
        T = length(Ũ⃗_indices)
        m = length(Ũ⃗_indices[1])
        V = Matrix{complex(eltype(z̃))}(undef, m, T)
        @inbounds for k in 1:T
            V[:, k] = @view z̃[(k-1)*m+1 : k*m]
        end
        V
    end

    function ℓ(Z::AbstractVector{<:Real})
        z̃ = packZ(Z)                   # only the entries that matter
        V  = build_V_from_packed(z̃)    # m×T
        G  = V' * V                    # T×T Gram
        s  = sum(abs2, G)              # ‖G‖_F^2 = Σ_{k,ℓ} |Tr(U_k U_ℓ†)|²
        return normalization * s - Q_t
    end


    ∇ℓ = Z -> ForwardDiff.gradient(ℓ, Z)

    function ∂²ℓ_structure()
        all_idx = vcat(Ũ⃗_indices...)
        return [(i, j) for i in all_idx for j in all_idx]
    end

    function ∂²ℓ(Z::AbstractVector{<:Real})
        z̃ = packZ(Z)
        H̃ = ForwardDiff.hessian(z -> begin
            V = build_V_from_packed(z)
            G = V' * V
            normalization * sum(abs2, G) - Q_t
        end, z̃)

        L = length(z̃)
        vals = Vector{eltype(H̃)}(undef, L*L)
        t = 1
        @inbounds for i in 1:L, j in 1:L
            vals[t] = H̃[i, j]
            t += 1
        end
        return vals
    end

    Objective(ℓ, ∇ℓ, ∂²ℓ, ∂²ℓ_structure)
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