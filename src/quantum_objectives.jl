module QuantumObjectives

export KetInfidelityObjective
export UnitaryInfidelityObjective
export DensityMatrixPureStateInfidelityObjective
export UnitarySensitivityObjective
export FirstOrderObjective
export UnitaryFreePhaseInfidelityObjective
export LeakageObjective
export UniversalObjective
export FastUniversalObjective
export TurboUniversalObjective
export UltraUniversalObjective
export TogglingObjective
export PertTogglingObjective
export AutoTogglingObjective

using LinearAlgebra
using NamedTrajectories
using PiccoloQuantumObjects
using DirectTrajOpt
using TestItems
using TrajectoryIndexingUtils
using ForwardDiff
using SparseArrays

using TrajectoryIndexingUtils
using ForwardDiff
using SparseArrays

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
    num_errors::Int
)
    Δt = traj.Δt[1]
    T = traj.T

    ℓ = Ũ⃗ -> scale^4 * sqrt(unitary_fidelity_loss(Ũ⃗)) / (Δt * T)^2 / num_errors

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
        D = length(toggle(Z, a_indices[1], Ũ⃗_indices[1]))
        for j in 1:D
            sum_terms = sum(toggle(Z, a_idx, U_idx)[j] for (a_idx, U_idx) in zip(a_indices, Ũ⃗_indices))
            push!(terms, sum_terms)
        end
        FO_obj = sum(real(norm(tr(term' * term), 2)) / real(traj.T^2 * H_scale^2) / D for term in terms) 
        return Q_t * FO_obj
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
    D = size(iso_vec_to_operator(traj.components.Ũ⃗[:,end]))[1]
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
        return Q_t * (s / (D^2 * T^4))
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
    d = size(iso_vec_to_operator(traj.components.Ũ⃗[:,end]))[1]
    Ũ⃗_indices  = [collect(slice(k, traj.components.Ũ⃗, traj.dim)) for k in 1:traj.T]
    normalization = Q_t / (d^2 * T^2)
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
        return normalization * s
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


#-----------------------------
# Toggling 
#-----------------------------
function TogglingObjective(
    H_err::Function,
    ∂H_err::Function, 
    traj::NamedTrajectory;
    Q_t::Float64=1.0
)
    a_indices  = [collect(slice(k, traj.components.a, traj.dim)) for k in 1:traj.T-1]
    Ũ⃗_indices  = [collect(slice(k, traj.components.Ũ⃗, traj.dim)) for k in 1:traj.T-1]
    a_ref = ones(length(traj.components.a))
    H_scale = norm(H_err(a_ref), 2)
    D = length(H_err(a_ref))

    @views function toggle(Z::AbstractVector, a_idx::AbstractVector{<:Int}, U_idx::AbstractVector{<:Int})
        a  = Z[a_idx]
        U  = iso_vec_to_operator(Z[U_idx])
        He_vec = H_err(a)
        return [U' * He * U for He in He_vec]
    end

    function ℓ(Z::AbstractVector{<:Real})
        terms = []
        for j in 1:length(toggle(Z, a_indices[1], Ũ⃗_indices[1]))
            term = [toggle(Z, a_idx, U_idx)[j] for (a_idx, U_idx) in zip(a_indices, Ũ⃗_indices)]
            sum_term = sum(term)
            push!(terms, sum_term)
        end
        FO_obj = sum(real(norm(tr(term' * term), 2)) / real(traj.T^2 * H_scale^2) / D for term in terms) 
        return Q_t * FO_obj
    end 

    function ∇toggle(Z::AbstractVector, a_idx::AbstractVector{<:Int}, U_idx::AbstractVector{<:Int})
        a  = Z[a_idx]
        U  = iso_vec_to_operator(Z[U_idx])
        He_vec = H_err(a)
        ∂He_vec = ∂H_err(a)
        n = Int(sqrt(length(U_idx) ÷ 2))
        grads = [spzeros(Matrix{ComplexF64}, length(Z)) for He in He_vec]

        for (g, He, ∂He) in zip(grads, He_vec, ∂He_vec)
            ### a derivatives ### 
            for (a_i, ∂Ha) in zip(a_idx, ∂He)
                g[a_i] = U' * ∂Ha * U
            end
            ### U derivatives ### 
            for U_i in U_idx
                idx = U_i - U_idx[1] + 1
                i = (idx % n) == 0 ? n : idx % n
                j =(idx - i) ÷ (2 * n) + 1
                re = (idx % (2*n) == 0 ? 2*n : idx%(2*n)) ≤ n
                d = spzeros(ComplexF64, (n,n))
                d[i,j] = re ? 1 : 1im 
                g[U_i] = U' * He * d + d' * He * U
            end 
        end
        return grads 
    end

    function ∇ℓ(Z::AbstractVector{<:Real})
        ### Get the "total" toggle for each alongside the gradient 
        terms = []
        grads = []
        for j in 1:length(toggle(Z, a_indices[1], Ũ⃗_indices[1]))
            sum_terms = sum(toggle(Z, a_idx, U_idx)[j] for (a_idx, U_idx) in zip(a_indices, Ũ⃗_indices))
            sum_grads = spzeros(Matrix{ComplexF64}, length(Z))
            for (a_idx, U_idx) in zip(a_indices, Ũ⃗_indices) 
                g = ∇toggle(Z, a_idx, U_idx)[j] 
                for (idx, val) in zip(g.nzind, g.nzval)
                    sum_grads[idx] = val
                end
            end
            push!(terms, sum_terms)
            push!(grads, sum_grads)
        end

        full_grad = spzeros(size(Z))

        for (t, g) in zip(terms, grads)
            for (idx, v) in zip(g.nzind, g.nzval)
                full_grad[idx] += sum(2 * real(t) .* real(v) + 2 * imag(t) .* imag(v))
            end 
        end
        return Q_t * full_grad / real(traj.T^2 * H_scale^2) / D
    end 

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

function AutoTogglingObjective(
    H_err::Function,
    traj::NamedTrajectory;
    Q_t::Float64=1.0
)
    a_indices  = [collect(slice(k, traj.components.a, traj.dim)) for k in 1:traj.T-1]
    Ũ⃗_indices  = [collect(slice(k, traj.components.Ũ⃗, traj.dim)) for k in 1:traj.T-1]
    a_ref = ones(length(traj.components.a))
    H_scale = norm(H_err(a_ref), 2)
    D = length(H_err(a_ref))

    @views function toggle(Z::AbstractVector, a_idx::AbstractVector{<:Int}, U_idx::AbstractVector{<:Int})
        a  = Z[a_idx]
        U  = iso_vec_to_operator(Z[U_idx])
        He_vec = H_err(a)
        return [U' * He * U for He in He_vec]
    end

    function ℓ(Z::AbstractVector{<:Real})
        terms = []
        for j in 1:length(toggle(Z, a_indices[1], Ũ⃗_indices[1]))
            term = [toggle(Z, a_idx, U_idx)[j] for (a_idx, U_idx) in zip(a_indices, Ũ⃗_indices)]
            sum_term = sum(term)
            push!(terms, sum_term)
        end
        FO_obj = sum(real(norm(tr(term' * term), 2)) / real(traj.T^2 * H_scale^2) / D for term in terms) 
        return Q_t * FO_obj
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

function AutoPertTogglingObjective(
    H_err::Function,
    traj::NamedTrajectory;
    Q_t::Float64=1.0
)
    a_indices  = [collect(slice(k, traj.components.a, traj.dim)) for k in 1:traj.T-1]
    Ũ⃗_indices  = [collect(slice(k, traj.components.Ũ⃗, traj.dim)) for k in 1:traj.T-1]
    a_ref = ones(length(traj.components.a))
    H_scale = norm(H_err(a_ref), 2)
    D = length(H_err(a_ref))

    @views function toggle(Z::AbstractVector, a_idx::AbstractVector{<:Int}, U_idx::AbstractVector{<:Int})
        a  = Z[a_idx]
        U  = iso_vec_to_operator(Z[U_idx])
        He_vec = H_err(a)
        return [U' * He * U for He in He_vec]
    end

    function pert_tog_obj(
        traj::NamedTrajectory, 
        H_drives::Vector{Matrix{ComplexF64}},
        H_error::Matrix{ComplexF64};
        order::Int=1,
        a_bound::Float64=a_bound
    )
        T = traj.T
        Δt = get_timesteps(traj)

        sys = QuantumSystem(H_drives)
        U = iso_vec_to_operator.(eachcol(unitary_rollout(traj, sys)))

        # toggle integral
        H_ti = zeros(ComplexF64, size(U[1]))

        # note: U_1 = I, so U[:, k] = U_{k-1}.
        # you need to go to T-1, only
        for k in 1:T-1
            Hₖ = sum(traj.a[l, k] / a_bound * H for (l, H) in enumerate(H_drives))
            adjⁿH_E = H_error
            Eₖ_n = H_error * Δt[k]
            
            # get the different orders of the Hadamard lemma
            for n in 2:order
                coef_n = ComplexF64(im^(n-1) * a_bound^(n-1) * Δt[k]^n / factorial(big(n)))
                adjⁿH_E = commutator(Hₖ, adjⁿH_E)
                # Eₖ_n = push!(Eₖ_n, coef_n * adjⁿH_E)
                Eₖ_n += coef_n * adjⁿH_E
            end

            # nth order toggle integral up to k
            H_ti += U[k]' * Eₖ_n * U[k]
        end

        d₁ = size(U[1], 1)
        Δt₁ = Δt[1]
        metric = norm(tr(H_ti'H_ti)) / (T * Δt₁)^2 / d₁
        return metric
    end

    function ℓ(Z::AbstractVector{<:Real})
        terms = []
        for j in 1:length(toggle(Z, a_indices[1], Ũ⃗_indices[1]))
            term = [toggle(Z, a_idx, U_idx)[j] for (a_idx, U_idx) in zip(a_indices, Ũ⃗_indices)]
            sum_term = sum(term)
            push!(terms, sum_term)
        end
        FO_obj = sum(real(norm(tr(term' * term), 2)) / real(traj.T^2 * H_scale^2) / D for term in terms) 
        return Q_t * FO_obj
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


# function PertTogglingObjective(
#     H_err::Function,
#     ∂H_err::Function, 
#     traj::NamedTrajectory;
#     Q_t::Float64=1.0,
#     order::Int=4,
#     a_bound::Float64=1.0,
#     H_drives::Vector{Matrix{ComplexF64}}=Matrix{ComplexF64}[]
# )   
#     T = traj.T
#     Δt = get_timesteps(traj)
#     a_indices  = [collect(slice(k, traj.components.a, traj.dim)) for k in 1:T]
#     Ũ⃗_indices  = [collect(slice(k, traj.components.Ũ⃗, traj.dim)) for k in 1:T]
#     a_ref = ones(length(traj.components.a))
#     H_scale = norm(H_err(a_ref), 2)
#     D = size(iso_vec_to_operator(traj.components.Ũ⃗[:,end]))[1]
#     n_drives = length(H_drives)
    
#     commutator(A::AbstractMatrix{<:Number}, B::AbstractMatrix{<:Number}) = A*B - B*A
    
#     # Precompute factorials to avoid repeated big() calls
#     factorial_cache = [factorial(big(n)) for n in 1:order]
    
#     @views function toggle(Z::AbstractVector, k::Int)
#         U = iso_vec_to_operator(Z[Ũ⃗_indices[k]])
        
#         a_k = Z[a_indices[k]]
#         He_vec = H_err(a_k)
        
#         # Build Hₖ for this timestep
#         Hₖ = sum(a_k[l] / a_bound * H for (l, H) in enumerate(H_drives))
        
#         # Compute perturbative expansion for each error Hamiltonian component
#         toggle_terms = []
#         for He in He_vec
#             adjⁿH_E = He
#             Eₖ_n = He * Δt[k]
            
#             # Get the different orders of the Hadamard lemma
#             for n in 2:order
#                 coef_n = ComplexF64(im^(n-1) * a_bound^(n-1) * Δt[k]^n / factorial_cache[n])
#                 adjⁿH_E = commutator(Hₖ, adjⁿH_E)
#                 Eₖ_n += coef_n * adjⁿH_E
#             end
            
#             # Apply unitary transformation
#             push!(toggle_terms, U' * Eₖ_n * U)
#         end
        
#         return toggle_terms
#     end

#     function ∇ℓ(Z::AbstractVector{<:Real})
#         normalization = real(T^2 * H_scale^2) * D
#         n_components = length(H_err(Z[a_indices[1]]))
        
#         # Forward pass: compute all toggles and accumulate sums
#         toggles = [toggle(Z, k) for k in 1:T-1]
#         H_ti = [sum(toggles[k][j] for k in 1:T-1) for j in 1:n_components]
        
#         # Initialize gradient
#         grad = zeros(length(Z))
        
#         # Backward pass: accumulate gradients
#         for j in 1:n_components
#             # Gradient of tr(H_ti[j]' * H_ti[j]) w.r.t H_ti[j]
#             # Since ∂/∂H tr(H'H) = 2H for complex matrices
#             ∇H_ti = 2 * H_ti[j] / normalization
            
#             # Backpropagate through sum of toggles
#             for k in 1:T-1
#                 ∇toggle!(grad, Z, k, j, ∇H_ti)
#             end
#         end
        
#         return Q_t * grad
#     end
    
#     function ∇toggle!(grad, Z, k::Int, j::Int, ∇out)
#         # Extract current parameters
#         U = iso_vec_to_operator(Z[Ũ⃗_indices[k]])
#         a_k = Z[a_indices[k]]
#         He_vec = H_err(a_k)
#         He = He_vec[j]
        
#         # Build Hₖ
#         Hₖ = sum(a_k[l] / a_bound * H for (l, H) in enumerate(H_drives))
        
#         # Compute Eₖ and its derivatives w.r.t amplitude parameters
#         Eₖ, ∂Eₖ_∂a = compute_Ek_with_grad(He, Hₖ, a_k, k)
        
#         # Gradient w.r.t U parameters
#         # toggle = U' * Eₖ * U
#         # We need ∂toggle/∂Ũ⃗ᵢ where Ũ⃗ is the iso_vec representation
#         grad_U_iso_vec!(grad[Ũ⃗_indices[k]], U, Eₖ, ∇out)
        
#         # Gradient w.r.t amplitude parameters
#         for l in 1:length(a_k)
#             # ∂toggle/∂aₗ = U' * (∂Eₖ/∂aₗ) * U
#             ∇a_contribution = U' * ∂Eₖ_∂a[:,:,l] * U
#             grad[a_indices[k][l]] += real(tr(∇out' * ∇a_contribution))
#         end
#     end
    
#     function grad_U_iso_vec!(grad_U, U, Eₖ, ∇out)
#         """
#         Compute gradient w.r.t iso_vec representation of U.
        
#         For toggle = U' * Eₖ * U, we have:
#         ∂toggle/∂U = (∂U'/∂U) * Eₖ * U + U' * Eₖ * (∂U/∂U)
#                   = Eₖ' * U + U' * Eₖ
        
#         But we need the gradient w.r.t the iso_vec representation Ũ⃗.
#         Using the chain rule: ∂L/∂Ũ⃗ = vec(∂L/∂U * ∂U/∂Ũ⃗)
#         """
        
#         # Compute gradient w.r.t U (complex matrix)
#         # ∂L/∂toggle = ∇out, and toggle = U' * Eₖ * U
#         # Using matrix calculus: ∂L/∂U* = U * ∇out * Eₖ'
#         # and ∂L/∂U = Eₖ * U * ∇out'
#         ∇U_complex = Eₖ * U * ∇out' + Eₖ' * U * ∇out
        
#         # Convert to gradient w.r.t iso_vec representation
#         # The iso_vec representation stacks [real; imag] for each column
#         N = size(U, 1)
#         for i = 0:N-1
#             col_grad = @view ∇U_complex[:, i+1]
#             grad_U[i*2N .+ (1:N)] .+= real(col_grad)
#             grad_U[i*2N .+ (N+1:2N)] .+= imag(col_grad)
#         end
#     end
    
#     function compute_Ek_with_grad(He, Hₖ, a_k, k)
#         """
#         Compute Eₖ and its gradient w.r.t amplitude parameters a_k.
        
#         Eₖ = He * Δt[k] + Σₙ (coef_n * adjⁿ(Hₖ)[He])
#         where adjⁿ is n nested commutators with Hₖ
#         """
        
#         # Initialize
#         D = size(He, 1)
#         n_params = length(a_k)
#         Eₖ = He * Δt[k]
#         ∂Eₖ_∂a = zeros(ComplexF64, D, D, n_params)
        
#         adjⁿH_E = He
#         ∂adjⁿH_E_∂a = zeros(ComplexF64, D, D, n_params)  # Initially zero since He doesn't depend on a
        
#         # Compute perturbative expansion and gradients
#         for n in 2:order
#             coef_n = ComplexF64(im^(n-1) * a_bound^(n-1) * Δt[k]^n / factorial_cache[n])
            
#             # Compute commutator: [Hₖ, adjⁿ⁻¹H_E]
#             comm = commutator(Hₖ, adjⁿH_E)
            
#             # Gradient of commutator w.r.t a_k
#             # ∂[Hₖ, B]/∂aₗ = [∂Hₖ/∂aₗ, B] + [Hₖ, ∂B/∂aₗ]
#             for l in 1:n_params
#                 ∂Hₖ_∂aₗ = H_drives[l] / a_bound
#                 # Product rule for commutator
#                 ∂comm_∂aₗ = commutator(∂Hₖ_∂aₗ, adjⁿH_E) + commutator(Hₖ, ∂adjⁿH_E_∂a[:,:,l])
                
#                 # Update gradient of adjⁿH_E for next iteration
#                 ∂adjⁿH_E_∂a[:,:,l] = ∂comm_∂aₗ
                
#                 # Accumulate contribution to Eₖ gradient
#                 ∂Eₖ_∂a[:,:,l] += coef_n * ∂comm_∂aₗ
#             end
            
#             # Add contribution to Eₖ
#             Eₖ += coef_n * comm
            
#             # Update for next iteration
#             adjⁿH_E = comm
#         end
        
#         return Eₖ, ∂Eₖ_∂a
#     end
    
#     # Helper function for commutator
#     function commutator(A, B)
#         return A * B - B * A
#     end
    
#     # Optional: More efficient version that preallocates workspace
#     struct GradientWorkspace
#         ∇U_complex::Matrix{ComplexF64}
#         ∂adjⁿH_E_∂a::Array{ComplexF64, 3}
#         ∂Eₖ_∂a::Array{ComplexF64, 3}
#         comm_temp::Matrix{ComplexF64}
#     end
    
#     function GradientWorkspace(D::Int, n_params::Int)
#         return GradientWorkspace(
#             zeros(ComplexF64, D, D),
#             zeros(ComplexF64, D, D, n_params),
#             zeros(ComplexF64, D, D, n_params),
#             zeros(ComplexF64, D, D)
#         )
#     end
    
#     # Version with preallocated workspace for better performance
#     function ∇ℓ_fast(Z::AbstractVector{<:Real}, workspace::GradientWorkspace)
#         normalization = real(T^2 * H_scale^2) * D
#         n_components = length(H_err(Z[a_indices[1]]))
        
#         # Forward pass: compute all toggles and accumulate sums
#         toggles = [toggle(Z, k) for k in 1:T-1]
#         H_ti = [sum(toggles[k][j] for k in 1:T-1) for j in 1:n_components]
        
#         # Initialize gradient
#         grad = zeros(length(Z))
        
#         # Backward pass: accumulate gradients
#         for j in 1:n_components
#             ∇H_ti = 2 * H_ti[j] / normalization
            
#             for k in 1:T-1
#                 ∇toggle_fast!(grad, Z, k, j, ∇H_ti, workspace)
#             end
#         end
        
#         return Q_t * grad
#     end

#     function ∂²toggle(Z::AbstractVector, k::Int, var_idx1::Int, var_idx2::Int)
#         """
#         Compute second derivative of toggle with respect to variables at indices var_idx1 and var_idx2
#         """
#         # Determine which variables we're differentiating with respect to
#         if k == 1
#             U = I(D)
#             U_idx = Int[]
#         else
#             U_idx = Ũ⃗_indices[k-1]
#             U = iso_vec_to_operator(Z[U_idx])
#         end
        
#         a_idx = a_indices[k]
#         a_k = Z[a_idx]
#         He_vec = H_err(a_k)
#         ∂He_vec = ∂H_err(a_k)
        
#         Hₖ = sum(a_k[l] / a_bound * H for (l, H) in enumerate(H_drives))
        
#         n = length(U_idx) > 0 ? Int(sqrt(length(U_idx) ÷ 2)) : D
#         n_components = length(He_vec)
        
#         hess_terms = [zeros(ComplexF64, D, D) for _ in 1:n_components]
        
#         # Determine variable types
#         is_a1 = var_idx1 in a_idx
#         is_a2 = var_idx2 in a_idx
#         is_U1 = var_idx1 in U_idx
#         is_U2 = var_idx2 in U_idx
        
#         # Early return if neither variable is in this timestep
#         if !is_a1 && !is_a2 && !is_U1 && !is_U2
#             return hess_terms
#         end
        
#         for (comp_idx, (He, ∂He_a)) in enumerate(zip(He_vec, ∂He_vec))
#             # Case 1: Both are control variables a
#             if is_a1 && is_a2
#                 i1 = findfirst(==(var_idx1), a_idx)
#                 i2 = findfirst(==(var_idx2), a_idx)
                
#                 if i1 !== nothing && i2 !== nothing && i1 <= n_drives && i2 <= n_drives
#                     # Compute ∂²Eₖ/∂a_i1∂a_i2
#                     ∂²Eₖ = zeros(ComplexF64, D, D)
                    
#                     # Derivatives of H_k
#                     ∂H1 = H_drives[i1] / a_bound
#                     ∂H2 = H_drives[i2] / a_bound
                    
#                     # Track nested commutators and their derivatives
#                     adjⁿH_E = He
                    
#                     # Handle scalar vs matrix for ∂He_a
#                     if isa(∂He_a[i1], Number)
#                         ∂adjⁿH_E_∂a1 = zeros(ComplexF64, D, D) * ∂He_a[i1]
#                     else
#                         ∂adjⁿH_E_∂a1 = ∂He_a[i1]
#                     end
                    
#                     if isa(∂He_a[i2], Number)
#                         ∂adjⁿH_E_∂a2 = zeros(ComplexF64, D, D) * ∂He_a[i2]
#                     else
#                         ∂adjⁿH_E_∂a2 = ∂He_a[i2]
#                     end
                    
#                     for n_order in 2:order
#                         coef_n = ComplexF64(im^(n_order-1) * a_bound^(n_order-1) * Δt[k]^n_order / factorial_cache[n_order])
                        
#                         # Second derivative of nested commutator
#                         # ∂²/∂a1∂a2 [Hₖ, f] = [∂²Hₖ/∂a1∂a2, f] + [∂Hₖ/∂a1, ∂f/∂a2] + [∂Hₖ/∂a2, ∂f/∂a1] + [Hₖ, ∂²f/∂a1∂a2]
#                         # Since ∂²Hₖ/∂a1∂a2 = 0 and we assume ∂²He/∂a1∂a2 = 0:
#                         ∂²adjⁿ = commutator(∂H1, ∂adjⁿH_E_∂a2) + commutator(∂H2, ∂adjⁿH_E_∂a1)
                        
#                         ∂²Eₖ += coef_n * ∂²adjⁿ
                        
#                         # Update derivatives for next iteration
#                         new_∂adjⁿH_E_∂a1 = commutator(∂H1, adjⁿH_E) + commutator(Hₖ, ∂adjⁿH_E_∂a1)
#                         new_∂adjⁿH_E_∂a2 = commutator(∂H2, adjⁿH_E) + commutator(Hₖ, ∂adjⁿH_E_∂a2)
#                         adjⁿH_E = commutator(Hₖ, adjⁿH_E)
                        
#                         ∂adjⁿH_E_∂a1 = new_∂adjⁿH_E_∂a1
#                         ∂adjⁿH_E_∂a2 = new_∂adjⁿH_E_∂a2
#                     end
                    
#                     hess_terms[comp_idx] = U' * ∂²Eₖ * U
#                 end
                
#             # Case 2: Both are U variables
#             elseif is_U1 && is_U2 && k > 1
#                 idx1 = var_idx1 - U_idx[1] + 1
#                 idx2 = var_idx2 - U_idx[1] + 1
                
#                 i1 = (idx1 - 1) % n + 1
#                 j1 = (idx1 - 1) ÷ (2 * n) + 1
#                 is_real1 = idx1 <= n * n
                
#                 i2 = (idx2 - 1) % n + 1
#                 j2 = (idx2 - 1) ÷ (2 * n) + 1
#                 is_real2 = idx2 <= n * n
                
#                 d1 = zeros(ComplexF64, n, n)
#                 d1[i1, j1] = is_real1 ? 1.0 : 1.0im
#                 d2 = zeros(ComplexF64, n, n)
#                 d2[i2, j2] = is_real2 ? 1.0 : 1.0im
                
#                 # Compute Eₖ for this component
#                 Eₖ_n = He * Δt[k]
#                 adjⁿH_E = He
#                 for n_order in 2:order
#                     coef_n = ComplexF64(im^(n_order-1) * a_bound^(n_order-1) * Δt[k]^n_order / factorial_cache[n_order])
#                     adjⁿH_E = commutator(Hₖ, adjⁿH_E)
#                     Eₖ_n += coef_n * adjⁿH_E
#                 end
                
#                 hess_terms[comp_idx] = d1' * Eₖ_n * d2 + d2' * Eₖ_n * d1
                
#             # Case 3: Mixed U and a variables
#             elseif (is_U1 && is_a2) || (is_a1 && is_U2)
#                 if is_U1 && is_a2
#                     U_var = var_idx1
#                     a_var = var_idx2
#                     a_i = findfirst(==(a_var), a_idx)
#                 else
#                     U_var = var_idx2
#                     a_var = var_idx1
#                     a_i = findfirst(==(a_var), a_idx)
#                 end
                
#                 if k > 1 && a_i !== nothing && a_i <= n_drives
#                     idx = U_var - U_idx[1] + 1
#                     i = (idx - 1) % n + 1
#                     j = (idx - 1) ÷ (2 * n) + 1
#                     is_real = idx <= n * n
                    
#                     d = zeros(ComplexF64, n, n)
#                     d[i, j] = is_real ? 1.0 : 1.0im
                    
#                     # Compute ∂Eₖ/∂a_i - handle scalar vs matrix case
#                     if isa(∂He_a[a_i], Number)
#                         ∂Eₖ_∂a = zeros(ComplexF64, D, D) * ∂He_a[a_i] * Δt[k]
#                         ∂adjⁿH_E = zeros(ComplexF64, D, D) * ∂He_a[a_i]
#                     else
#                         ∂Eₖ_∂a = ∂He_a[a_i] * Δt[k]
#                         ∂adjⁿH_E = ∂He_a[a_i]
#                     end
                    
#                     adjⁿH_E = He
#                     ∂Hₖ = H_drives[a_i] / a_bound
                    
#                     for n_order in 2:order
#                         coef_n = ComplexF64(im^(n_order-1) * a_bound^(n_order-1) * Δt[k]^n_order / factorial_cache[n_order])
#                         new_∂adjⁿH_E = commutator(∂Hₖ, adjⁿH_E) + commutator(Hₖ, ∂adjⁿH_E)
#                         adjⁿH_E = commutator(Hₖ, adjⁿH_E)
#                         ∂Eₖ_∂a += coef_n * new_∂adjⁿH_E
#                         ∂adjⁿH_E = new_∂adjⁿH_E
#                     end
                    
#                     hess_terms[comp_idx] = U' * ∂Eₖ_∂a * d + d' * ∂Eₖ_∂a * U
#                 end
#             end
#         end
        
#         return hess_terms
#     end

#     function ∂²ℓ_structure()
#         Z_dim = traj.dim * traj.T + traj.global_dim
        
#         # Get all relevant indices (only U_{1:T-1} and a_{1:T-1})
#         relevant_indices = Int[]
#         for k in 1:T-1
#             if k > 1  # U derivatives only for k > 1
#                 append!(relevant_indices, Ũ⃗_indices[k-1])
#             end
#             append!(relevant_indices, a_indices[k][1:min(n_drives, length(a_indices[k]))])
#         end
        
#         # Create pairs for upper triangular part only (Hessian is symmetric)
#         structure_pairs = Tuple{Int,Int}[]
#         for i in relevant_indices
#             for j in relevant_indices
#                 if j >= i
#                     push!(structure_pairs, (i, j))
#                 end
#             end
#         end
        
#         return structure_pairs
#     end

#     function ∂²ℓ(Z::AbstractVector{<:Real})
#         normalization = real(T^2 * H_scale^2) * D
#         structure_pairs = ∂²ℓ_structure()
        
#         ∂²ℓ_values = zeros(length(structure_pairs))
#         n_components = length(H_err(Z[a_indices[1]]))
        
#         for (pair_idx, (i, j)) in enumerate(structure_pairs)
#             hess_ij = 0.0
            
#             for comp_idx in 1:n_components
#                 # First, compute H_ti for this component
#                 H_ti = zeros(ComplexF64, D, D)
#                 for k in 1:T-1
#                     H_ti += toggle(Z, k)[comp_idx]
#                 end
                
#                 # Compute first derivatives
#                 ∂H_ti_∂i = zeros(ComplexF64, D, D)
#                 ∂H_ti_∂j = zeros(ComplexF64, D, D)
                
#                 for k in 1:T-1
#                     grad_k = ∇toggle(Z, k)[comp_idx]  # This is a dictionary
#                     if haskey(grad_k, i)
#                         ∂H_ti_∂i += grad_k[i]
#                     end
#                     if haskey(grad_k, j)
#                         ∂H_ti_∂j += grad_k[j]
#                     end
#                 end
                
#                 # Product term: 2 Re(tr(∂H_ti/∂i)' * ∂H_ti/∂j))
#                 hess_ij += 2 * real(tr(∂H_ti_∂i' * ∂H_ti_∂j))
                
#                 # Second derivative term: 2 Re(tr(H_ti' * ∂²H_ti/∂i∂j))
#                 if i == j
#                     # Diagonal terms need special care for the second derivative
#                     for k in 1:T-1
#                         hess_k_terms = ∂²toggle(Z, k, i, j)
#                         if !iszero(hess_k_terms[comp_idx])
#                             hess_ij += 2 * real(tr(H_ti' * hess_k_terms[comp_idx]))
#                         end
#                     end
#                 else
#                     # Off-diagonal terms
#                     for k in 1:T-1
#                         hess_k_terms = ∂²toggle(Z, k, i, j)
#                         if !iszero(hess_k_terms[comp_idx])
#                             hess_ij += 2 * real(tr(H_ti' * hess_k_terms[comp_idx]))
#                         end
#                     end
#                 end
#             end
            
#             ∂²ℓ_values[pair_idx] = Q_t * hess_ij / normalization
#         end
        
#         return ∂²ℓ_values
#     end

#     return Objective(ℓ, ∇ℓ, ∂²ℓ, ∂²ℓ_structure)
# end


# function TogglingObjectiveTest(
#     H_err::Function,
#     ∂H_err::Function, 
#     traj::NamedTrajectory;
#     Q_t::Float64=1.0,
#     order::Int=4,
#     a_bound::Float64=1.0,
#     H_drives::Vector{Matrix{ComplexF64}}=Matrix{ComplexF64}[]
# )   
#     T = traj.T
#     Δt = get_timesteps(traj)
#     a_indices  = [collect(slice(k, traj.components.a, traj.dim)) for k in 1:T]
#     Ũ⃗_indices  = [collect(slice(k, traj.components.Ũ⃗, traj.dim)) for k in 1:T]
#     a_ref = ones(length(traj.components.a))
#     H_scale = norm(H_err(a_ref), 2)
#     D = size(iso_vec_to_operator(traj.components.Ũ⃗[:,end]))[1]
#     commutator(A::AbstractMatrix{<:Number}, B::AbstractMatrix{<:Number}) = A*B - B*A

#     @views function toggle(Z::AbstractVector, k::Int)
#         # For k=1, U_1 = I (identity), for k>1, use U_{k-1}
#         if k == 1
#             U = I(D)
#         else
#             U = iso_vec_to_operator(Z[Ũ⃗_indices[k-1]])
#         end
        
#         a_k = Z[a_indices[k]]
#         He_vec = H_err(a_k)
        
#         # Build Hₖ for this timestep
#         Hₖ = sum(a_k[l] / a_bound * H for (l, H) in enumerate(H_drives))
        
#         # Compute perturbative expansion for each error Hamiltonian component
#         toggle_terms = []
#         for He in He_vec
#             adjⁿH_E = He
#             Eₖ_n = He * Δt[k]
            
#             # Get the different orders of the Hadamard lemma
#             for n in 2:order
#                 coef_n = ComplexF64(im^(n-1) * a_bound^(n-1) * Δt[k]^n / factorial(big(n)))
#                 adjⁿH_E = commutator(Hₖ, adjⁿH_E)
#                 Eₖ_n += coef_n * adjⁿH_E
#             end
            
#             # Apply unitary transformation
#             push!(toggle_terms, U' * Eₖ_n * U)
#         end
        
#         return toggle_terms
#     end

#     function ℓ(Z::AbstractVector{<:Real})
#         normalization = real(T^2 * H_scale^2) * D
        
#         # Sum toggles over time (only to T-1 since last U doesn't contribute)
#         H_ti_components = []
#         n_components = length(H_err(Z[a_indices[1]]))
        
#         for j in 1:n_components
#             H_ti = zeros(ComplexF64, D, D)
#             for k in 1:T-1  # Only go to T-1
#                 toggle_k = toggle(Z, k)[j]
#                 H_ti += toggle_k
#             end
#             push!(H_ti_components, H_ti)
#         end
        
#         # Objective: sum_j ||H_ti_j||_F^2 / normalization
#         FO_obj = sum(real(tr(H_ti' * H_ti)) / normalization for H_ti in H_ti_components)
        
#         return Q_t * FO_obj
#     end 

#     function ∇toggle(Z::AbstractVector, k::Int)
#         # For k=1, U_1 = I, for k>1, use U_{k-1}
#         if k == 1
#             U = I(D)
#             U_idx = Int[]  # No U indices for identity
#         else
#             U_idx = Ũ⃗_indices[k-1]
#             U = iso_vec_to_operator(Z[U_idx])
#         end
        
#         a_idx = a_indices[k]
#         a_k = Z[a_idx]
#         He_vec = H_err(a_k)
#         ∂He_vec = ∂H_err(a_k)
        
#         Hₖ = sum(a_k[l] / a_bound * H for (l, H) in enumerate(H_drives))
        
#         n = Int(sqrt(length(U_idx) ÷ 2))
#         grads = [spzeros(ComplexF64, length(Z)) for _ in He_vec]
        
#         for (g, He, ∂He_a) in zip(grads, He_vec, ∂He_vec)
#             # Compute Eₖ_n with perturbative expansion
#             adjⁿH_E = He
#             Eₖ_n = He * Δt[k]
            
#             # Store derivatives for each order
#             ∂Eₖ_n_∂a = [∂Ha * Δt[k] for ∂Ha in ∂He_a]
            
#             for n_order in 2:order
#                 coef_n = ComplexF64(im^(n_order-1) * a_bound^(n_order-1) * Δt[k]^n_order / factorial(big(n_order)))
                
#                 # Update the nested commutator
#                 adjⁿH_E = commutator(Hₖ, adjⁿH_E)
#                 Eₖ_n += coef_n * adjⁿH_E
                
#                 # Update derivatives with respect to a
#                 for (l, ∂Ha) in enumerate(∂He_a)
#                     # Derivative of Hₖ w.r.t. a[l]
#                     ∂Hₖ_∂al = H_drives[l] ./ a_bound
                    
#                     # Apply product rule for nested commutators
#                     ∂adjⁿH_E_∂al = commutator(∂Hₖ_∂al, adjⁿH_E)
#                     # Note: Full derivative requires chain rule through all nested commutators
#                     # This is simplified here - full implementation would track all terms
                    
#                     ∂Eₖ_n_∂a[l] += coef_n * ∂adjⁿH_E_∂al
#                 end
#             end
            
#             ### a derivatives ###
#             for (i, l) in enumerate(a_idx)
#                 g[l] = U' * ∂Eₖ_n_∂a[i] * U
#             end
            
#             ### U derivatives (only if k > 1) ###
#             if k > 1
#                 for U_i in U_idx
#                     idx = U_i - U_idx[1] + 1
#                     i = (idx % n) == 0 ? n : idx % n
#                     j = (idx - i) ÷ (2 * n) + 1
#                     re = (idx % (2*n) == 0 ? 2*n : idx%(2*n)) ≤ n
#                     d = spzeros(ComplexF64, (n,n))
#                     d[i,j] = re ? 1 : 1im 
#                     g[U_i] = U' * Eₖ_n * d + d' * Eₖ_n * U
#                 end
#             end
#         end
#         return grads 
#     end

#     function ∇ℓ(Z::AbstractVector{<:Real})
#         normalization = real(T^2 * H_scale^2) * D
        
#         full_grad = spzeros(length(Z))
#         n_components = length(H_err(Z[a_indices[1]]))
        
#         for j in 1:n_components
#             # Accumulate toggle sum
#             H_ti = zeros(ComplexF64, D, D)
#             for k in 1:T-1
#                 H_ti += toggle(Z, k)[j]
#             end
            
#             # Compute gradient contribution from each timestep
#             for k in 1:T-1
#                 grad_k = ∇toggle(Z, k)[j]
                
#                 # Chain rule: ∂||H_ti||²/∂Z = 2 Re(tr(H_ti' * ∂H_ti/∂Z))
#                 for (idx, v) in zip(findnz(grad_k)...)
#                     full_grad[idx] += 2 * real(tr(H_ti' * v))
#                 end
#             end
#         end
        
#         return Q_t * full_grad / normalization
#     end 

#     function ∂²ℓ_structure()
#         Z_dim = traj.dim * traj.T + traj.global_dim
#         structure = spzeros(Z_dim, Z_dim)
        
#         # Get all relevant indices (only U_{1:T-1} and a_{1:T-1})
#         relevant_Ũ⃗_indices = vcat(Ũ⃗_indices[1:T-1]...)
#         relevant_a_indices = vcat(a_indices[1:T-1]...)
#         all_indices = vcat(relevant_Ũ⃗_indices, relevant_a_indices)
        
#         for i in all_indices
#             for j in all_indices
#                 structure[i, j] = 1.0
#             end
#         end
        
#         structure_pairs = collect(zip(findnz(structure)[1:2]...))
#         return structure_pairs
#     end

#     function ∂²ℓ(Z::AbstractVector{<:Real})
#         normalization = real(T^2 * H_scale^2) * D
#         structure_pairs = ∂²ℓ_structure()
        
#         Z_dim = length(Z)
#         H_full = spzeros(Z_dim, Z_dim)
        
#         n_components = length(H_err(Z[a_indices[1]]))
        
#         for j in 1:n_components
#             # Get H_ti sum
#             H_ti = zeros(ComplexF64, D, D)
#             for k in 1:T-1
#                 H_ti += toggle(Z, k)[j]
#             end
            
#             # Get gradients for this component from timesteps 1:T-1
#             grad_components = [∇toggle(Z, k)[j] for k in 1:T-1]
            
#             # Hessian contribution
#             for (i, j_idx) in structure_pairs
#                 grad_i = sum(g[i] for g in grad_components if i in findnz(g)[1])
#                 grad_j = sum(g[j_idx] for g in grad_components if j_idx in findnz(g)[1])
                
#                 H_full[i, j_idx] += 2 * real(tr(grad_i' * grad_j))
#             end
#         end
        
#         ∂²ℓ_values = [(Q_t / normalization) * H_full[i, j] for (i, j) in structure_pairs]
        
#         return ∂²ℓ_values
#     end

#     return Objective(ℓ, ∇ℓ, ∂²ℓ, ∂²ℓ_structure)
# end

# function TogglingObjectiveTest(
#     H_err::Function,
#     ∂H_err::Function, 
#     traj::NamedTrajectory;
#     Q_t::Float64=1.0,
#     order::Int=4,
# )   
#     T = traj.T
#     Δt = get_timesteps(traj)
#     a_indices  = [collect(slice(k, traj.components.a, traj.dim)) for k in 1:traj.T-1]
#     Ũ⃗_indices  = [collect(slice(k, traj.components.Ũ⃗, traj.dim)) for k in 1:traj.T-1]
#     a_ref = ones(length(traj.components.a))
#     H_scale = norm(H_err(a_ref), 2)
#     D = size(iso_vec_to_operator(traj.components.Ũ⃗[:,end]))[1]

#     @views function toggle(Z::AbstractVector, a_idx::AbstractVector{<:Int}, U_idx::AbstractVector{<:Int}, Eₖ_n::Matrix{ComplexF64})
#         a  = Z[a_idx]
#         U  = iso_vec_to_operator(Z[U_idx])
#         He_vec = H_err(a)
#         return [U' * He * U for He in He_vec]
#     end

#     function ℓ(Z::AbstractVector{<:Real})
#         normalization = real(traj.T^2 * H_scale^2) * D
        
#         # Compute sum of toggles over time for each Hamiltonian component
#         terms = []
#         for j in 1:length(toggle(Z, a_indices[1], Ũ⃗_indices[1]))
#             Hₖ = sum(traj.a[l, k] / a_bound * H for (l, H) in enumerate(H_drives))
#             adjⁿH_E = H_err(a)[j]
#             Eₖ_n = H_err(a)[j] * Δt
#             for n in 2:order
#                 coef_n = ComplexF64(im^(n-1) * a_bound^(n-1) * Δt^n / factorial(big(n)))
#                 adjⁿH_E = commutator(Hₖ, adjⁿH_E)
#                 # Eₖ_n = push!(Eₖ_n, coef_n * adjⁿH_E)
#                 Eₖ_n += coef_n * adjⁿH_E
#             end
#             term = [toggle(Z, a_idx, U_idx, Eₖ_n)[j] for (a_idx, U_idx) in zip(a_indices, Ũ⃗_indices)]
#             sum_term = sum(term)
#             push!(terms, sum_term)
#         end

#         # Objective: sum_j ||terms[j]||_F^2 / normalization
#         # where ||A||_F^2 = tr(A' * A) = sum of squared absolute values
#         FO_obj = sum(real(norm(tr(term' * term), 2)) / normalization for term in terms)
        
#         return Q_t * FO_obj
#     end 

#     function ∇toggle(Z::AbstractVector, a_idx::AbstractVector{<:Int}, U_idx::AbstractVector{<:Int})
#         a  = Z[a_idx]
#         U  = iso_vec_to_operator(Z[U_idx])
#         He_vec = H_err(a)
#         ∂He_vec = ∂H_err(a)
#         n = Int(sqrt(length(U_idx) ÷ 2))
#         grads = [spzeros(Matrix{ComplexF64}, length(Z)) for He in He_vec]

#         for (g, He, ∂He) in zip(grads, He_vec, ∂He_vec)
#             ### a derivatives ### 
#             for (a_i, ∂Ha) in zip(a_idx, ∂He)
#                 g[a_i] = U' * ∂Ha * U
#             end
#             ### U derivatives ### 
#             for U_i in U_idx
#                 idx = U_i - U_idx[1] + 1
#                 i = (idx % n) == 0 ? n : idx % n
#                 j = (idx - i) ÷ (2 * n) + 1
#                 re = (idx % (2*n) == 0 ? 2*n : idx%(2*n)) ≤ n
#                 d = spzeros(ComplexF64, (n,n))
#                 d[i,j] = re ? 1 : 1im 
#                 g[U_i] = U' * He * d + d' * He * U
#             end 
#         end
#         return grads 
#     end


#     function ∇ℓ(Z::AbstractVector{<:Real})
#         normalization = real(traj.T^2 * H_scale^2) * D
        
#         # Compute sum of toggles and their gradients over time for each component
#         terms = []
#         grads = []
#         for j in 1:length(toggle(Z, a_indices[1], Ũ⃗_indices[1]))
#             sum_term = sum(toggle(Z, a_idx, U_idx)[j] for (a_idx, U_idx) in zip(a_indices, Ũ⃗_indices))
#             sum_grad = spzeros(Matrix{ComplexF64}, length(Z))
            
#             for (a_idx, U_idx) in zip(a_indices, Ũ⃗_indices) 
#                 g = ∇toggle(Z, a_idx, U_idx)[j] 
#                 for (idx, val) in zip(g.nzind, g.nzval)
#                     sum_grad[idx] = val
#                 end
#             end
#             push!(terms, sum_term)
#             push!(grads, sum_grad)
#         end

#         # Gradient of ||M||_F^2 = tr(M' * M) w.r.t. M is 2*M
#         # For complex M: d/dZ tr(M'*M) = d/dZ sum_ij |M_ij|^2
#         # Chain rule: ∂ℓ/∂Z = sum_j (2 * Re(terms[j]) .* Re(∇terms[j]) + 2 * Im(terms[j]) .* Im(∇terms[j]))
#         full_grad = spzeros(size(Z))

#         for (t, g) in zip(terms, grads)
#             for (idx, v) in zip(g.nzind, g.nzval)
#                 full_grad[idx] += sum(2 * real(t) .* real(v) + 2 * imag(t) .* imag(v))
#             end 
#         end
#         return Q_t * full_grad / normalization
#     end 

#     function ∂²ℓ_structure()
#         Z_dim = traj.dim * traj.T + traj.global_dim
#         structure = spzeros(Z_dim, Z_dim)
        
#         # Hessian couples all U variables and a variables
#         all_Ũ⃗_indices = vcat(Ũ⃗_indices...)
#         all_a_indices = vcat(a_indices...)
#         all_indices = vcat(all_Ũ⃗_indices, all_a_indices)
        
#         for i in all_indices
#             for j in all_indices
#                 structure[i, j] = 1.0
#             end
#         end
        
#         structure_pairs = collect(zip(findnz(structure)[1:2]...))
#         return structure_pairs
#     end

#     function ∂²ℓ(Z::AbstractVector{<:Real})
#         normalization = real(traj.T^2 * H_scale^2) * D
#         structure_pairs = ∂²ℓ_structure()
        
#         Z_dim = length(Z)
#         H_full = spzeros(Z_dim, Z_dim)
        
#         # Compute terms and first/second derivatives
#         n_components = length(toggle(Z, a_indices[1], Ũ⃗_indices[1]))
        
#         for j in 1:n_components
#             # Get sum term and its first derivative
#             sum_term = sum(toggle(Z, a_idx, U_idx)[j] for (a_idx, U_idx) in zip(a_indices, Ũ⃗_indices))
            
#             # Get gradients for this component from all time steps
#             grad_components = [∇toggle(Z, a_idx, U_idx)[j] for (a_idx, U_idx) in zip(a_indices, Ũ⃗_indices)]
            
#             # Hessian contribution: ∂²/∂Z_i∂Z_k tr(M'*M) = 2*Re(tr(∂M'/∂Z_i * ∂M/∂Z_k)) + 2*Re(tr(M' * ∂²M/∂Z_i∂Z_k))
#             # For our case, second derivatives of toggle are zero (linear in each variable)
#             # So: H[i,k] = 2*Re(tr(∂term'/∂Z_i * ∂term/∂Z_k))
            
#             for (i, j_val) in structure_pairs
#                 grad_i = sum(g[i] for g in grad_components if i in g.nzind)
#                 grad_j = sum(g[j_val] for g in grad_components if j_val in g.nzind)
                
#                 H_full[i, j_val] += 2 * real(tr(grad_i' * grad_j))
#             end
#         end
        
#         ∂²ℓ_values = [(Q_t / normalization) * H_full[i, j] for (i, j) in structure_pairs]
        
#         return ∂²ℓ_values
#     end

#     return Objective(ℓ, ∇ℓ, ∂²ℓ, ∂²ℓ_structure)
# end

end