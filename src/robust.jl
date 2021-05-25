using LinearAlgebra
using Roots
using Intervals


"""
    GammaEqn

Save data for solving for gamma
"""
struct GammaEqn{T}
    ϵ::T 
    λ::Array{T, 1}
    m::Int
end
GammaEqn(e::Int, l::AbstractArray, m::Int) = GammaEqn(Float64(e), l, m)
function GammaEqn(ϵ::Union{Real, Int}, λ)
    if !all( λ .>= 0 )
        throw(ArgumentError("Eigenvalues must be nonnegative"))
    else
        return GammaEqn(ϵ, λ, length(λ))
    end
end
function (eq::GammaEqn)(expr, γ)
    ϵ = eq.ϵ 
    λ = eq.λ
    m = eq.m
    z = γ[1]
    expr[1] = (ϵ^2 .- 0.5*sum(λ)) .* z - m + 0.5 * sum(sqrt.(z.^2 .* λ.^2 .+ 4 .* λ .* z))
end

# Computes the Wasserstein-DR precision matrix of X wrt the Stein loss logdet(C*P) - <C*P> - p
"""
    LinearShrinkage(target, shrinkage; corrected=false)

Linear shrinkage estimator described by equation
``(1 - \\lambda) S + \\lambda F`` where ``S`` is standard covariance matrix,
``F`` is shrinkage target described by argument `target` and ``\\lambda`` is a
shrinkage parameter, either given explicitly in `shrinkage` or automatically
determined according to one of the supported methods.

The corrected estimator is used if `corrected` is true.
"""
struct WassersteinShrinkage <:CovarianceEstimator
    eps::Float64 
end


function cov(sre::WassersteinShrinkage, X::AbstractMatrix{<:Real};
             dims::Int=1, mean=nothing)
    inv_cov = prec(sre, X)
    return inv(inv_cov)
end

function prec(sre::WassersteinShrinkage, X::AbstractMatrix{<:Real};
        dims::Int=1, mean=nothing)
    return wasserstein_pm(X, sre.eps)
end

"""
    linear_shrinkage(::DiagonalUnitVariance, Xc, S, λ, n, p, corrected)

Compute the shrinkage estimator where the target is a `DiagonalUnitVariance`.
"""
function wasserstein_pm(X, ϵ)
    C = Symmetric(cov(X))
    λ, v = eigen(C)
    m = length(λ)
    λ = max.(1e-10, λ)
    eq = GammaEqn(ϵ, λ)
    ϕ(z) = (ϵ^2 .- 0.5*sum(λ)) .* z - m + 0.5 * sum(sqrt.(z.^2 .* λ.^2 .+ 4 .* λ .* z))
    bounds = γ_bounds(eq)
    ϕ.(bounds)
    γ = find_zero(ϕ, bounds, Roots.Bisection())
    u = λ*γ
    # x = γ .* (1 .- 2. ./ (1 .+ sqrt.(1 .+ 4 ./ u)))
    x = γ.*(1 .- 0.5 *(sqrt.(max.(0., u.^2 .+ 4*u)) .- u)) 
    for i = 1:length(λ)
        # NOTE: Numerically stable alternative for large values
        if λ[i]*γ > 1e8
            x[i] = γ .* (1 .- 2. ./ (1 .+ sqrt.(1 .+ 4 ./ λ[i]*γ)))
            # x[i] = γ*(1 - 0.5 *(sqrt((γ*λ[i])^2 + 4*λ[i]*γ)) - λ[i]*γ) 
        end
    end
    est = v*diagm(x)*v'
    return Symmetric(est)
end
function γ_bounds(eq::GammaEqn)
    m = eq.m
    L = eq.λ[end]
    ϵ = eq.ϵ
    # NOTE: Lower bound from original paper has an error?
    lower = 1e-15 
    upper = min(m/ϵ^2, 1/(ϵ + 1e-15)*sqrt(sum(eq.λ.^(-1))))
    return lower, upper
end




