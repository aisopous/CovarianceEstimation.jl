module CovarianceEstimation

using Statistics
using StatsBase
using LinearAlgebra
import StatsBase: cov

export cov
export prec
export CovarianceEstimator, SimpleCovariance,
    LinearShrinkage,
    # Targets for linear shrinkage
    DiagonalUnitVariance, DiagonalCommonVariance, DiagonalUnequalVariance,
    CommonCovariance, PerfectPositiveCorrelation, ConstantCorrelation,
    # Eigendecomposition-based methods
    AnalyticalNonlinearShrinkage,
    WassersteinShrinkage
    
include("utils.jl")
include("linearshrinkage.jl")
include("nonlinearshrinkage.jl")
include("robust.jl")

end # module
