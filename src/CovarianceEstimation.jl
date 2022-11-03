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
    WassersteinShrinkage,
    AnalyticalNonlinearShrinkage,
    # Biweight midcovariance
    BiweightMidcovariance

include("utils.jl")
include("biweight.jl")
include("linearshrinkage.jl")
include("nonlinearshrinkage.jl")
include("wasserstein.jl")

end # module
