@testset "Wshrink: Basic properties" begin

    GammaEqn = CovarianceEstimation.GammaEqn

    # At γ -> ∞ the expression in GammaEq approaches ϵ² 
    p = ifelse(endswith(pwd(), "CovarianceEstimation"), "test", "")

    test_mat1 = readdlm(joinpath(p, "test_matrices/20x100.csv"))
    ref_cov1  = readdlm(joinpath(p, "test_matrices/20x100_corpcor.csv"))
    test_mat2 = readdlm(joinpath(p, "test_matrices/100x20.csv"))
    ref_cov2  = readdlm(joinpath(p, "test_matrices/100x20_corpcor.csv"))
    test_mat3 = readdlm(joinpath(p, "test_matrices/50x50.csv"))
    ref_cov3  = readdlm(joinpath(p, "test_matrices/50x50_corpcor.csv"))
    
    test_matrices = [test_mat1, test_mat2, test_mat3]
    ref_covs = [ref_cov1, ref_cov2, ref_cov3]

    estimators = WassersteinShrinkage.([10., 5., 1., 0.1, 0.01, 0.001, 0.0001, 1e-5])
    for (X, C) ∈ zip(test_matrices, ref_covs)
        estimates = cov.(estimators, [X]);
        norm.(estimates .- [cov(X)])
        norm.(estimates .- [C])
        λ = max.(10e-10, eigen(cov(X)).values)
        @test sqrt(GammaEqn(0.5, λ)([0.], [1e10])/1e10) ≈ 0.5
        @test sqrt(GammaEqn(0.1, λ)([0.], [1e15])/1e15) ≈ 0.1
        @test sqrt(GammaEqn(0.01, λ)([0.], [1e20])/1e20) ≈ 0.01
        n = length(λ)
        @test GammaEqn(1., λ, n)([0.], [0]) == -n
        @test GammaEqn(1., λ, n)([0.], [0]) == -n
        @test all(inv.(prec.(estimators, [X])) .== estimates)
    end
end
