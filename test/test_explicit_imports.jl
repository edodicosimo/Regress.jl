@testitem "ExplicitImports" tags = [:explicit_imports] begin
    using ExplicitImports
    using Regress

    # Deliberate reliance on non-public names:
    # - PValue, NoQuote (StatsBase): coefficient-table formatting, same usage as GLM.jl
    # - Schema, FullRank, collect_matrix_terms, missing_omit, needs_schema,
    #   termvars (StatsModels): the standard downstream apply_schema integration
    # - istable, getcolumn, ColumnTable (Tables): documented interface, not
    #   declared `public` by Tables.jl
    # - Clustering, bread, leverage, mask, momentresiduals (CovarianceMatrices,
    #   public): the protocol a model implements to be usable with its variance
    #   estimators
    # - numobs, residual_adjustment, _kernelweights (CovarianceMatrices, internal):
    #   the observation count, the HC2–HC5 residual adjustments, and the
    #   bandwidth-selection weights of the design matrix, which this package's own
    #   moment-matrix paths need
    # - alignment, print_matrix_row (Base): custom coeftable show
    # - QRCompactWY (LinearAlgebra): concrete type of the stored QR factorization
    # - update_weights! (FixedEffects): reweights the fixed-effect solver in place
    #   across probit IRLS iterations
    test_explicit_imports(Regress;
        ignore = (:PValue, :NoQuote,
            :Schema, :FullRank, :collect_matrix_terms, :missing_omit,
            :needs_schema, :termvars,
            :istable, :getcolumn, :ColumnTable,
            :Clustering, :bread, :leverage, :mask, :momentresiduals,
            :numobs, :residual_adjustment, :_kernelweights,
            :alignment, :print_matrix_row,
            :QRCompactWY, :update_weights!))
end
