"""
Options for all functions of package SparseTools.
"""
mutable struct Options{T<:AbstractFloat}
    # options for all functions (spqr_pinv uses all options)
    get_details::Int    # 0: basic statistics (default)
        # 1: detailed statistics:  basic stats, plus input options, time taken by
        #    various phases, statistics from spqr and spqr_rank subfunctions called,
        #    and other details.  Normally of interest only to the developers.
        # 2: basic statistics and a few additional statistics.  Used internally
        #    by some routines to pass needed information.
    repeatable::Bool    # by default, results are repeatable

    # options for spqr_basic, spqr_cod, and spqr_ssi
    tol::T        # a negative number means the default tolerance should be computed
    tol_norm_type::Int  # 1: use norm(A, 1) to compute the default tol
                        # 2: use normest(A, 0.01) - This is the default

    nsvals_large::Int   # default number of large singular values to estimate
    
    # options for spqr_basic, spqr_null, spqr_cod, and spqr
    nvals_small::Int    # default number of small singular values to estimate
    implicit_null_space_basis::Bool # default true

    start_with_A_transpose::Bool    # default false

    # options for spqr_ssi (called by spqr_basic, spqr_null, spqr_cod and spqr_pinv)
    ssi_tol::T    # default like tol
    ssi_min_block::Int  # 3
    ssi_max_block::Int  # 10
    ssi_min_iters::Int  # 3
    ssi_max_iters::Int  # 100
    ssi_nblock_increment::Int   # 5
    ssi_convergence_factor::T # 0.1
    
    # options for spqr_ssp (called by spqr_basic, spqr_null, spqr_cod and spqr_pinv)
    k::Int              # number of singular values to compute
    ssp_min_iters::Int  # 4
    ssp_max_iters::Int  # 10
    ssp_convergence_factor::T # 0.1

    # no option 
    normest_A::T
end

"""
    `Options([A, [k]])`
    
Set default options for all sparse tools.
If `A` is given, calculate absolute tolerance from norm of matrix `A`.
If also `k`is given, use as default valus of option `k` for ssi.
"""
Options{T}() where T = Options{T}(0, true,
                          -1.0, 2,
                          1, 1, true, false,
                          -1.0, 3, 10, 3, 100, 5, 0.1,
                          1, 4, 3, 0.1,
                         -1.0)

function Options(A::AbstractMatrix{T}, k::Int=1) where T
    opts = Options{real(T)}()
    opts.k = k
    opts.normest_A = opts.tol_norm_type == 2 ? norm(A, 1) : normest(A, 0.01) ## TODO
    opts.tol = opts.normest_A * max(size(A)...) * eps(real(T))
    opts.ssi_tol = opts.tol
    opts
end

mutable struct Statistics{T<:AbstractFloat}
    flag::Int   # -1 mean 'not yet computed'
    rank::Int
    tol::T    # to be copied from opts
    normest_A::T  # to be copied from options
    tol_alt::T
    est_sval_upper_bounds::Vector{T}
    est_sval_lower_bounds::Vector{T}

    est_svals::Vector{T}
    est_svals_of_R::Vector{T}
    est_error_bounds::Vector{T}
    iters::Int
    opts_used::Options
    time::Int64
    time_initialize::Int64
    time_iters::Int64
    time_est_error_bounds::Int64
    time_svd::Int64
    norm_R_times_N::T
    norm_R_transpose_times_NT::T
    nsvals_large_found::Int
    final_blocksize::Int
    ssi_min_block_used::Int
    ssi_max_block_used::Int
end

function Statistics(::Type{T}; tol=0.0, normest_A=1.0) where T
    Statistics{T}(-1, -1,
                          tol, normest_A, -1.0,
                          T[], T[], T[], T[], T[],
                          0, Options{T}(),
                          0, 0, 0, 0, 0, 0.0, 0.0, 0, -1, -1, -1)
end

function Statistics(opts::Options{T}) where T
    Statistics{T}(-1, -1,
                          opts.tol, opts.normest_A, -1.0,
                          T[], T[], T[], T[], T[],
                          0, opts,
                          0, 0, 0, 0, 0, 0.0, 0.0, 0)
end

