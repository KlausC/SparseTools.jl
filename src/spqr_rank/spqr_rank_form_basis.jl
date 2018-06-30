"""
SPQR_RANK_FORM_BASIS forms the basis for the null space of a matrix.

 Called from spqr_basic and spqr_cod after these routines call spqr and
 spqr_rank_ssi.  The input parameters are as used in spqr_basic and spqr_cod.
 The parameter call_from indicates the type of call:
      call_from = 1 -- a call from spqr_basic, form null space basis for A'
      call_from = 2 -- a call from spqr_cod, form null space basis for A
      call_from = 3 -- a call from spqr_cod, form null space basis for A'
 Output:
   N -- basis for the null space in implicit form
   stats -- an update of the stats vector in spqr_basic or spqr_cod
   stats_ssp -- information about the call to spqr_ssp
   est_sval_upper_bounds -- an update to the estimated singular value upper
      bounds
"""
function spqr_rank_form_basis(call_from, A::AbstractMatrix{T},
                              U::AbstractMatrix{T}, V::AbstractMatrix{T},
                              Q1::AbstractMatrix{T}, prow::AbstractVector{Int},
                              rank_spqr::Int, numerical_rank::Int,
                              stats::Statistics, opts,
                              est_sval_upper_bounds::AbstractVector{S},
                              nsvals_small::Int, nsvals_large::Int,
                              p1::AbstractVector{Int} = 1:0,
                              Q2::AbstractMatrix{T} = zeros(0,0),
                              p2::AbstractVector{Int} = 1:0
                             ) where {T<:Number, S<:Real}

# Copyright 2012, Leslie Foster and Timothy A Davis.

    opts, get_details, start_with_A_transpose, k =
    get_opts(opts, :get_details, :start_with_A_transpose, :k)

    if get_details == 1
        t = time_ns()
    end
    m , n = size(A)
    @assert rank_spqr == size(U, 1)
    @assert rank_spqr == size(V, 1)
    nullity_R11 = rank_spqr - numerical_rank

    #-------------------------------------------------------------------------------
    # form null space basis for A or A' using X and Q from QR factorization
    #-------------------------------------------------------------------------------

    N = if call_from == 1
        ImplicitProd(invperm(prow), Q1, U)
    elseif start_with_A_transpose == (call_from == 2) 
        ImplicitProd(Q1, invperm(p2), V) 
    else
        ImplicitProd(invperm(p1), Q2, U)
    end

    if get_details == 1
        stats.time_basis = time_ns() - t
    end

    #-------------------------------------------------------------------------------
    # call spqr_ssp to enhance, potentially, est_sval_upper_bounds, the estimated
    #    upper bounds on the singular values
    # and / or
    #    to estimate ||A * N || or || A' * N ||
    #-------------------------------------------------------------------------------

    # Note: nullity = m - numerical_rank;   # the number of columns in X and N

    kk = max(nsvals_small, opts.data.k)

    if call_from == 1    # call from spqr_basic (for null space basis for A')

        # Note: opts.k is not the same as k in the algorithm description above.

        # Note that, by the interleave theorem for singular values, for
        # i=1:nullity, singular value i of A'*N will be an upper bound on singular
        # value numerical_rank + i of A.  S[i] is an estimate of singular value i
        # of A'*N with an estimated accuracy of stats_ssp.est_error_bounds[i].
        # Therefore let
        s_ssp, stats_ssp = spqr_ssp(A', N; nargout=2, opts..., k = kk)

    elseif call_from == 2    # call from spqr_cod (for null space basis of A)

        # Note that, by the interleave theorem for singular
        # values, for i = 1, ..., nullity, singular value i of A*N will be
        # an upper bound on singular value numerical_rank + i of A.
        # S[i] is an estimate of singular value i of A*N with an estimated
        # accuracy of stats_ssp.est_error_bounds[i].  Therefore let
        s_ssp, stats_ssp = spqr_ssp(A, N; nargout=2, opts..., k = kk)
    end

    if call_from == 1 || call_from == 2

        # By the comments prior to the call to spqr_ssp we have
        # s_ssp + stats_ssp.est_error_bounds are estimated upper bounds
        # for singular values (numerical_rank+1):(numerical_rank+nsvals_small)
        # of A.  We have two estimates for upper bounds on these singular
        # values of A.  Choose the smaller of the two:
        if nsvals_small > 0
            est_sval_upper_bounds[nsvals_large+1:end] .=
                min.(est_sval_upper_bounds[nsvals_large+1:end],
                vec(s_ssp[1:nsvals_small]) + stats_ssp.est_error_bounds[1:nsvals_small])
        end

    elseif call_from == 3  # call from spqr_cod (for null space basis for A')

        # call ssp to estimate nsvals_small sing. values of A' * N
        #    (useful to estimate the norm of A' * N)
        _, stats_ssp = spqr_ssp(A', N; nargout=2, opts..., nsvals_small = nsvals_small)   ##ok

    end

    if (get_details == 1)
        if call_from == 1 || call_from == 3
            stats.stats_ssp_NT = stats_ssp
        elseif call_from == 2
            stats.stats_ssp_N = stats_ssp
        end
        stats.time_svd = stats.time_svd + stats_ssp.time_svd
    end
    
    N , stats, stats_ssp, est_sval_upper_bounds
end
