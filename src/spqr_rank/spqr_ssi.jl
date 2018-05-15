using Random
using LinearAlgebra

export spqr_ssi

#==
SPQR_SSI block power method or subspace iteration applied to inv(R)
 to estimate rank, smallest singular values, and left/right singular vectors.

 [U, S, V, stats] = spqr_ssi(R, opts...)

 Uses the block power method or subspace iteration applied to the inverse of
 the triangular matrix R (implicitly) to estimate the numerical rank, smallest
 singular values, and left and right singular vectors R.  The algorithm will
 be efficient in the case the nullity of R is relatively low and inefficient
 otherwise.

 Input:

   R -- an n by n nonsingular triangular matrix
   opts (optional) -- see 'help spqr_rank_opts' for details.

 Output:

   U  -- n by k matrix containing estimates of the left singular vectors of R
       corresponding to the singular values in S. When stats.flag is 0 (and
       when the optional parameter opts.nsvals_large has its default value 1),
       r = n - k + 1 is the estimated rank of R. Also NT=U(:,2:k) is an
       orthonormal basis for the numerical null space of R'.
   S -- A k by k diagonal matrix whose diagonal entries are the estimated
       smallest k singular values of R, with k as described above.  For i=1:k,
       S(i,i) is an an estimate of singular value (r + i - 1) of R.
       Note that, unless stats.flag = 3 (see below), S(1,1)) > tol and for
       i =2:k, S(i,i) <= tol.
   V  -- n by k matrix containing estimates of the right singular vectors of R
       corresponding to the singular values in S.   When stats.flag is 0,
       N=V(:,2:k) is an orthonormal basis for the numerical null space of R.
   stats -- statistics, type 'help spqr_rank_stats' for details.

 Note that U' * R = S * V' (in exact arithmetic) and that R * V is
 approximately equal to U * S.

 output (for one or two output arguments): [s,stats] = spqr_ssi (...)
    s -- diag(S), with S as above.

 Example:
    R = sparse(matrixdepot('kahan', 100))
    U,S,V = spqr_ssi(R)
    norm_of_residual = norm( U' * R - S * V' ) # should be near zero
     or
    U,S,V,stats = spqr_ssi(R, nothing, stats)
    N = V(:,2:end)   # orthonormal basis for numerical null space of R
    NT = U(:,2:end)  # orthonormal basis for numerical null space of R'
    norm_R_times_N = norm(R*N)
    norm_R_transpose_times_NT = norm(R'*NT)
     or
    opts = struct('tol',1.e-5,'nsvals_large',3)
    s,stats = spqr_ssi(R, opts, stats)
    stats  # information about several singular values

 See also spqr_basic, spqr_null, spqr_pinv, spqr_cod.

 Copyright 2012, Leslie Foster and Timothy A Davis.

 Outline of algorithm:
    let b = initial block size
    U = n by b random matrix with orthonormal columns
    repeat
       V1 = R \ U
       V,D1,X1 = svd(V1) = the compact svd of V1
       U1 = R' \ V
       U,D2,X2 = svd(U1) = the compact svd of U1
       sdot = diag(D2)
       s = 1 ./ sdot = estimates of singular values of R
       if s(end) <= tol
          increase the number of columns of U and repeat loop
       else
          repeat loop until stopping criteria (see code) is met
       end
    end
    k = smallest i with  sdot(i) > tol
    V = V*X2 (see code)
    V = first k columns of V in reverse order
    U = first k columns of U in reverse order
    s = 1 ./ (first k entries in sdot in reverse order)
    S = diag(s)

    Note for i = 1, 2, ..., k, S(i,i) is an upper bound on singular
    value (r + i - 1) of R in exact arithmetic but not necessarily
    in finite precision arithmetic.
==#

"""
start block power method to estimate the smallest singular
     values and the corresponding right singular vectors (in the
     r by nblock matrix V) and left singular vectors (in the r by
     nblock matrix U) of the n by n matrix R
"""
function spqr_ssi(R::AbstractMatrix{T};
                get_details::Int = 1,   # 0: basic statistics (default)
        # 1: detailed statistics:  basic stats, plus input options, time taken by
        #    various phases, statistics from spqr and spqr_rank subfunctions called,
        #    and other details.  Normally of interest only to the developers.
        # 2: basic statistics and a few additional statistics.  Used internally
        #    by some routines to pass needed information.
                repeatable::Bool = true, # by default, results are repeatable
                tol::Real = -1.0,        # a negative number means the default
        # tolerance should be computed
                tol_norm_type::Int = 2, # 1: use norm(A, 1) to compute the default tol
                                        # 2: use normest(A, 0.01)
                nsvals_large::Int = 1,  # default number of large singular values to estimate
                min_block::Int = 3,     #
                max_block::Int = 10,    #
                min_iters::Int = 3,     #
                max_iters::Int = 100,   #
                nblock_increment = 5,   #
                convergence_factor::Real = 0.1  #
        ) where T<:Number

    start_tic = time_ns()
    stats = Statistics(real(T))
    stats.time_initialize = time_ns() - start_tic

    m, n = size(R)
    if m != n
        error("R must be square")
    end

    normest_A = tol >= 0 ? 1.0 : tol_norm_type == 1 ? norm(R, 1) : normest(R, 0.01)
    stats.normest_A = normest_A
    tol = real(T)(tol)
    tol = tol >= 0 ? tol : normest_A * eps(real(T)) * n

    private_stream = repeatable ? MersenneTwister(1) : Random.GLOBAL_RNG

    #-------------------------------------------------------------------------------
    # adjust the options to reasonable ranges
    #-------------------------------------------------------------------------------

    # cannot compute more than n large singular values, where R is n-by-n
    nsvals_large = min(nsvals_large, n)

    # make the block size large enough so that there is a possiblity of
    # calculating all nsvals_large singular values
    max_block = max(max_block, nsvals_large)

    # max_block cannot be larger than n
    max_block = min(max_block, n)

    # min_block cannot be larger than n
    min_block = min(min_block, n)

    # start with nblock = min_block
    nblock = min_block

    #-------------------------------------------------------------------------------
    # initializations
    #-------------------------------------------------------------------------------

    # stats.flag and stats.rank initialized in spqr_rank_get_inputs
    # set the order of the remaining stats fields
    stats.tol = tol
    stats.tol_alt = -1.0   # removed later if remains -1
    if get_details == 2
        stats.ssi_max_block_used = -1
        stats.ssi_min_block_used = -1
    end
    if (get_details == 1)
        stats.norm_R_times_N = -1
        stats.norm_R_transpose_times_NT = -1
        stats.iters = -1
        stats.nsvals_large_found = 0
        stats.final_blocksize = -1
        stats.ssi_max_block_used = -1
        stats.ssi_min_block_used = -1
        # stats.opts_used = opts
        stats.time = 0
        stats.time_iters = 0
        stats.time_est_error_bounds = 0
        stats.time_svd = 0
    end

    stats.rank = 0
    stats.tol = tol
    if get_details == 1 || get_details == 2
        stats.ssi_max_block_used = max_block
        stats.ssi_min_block_used = nblock
    end

    if (get_details == 1)
        start_iters_tic = time_ns()
    end

    U = randn(private_stream, n, nblock)
    U = qr(U)[1]    # random orthogonal matrix
    # est_error_bound_calculated = 0      # set to 1 later if bound calculated
    flag_overflow = 0                     # set to 1 later if overflow occurs

    #-------------------------------------------------------------------------------
    # iterations
    #-------------------------------------------------------------------------------
    D2 = T[]
    X2 = T[]
    est_error_bound = T(0)
    
    iters = 0
    while iters < max_iters
        iters += 1

        U0= U
        V1 = R \ U
        if !all(isfinite.(V1))
            flag_overflow = 1
            break   # *************>>>>> early exit from for loop
        end

        if get_details == 1
            time_svd = time_ns()
        end

        V, D1, X1 = svd(V1)

        if get_details == 1
            stats.time_svd += time_ns() - time_svd
        end

        U1 = R' \ V
        # Note: with the inverse power method overflow is a potential concern
        #     in extreme cases (SJid = 137 or UFid = 1239 is close)
        if !all(isfinite.(U1))
            # *************>>>>> early exit from for loop
            # We know of no matrix that triggers this condition, so the next
            # two lines are untested.
            flag_overflow = 1      # untested
            break                   # untested
        end

        if get_details == 1
            time_svd = time_ns()
        end

        U, D2, X2 = svd(U1)

        if (get_details == 1)
            stats.time_svd += time_ns() - time_svd
        end
        
        k = findfirst(D2 .< tol^-1)
        if k == nothing && nblock == n
            # success since the numerical rank of R is zero
            break   # *************>>>>> early exit from for loop
        end

        errbound(k::Int) = norm(U0 * (X1 * (X2[:,k] ./ D1)) - U[:,k] ./ D2[k]) / sqrt(2)

        if k != nothing
            # k equals the calculated nullity + 1, k corresponds to
            # the first singular value of R that is greater than tol
            # reduce nsvals_large if necessary due to max_block limit:
            nsvals_large_old = nsvals_large
            nsvals_large = min(nsvals_large, max_block - k + 1)

            if nblock >= k + nsvals_large - 1
                # estimate error bound for singular value n - k + 1 of R:
                k2 = k + nsvals_large - 1
                est_error_bound = errbound(k)
                # When nsvals_large is 1, k and k2 are the same.  When
                # nsvals_large > 1, k2 corresponds to the largest singular
                # value of R that is returned.
                est_error_bound2 = k == k2 ? est_error_bound : errbound(k2)
            end

            # Note that
            #     [ 0     R ] [  U   ]  =    [ U0 * X1 * ( D1 \ X2 ) ]
            #     [ R'    0 ] [ V*X2 ]       [     V * ( X2 / D2 )   ]
            # It follows that, by Demmel's Applied Numerical Linear Algebra,
            # Theorem 5.5, some eigenvalue of
            #     B  =  [ 0    R ]
            #           [ R'   0 ]
            # will be within a distance of est_error_bound of 1 / d2(k), where s(1)
            # = 1 / d2(k) is our estimate of singular value n - k + 1 of R.
            # Typically the eigenvalue of B is singular value number n - k + 1 of
            # R, although this is not guaranteed.  est_error_bound is our estimate
            # of a bound on the error in using s(1) to approximate singular value
            # number n - k + 1 of R.

            if nblock >= k + nsvals_large - 1 &&
                est_error_bound <= convergence_factor * abs(1 / D2[k] - tol) &&
                est_error_bound2 <= convergence_factor * abs( 1/D2[k2] ) &&
                iters >= min_iters

                # Motivation for the tests:
                # The first test in the if statement is an attempt to insure
                # that nsvals_large singular values of R larger than tol are
                # calculated.
                #
                # Goal of the power method is to increase nblock until we find
                # sigma = (singular value n - k + 1 of R) is > tol.  If
                # this is true it is guaranteed that n - k + 1 is the
                # correct numerical rank of R.  If we let s(1) = 1 / d2(k)
                # then s(1) > tol.  However s(1) is only an estimate of sigma.
                # However, in most cases
                #     | s(1) - sigma | <= est_error_bound                   (1)
                # and to be conservative assume only that
                #  |s(1) - sigma|<= (1/convergence_factor)*est_error_bound  (2)
                # where convergence_factor<=1. By the second test in the if
                # statement
                #  est_error_bound <= convergence_factor * | s(1) - tol |   (3)
                # Equations (2) and (3) imply that
                #      | s(1) - sigma | <= | s(1) - tol |
                # This result and s(1) > tol imply that sigma > tol, as
                # desired.  Thus the second test in the if statement attempts
                # to carry out enough iterations to insure that the calculated
                # numerical rank is correct.
                #
                # The third test in the if statement checks on the accuracy of
                # the estimate for singular values n - k2 + 1.  Let sigma2 be
                # singular value n - k2 + 1 of R.  Usually it is true
                # that
                #     | s( k2 ) - sigma2 | <= est_error_bound2.             (4)
                # Assuming (4) and the third test in the if statement it
                # follows that the estimated relative
                # error in s(k2),  as measured by est_error_bound2 / s( k2) ,
                # is less that or equal to convergence_factor.  Therefore
                # the third test in the if statement attempts to insure that
                # the largest singular value returned by ssi has a relative
                # error <= convergence_factor.
                #
                # SUCCESS!!!, found singular value or R larger than tol
                break  # *************>>>>> early exit from for loop
            end
            nsvals_large = nsvals_large_old  # restore original value
        end

        if nblock == max_block && iters >= min_iters && k == nothing
            # reached max_block block size without encountering any
            # singular values of R larger than tolerance
            break    # *************>>>>> early exit from for loop
        end

        if 1 <= iters && iters < max_iters && 
            ( k == nothing  || ( k != nothing && nblock < k + nsvals_large - 1) )

            # increase block size
            nblock_prev = nblock
            if k == nothing
                nblock = min(nblock + nblock_increment, max_block)
            else
                nblock = min(k + nsvals_large - 1, max_block )
            end
            if nblock > nblock_prev
                Y = randn(private_stream, n, nblock-nblock_prev)
                Y = Y - U * ( U' * Y )
                Y = qr(Y)[1]                                            ##ok
                U = [U Y]      ##ok
            end
        end
    end

    if get_details == 1
        stats.final_blocksize = nblock    # final block size
        stats.time_iters = time_ns() - start_iters_tic
        stats.iters = iters               # number of iterations taken in ssi
    end

    #-------------------------------------------------------------------------------
    # check for early return
    #-------------------------------------------------------------------------------

    if flag_overflow == 1
        warning("spqr_rank:overflow", "overflow in inverse power method")
        stats, U, S, V = spqr_failure(4, stats, get_details, start_tic)
        return U, S, V, stats
    end

    #-------------------------------------------------------------------------------
    # determine estimated singular values of R
    #-------------------------------------------------------------------------------

    nkeep = k != nothing ? min(nblock, k + nsvals_large - 1) : nblock
    est_error_bounds = Vector{T}(undef, nkeep)

    if k != nothing
        # Note: in this case nullity = k - 1 and rank = n - k + 1
        est_error_bounds[nsvals_large] = est_error_bound
        numerical_rank = n - k + 1
    else
        k = nblock
        if nblock == n
            numerical_rank = 0
            # success since numerical rank is 0
        else
            # in this case rank not successfully determined
            # Note: In this case k is a lower bound on the nullity and
            #       n - k is an upper bound on the rank
            numerical_rank = n - k  #upper bound on numerical rank
            nsvals_large = 0 # calculated no singular values > tol
        end
    end

    S = T(1) ./ D2[nkeep:-1:1]

    stats.rank = numerical_rank

    if get_details == 1
        stats.nsvals_large_found = nsvals_large
    end


    #-------------------------------------------------------------------------------
    # adjust V so that R'*U = V*S (in exact arithmetic)
    #-------------------------------------------------------------------------------

    V = V * X2
    # reverse order of U and V and keep only nkeep singular vectors
    V = V[:,nkeep:-1:1]
    U = U[:,nkeep:-1:1]

    if get_details == 1
        t = time_ns()
    end

    if nsvals_large > 0
        # this recalculates est_error_bounds(nsvals_large)
        U0 = R * V[:,1:nsvals_large] - U[:,1:nsvals_large] * Diagonal(S[1:nsvals_large])
        U0 = [U0; R' * U[:,1:nsvals_large] - V[:,1:nsvals_large] * Diagonal(S[1:nsvals_large])]
        est_error_bounds[1:nsvals_large] .= vec(sqrt.(sum(abs2, U0, dims=1))) / sqrt(2)
    end

    # this code calculates estimated error bounds for singular values
    #    nsvals_large+1 to nkeep
    ibegin = nsvals_large + 1
    U0 = R * V[:,ibegin:nkeep] - U[:,ibegin:nkeep] * Diagonal(S[ibegin:nkeep])
    U0 = [U0; R' * U[:,ibegin:nkeep] - V[:,ibegin:nkeep] * Diagonal(S[ibegin:nkeep])]
    est_error_bounds[ibegin:nkeep] .= vec(sqrt.(sum(abs2, U0, dims=1))) / sqrt(2)
    # Note that
    #    [ 0      R ]  [ U ]   -    [ U ] * S = [ R * V - U * S ]
    #    [ R'     0 ]  [ V ]   -    [ V ]       [      0        ].
    # It follows, by Demmel's Applied Numerical Linear Algebra,
    # Theorem 5.5, for i = 2, . . .,  k, some eigenvalue of
    #     B  =  [ 0    R ]
    #           [ R'   0 ]
    # will be within a distance of the norm of the ith column of
    # [ R * V - U * S R'*U - V*S] / sqrt(2) from S(i,i). Typically this
    # eigenvalue will be singular value number n - k + i of R,
    # although it is not guaranteed that the singular value number
    # is correct.  est_error_bounds(i) is the norm of the ith column
    # of [ R * V - U * S; R' * U - V * S ] / sqrt(2).

    if get_details == 1
        # Note that stats.time_est_error_bounds includes the time for the error
        #   bound calculations done outside of the subspace iterations loop
        stats.time_est_error_bounds = time_ns() - t
    end

    nr1 = numerical_rank - nsvals_large
    ## stats.sval_numbers_for_bounds = nr1 + 1 : nr1 + length(est_error_bounds)

    #-------------------------------------------------------------------------------
    # compute norm R*N and R'*NT
    #-------------------------------------------------------------------------------

    # compute norm R*N where N = V(:,nsvals_large+1:end) is the approximate
    #         null space of R and
    #         norm R'*NT where NT = U(:,nsvals_large+1:end) is the approximate
    #         null space of R'

    if get_details == 1
        t = time_ns()
    end

    # svals_R_times_N = svd(R*V[:,nsvals_large+1:end])'
    norm_R_times_N = norm(R * V[:,nsvals_large+1:end])
    # svals_R_transpose_times_NT = svd(R'*U[:,nsvals_large+1:end])'
    norm_R_transpose_times_NT = norm(R' * U[:,nsvals_large+1:end])

    if get_details == 1
        stats.norm_R_times_N = norm_R_times_N
        stats.norm_R_transpose_times_NT = norm_R_transpose_times_NT
        stats.time_svd += time_ns() - t
    end

    # Note: norm_R_times_N is an upper bound on sing. val. rank1+1 of R
    # and norm_R_transpose_times_NT is also an upper bound on sing. val.
    #      rank1+1 of R
    max_norm_RN_RTNT = max(norm_R_times_N, norm_R_transpose_times_NT)
    # choose max here to insure that both null spaces are good

    #-------------------------------------------------------------------------------
    # determine flag indicating the accuracy of the rank calculation
    #-------------------------------------------------------------------------------

    if numerical_rank == 0
        # numerical rank is 0 in this case
        stats.flag = 0
    elseif (numerical_rank == n  || max_norm_RN_RTNT <= tol) &&
           (nsvals_large > 0 && S[nsvals_large] - est_error_bounds[nsvals_large] > tol )
        # in this case, assuming est_error_bounds(nsvals_large) is a true
        # error bound, then the numerical rank is correct.  Also
        # N = V(:,nsvals_large+1:end) and NT = U(:,nsvals_large+1:end)
        # are bases for the numerical null space or R and R', respectively
        stats.flag = 0
    elseif ( nsvals_large > 0 && numerical_rank == n ) ||
            ( nsvals_large > 0 &&
             S[nsvals_large] - est_error_bounds[nsvals_large] > max_norm_RN_RTNT )
        # in this case, assuming est_error_bounds(nsvals_large) is a true
        # error bound, then the numerical rank is correct with a modified
        # tolerance.  This is a rare case.
        stats.flag = 1
        tol_alt = ( S[nsvals_large] - est_error_bounds[nsvals_large] )
        tol_alt = tol_alt - eps(tol_alt) # so that tol_alt satisfies the >
                                         # portion of the inequality below
        # tol_alt = max_norm_RN_RTNT
        # Note: satisfactory values of tol_alt are in the range
        #    S[nsvals_large] - est_error_bounds[nsvals_large] > tol_alt
        #    >= max_norm_RN_RTNT
        stats.tol_alt = tol_alt
    elseif  nsvals_large > 0 && s(nsvals_large) > tol &&
            max_norm_RN_RTNT <= tol &&
            S[nsvals_large] - est_error_bounds[nsvals_large] <= max_norm_RN_RTNT
        # in this case, assuming est_error_bounds(nsvals_large) is a true
        # error bound, the error bound is too large to determine the
        # numerical rank.  This case is very rare.
        stats.flag = 2
    else
        # in this case all the calculated singular values are
        # smaller than tol or either N is not a basis for the numerical
        # null space of R with tolerance tol or NT is not such a basis for R'.
        # stats.rank is an upper bound on the numerical rank.
        stats.flag = 3
    end

    #-------------------------------------------------------------------------------
    # return results
    #-------------------------------------------------------------------------------
    stats.est_svals_of_R = S
    stats.est_error_bounds = est_error_bounds

    if get_details == 1
        stats.time = time_ns() - start_tic
    end

    U, S, V, stats
end
