using Random
using LinearAlgebra

export spqr_ssp, normest, normestinv

#==
SPQR_SSP block power method or subspace iteration applied to A or A*N

 U,S,V,stats = spqr_ssp(A, N, k, opts)

 Uses the block power method or subspace iteration to estimate the largest k
 singular values, and left and right singular vectors an m-by-n matrix A or,
 if the n-by-p matrix N is included in the input, of A*N.

 Alternate input syntaxes (opts are set to defaults if not present):

       spqr_ssp(A)         k = 1, N is [ ]
       spqr_ssp(A,k)       k is a scalar
       spqr_ssp(A,N)       N is a matrix or struct
       spqr_ssp(A,N,k)     k is a scalar
       spqr_ssp(A,N,opts)  opts is a struct

 Input:
   A -- an m-by-n matrix
   N (optional) -- an n-by-p matrix representing the null space of A, stored
       either explicitly as a matrix or implicitly as a structure, and
       returned by spqr_basic, spqr_null, spqr_cod, or spqr_pinv.
       N is ignored if [ ].
   k (optional) -- the number of singular values to calculate. Also the block
       size in the block power method.  Also can be specified as opts.k.
       Default: 1.
   opts (optional) -- type 'help spqr_rank_opts' for details

 Output:
   U  -- m-by-k matrix containing estimates of the left singular vectors of A
       (or A*N if N is included in the input) corresponding to the singular
       values in S.
   S -- k-by-k diagonal matrix whose diagonal entries are the estimated
       singular values of A (or A*N). Also for i = 1:nsval, S[i,i] is a lower
       bound on singular value i of A (or A*N).
   V -- n-by-k matrix containing estimates of the right singular vectors of A
       corresponding to the singular values in S.
   stats -- type 'help spqr_rank_stats'

 Note that A*V == U*S (or A*N*V == U*S) and that U'*A (or U'*A*N)
 is approximately equal to S*V'.

 Output (for one or two output arguments): s,stats= spqr_ssp(...)
   s -- diag(S), with S as above

 Example:
    A = MatrixDepot.matrixdepot("kahan", 100))
    U,S,V = spqr_ssp(A)   # singular triple for largest singular value
    # or
    U,S,V,stats = spqr_ssp(A,4)  # same for largest 4 singular values
    # or
    s,stats = spqr_ssp(A,4)
    stats     # stats has information for largest 4 singular values
    # or
    N = spqr_null(A)   # N has a basis for the null space of A
    s = spqr_ssp(A,N)
    s     # s has estimate of largest singular value (or norm) of A*N

 See also spqr_basic, spqr_null, spqr_pinv, spqr_cod.

 Copyright 2012, Leslie Foster and Timothy A Davis.

 Outline of algorithm:
    Let B = A or B = A*N, if the optional input N is included
    where A is m-by-n and N is n-by-p. Note that the code is written so
    that B is not explicitly formed.  Let n = p if N is not input.
    Let k = block size = min(k,m,p)
    U = m-by-k random matrix with orthonormal columns
    repeat
       V1 = B' * U
       V, D1, X1= svd(V1) = the compact svd of V1
       U1 = B * V
       U, D2, X2 = svd(U1) = the compact svd of U1

       kth_est_error_bound = norm(B'*U[:,k] - V*(X2[:,k]*D2[k,k])) / sqrt(2)
       if kth_est_error_bound <= convergence_factor*s[k] && iters >= min_iters
            flag = 0
            exit loop
       elseif number of iteration > max number of iterations
            flag = 1
            exit loop
       else
            continue loop
       end
    end
    V = V*X2 # adjust V so that A*V = U*S
    calculate estimated error bounds for singular values 1:k-1
       as discussed in the code
    S = diag(s)
==#

"""
start block power method to estimate the largest singular values and the
    corresponding right singular vectors (in the n-by-k matrix V) and left
    singular vectors (in the m-by-k matrix U) of the m-by-n matrix A
"""
function spqr_ssp(A::AbstractMatrix{T}, N::Union{UniformScaling{T}, AbstractArray{T},NamedTuple}=one(T)I;
                  nargout=4, opts...) where T<:Number

##           k::Int = 1,   # the # of singular values to compute.
##           min_iters::Int = 4, # min # of iterations before checking convergence.
##           max_iters::Int = 10, # max # of iterations before stopping the iterations.
#       The default value = 10 appears, with the default value of opts.tol,
#       to provide sufficient accuracy to correctly determine the numerical
#       rank when spqr_ssi is called by spqr_basic, spqr_null, spqr_pinv or
#       spqr_cod in almost all cases, assuming that stats.flag is 0. For
#       values of opts.tol larger than the default value, a larger value of
#       opts.ssp_max_iters, for example 100, may be useful.
##            convergence_factor::Real = 0.1, # continue power method iterations until an
#       estimated bound on the relative error in the approximation
#       S[k] to singular value number k of A is <= convergence_factor.
#       The default value = 0.1 appears, with the default value of
#       opts.tol, to provide sufficient accuracy to correctly determine
#       the numerical rank in spqr_basic, spqr_null, spqr_pinv or spqr_cod
#       in almost all cases, assuming that stats.flag is 0.  For values
#       of opts.tol larger than the default value, a smaller value of
#       opts.ssp_convergence_factor, for example 0.01, may be useful.      
##            get_details::Int = 1, # determine what statistics to return.
#       0: basic statistics
#       1: extensive statistics.
#       2: basic statistics and a few addional statistics.
#       See 'help spqr_rank_stats' for details.
##            repeatable::Bool = true # controls the random stream.
#       false: use the current random number stream.  The state of the stream will
#           be different when the method returns.  Repeated calls may generate
#           different results.  Faster for small matrices, however.
#       true: use a repeatable internal random number stream.  The current stream
#           will not be affected.  Repeated calls generate the same results.

    start_tic = time_ns()
    opts, get_details, repeatable, k, min_iters, max_iters, convergence_factor =
    get_opts(opts, :get_details, :repeatable, :k, :ssp_min_iters, :ssp_max_iters, :ssp_convergence_factor) 
    stats = Statistics(real(T))
    stats.time_initialize = time_ns() - start_tic

    private_stream = repeatable ? MersenneTwister(1) : Random.GLOBAL_RNG
    #-------------------------------------------------------------------------------
    # initializations
    #-------------------------------------------------------------------------------

    m, n = size(A)
    n1 = N isa UniformScaling ? n : size(N, 1)
    n2 = N isa UniformScaling ? n : size(N, 2)
    k = max(k, 0)
    k = min(k, m, n2)     # number of singular values to compute
    max_iters = max(1, max_iters)
    
    stats.flag = 1
    stats.est_error_bounds = zeros(T, max(k, 1))
    #stats.sval_numbers_for_bounds = 1:k
    stats.opts_used = opts.data

    if k <= 0
        # quick return.  This is not an error condition.
        stats.flag = 0
        stats.est_svals = T[]
        stats.sval_numbers_for_bounds = 1:0
        U = zeros(m, 0)
        S = zeros(0)
        V = zeros(n, 0)
        return nargout == 1 ? S : nargout <= 2 ? (S, stats) : (U, S, V, stats)
    end
    
    U = randn(private_stream, m, k)
    U = qr(U).Q * Matrix{T}(I, m, k)  # iteration start - U has k orthonormal columns

    #-------------------------------------------------------------------------------
    # block power iterations
    #-------------------------------------------------------------------------------
    get_details >= 1 && ( t = time_ns() )

    D2 = T[]
    X2 = T[]
    kth_est_error_bound = T(0)
    
    ASU = similar(U, n, k)
    V1 = similar(U, n2, k)
    NV = similar(U, n1, k)
    U1 = similar(U)
    ASU1 = similar(U, n)
    V11 = similar(U, size(V1, 1))

    crit = eps(norm(A)) * min(m, n) 

    iters = 0
    while iters < max_iters
        iters += 1
        mul!(ASU, A', U)
        mul!(V1, N', ASU)       

        get_details == 1 && (time_svd = time_ns())
        V = svd(V1).U
        get_details == 1 && (stats.time_svd += time_ns() - time_svd)

        mul!(NV, N, V)
        mul!(U1, A, NV)

        get_details == 1 && (time_svd = time_ns())
        U, D2, X2 = svd(U1)
        get_details == 1 && (stats.time_svd += time_ns() - time_svd)

        # estimate error bound on kth singular value
        mul!(ASU1, A', U[:,k])
        mul!(V11, N', ASU1)
        V11 .-= V * ( X2[:,k] * D2[k] )
        kth_est_error_bound = norm(V11) / sqrt(2)

        # Note that
        #     [ 0     B ] [  U   ]  =    [       U * D2          ]
        #     [ B'    0 ] [ V*X2 ]       [     V * ( X2 * D2 )   ]
        # where B = A or B = A*N (if N is included in the input).  It follows that,
        # by Demmel's Applied Numerical Linear Algebra, Theorem 5.5, some
        # eigenvalue of
        #     C  =  [ 0    B ]
        #           [ B'   0 ]
        # will be within a distance of kth_est_error_bound of D2[k,k].  Typically
        # the eigenvalue of C is the kth singular value of B, although this is not
        # guaranteed.  kth_est_error_bound is our estimate of a bound on the error
        # in using D2[k,k] to approximate kth singular value of B.
        
        crit = max(convergence_factor * D2[k], crit)

        println("ssp: kth_est[$iters] = $kth_est_error_bound crit = $crit D2[$k] = $(D2[k])" )

        if kth_est_error_bound <= crit && iters >= min_iters
            # The test indicates that the estimated relative error in
            # the estimate, D2[k,k] for the kth singular value is smaller than
            # convergence_factor, which is the goal.
            stats.flag = 0
            break
        end
    end

    if get_details == 1
        stats.iters = iters
        stats.time_iters = time_ns() - t
    end

    S = D2[1:k]
    stats.est_error_bounds[k] = kth_est_error_bound
    stats.est_svals = S


    # adjust V so that A*V = U*S
    V = V*X2

    #-------------------------------------------------------------------------------
    # estimate error bounds for the 1st through (k-1)st singular values
    #-------------------------------------------------------------------------------

    if get_details >= 1
        t = time_ns()
    end

    if k > 1

        V1 = N' * (A' * U[:,1:k-1])
        U0 = V1 - V[:,1:k-1] * Diagonal(S[1:k-1])
        stats.est_error_bounds[1:k-1] .= vec(sqrt.(sum(abs2, U0, dims=1))) / sqrt(2)

        # Note (with V adjusted as above) that
        #    [ 0      B ]  [ U ]   -    [ U ] * S = [       0       ]
        #    [ B'     0 ]  [ V ]   -    [ V ]       [   B'*U - V*S  ]
        # where B = A or B = A*N (if N is included in the input).  It follows, by
        # Demmel's Applied Numerical Linear Algebra, Theorem 5.5, for i = 1:k, some
        # eigenvalue of
        #     C  =  [ 0    B ]
        #           [ B'   0 ]
        # will be within a distance of the norm of the ith column of [ B' * U - V *
        # S ] / sqrt(2) from S[i,i]. Typically this eigenvalue will be the ith
        # singular value number of B, although it is not guaranteed that the
        # singular value number is correct.  stats.est_error_bounds[i] is the norm
        # of the ith column of [ B' * U - V * S ] / sqrt(2).

    end

    #-------------------------------------------------------------------------------
    # return results
    #-------------------------------------------------------------------------------

    if get_details >= 1
        stats.time_est_error_bounds = time_ns() - t
        stats.time = time_ns() - start_tic
    end

    nargout  == 1 ? S : nargout <= 2 ? (S, stats) : (U, S, V, stats)
end


"""
    normest(A; [tol=convergence_factor], [max_iters=...])

Estimate 2-norm of matrix A, power-iterating `A'A` up to given tolerance.
"""
function normest(A::AbstractMatrix{T}; tol::AbstractFloat = 1e-6,
                 max_iters::Integer = 100) where T <:Number

    min_iters = 2
    max_iters = max(max_iters, min_iters)
    tol = max(tol, eps())
    ncols = size(A, 2)
    y = randn(T, ncols)
    k = argmax(sum(abs, A, dims=1))[2]
    y[k] += sign(y[k]) * 0.1
    lmul!(1/norm(y), y)
    i = 0
    n = 0.0
    n0 = n
    x = Vector{T}(undef, size(A,1))
    while ( abs(n - n0) > tol * n || i < min_iters ) && i < max_iters
        i += 1
        n0 = n
        mul!(x, A, y)
        normx = norm(x)
        if normx == 0
            x = rand(T, ncols)
        end
        lmul!(1/norm(x), x)
        mul!(y, A', x)
        n = norm(y)
    end
    n
end

"""
    normestinv(A; [tol=convergence_factor], [max_iters=...])

Estimate 2-norm of inverse of matrix A, power-iterating `A⁻¹'A⁻¹` up to given tolerance.
"""
function normestinv(A::AbstractMatrix{T}; tol::AbstractFloat = 1e-6,
                    max_iters::Integer = 100) where T <:Number

    if istriu(A)
        _normestinv(UpperTriangular(A), tol, max_iters)
    elseif istril(A)
        _normestinv(LowerTriangular(A), tol, max_iters)
    else
        _normestinv(A, tol, max_iters)
    end
end

function _normestinv(A::Union{Adjoint{<:Any,<:AbstractSparseMatrix},
                              Transpose{<:Any,<:AbstractSparseMatrix}},
                              tol::AbstractFloat, max_iters::Integer)

    _normestinv(A.parent, tol, max_iters)
end

function _normestinv(A::AbstractSparseMatrix{T}, tol::AbstractFloat = 1e-6,
                    max_iters::Integer = 100) where T <:Number

    m, n = size(A)
    tol2 = max(tol^2, eps())
    if m < n
        A = sparse(A')
    end
    qrf = qr(A, tol=tol^2) # TODO that is not clean
    
    R = qrf.R
    m1, n1 = size(R)
    if min(m1, n1) < min(m, n)
        return Inf
    elseif m1 > n1 
        R = SparseMatrixCSC(n1, n1, R.colptr, R.rowval, R.nzval)
    elseif m1 < n1
        R = SparseMatrixCSC(m1, m1, R.colptr[1:m1+1], R.rowval, R.nzval)
    end
    _normestinv(UpperTriangular(R), tol, max_iters)
end

function _normestinv(A::AbstractMatrix{T}, tol::AbstractFloat = 1e-6,
                    max_iters::Integer = 100) where T <:Number
    
    m, n = size(A)
    if m < n
        A = A'
    end
    qrf = qr(A, Val(true))
    R = qrf.R
    m1, n1 = size(R)
    if min(m1, n1) < min(m, n)
        Inf
    else
        R = view(R, 1:m1, 1:m1)
    end
    _normestinv(UpperTriangular(R), tol, max_iters)
end

function normestinv(A::LinearAlgebra.AbstractTriangular{T}; tol::AbstractFloat = 1e-6,
                    max_iters::Integer = 100) where T <:Number
    _normestinv(A, tol, max_iters)
end

function _normestinv(A::LinearAlgebra.AbstractTriangular{T}, tol::AbstractFloat,
                    max_iters::Integer) where T <:Number

    min_iters = 2
    max_iters = max(max_iters, min_iters)
    tol = max(tol, eps())
    ncols = size(A, 2)
    y = randn(T, ncols)
    lmul!(1/norm(y), y)
    i = 0
    n = 0.0
    n0 = n
    x = Vector{T}(undef, size(A,1))
    while ( abs(n - n0) > tol * n || i < min_iters ) && i < max_iters
        i += 1
        n0 = n
        # ldiv!(x, A, y)
        x = A \ y
        normx = norm(x)
        if normx == 0
            x = rand(T, ncols)
        end
        lmul!(1/norm(x), x)
        #ldiv!(y, A', x)
        y = A' \ x
        n = norm(y)
    end
    n
end
