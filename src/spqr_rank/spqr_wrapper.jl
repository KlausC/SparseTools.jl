"""
    Q, R, C, pcol, prow, info = spqr_wrapper(A, B, tol, get_details)

SPQR_WRAPPER wrapper around spqr to get additional statistics.
The ouput `R` has reduced rown count, if final diagonal elements would become below `tol`.
Note: In addition to the original, a row pivoting array is returned
Maybe that should be absorbed into Q in the sparse case of `qr()`.
"""
function spqr_wrapper(A::AbstractArray{T}, B::AbstractArray{T}, tol, get_details) where T
    # Copyright 2012, Leslie Foster and Timothy A Davis.

    if get_details != 0
        # get detailed statistics for time and memory usage
        t = time_ns()
    else
        t = UInt(0)
    end

    m, n = size(A)
    info = NamedTuple()

    gt(tol::AbstractFloat) = x::AbstractFloat -> abs(x) > tol

    function _spqr_wrapper(A::AbstractSparseArray, tol)
        m, n = size(A)
        qrf = qr(A, tol=tol)
        Q, R, pcol, prow = qrf.Q, qrf.R, qrf.pcol, qrf.prow
        rk = size(R, 1)
        norm_E_fro = 0.0 #norm(A[prow,pcol] - Q * [R; spzeros(m-rk,n)], 2) 
        Q, R, pcol, prow, maximum(R.rowval), norm_E_fro
    end

    function _spqr_wrapper(A::AbstractArray, tol)
        qrf = qr(A, Val(true)) # enable column pivoting
        Q, R, pcol, prow = qrf.Q, qrf.R, qrf.p, [1:size(A, 1)...]
        k = findlast(gt(tol), diag(R))
        rank = k == nothing ? 0 : k
        R[k+1:end,:] = 0
        # compute Q*R = A[prow,pcol]
        norm_E_fro = norm(A[prow,pcol] - Q * R, 2) 
        Q, R, prow, pcol, rank, norm_E_fro
    end

    Q, R, pcol, prow, rank, norm_E_fro = _spqr_wrapper(A, tol)
    R = striprows(R, rank)

    if isempty(B)
        # C is empty
        C = zeros(T, m, 0)
    else
        # also compute C = Q'*B if B is present
        C = Q' * B
    end

    t = get_details != 0 ? time_ns() - t : t
    info = (rank_A_estimate=rank, norm_E_fro=norm_E_fro, time=t)

    Q, R, C, pcol, prow, info
end

function striprows(A::SparseMatrixCSC, k::Int)
    m, n = size(A)
    if k < m
        SparseMatrixCSC(k, n, A.colptr, A.rowval, A.nzval)
    else
        A
    end
end

function striprows(A::AbstractMatrix, k::Int)
    m, n = size(A)
    if k < m
        A[1:k,:]
    else
        A
    end
end

