"""
    Q, R, C, prow, pcol, info = spqr_wrapper(A, B, tol, get_details)

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
        Q, R, prow, pcol = qrf.Q, qrf.R, qrf.prow, qrf.pcol
        rk = maximum(R.rowval)
        B = Matrix(A)
        norm_E_fro = errornorm(Q, A, rk, prow, pcol)
        Q, R, prow, pcol, rk, norm_E_fro
    end

    function _spqr_wrapper(A::AbstractArray, tol)
        qrf = qr(A, Val(true)) # enable column pivoting
        Q, R, prow, pcol = qrf.Q, qrf.R, [1:size(A, 1)...], qrf.p
        rm, rn = size(R)
        k = findlast(gt(tol), diag(R))
        rk = k == nothing ? 0 : k
        norm_E_fro = norm(view(R, k+1:rm, k+1:rn))
        R[k+1:end,:] .= 0
        # compute Q*R = A[prow,pcol]
        norm_E_fro1 = norm(A[prow,pcol] - Q * R, 2)
        Q, R, prow, pcol, rk, norm_E_fro
    end

    Q, R, prow, pcol, rank, norm_E_fro = _spqr_wrapper(A, tol)
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

    Q, R, C, prow, pcol, info
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

"""
    Calculate `norm((Q' * A[prow,pcol])[k+1:end,k+1:end])` 
"""
function errornorm(Q::AbstractMatrix, A::AbstractMatrix{T}, k::Int, prow::Vector{Int}, pcol::Vector{Int}) where T<:Number

    m, n = size(A)
    B = Vector{T}(undef, m)
    sum = real(T)(0)
    if k < m
        for i = k+1:n
            copyto!(B, view(A, prow, pcol[i]))
            if norm(B, Inf) != 0
                lmul!(Q', B)
                j = k
                while j < m
                    j += 1
                    sum += abs2(B[j])
                end
            end
        end
    end
    sqrt(sum)
end







