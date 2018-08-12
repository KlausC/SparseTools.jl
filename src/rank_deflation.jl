"""
    x = spqr_rank_deflation(...)

SPQR_RANK_DEFLATION constructs pseudoinverse or basic solution using deflation.

 Called from spqr_basic and spqr_cod after these routines call spqr,
 spqr_rank_ssi and spqr_rank_form_basis.  The input parameters are as used
 in spqr_basic and spqr_cod.
 The parameter call_from indicates the type of call:
      call_from = 1 indicates a call from spqr_basic
      call_from = 2 indicates a call from spqr_cod
 Output:
   x -- for call_from = 1 x is a basic solution to
                 min || b - A * x ||.               (1)
        The basic solution has n - (rank returned by spqr) zero components.
        For call_from = 2 x is an approximate pseudoinverse solution to (1).
 Not user-callable.

 Algorithm:   R * wh = ch or R' * wh = ch is solved where ch is described
              in the code and R comes from the QR factorizations in
              spqr_basic or spqr_cod. R is triangular and potentially
              numerically singular with left and right singular vectors for
              small singular values stored in U and V.  When R is numerically
              singular deflation (see SIAM SISC, 11:519-530, 1990) to
              calculate an approximate truncated singular value solution to
              the triangular system.  Orthogonal transformations
              are applied to wh to obtain the solutions x to (1).

"""
function spqr_rank_deflation(call_from, R::AbstractMatrix{T}, U, V, C, m, n, rank_spqr,
                    numerical_rank, prow, pcol, Q::AbstractMatrix{T}; opts...) where T

# disable nearly-singular matrix warnings, and save the current state
## TODO: how? warning_state = warning ('off', 'MATLAB:nearlySingularMatrix') ;

    opts, nsvals_large, start_with_A_transpose =
        get_opts(opts, :nsvals_large, :start_with_A_transpose) 

    if isempty(C)
        return zeros(T, m)
    end
    # restrict to columns of U, V belonging to small singular Values of R (< tol)
    if numerical_rank < rank_spqr
        k = size(U, 2)
        U = view(U, 1:m, nsvals_large+1:k)
        V = view(V, 1:m, nsvals_large+1:k)
    end

   if start_with_A_transpose || call_from == 1

       ch = view(C, 1:rank_spqr, 1:size(C,2))
       if numerical_rank == rank_spqr
           wh = R \ ch
       else
           # use deflation (see SIAM SISC, 11:519-530, 1990) to calculate an
           # approximate truncated singular value solution to R * wh = ch
           wh = ch - U * (U' * ch)
           wh = R \ wh
           wh = wh - V * (V' * wh)
        end
        if call_from == 2
            wh[prow,:] = wh;
        end
        wh = [wh ; zeros(T, n - rank_spqr,size(C,2)) ];
        if call_from == 1
            x = wh
            x[pcol,:] = x
        else
            x = spqr_qmult(Q, wh, 1)
        end

    else

        ch = view(C, prow, 1:size(C,2))
        if numerical_rank == rank_spqr
            wh = R' \ ch
        else
            # use deflation (see SIAM SISC, 11:519-530, 1990) to calculate an
            # approximate truncated singular value solution to R' * wh = ch
            wh = ch - V * (V' * ch)
            wh = R' \ wh
            wh = wh - U * (U' * wh)
        end
        wh = [wh ; zeros(T, n - rank_spqr, size(C,2)) ];
        wh = spqr_qmult(Q, wh, 1)
        x[pcol,:] = wh

    end

    # restore the warning back to what it was
    ## TODO: warning (warning_state) ;

    x
end
