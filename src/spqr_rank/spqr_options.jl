"""
    Statistics


"""
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
    opts_used::NamedTuple
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
    sval_numbers_for_bounds::AbstractUnitRange{Int}
end

function Statistics(::Type{T}; tol::T=0.0, normest_A::T=0.0) where T
    Statistics{T}(-1, -1,
                          tol, normest_A, -1.0,
                          T[], T[], T[], T[], T[],
                          0, NamedTuple(),
                          0, 0, 0, 0, 0, 0.0, 0.0, 0, -1, -1, -1, 0:-1)
end

