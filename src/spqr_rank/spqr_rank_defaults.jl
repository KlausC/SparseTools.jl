export get_opts, get_tol_norm
import Base.Iterators: Pairs

const SPQR_DEFAULTS = pairs((
                             get_details = 0,
                             repeatable = true,
                             k = 1,
                             tol = -1.0,
                             atol = 0.0,
                             rtol = 0.0,
                             tol_norm_type = 2,
                             nsvals_large = 1,
                             nsvals_small = 1,
                             implicit_null_space_basis = true,
                             start_with_A_transpose = false,
                             ssi_min_block = 3,
                             ssi_max_block = 10,
                             ssi_nblock_increment = 5,
                             ssi_min_iters = 3,
                             ssi_max_iters = 100,
                             ssi_convergence_factor = 0.1, # also 0.25 is often safe
                             ssp_min_iters = 4,
                             ssp_max_iters = 10,
                             ssp_convergence_factor = 0.1,
));

function get_opts(opts::Pairs, symbols::Symbol...)
    res = []
    for sym in symbols
        push!(res, get(opts, sym) do; getindex(SPQR_DEFAULTS, sym); end)
    end
    pushfirst!(res, merge_opts(opts, zip_named(symbols, res))) 
    length(res) == 1 ? res[1] : Tuple(res)
end

function get_tol_norm(opts::Pairs, A::AbstractMatrix{T}) where T
    opts, tol = get_opts(opts, :tol)
    normest_A = -1.0
    if tol == -1.0
        opts, atol, rtol, tol_norm_type = get_opts(opts, :atol, :rtol, :tol_norm_type)
        if ( atol <= 0.0 && rtol <= 0.0 )
            rtol = max(size(A)...) * eps(real(T))
        end
        if rtol > 0.0 
            normest_A = tol_norm_type == 1 ? opnorm(A, 1) : normest(A, 0.01) 
            xtol = rtol * normest_A
        end
        tol = max(atol, xtol)
        opts = merge_opts(opts, tol=tol)
    end
    opts, tol, normest_A
end

function zip_named(vs, vv)
    Pairs(NamedTuple{Tuple(vs)}(Tuple(vv)), Tuple(vs))
end

function merge_opts(x::Pairs; kw...)
    merge_opts(x, kw)
end

function merge_opts(x::Pairs, kw::Pairs)
    vs = Symbol[]
    vv = Any[]
    vx = Symbol[]
    for (s, v) in x
        if s in keys(kw)
            push!(vx, s)
            if kw[s] != nothing
                push!(vs, s)
                push!(vv, kw[s])
            end
        else
            push!(vs, s);
            push!(vv, v);
        end
    end
    for (s, v) in kw
        if s ∉ vx
            push!(vs, s); push!(vv, v)
        end
    end
    zip_named(vs, vv)
end
