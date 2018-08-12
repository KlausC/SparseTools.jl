module SparseTools

using SparseArrays
using SuiteSparse
using LinearAlgebra

# SPQR begin
include("options.jl")

include("support.jl")
include("wrapper.jl")
include("basic.jl")
include("null.jl")
include("pinv.jl")
include("rank_defaults.jl")
include("rank_deflation.jl")
include("rank_form_basis.jl")
include("ssi.jl")
include("ssp.jl")
# SPQR end

end # module
