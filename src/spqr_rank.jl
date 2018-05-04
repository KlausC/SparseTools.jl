
dir = @__FILE__; dir = dir[1:length(dir)-3]
for f in filter(f->endswith(f, ".jl"), readdir(dir))
    include(joinpath(dir, f))
end

#include(joinpath("spqr_rank", "spqr_options.jl"))
#include(joinpath("spqr_rank", "spqr_ssi.jl"))
#include(joinpath("spqr_rank", "spqr_ssp.jl"))

