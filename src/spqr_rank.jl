
dir = @__FILE__; dir = dir[1:length(dir)-3]
include(joinpath(dir, "spqr_options.jl"))
for f in filter(f->endswith(f, ".jl"), readdir(dir))
    if f != "spqr_options.jl"
        include(joinpath(dir, f))
    end
end

