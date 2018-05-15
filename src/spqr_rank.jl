
dir = @__FILE__; dir = dir[1:length(dir)-3]
for f in filter(f->endswith(f, ".jl"), readdir(dir))
    include(joinpath(dir, f))
end

