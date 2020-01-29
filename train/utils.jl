module NNUtils

using CUDAdrv, Flux, MAT
using CUDAdrv.Mem

export gpu_mem_util, save_predict, build_model

# GPU memory utilization rate
gpu_mem_util() = @info("GPU memory utilization rate: " *
        "$(round(1 - (Mem.info()[1]/Mem.info()[2]); digits=4)*100)%")


"""
    save_predict(case::String, T::Float64, va::Array{Float32}, vm::Array{Float32})
Saves predicted results (va in degrees, vm .+ 1.0) and T value to .mat file
"""
function save_predict(case::String, T::Int64, va::Array{Float32}, vm::Array{Float32})
    matwrite("$(case)_predict_T=$(T).mat", Dict{String, Any}(
        "case_name" => case,
        "T" => T,
        "V_A" => rad2deg.(va),
        "V_M" => vm .+ 1.0
    ))
end


"""
    build_model(numBus::Int64)
Returns a 1D ConvNet with initial C=4, and final C=1
"""
function build_model(numBus::Int64)
    actfn = elu
    # W = 1, H = numBus, C = 4 , N = numSample
    # data needs to be in WHCN format (so conv filter is 1 by x)
    channels = [8,8,8,8,8]
    final_hidden = convert(Int32, numBus*channels[end])

    # for cases with more than 300 buses, first kernel length is 7 instead
    # of 3 for smaller cases
    if numBus <= 300
        model = Chain(
            Conv((1,3), 4=>channels[1], pad=(0,1), actfn),
            Conv((1,3), channels[1]=>channels[2], pad=(0,1), actfn),
            Conv((1,3), channels[2]=>channels[3], pad=(0,1), actfn),
            x -> reshape(x, (:, size(x, 4))),  # size(x, 4) = N, reshape to W*H*C by N
            Dense(final_hidden, numBus)
        )
    else
        model = Chain(
            Conv((1,7), K=>channels[1], pad=(0,3), actfn),
            Conv((1,3), channels[1]=>channels[2], pad=(0,1), actfn),
            Conv((1,3), channels[2]=>channels[3], pad=(0,1), actfn),
            Conv((1,3), channels[3]=>channels[4], pad=(0,1), actfn),
            Conv((1,3), channels[4]=>channels[5], pad=(0,1), actfn),
            x -> reshape(x, (:, size(x, 4))),  # size(x, 4) = N, reshape to W*H*C by N
            Dense(final_hidden, numBus)
        )
    end
    return model
end

end
