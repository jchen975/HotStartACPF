using CUDAdrv, CuArrays, Flux, ForwardDiff  # GPU, ML libraries
using Flux.Optimise: ADAM
using LinearAlgebra
using BSON, MAT  # saving and loading files
using BSON: @save, @load
using Dates
import Base.time

include("./utils.jl")
using .NNUtils

CuArrays.culiteral_pow(::typeof(^), x::ForwardDiff.Dual{Nothing,Float32,1}, ::Val{2}) = x*x

"""
    inference(data::Array{Float32}, model_name::String, failmode::Bool=false)

Build model with untrained weights and load trained weights to model, then
perform forward pass and save predicted values. If `failmode` is true, then the
predictions are hot-start points that will potentially turn
forward pass on the samples in caseX_failed.mat, samples that failed to converge.
"""
function inference(data::Array{Float32}, model_name::String, failmode::Bool=false)
    case = split(model_name, "_")[1]
    inferlog = open("$(case)_infer_output.log", "a")
    println(inferlog, "Performing inference pass for $(case) at $(now())")
    close(inferlog)

    @load model_name model
    model = model |> gpu
    data = data |> gpu

    println(typeof(model))
    t = time()
    predict = model(data)
    fptime = time() - t

    predict = cpu(predict)
    return (predict, fptime)
end

"""
    main(args::Array{String})
Predictions for hot start without true target. If called with "failmode", try
to predict values that might solve infeasible cases (e.g. case300)

Example usage
- in REPL: main(["case118", "2000", "failmode"])
- on command line: julia inference.jl "case118" "2000" "failmode"
"""
function main(args::Array{String})
    case = args[1]  # case name, is String
    T = parse(Int64, args[2])  # to identify model to load
    failmode = (args[3]=="failmode") ? true : false  # if also predicting for failed to converge samples
    if length(args) != 3
        @warn("Incorrect number of arguments provided. Expected 3, received $(length(args))")
        return
    elseif !isfile("$(case)_dataset.mat") && !isfile("$(case)_failed.mat")
        @warn("$case dataset not found in current directory. Exiting...")
        return
    elseif !isfile("$(case)_va_model_T=$(T).bson") || !isfile("$(case)_vm_model_T=$(T).bson")
        @warn("Did not find both trained va and vm models")
        return
    elseif failmode && !isfile("$(case)_failed.mat")
        @warn("Failed samples for $case not found in current directory. Exiting...")
        return
    end

    inferlog = open("$(case)_infer_output.log", "a")
    println(inferlog, "Performing forward pass for $case at $(now())...")
    close(inferlog)

    if failmode
        data = matread("$(case)_failed.mat")["fdata"]
        @info("Forward pass for infeasible samples")
    else
        data = matread("$(case)_dataset.mat")["data"]
        @info("Forward pass for DCPF results")
    end
    @info("Dataset loaded at $(now())")
    success = false

    data = permutedims(data, [1,3,2])
    data = reshape(data, (1, size(data,1), size(data,2), size(data,3)))  # WHCN
    numBus, K = size(data)[2:3]

    ret_va = inference(data, "$(case)_va_model_T=$(T).bson", failmode)
    ret_vm = inference(data, "$(case)_vm_model_T=$(T).bson", failmode)
    save_predict(case, T, ret_va[2], ret_vm[2])

    fptime = ret_vm[3] + ret_va[3]
    inferlog = open("$(case)_infer_output.log", "a")
    println(inferlog, "  >> Forward pass: $(round(fptime; digits=7)) seconds")
    println(inferlog, "")  # new line
    close(inferlog)

    @info("Program finished at $(now()). Exiting...")
end

main(ARGS)  # uncomment this when running on command line
