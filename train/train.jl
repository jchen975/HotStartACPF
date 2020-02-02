################################################################################
# We assume there is CUDA hardware; if error(s) are reported, comment
# `using CuArrays` out and `|> gpu` will be no-ops
#
# This file requires Flux v0.10.0 since I removed all `Tracker` related code,
# and it works best with the most recent version of CuArrays on Julia 1.3.0
# as of January 2020
################################################################################
using CuArrays, Flux, ForwardDiff  # GPU, ML libraries
using Flux: train!, params
using Flux.Optimise: ADAM
using Random  # shuffling for mini batch; not setting a random seed here
using LinearAlgebra, Statistics
using BSON, MAT  # saving and loading files
using BSON: @save, @load
using Dates
import Base.time

include("./utils.jl")
using .NNUtils

Random.seed!(7)  # mini-batch shuffling

# include("./BerhuLoss.jl")

# Fix culiteral_pow() error; check later to see if fix is merged
CuArrays.culiteral_pow(::typeof(^), x::ForwardDiff.Dual{Nothing,Float32,1}, ::Val{2}) = x*x

"""
    split_dataset(data::Array{Float32}, target::Array{Float32},
            nn_type::String, trainSplit::Int64, valSplit::Int64)
Separate out training + validation set, and N ∖ T for "test set"
"""
function split_dataset(data::Array{Float32}, target::Array{Float32},
            trainSplit::Int64, valSplit::Int64)
    # data is currently (numBus, N, numFeature) == HNC, need to add W
    # axis and flip N and C to be WHCN, don't need to touch target
    data = permutedims(data, [1,3,2])
    data = reshape(data, (1, size(data,1), size(data,2), size(data,3)))
    trainData = data[:, :, :, 1:trainSplit]
    trainTarget = target[:, 1:trainSplit]  # remove last dimension (is 1)
    valData = data[:, :, :, trainSplit+1:valSplit]
    valTarget = target[:, trainSplit+1:valSplit]
    testData = data[:, :, :, valSplit+1:end]
    testTarget = target[:, valSplit+1:end]
    return (trainData, trainTarget, valData, valTarget, testData, testTarget)
end


"""
    train_net(data::Array{Float32}, target::Array{Float32}, case::String,
            T::Int64, failmode::Bool=false, lr::Float64=1e-3, epochs::Int64=5,
            bs::Int64=32)
Train an 1D ConvNet with provided dataset. Saves trained model as well as the
loss and accuracy data in current directory. If `failmode` is true, then also
compute inference before saving the model, and save its fp time.
"""
function train_net(data::Array{Float32}, target::Array{Float32}, case::String,
            T::Int64, failmode::Bool=false, lr::Float64=1e-3, epochs::Int64=500,
            bs::Int64=32)
    N = size(data, 2)
    trainSplit = round(Int64, T*0.9)
    valSplit = T
    split_data = split_dataset(data, target, trainSplit, valSplit)

    trainData = split_data[1] |> gpu
    trainTarget = split_data[2] |> gpu
    valData = split_data[3] |> gpu
    valTarget = split_data[4] |> gpu
    testData = split_data[5] |> gpu
    testTarget = split_data[6] |> gpu

    model = build_model(size(trainData,2))
    model = model |> gpu
    @info(model)

    # log model architecture
    casename = split(case, "_")[1]  # ex. case30_va => case30
    trainlog = open("$(casename)_train_output.log", "a")
    println(trainlog, "Model for $case built at $(now()):")
    println(trainlog, "  >> $model")
    close(trainlog)

    opt = ADAM(lr)
    loss(x, y) = sum((y - model(x)).^2)

    # "accuracy": norm and Δnorm (as percentage of initial)
    pred_norm(x::AbstractArray, y::AbstractArray) = norm(y - model(x))
    Δpred_norm(x::AbstractArray, y::AbstractArray, init::Float32) = pred_norm(x, y) / init

    # record loss/accuracy data for three
    init_train_norm = pred_norm(trainData, trainTarget)
    init_val_norm = pred_norm(valData, valTarget)  # for early stopping
    init_test_norm = pred_norm(testData, testTarget)

    trainLoss, valLoss = Float32[], Float32[]
    epochTrainLoss, epochTrainErr = 0.0, 0.0
    batchX, batchY = Nothing, Nothing

    elapsedEpochs = 0
    training_time = 0.0  # excludes early stop condition, checkpointing, etc. overhead

    # train with mini batches; if bs = 0, train with full batch
    randIdx = collect(1:1:trainSplit)
    if bs > 0
        numBatches = round(Int, floor(trainSplit / bs - 1))
    else
        numBatches = 1
        bs = trainSplit - 1
    end

    ############################# training ##################################
    # consecutive lr decays without improvements, stop training if == 3
    last_improved_epoch, consec_decays_wo_imp = 1, 0
    train_norm = init_train_norm
    for epoch = 1:epochs
        # record validation set loss/err values of current epoch before training
        # so that we know the val loss in the beginning
        push!(valLoss, loss(valData, valTarget))
        epochTrainLoss, epochTrainErr = 0.0, 0.0   # reset values
        Random.shuffle!(randIdx) # to shuffle training set
        i = 1
        for j = 1:numBatches
            batchX = trainData[:,:,:,randIdx[i:i+bs-1]]
            batchY = trainTarget[:, randIdx[i:i+bs-1]]
            # record training time, excluding all overhead except train!()'s own
            t = time()
            train!(loss, params(model), [(batchX, batchY)], opt)
            training_time += time() - t
            # record training set loss every mini-batch
            push!(trainLoss, loss(batchX, batchY))
            i += bs
            if i + bs > trainSplit  # out of bounds indexing check
                break
            end
        end
        epoch % 50 == 0 && @info("epoch $epoch, loss = $(trainLoss[end])")
        elapsedEpochs = epoch

        # epoch % 50 == 0 && gpu_mem_util()

        # stop training when validation set norm is 0.1% of the initial
        if Δpred_norm(valData, valTarget, init_val_norm) <= 1e-4
            @info("Validation set norm is 0.1% or lower than initial; training "*
                "completed at epoch $epoch.")
            break
        end

        # lr decay: if train loss doesn't decrease for 5 consecutive epochs,
        # otherwise update last improved epoch to be current epoch, and current
        # train_norm which initially equals to init_train_norm
        if Δpred_norm(trainData, trainTarget, train_norm) < 1.
            train_norm = pred_norm(trainData, trainTarget)
            last_improved_epoch = epoch
            consec_decays_wo_imp = 0
        elseif (epoch - last_improved_epoch) >= 5 && opt.eta > 1e-9
            opt.eta /= 10.0
            @info("No improvements for the last 5 epochs. Decreased lr to" *
                    " $(opt.eta) at epoch $epoch.")
            last_improved_epoch = epoch
            consec_decays_wo_imp += 1
        # elseif consec_decays_wo_imp == 3 || opt.eta <= 1e-9
        #     @info("No improvements for the last 15 epoches or reached min lr" *
        #             " of 1e-9. Training stopped at epoch $epoch.")
        #     break
        end
    end

    if elapsedEpochs == epochs
        @info("Training stopped at max epochs $epochs.")
    end

    # calculate change in test norm compared to init
    Δtest_norm = Δpred_norm(testData, testTarget, init_test_norm)

    # forward pass
    t = time()
    testPredict = model(testData)
    fptime = time() - t
    testPredict = cpu(testPredict)

    ffptime, fpredict = 0.0, zeros(0)
    if failmode
        if isfile("$(casename)_failed.mat")
            fdata = matread("$(casename)_failed.mat")["fdata"]
            fdata = permutedims(fdata, [1,3,2])
            fdata = reshape(fdata, (1, size(fdata,1), size(fdata,2), size(fdata,3)))
            fdata = fdata |> gpu
            tf = time()
            fpredict = model(fdata)
            ffptime = time() - tf
            fpredict = cpu(fpredict)
        else
            @info("No failed dataset found, ignoring `failmode == true`")
        end
    end

    # save final model
    model = cpu(model)
    BSON.@save "$(case)_model_T=$(T).bson" model

    # save loss and accuracy data
    matwrite("$(case)_loss_T=$(T).mat", Dict{String, Any}(
        "case_name" => case,
        "T" => T,
        "trainLoss" => trainLoss,
        "valLoss" => valLoss
    ))
    return (testPredict, Δtest_norm, (training_time, fptime, elapsedEpochs), (fpredict, ffptime))
end

"""
    main(args::Array{String})

Example usage
- in REPL: main(["case118", "2000", "retrain", "failmode"])
- on command line: julia inference.jl "case118" "2000" "retrain" "failmode"
"""
function main(args::Array{String})
    case = args[1]  # case name, is String
    T = parse(Int64, args[2])  # number of samples in training+val set (default = 2000)
    retrain = (args[3]=="retrain") ? true : false
    failmode = (args[4]=="failmode") ? true : false

    if length(args) != 4
        @warn("Incorrect number of arguments provided. Expected 4, received $(length(args))")
        return
    elseif !isfile("$(case)_dataset.mat")
        @warn("$case dataset not found in current directory. Exiting...")
        return
    end

    # load dataset from local directory
    data = matread("$(case)_dataset.mat")["data"]
    target = matread("$(case)_dataset.mat")["target"]
    if typeof(data) != Array{Float32, 2}  # Float32 better on gaming cards
        data = convert(Array{Float32}, data)
    end
    if typeof(target) != Array{Float32, 2}
        target = convert(Array{Float32}, target)
    end
    @info("Dataset loaded at $(now())")

    trainlog = open("$(case)_train_output.log", "a")
    println(trainlog, "Training a 1D CNN model for $case at $(now())...")
    close(trainlog)

    numBus = size(data, 1)
    if !retrain
        if !isfile("$(case)_va_model_T=$(T).bson") || !isfile("$(case)_vm_model_T=$(T).bson")
            @info("At least one of the two models not found, retraining...")
        else
            @info("Trained models found, use inference.jl instead.")
            return
        end
    end

    # training functions
    @time ret_va = train_net(data, target[:,:,1], case*"_va", T, failmode)
    @time ret_vm = train_net(data, target[:,:,2], case*"_vm", T, failmode)

    save_predict(case, T, ret_va[1], ret_vm[1])
    Δtest_norm_va = ret_va[2]
    Δtest_norm_vm = ret_vm[2]
    fptime = ret_va[3][2] + ret_vm[3][2]  # total foward pass time

    if failmode
        save_predict(case*"_failed", T, ret_va[4][1], ret_vm[4][1])
        ffptime =  ret_va[4][2] + ret_vm[4][2]
    end
    training_time = ret_va[3][1] + ret_vm[3][1]

    # write result to file
    trainlog = open("$(case)_train_output.log", "a")
    println(trainlog, "Finished training on $T samples after "
            * "$(round(training_time; digits=5)) seconds")
    println(trainlog, "Test set results: ")
    println(trainlog, "  >> Forward pass: $(round(fptime; digits=7)) seconds")
    println(trainlog, "  >> L2 norm of (true - predict) VA = $(Δtest_norm_va*100)% of initial")
    println(trainlog, "  >> L2 norm of (true - predict) VM = $(Δtest_norm_vm*100)% of initial")
    if failmode
        println(trainlog, "Infeasible dataset forward pass: " *
                "$(round(fptime; digits=7)) seconds")
    end
    println(trainlog, "")  # new line
    close(trainlog)

    @info("Program finished at $(now()). Exiting...")
end

main(ARGS)  # uncomment this when running on command line
