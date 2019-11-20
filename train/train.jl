"""
    `using CuArrays`
We assume there is CUDA hardware; if error(s) are reported, comment it out and
`|> gpu` will be no-ops
"""

using CuArrays, Flux  # GPU, ML libraries
using Flux.Optimise: ADAM
using Statistics # to use mean()
using Random  # shuffling for mini batch; not setting a random seed here
using LinearAlgebra
using BSON, MAT  # saving and loading files
using BSON: @save, @load
using Dates

Random.seed!(7)  # mini-batch shuffling

# Fix culiteral_pow() error; check later to see if fix is merged
using ForwardDiff
CuArrays.culiteral_pow(::typeof(^), x::ForwardDiff.Dual{Nothing,Float32,1}, ::Val{2}) = x*x

"""
    save_predict(case::String, T::Float64, va::Array{Float32}, vm::Array{Float32})
Saves predicted results (va in degrees, vm .+ 1.0) and T value to .mat file
"""
function save_predict(case::String, T::Float64, va::Array{Float32}, vm::Array{Float32})
    matwrite("$(case)_predict_T=$(T).mat", Dict{String, Any}(
        "case_name" => case,
        "T" => T,
        "V_A" => rad2deg.(va),
        "V_M" => vm .+ 1.0
    ))
end


"""
    build_model(type::String, Fx::Int64, Fy::Int64, K::Int64)
Returns the NN model: MLP with 2 hidden layers or 1D ConvNet

ARGUMENTS:

    type: a string, either "mlp" or "conv"
    Fx, Fy: integers, input and output feature size
    K: an integer, hidden layer size for MLP or input channel number for ConvNet

"""
function build_model(type::String, Fx::Int64, Fy::Int64, K::Int64)
    actfn = elu  # ELU activation function; seems to work better than ReLU/LeakyReLU
    if type == "mlp"
        # Dense() are fully connected layers, i.e. weight matrices connecting
        # in/output and hidden layers
        model = Chain(
            Dense(Fx, K, actfn),
            Dense(K, K, actfn),
            Dense(K, Fy)
        )
    elseif type == "conv"
        # W = 1, H = numBus (Fx, if conv)
        # C = numChannel (maximum 4, va, vm, p, q), N = numSample
        # data needs to be in WHCN format (so conv filter is 1 by x)
        channels = [8,8,8]
        final_hidden = convert(Int32, Fy*channels[end])

        # for cases with more than 300 buses, first kernel length is 5 instead
        # of 3 for smaller cases
        if Fx > 300
            conv1 = Conv((1,5), K=>channels[1], pad=(0,2), actfn)
        else
            conv1 = Conv((1,3), K=>channels[1], pad=(0,1), actfn)
        end
        model = Chain(
            conv1,
            Conv((1,3), channels[1]=>channels[2], pad=(0,1), actfn),
            Conv((1,3), channels[2]=>channels[3], pad=(0,1), actfn),  # H unchanged with 1 padding
            x -> reshape(x, (:, size(x, 4))),  # size(x, 4) = N, reshape to W*H*C by N
            Dense(final_hidden, Fy)
        )
    else
        @warn("Type of NN unspecified. Failed to build model.")
        model = Nothing
    end
    return model
end


"""
    split_dataset(data::Array{Float32}, target::Array{Float32}, nn_type::String,
            T::Float64)
Separate out training + validation set, and N ∖ T for "test set"
"""
function split_dataset(data::Array{Float32}, target::Array{Float32},
            nn_type::String, trainSplit::Int64, valSplit::Int64)
    if nn_type == "mlp"  # flatten 3d data/target into 2d, [numBus * 4, N]
        data = vcat(data[:,:,1], data[:,:,2], data[:,:,3], data[:,:,4])
        if size(target,3) > 1  # if not trained separately
            target = vcat(target[:,:,1], target[:,:,2])
        end
        trainData = data[:, 1:trainSplit]
        trainTarget = target[:, 1:trainSplit]
        valData = data[:, trainSplit+1:valSplit]
        valTarget = target[:, trainSplit+1:valSplit]
        testData = data[:, valSplit+1:end]
        testTarget = target[:, valSplit+1:end]
    elseif nn_type == "conv"
        # data is currently (numBus, N, numFeature) == HNC, need to add W
        # axis and flip N and C to be WHCN, don't need to touch target
        data = permutedims(data, [1,3,2])
        data = reshape(data, (1, size(data,1), size(data,2), size(data,3)))
        if size(target,3) > 1  # if not trained separately
            target = vcat(target[:,:,1], target[:,:,2])
        end
        trainData = data[:, :, :, 1:trainSplit]
        trainTarget = target[:, 1:trainSplit]  # remove last dimension (is 1)
        valData = data[:, :, :, trainSplit+1:valSplit]
        valTarget = target[:, trainSplit+1:valSplit]
        testData = data[:, :, :, valSplit+1:end]
        testTarget = target[:, valSplit+1:end]
    else
        @warn("Type of NN unspecified. Failed to split dataset.")
        return Nothing
    end
    return (trainData, trainTarget, valData, valTarget, testData, testTarget)
end


"""
    train_net(data::Array{Float32}, target::Array{Float32}, case::String,
            nn_type::String, T::Float64, K::Int64, lr::Float64=1e-3,
            epochs::Int64=1000, bs::Int64=32)
Train an MLP/1D ConvNet with provided dataset. Saves trained model weights
as well as the loss and accuracy data in current directory.

ARGUMENTS:

    case: matpower case file name as a string; e.g. case118
    data, target: dataset and ground truth, must be of dimension (K, N) where N
        = number of samples and K = number of features
    nn_type: type of neural network, MLP w/ 2 hidden layers or 1D ConvNet
    T: the ratio of training + validation set in N samples
    K: a positive integer, hidden layer size if nn_type is "mlp", or number of
        "channels", i.e. features on each bus (maximum = 4 if va, vm, P, Q all
        present)

OPTIONAL ARGUMENTS:

    lr: learning rate, default = 1e-3
    epochs: default = 1000
    bs: a positive integer, default 64

RETURN VALUES:

    A tuple, (testPredict, Δtest_norm, fptime), where testPredict is a matrix
    with size numBus by numTestSamples, Δtest_norm is a number between 0 and 1
    representing the final norm of (true - testPredict), and fptime is the time
    for test set forward pass at the end of training.

Note that there might not be enough memory on the GPU if not running on clusters,
    but anything with a 6GB VRAM or higher should work. Ideally more though.
"""
function train_net(data::Array{Float32}, target::Array{Float32}, case::String,
            nn_type::String, T::Float64, K::Int64, lr::Float64=1e-3,
            epochs::Int64=500, bs::Int64=32)
    N = size(data, 2)
    trainSplit = round(Int64, N * T * 0.9)
    valSplit = round(Int64, N * T)
    split_data = split_dataset(data, target, nn_type, trainSplit, valSplit)
    if split_data == Nothing
        return ()
    end
    trainData = split_data[1] |> gpu
    trainTarget = split_data[2] |> gpu
    valData = split_data[3] |> gpu
    valTarget = split_data[4] |> gpu
    testData = split_data[5] |> gpu
    testTarget = split_data[6] |> gpu

    nn_type == "conv" ? Fx = size(trainData,2) : Fx = size(trainData,1)
    model = build_model(nn_type, Fx, size(trainTarget,1), K)
    if model == Nothing
        return ()
    end
    model = model |> gpu

    # log model architecture; if va, vm trained separately, only want case name
    casename = split(case, "_")[1]  # ex. case30_va => case30
    trainlog = open("$(casename)_train_output.log", "a")
    println(trainlog, "Model for $case built at $(now()):")
    println(trainlog, "  >> $model")
    close(trainlog)

    opt = ADAM(lr)
    loss(x, y) = norm(y - model(x))  # Flux.mse(model(x), y)

    # "accuracy": norm and Δnorm (as percentage of initial, return as untracked
    pred_norm(x::AbstractArray, y::AbstractArray) = Tracker.data(norm(y - model(x)))
    Δpred_norm(x::AbstractArray, y::AbstractArray, init::Float32) = pred_norm(x, y) / init

    # record loss/accuracy data for three
    init_train_norm = pred_norm(trainData, trainTarget)
    init_val_norm = pred_norm(valData, valTarget)  # for early stopping
    init_test_norm = pred_norm(testData, testTarget)

    trainLoss, valLoss, = Float32[], Float32[]
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
        # println("epoch: $epoch")
        # record validation set loss/err values of current epoch before training
        # so that we know the val loss in the beginning
        push!(valLoss, Tracker.data(loss(valData, valTarget)))
        epochTrainLoss, epochTrainErr = 0.0, 0.0   # reset values
        Random.shuffle!(randIdx) # to shuffle training set
        i = 1
        for j = 1:numBatches
            # get batch data, target
            nn_type == "conv" ? batchX = trainData[:,:,:,randIdx[i:i+bs]] : batchX = trainData[:,randIdx[i:i+bs]]
            batchY = trainTarget[:, randIdx[i:i+bs]]
            # record training time, excluding all overhead except train!()'s own
            time = Base.time()
            Flux.train!(loss, Flux.params(model), [(batchX, batchY)], opt)
            training_time += Base.time() - time
            # record training set loss every mini-batch
            push!(trainLoss, Tracker.data(loss(batchX, batchY)))
            i += bs + 1  # without +1 there will be overlap
            if i + bs > trainSplit  # out of bounds indexing check
                break
            end
        end
        elapsedEpochs = epoch

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
        elseif consec_decays_wo_imp == 3 || opt.eta <= 1e-9
            @info("No improvements for the last 15 epoches or reached min lr" *
                    " of 1e-9. Training stopped at epoch $epoch.")
            break
        end
    end

    if elapsedEpochs == epochs
        @info("Training stopped at max epochs $epochs.")
    end

    # calculate change in test norm compared to init
    Δtest_norm = Δpred_norm(testData, testTarget, init_test_norm)
    println(Δtest_norm)

    # forward pass
    time = Base.time()
    testPredict = model(testData)
    fptime = Base.time() - time

    # get predicted values (untracked)
    testPredict = cpu(Tracker.data(testPredict))

    # save final model weights (untracked)
    # model_weights = Flux.params(model)
    BSON.@save "$(case)_model_T=$(T).bson" model

    # save loss and accuracy data
    matwrite("$(case)_loss_T=$(T).mat", Dict{String, Any}(
        "case_name" => case,
        "T" => T,
        "trainLoss" => trainLoss,
        "valLoss" => valLoss
    ))
    return (testPredict, Δtest_norm, (training_time, fptime, elapsedEpochs))
end


"""
    forward_pass(data::Array{Float32}, target::Array{Float32}, case::String,
                nn_type::String, K::Int)

Build model with untrained weights and load trained weights to model, then
perform forward pass and save predicted values. By default, `split` which
represents whether `data` are test set already split, is false. No `target`
here because no training involved. Refer to `train_net` and `build_model` for
arguement types and meanings.
"""
function forward_pass(data::Array{Float32}, model_name::String, nn_type::String,
                T::Float64, K::Int64, split::Bool=false)
    if !isfile("$(model_name).bson")
        return false
    end
    trainlog = open("$(case)_train_output.log", "a")
    println(trainlog, "Performing forward pass for $(split(model_name, "_")) at $(now())")

    # build model with untrained weights
    model = build_model(nn_type, size(data,1), size(target,1), K)
    if model == Nothing
        return (false,)
    end

    # load trained weights to model and send to GPU
    @load "$(model_name).bson" weights
    Flux.loadparams!(model, weights)
    model = model |> gpu

    # get the test set from data if split == false and send to GPU
    if !split
        data = data[:, round(Int, T*size(data,2))+1:end] |> gpu  # N \ T
    else
        data = data |> gpu
    end

    t = time()
    predict = model(data)
    println(trainlog, "Forward pass with $(size(data)[2]) samples took " *
            "$(round(time() - t, digits=5)) seconds")

    predict = cpu(Tracker.data(predict))
    close(trainlog)
    return (true, predict)
end


"""
    main(args::Array{String})

Running in REPL:

    1. default hyperparameters:
        ex. main(["case30", "0.2", "conv", "Y", "Y"])
    2. custom hyperparameters:
        ex. main(["case30", "0.2", "conv", "Y", "Y", "5e-4", "30", "64"])

Running on command line (assuming train.jl is in current directory):

    1. default hyperparameters:
    `/path/to/julia run_train.jl <case name> <train ratio> <nn_type>
        <train separately Y/N> <retrain Y/N>`
    2. custom hyperparameters (currently must be complete):
    `/path/to/julia run_train.jl <case name> <train ratio> <nn_type>
        <train separately Y/N> <retrain Y/N> <learning rate> <epochs> <batch size>`

TODO: accommodate vararg hyperparameter set
"""
function main(args::Array{String})
    error = false
    case = args[1]  # case name, is String
    if length(args) != 8 && length(args) != 5
        @warn("Incorrect number of arguments provided. Expected 5 or 8, received $(length(args))")
        error = true
    elseif !isfile("$(case)_dataset.mat")
        @warn("$case dataset not found in current directory. Exiting...")
        error = true
    end
    if error
        return
    end

    # parse other arguments
    T = parse(Float64, args[2])  # ratio of training/val set (default = 0.2)
    nn_type = args[3]  # is a string
    separate = (args[4]=="Y" || args[4]=="y") ? true : false  # train two models, one each for vm, va
    retrain = (args[5]=="Y" || args[5]=="y") ? true : false  # if model(s) already trained
    default_param = true
    if length(args) == 8
        lr = parse(Float64, args[6])
        epochs = parse(Int64, args[7])
        bs = parse(Int64, args[8])
        default_param = false
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
    println("Dataset loaded at $(now())")

    trainlog = open("$(case)_train_output.log", "a")
    println(trainlog, "Training a $nn_type model for $case at $(now())...")
    close(trainlog)

    # hidden layer size if nn_type is mlp, or 4 if ConvNet
    K = (nn_type == "mlp") ? size(target, 1)*2 : 4
    numBus = size(data, 1)

    # if model already trained, try forward pass
    if !retrain
        success = false
        if separate
            ret_va = forward_pass(data, "$(case)_va_model_T=$(T).bson", nn_type, T, K)
            ret_vm = forward_pass(data, "$(case)_vm_model_T=$(T).bson", nn_type, T, K)
            success = ret_va[1] & ret_vm[1]
            if success
                save_predict(case, T, ret_va[2], ret_vm[2])
                return
            end
        else
            ret = forward_pass(data, "$(case)_model_T=$(T).bson", nn_type, T, K)
            success = ret[1]
            if success
                save_predict(case, T, ret[2][1:numBus, :], ret[2][numBus+1:end, :])
                return
            end
        end
    end

    # training functions, will enter from forward pass above if unsuccessful
    if separate
        if default_param
            @time ret_va = train_net(data, target[:,:,1], case*"_va", nn_type, T, K)
            @time ret_vm = train_net(data, target[:,:,2], case*"_vm", nn_type, T, K)
        else
            @time ret_va = train_net(data, target[:,:,1], case*"_va", nn_type,
                                        T, K, lr, epochs, bs)
            @time ret_vm = train_net(data, target[:,:,2], case*"_vm", nn_type,
                                        T, K, lr, epochs, bs)
        end
        if ret_va == () || ret_vm == ()  # build_model failed
            @warn("build_model() failed, exiting at $(now())")
            return
        end
        # save as .mat
        save_predict(case, T, ret_va[1], ret_vm[1])
        Δtest_norm_va = ret_va[2]
        Δtest_norm_vm = ret_vm[2]
        training_time = ret_va[3][1] + ret_vm[3][1]
        fptime = ret_va[3][2] + ret_vm[3][2]  # total foward pass time
    else
        if default_param
            @time ret = train_net(data, target, case, nn_type, T, K)
        else
            @time ret = train_net(data, target, case, nn_type, T, K, lr, epochs, bs)
        end
        if ret == ()  # build_model failed
            @warn("build_model() failed, exiting at $(now())")
            return
        end
        save_predict(case, T, ret[1][1:numBus, :], ret[1][numBus+1:end, :])
        Δtest_norm = ret[2]
        fptime = ret[3]
    end

    # write result to file
    trainlog = open("$(case)_train_output.log", "a")
    println(trainlog, "Finished training after $(round(training_time, digits=5)) seconds")
    if default_param
        println(trainlog, "Hyperparameters used: T = $T, learning rate = 1e-3, "
            * "bs = 32")
    else
        println(trainlog, "Hyperparameters used: T = $T, learning rate = $lr, "
            * "bs = $bs")
    end
    println(trainlog, "Test set Results: ")
    println(trainlog, "  >> Forward pass: $(round(fptime, digits=7)) seconds")
    if separate
        println(trainlog, "  >> L2 norm of (true - predict) VA = $(Δtest_norm_va*100)% of initial")
        println(trainlog, "  >> L2 norm of (true - predict) VM = $(Δtest_norm_vm*100)% of initial")
    else
        println(trainlog, "  >> L2 norm of (true - predict) = $(Δtest_norm*100)% of initial")
    end
    println(trainlog, "")  # new line
    close(trainlog)

    println("Program finished at $(now()). Exiting...")
end

# main(ARGS)  # uncomment this when running remotely
