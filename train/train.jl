"""
	`using CuArrays`
We assume there is CUDA hardware; if error(s) are reported, comment it out and
`|> gpu` will be no-ops
"""

using CuArrays, Flux  # GPU, ML libraries
using Flux.Optimise: ADAM
using Flux: testmode!  # deactivate dropout layer
using Statistics # to use mean()
using Random  # shuffling for mini batch; not setting a random seed here
using LinearAlgebra
using BSON, MAT  # saving and loading files
using BSON: @save, @load
using Dates

# Fix culiteral_pow() error; check later to see if it is merged
using ForwardDiff
CuArrays.culiteral_pow(::typeof(^), x::ForwardDiff.Dual{Nothing,Float32,1}, ::Val{2}) = x*x

# # ELU activation function rotated CW by 180 degrees
# # VM (after shifting) and VA both (usually) have negative mean, and max VM
# # (not shifted) is just over 1, so let α = 1.5
# invELU(x, α=1.5) = x < 0 ? x : α*(1 - exp(-x))

"""
	train_net(case::String, data::Array{Float32}, target::Array{Float32},
		   vm_mean_shift::Array{Float32}, T::Float64, λ::Float64, K1::Int64,
		   K2::Int64=0, lr::Float64=1e-3, epochs::Int64=200,
		   batch_size::Int64=32, retrain::Bool=false)
Train an MLP with provided dataset or load trained model. Saves trained model
with checkpointing, as well as the loss and accuracy data in current directory.

ARGUMENTS:

	case: matpower case file name as a string; e.g. case118
	data, target: dataset and ground truth, must be of dimension (K, N) where N
		= number of samples and K = number of features
	vm_mean_shift: the difference between mean-col-norm of (ACVM-DCVM) and that
		of (ACVA-DCVA). Need to add back to ACVM before running hot start ACPF
	T: the ratio of training + validation set in N samples
	λ: additional penalty term on voltage angle mismatches for MSE
	K1: a positive integer, first hidden layer size

OPTIONAL ARGUMENTS:

	K2 : a non-negative integer, second hidden layer size (default = 0)
	lr: learning rate, default = 1e-3
	epochs: default = 100
	batch_size: a positive integer, default 64
	retrain: if `true` and a trained model already exists in current directory, then
		load the model, perform forward pass with data and target and exit. Default
		is `false`

Note that there might not be enough memory on the GPU if not running on clusters,
	in which case... you should probably find a GPU with enough memory to ensure
	model is well trained. Anything with a 6GB VRAM or higher should
	work.
"""
function train_net(case::String, data::Array{Float32}, target::Array{Float32},
			vm_mean_shift::Array{Float32}, T::Float64, λ::Float64, K1::Int64,
			K2::Int64=0, lr::Float64=1e-3, epochs::Int64=200,
			batch_size::Int64=32, retrain::Bool=false)

	# separate out training + validation (80/20) set, and N \ T for "test set"
	# also get the start and end indices of VA
	L, N = size(data)  # L = numPQ * 4, N = num samples
	va_end_idx = Int(L/2)  # will always be int, so no need to call round()
	# vm_end_idx = Int(L/2)
	trainSplit = round(Int32, N * T * 0.8)
	valSplit = round(Int32, N * T)

	trainData = data[:, 1:trainSplit] |> gpu
	trainTarget = target[:, 1:trainSplit] |> gpu
	valData = data[:, trainSplit+1:valSplit] |> gpu
	valTarget = target[:, trainSplit+1:valSplit] |> gpu
	testData = data[:, valSplit+1:end] |> gpu
	testTarget = target[:, valSplit+1:end] |> gpu

	# network model: if K2 is nonzero, there are two hidden layers, otherwise 1
	nlayers = 1  # number of hidden layer(s)
	W1 = Dense(size(data)[1], K1, relu)  # weight matrix 1
	W2 = Dense(K1, size(target)[1]) # weight matrix 2
	model = Chain(W1, Dropout(0.1), W2) |> gpu # foward pass: ŷ = model(x)
	if K2 != 0  # second hidden layer, optional
		W2 = Dense(K1, K2, relu)
		W3 = Dense(K2, size(target)[1])
		model = Chain(W1, Dropout(0.1), W2, Dropout(0.1), W3) |> gpu
		nlayers = 2
	end

	println(model)

	opt = ADAM(lr)  # ADAM optimizer

	######################## lambda functions start ############################
	# loss function and "accuracy" measure
	lossva(x, y) = λ * Flux.mse(model(x)[1:va_end_idx, :], y[1:va_end_idx, :])
	lossvm(x, y) = Flux.mse(model(x)[va_end_idx+1:end, :], y[va_end_idx+1:end, :])
	loss(x, y) =  lossva(x, y) + lossvm(x, y)

	norm_va(x, y, end_idx::Int64) = norm((y - model(x))[1:end_idx, :])  # L2 norm of (AC - predict) va
	norm_vm(x, y, start_idx::Int64) = norm((y - model(x))[start_idx:end, :])  # ... vm, :end because y only contains vm, va
	Δ_va(x, y, end_idx::Int64, init::Float32) = norm_va(x, y, end_idx) / init  # current norm_va compared to initial value
	Δ_vm(x, y, start_idx::Int64, init::Float32) = norm_vm(x, y, start_idx) / init  # ... vm
	######################## lambda functions end ##############################

	# record loss/accuracy data for three sets
	# println(va_end_idx)
	init_test_va_norm = Tracker.data(norm_va(testData, testTarget, va_end_idx))
	init_test_vm_norm = Tracker.data(norm_vm(testData, testTarget, va_end_idx+1))
	init_val_va_norm = Tracker.data(norm_va(valData, valTarget, va_end_idx))  # for early stopping
	init_val_vm_norm = Tracker.data(norm_vm(valData, valTarget, va_end_idx+1))
	init_train_va_norm = Tracker.data(norm_va(trainData, trainTarget, va_end_idx))
	init_train_vm_norm = Tracker.data(norm_vm(trainData, trainTarget, va_end_idx+1))

	trainLoss, valLoss, = Float32[], Float32[]
	epochTrainLoss, epochTrainErr = 0.0, 0.0
	batchData, batchTarget = undef, undef

	elapsedEpochs = 0
	training_time = 0.0  # excludes early stop condition & checkpointing overhead
	# earlyStopInd = Int16[]  # An indicator array, push an 1 if val loss increases

	# train with mini batches; if batch_size = 0, train with full batch
	randIdx = collect(1:1:trainSplit)
	if batch_size > 0
		numBatches = round(Int, floor(trainSplit / batch_size - 1))
	else
		numBatches = 1
		batch_size = trainSplit - 1
    end

	############################# training ##################################
	train_va_norm = init_train_va_norm
	last_improved_epoch = 1
	for epoch = 1:epochs
		# println("epoch: $epoch")
		# record validation set loss/err values of current epoch
		# before training so that we know the val loss/err in the beginning
		push!(valLoss, Tracker.data(loss(valData, valTarget)))
		epochTrainLoss, epochTrainErr = 0.0, 0.0   # reset values
		Random.shuffle!(randIdx) # to shuffle training set
		i = 1
		time = Base.time()
		for j = 1:numBatches
			batchData = trainData[:, randIdx[i:i+batch_size]]
			batchTarget = trainTarget[:, randIdx[i:i+batch_size]]
			Flux.train!(loss, Flux.params(model), [(batchData, batchTarget)], opt)
			# record training set loss every mini-batch
			push!(trainLoss, Tracker.data(loss(batchData, batchTarget)))
			i += batch_size+1  # without +1 there will be overlap
			if i+batch_size > trainSplit  # out of bounds indexing check
				break
			end
		end
		training_time += Base.time() - time
		elapsedEpochs = epoch

		testmode!(model)  # early stopping criteria, so put in test mode
		if Δ_va(valData, valTarget, va_end_idx, init_val_va_norm) <= 0.005
			break
		end

		# lr decay: if train loss doesn't decrease for 5 consecutive epochs,
		# otherwise update last improved epoch to be current epoch, and current
		# train_va_norm which initially equals to init_train_va_norm
		if Δ_va(trainData, trainTarget, va_end_idx, train_va_norm) < 1.
			train_va_norm = Tracker.data(norm_va(trainData, trainTarget, va_end_idx))
			last_improved_epoch = epoch
		elseif (epoch - last_improved_epoch) >= 5 && opt.eta > 1e-8
			opt.eta /= 10.0
			@warn("No improvements for the last 5 epochs. Decreased lr to" *
			 	" $(opt.eta) at epoch $epoch.")
			last_improved_epoch = epoch
		end
		testmode!(model, false)  # back to training mode
	end

	testmode!(model)  # done training, change to test mode
	println(model)

	# calculate change in test_vm, test_va compared to init_...
	Δtest_va = Tracker.data(Δ_va(testData, testTarget, va_end_idx, init_test_va_norm))
	Δtest_vm = Tracker.data(Δ_vm(testData, testTarget, va_end_idx+1, init_test_vm_norm))

	Δtrain_va = Tracker.data(Δ_va(trainData, trainTarget, va_end_idx, init_train_va_norm))
	Δtrain_vm = Tracker.data(Δ_vm(trainData, trainTarget, va_end_idx+1, init_train_vm_norm))

	println("Test: Δvm = $Δtest_vm, Δva = $Δtest_va")
	println("Train: Δvm = $Δtrain_vm, Δva = $Δtrain_va")

	# forward pass
	time = Base.time()
	testPredict = model(testData)
	fptime = Base.time() - time

	# save predicted vm (plus shifted mean), va values for N \ T
	testPredict = cpu(Tracker.data(testPredict))
	testPredict[va_end_idx+1:end, :] .+= vm_mean_shift[:, valSplit+1:end]
	# testData = cpu(testData[vm_end_idx+1:end, :])  # PD, QD values
	matwrite("$(case)_predict_$(T)T_$(λ)lambda.mat", Dict{String, Any}(
		"case_name" => case,
		"T" => T,
		"lambda" => λ,
		"vpredict" => testPredict#,
		# "pq" => testData
	))  # save predicted

	# save final model
	model = cpu(model)
	@save "$(case)_model_$(T)T_$(λ)λ.bson" model  # to get weights, use Tracker.data()

	# save loss and accuracy data
	matwrite("$(case)_loss_acc_$(T)T_$(λ)lambda.mat", Dict{String, Any}(
		"case_name" => case,
		"T" => T,
		"lambda" => λ,
		"trainLoss" => trainLoss,
		"valLoss" => valLoss,
	))

	# write result to file
	trainlog = open("$(case)_train_output.log", "a")
	println(trainlog, "Finished training after $elapsedEpochs epochs and $(round(
			training_time, digits=5)) seconds")
	println(trainlog, "Hyperparameters used: T = $T, λ = $λ, learning rate = $lr, "
			* "batch_size = $batch_size, hidden layers = $nlayers")
	println(trainlog, "Test set Results: ")
	println(trainlog, "  >> Forward pass: $(round(fptime, digits=5)) seconds")
	println(trainlog, "  >> L2 norm of (true - predict) VA = $(Δtest_va*100)% of initial")
	println(trainlog, "  >> L2 norm of (true - predict) VM = $(Δtest_vm*100)% of initial")
	println(trainlog, "")  # new line
	close(trainlog)
	println("Total training performance:")
end


"""
Running on command line (assuming train.jl is in current directory):

1. default hyperparameters:
`/path/to/julia run_train.jl <case name> <train ratio> <λ>

2. custom hyperparameters (currently must be complete):
`/path/to/julia run_train.jl <case name> <train ratio> <λ> <retrain Y/N>
 	<learning rate> <epochs <batch size>`
TODO: accommodate vararg hyperparameter set
"""

function main(args::Array{String})
	error = false
	case = args[1]  # case name, is String
	if length(args) != 7 && length(args) != 3
		println("Incorrect number of arguments provided. Expected 3 or 7, received $(length(args))")
		error = true
	elseif isfile("$(case)_dataset.mat") == false
		println("$case dataset not found in current directory. Exiting...")
		error = true
	end

	if error == true
		return Nothing
	end
	# load dataset from local directory
	data = matread("$(case)_dataset.mat")["data"]
	target = matread("$(case)_dataset.mat")["target"]
	vm_mean_shift = matread("$(case)_dataset.mat")["diff"]
	# # mean L2 norm of column wise ACVA - DCVA, ACVM - DCVM
	# norm_va = matread("$(case)_perf_cs.mat")["norm_va"]
	# norm_vm = matread("$(case)_perf_cs.mat")["norm_vm"]
	@assert(size(data)[1] == size(target)[1])

	println("Dataset loaded at $(now())")

	# Float32 should decrease memory allocation demand and run much faster on
	# non professional GPUs
	if typeof(data) != Array{Float32, 2}
		data = convert(Array{Float32}, data)
	end
	if typeof(target) != Array{Float32, 2}
		target = convert(Array{Float32}, target)
	end

	# calculate hidden layer size(s) based on dataset feature size
	K1 = round(Int, size(data)[1] * 2)
	T = parse(Float64, args[2])
	λ = parse(Float64, args[3])
	default_param = true
	if length(args) == 7
		retrain = (args[4]=="Y" || args[2]=="y") ? true : false
		lr = parse(Float64, args[5])
		epochs = parse(Int64, args[6])
		bs = parse(Int64, args[7])
		default_param = false
	end

	# check if model is trained and does not need to be retrained
	# if isfile("$(case)_model_$(T)T_$(λ)λ.bson") == true &&
	# 		(default_param == true || retrain == false)
	# 	trainlog = open("$(case)_train_output.log", "a")
	# 	println(trainlog, "Model already trained! Performing forward pass...")
	#
	# 	# load model to GPU
	# 	@load "$(case)_model_$(T)T_$(λ)λ.bson" model
	# 	model = model |> gpu
	# 	data = data[:, round(Int, T*size(data)[2])+1:end] |> gpu  # take N \ T for forward pass
	#
	# 	t = time()
	# 	predict = model(data)
	# 	println(trainlog, "Forward pass with $(size(data)[2]) samples took " *
	# 			"$(round(time() - t, digits=5)) seconds")
	#
	# 	# save predicted vm, va and the corresponding P, Q
	# 	pq_idx = Int(size(data)[1] / 2)  # pq_idx+1:end = index range of P, Q features
	# 	predict = cpu(Tracker.data(predict))
	# 	data = cpu(data[pq_idx+1:end, :])
	# 	matwrite("$(case)_predict_$(T)T_$(λ)lambda.mat", Dict{String, Any}(
	# 		"case_name" => case,
	# 		"T" => T,
	# 		"lambda" => λ,
	# 		"vpredict" => predict,
	# 		"pq" => data
	# 	))
	# 	close(trainlog)
	# 	println("Program finished at $(now()). Exiting...")
	# 	return nothing
    # end

	trainlog = open("$(case)_train_output.log", "a")
	println(trainlog, "Training a model for $case at $(now())...")
	close(trainlog)

	# train
	if default_param == true
		@time train_net(case, data, target, vm_mean_shift, T, λ, K1, K1)
	else
		@time train_net(case, data, target, vm_mean_shift, T, λ, K1, lr, epochs, bs, retrain)
	end
	println("Program finished at $(now()). Exiting...")
end

# main(ARGS)
# main(["case30", "0.2", "5.0"])
