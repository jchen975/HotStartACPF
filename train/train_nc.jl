"""
	`using CuArrays`
We assume there is CUDA hardware; if error(s) are reported, comment it out and
`|> gpu` will be no-ops
"""

using CuArrays, Flux  # GPU, ML libraries
using Flux.Optimise: ADAM
using Statistics # to use mean()
using Random  # shuffling for mini batch; not setting a randome seed here
using JLD2, BSON, FileIO  # saving and loading files
using BSON: @save
using Dates

# Fix culiteral_pow() error; check later to see if it is merged
using ForwardDiff
CuArrays.culiteral_pow(::typeof(^), x::ForwardDiff.Dual{Nothing,Float32,1}, ::Val{2}) = x*x

"""
	train_net(case::String, data::Array{Float32}, target::Array{Float32},
				T::Float64, K1::Int64, lr::Float64=1e-4, epochs::Int64=100,
				batch_size::Int64=32, λ::Float32=2.0, retrain::Bool=false)
Train an MLP with provided dataset or load trained model. Saves trained model
with checkpointing, as well as the loss and accuracy data in current directory.

ARGUMENTS:

	case: matpower case file name as a string; e.g. case118
	data, target: dataset and ground truth, must be of dimension (K, N) where N
		= number of samples and K = number of features
	T: a ratio of training + validation set in N samples
	K1: a positive integer, first hidden layer size

OPTIONAL ARGUMENTS:

	lr: learning rate, default = 1e-3
	epochs: default = 30
	batch_size: a positive integer, default 64
	λ: additional penalty term on voltage angle mismatches for MSE
	retrain: if `true` and a trained model already exists in current directory, then
		load the model, perform forward pass with data and target and exit. default
		is `false`

Note that there might not be enough memory on the GPU if not running on clusters,
	in which case... you should probably find a GPU with enough memory to ensure
	model is well trained. Anything with a 6GB VRAM or higher should
	work.
"""
function train_net(case::String, data::Array{Float32}, target::Array{Float32},
			T::Float64, K1::Int64, lr::Float64=1e-4, epochs::Int64=100,
			batch_size::Int64=32, λ::Float32=2.0, retrain::Bool=false)

	# check if model is trained and does not need to be retrained
	if isfile("$(case)_model.bson") == true && retrain == false
		trainlog = open("$(case)_train_output.log", "a")
		println(trainlog, "Model already trained! Run hot start acpf Instead.")
		close(trainlog)
		return nothing
    end

	# separate out training + validation (80/20) set, and N \ T for "test set"
	# also get the start and end indices of VA
	L, N = size(data)  # L = numPQ * 4, N = num samples
	vaSplit = Int(L/4)  # will always be int, so no need to call round()
	trainSplit = round(Int32, N * T * 0.8)
	valSplit = round(Int32, N * T)
	trainData = data[:, 1:trainSplit] |> gpu
	trainTarget = target[:, 1:trainSplit] |> gpu
	valData = data[:, trainSplit+1:valSplit] |> gpu
	valTarget = target[:, trainSplit+1:valSplit] |> gpu
	testData = data[:, valSplit+1:end] |> gpu
	testTarget = target[:, valSplit+1:end] |> gpu

	# network model: if K2 is nonzero, there are two hidden layers, otherwise 1
	layer1 = Dense(size(data)[1], K1, leakyrelu)  # weight matrix 1
	layer2 = Dense(K1, size(target)[1]) # weight matrix 2
	model = Chain(layer1, layer2) |> gpu  # foward pass: ŷ = model(x)
	# nlayers = 1  # number of hidden layer(s)
	# if K2 != 0  # second hidden layer, optional
	# 	layer2 = Dense(K1, K2, leakyrelu)
	# 	layer3 = Dense(K2, size(target)[1])
	# 	model = Chain(layer1, layer2, layer3) |> gpu
	# 	nlayers = 2
	# end

	# record loss/accuracy data for three sets
	testInitErr = mean(testData[1:vaSplit*2, :] - testTarget).^2  # initial mse
	trainLoss, trainErr, valLoss, valErr = Float32[], Float32[], Float32[], Float32[]
	epochTrainLoss, epochTrainErr = 0.0, 0.0
	batchData, batchTarget = undef, undef

	# loss function and accuracy measure
	lossva(x, y) = λ*Flux.mse(model(x)[1:vaSplit, :], y[1:vaSplit, :])
	lossvm(x, y) = Flux.mse(model(x)[vaSplit+1:end, :], y[vaSplit+1:end, :])
	loss(x, y) =  lossva(x, y) + lossvm(x, y)

	ϵ(x, y) = Flux.mse(model(x), y)  # plain MSE
	opt = ADAM(lr)  # ADAM optimizer

	elapsedEpochs = 0
	training_time = 0.0  # excludes early stop condition & checkpointing overhead
	earlyStopInd = Int16[]  # An indicator array, push an 1 if val loss increases

	# train with mini batches; if batch_size = 0, train with full batch
	randIdx = collect(1:1:trainSplit)
	if batch_size > 0
		numBatches = round(Int, floor(trainSplit / batch_size - 1))
	else
		numBatches = 1
		batch_size = trainSplit - 1
    end
	for epoch = 1:epochs
		time = Base.time()
		# println("epoch: $epoch")

		# record validation set loss/err values of current epoch
		# before training so that we know the val loss/err in the beginning
		push!(valLoss, Tracker.data(loss(valData, valTarget)))
		push!(valErr, Tracker.data(ϵ(valData, valTarget)))
		epochTrainLoss, epochTrainErr = 0.0, 0.0   # reset values
		Random.shuffle!(randIdx) # to shuffle training set
		i = 1
		for j = 1:numBatches
			batchData = trainData[:, randIdx[i:i+batch_size]]
			batchTarget = trainTarget[:, randIdx[i:i+batch_size]]
			Flux.train!(loss, Flux.params(model), [(batchData, batchTarget)], opt)
			# record train set loss/err every mini-batch
			push!(trainLoss, Tracker.data(loss(batchData, batchTarget))
			push!(trainErr, Tracker.data(ϵ(batchData, batchTarget))
			i += batch_size
		end
		training_time += Base.time() - time
		elapsedEpochs = epoch

		# checkpoint if val err decreased; to load, change @save to @load
		if epoch % 5 == 0 && epoch > 1 && valErr[end] < valErr[end-1]
			model_checkpt = cpu(model)
			@save "$(case)_model_ep$epoch.bson" model_checkpt  # to get weights, use Tracker.data()
		end
		# early exit condition on val loss
		if length(valLoss) > 1 && valLoss[end] > valLoss[end-1]
			push!(earlyStopInd, 1)
			# val loss increased over the last 3 epochs, overfitting starts
			if length(earlyStopInd) > 3 && sum(earlyStopInd[end-2:end]) == 3
				break
			end
		else
			push!(earlyStopInd, 0)  # indicates val loss decreased, good
		end
	end
	time = Base.time()
	testFinalErr = Tracker.data(ϵ(testData, testTarget))
	fptime = Base.time() - time
	testErr = Tracker.data(abs((testInitError - testFinalErr) / testInitError)) * 100  # percentage test err decreased

	# save PQ, predicted vm, va values for N \ T
	testPredict = model(testData)
	save("$(case)_predict.jld2", "predict", testPredict, "PQ", testData[vaSplit*2+1:end, :])

	# write result to file
	trainlog = open("$(case)_train_output.log", "a")
	println(trainlog, "Finished training after $elapsedEpochs epochs and $(round(
			training_time, digits=3)) seconds")
	println(trainlog, "Hyperparameters used: learning rate = $lr, "
			* "batch_size = $batch_size, number of hidden layers = $nlayers")
	println(trainlog, "Total samples in dataset: $N")
	println(trainlog, "  => Percentage of training + validation samples: $(T*100)%")
	println(trainlog, "  => Percentage of test set samples (forward pass): $((1-T)*100)%")
	println(trainlog, "Test set forward pass took $(round(fptime, digits=4)) seconds.")
	println(trainlog, "Initial test error = $testInitErr Final test error = $testFinalErr.")
	println(trainlog, "Test error Decreased by $(testErr)% after forward pass.")
	println(trainlog, "")  # new line
	close(trainlog)

	# save final model
	model = cpu(model)
	@save "$(case)_model.bson" model  # to get weights, use Tracker.data()
	# save loss and accuracy data
	save("$(case)_loss_acc_data.jld2", "trainLoss", trainLoss,
		"trainErr", trainErr, "valLoss", valLoss, "valErr", valErr)
	# plot_results(trainLoss, trainErr, valLoss, valErr, case)
	println("Total training performance:")
end


"""
Running on command line (assuming train.jl is in current directory):

1. default hyperparameters:
`/path/to/julia run_train.jl <case name> -d

2. custom hyperparameters (currently must be complete):
`/path/to/julia run_train.jl <case name> <retrain Y/N> <learning rate> <epochs>
 	<batch size> <train ratio>`
TODO: accommodate vararg hyperparameter set
"""

function main(args::Array{String})
	error = false
	case = args[1]  # case name, is String
	if length(args) != 7 && length(args) != 2
		println("Incorrect number of arguments provided. Expected 2 or 7, received $(length(args))")
		error = true
	elseif length(args) == 2 && args[2] != "-d"
		println("Unknown option. Expected -d or custom hyperparameters, received \"$(args[2])\"")
		error = true
	elseif isfile("$(case)_pf_results.jld2") == false
		println("Dataset $(case)_pf_results.jld2 not found in current directory. Exiting...")
		error = true
	end

	if error == true
		return Nothing
	end
	# load dataset from local directory
	data = FileIO.load("$(case)_pf_results.jld2")["data"]
	target = FileIO.load("$(case)_pf_results.jld2")["target"]
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
	K1 = round(Int, (size(data)[2] + size(target)[2]) / 2)
	default_param = true
	if length(args) == 7
		retrain = (args[2]=="Y" || args[2]=="y") ? true : false
		lr = parse(Float64, args[3])
		epochs = parse(Int64, args[4])
		bs = parse(Int64, args[5])
		T = parse(Float64, args[6])
		λ = parse(Float64, args[7])
		default_param = false
	end

	trainlog = open("$(case)_train_output.log", "a")
	println(trainlog, "Training a model for $case at $(now())...")
	close(trainlog)

	# train
	if default_param == true
		@time train_net(case, data, target, .2, K1)  # K2 is same size as K1
	else
		@time train_net(case, data, target, T, K1, lr, epochs, bs, λ, retrain)
	end
	println("Program finished at $(now()). Exiting...")
end

# main(ARGS)
