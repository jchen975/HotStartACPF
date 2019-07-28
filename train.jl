"""
	`using CuArrays`
We assume there is CUDA hardware; if error(s) are reported, comment it out and
`|> gpu` will be no-ops
"""

using CuArrays
using Flux  # ML library
using Flux.Optimise: ADAM
using JLD2, BSON  # saving and loading files
using BSON: @save
using Plots  # plot loss and acc curves

"""
	train_net(case::String, data::Array{AbstractFloat},
		target::Array{AbstractFloat}, K1::Int64, K2::Int64=0, lr::Float64=1e-3,
		epochs::Int64=30, batch_size::Int64=64, retrain::Bool=false)
Train an MLP with provided dataset or load trained model. Saves trained model
with checkpointing, as well as the loss and accuracy data in current directory.

ARGUMENTS:

	case: matpower case file name as a string; e.g. case118
	data, target: dataset and ground truth, must be of dimension (K, N) where N
		= number of samples and K = number of features
	K1: a positive integer, first hidden layer size

OPTIONAL ARGUMENTS:

	K2: second hidden layer size, default is 0. In almost all cases having two hidden
		layers is sufficient, so no need for a K3
	lr: learning rate, default = 1e-3
	epochs: default = 30
	batch_size: a positive integer, default 64
	retrain: if `true` and a trained model already exists in current directory, then
		load the model, perform forward pass with data and target and exit. default
		is `false`

Note that there might not be enough memory on the GPU if not running on clusters,
	in which case... you should probably find a GPU with enough memory to ensure
	model is well trained. Otherwise, train on a CPU. NVIDIA GTX960M is known to
	have insufficient memory (2GB); anything with a 6GB VRAM or higher should
	work.
"""
function train_net(case::String, data::Array{AbstractFloat},
	target::Array{AbstractFloat}, K1::Int64, K2::Int64=0, lr::Float64=1e-3,
	epochs::Int64=30, batch_size::Int64=64, retrain::Bool=false)
	# Float32 should decrease memory allocation demand and run much faster on
	# non professional GPUs
	if typeof(data) != Array{Float32, 2}
		data = convert(Array{Float32}, data)
	end
	if typeof(target) != Array{Float32, 2}
		target = convert(Array{Float32}, target)
	end
	f = string(case, "_model.bson")

	# check if model is trained and does not need to be retrained
	if isfile(f) == true && retrain == false
		log = open("$(case)_train_output.log", "a")
		println(log, "Model already trained! Loading model for forward pass...")
		start = Base.time()
		BSON.@load string(filename, "_model.bson") model
		model |> gpu
		data |> gpu
		predict = model(data)
		elapsed = Base.time() - start
		println(log, "Loading model and forward pass took $(round(elapsed, digits=3)) seconds.")
		# save predict to current directory
		save("$(case)_predict.jld2", "predict", predict)
		println(log, "Saved predicted results to current directory as $(case)_predict.jld2")
		close(log)
    end

	# separate out train (70%), validation (15%) and test (15%) data
	N = size(data)[2]
	trainSplit = round(Int32, N*0.7)
	valSplit = round(Int32, N*0.85)
	trainData = data[:, 1:trainSplit] |> gpu
	trainTarget = target[:, 1:trainSplit] |> gpu
	valData = data[:, trainSplit:valSplit] |> gpu
	valTarget = target[:, trainSplit:valSplit] |> gpu
	testData = data[:, valSplit:end] |> gpu
	testTarget = target[:, valSplit:end] |> gpu

	# network model: if K2 is nonzero, there are two hidden layers, otherwise 1
	layer1 = Dense(size(data)[1], K1, relu)  # weight matrix 1
	layer2 = Dense(K1, size(target)[1]) # weight matrix 2
	model = Chain(layer1, layer2) |> gpu  # foward pass: ŷ = model(x)
	nlayers = 1  # number of hidden layer(s)
	if K2 != 0  # second hidden layer, optional
		layer2 = Dense(K1, K2, relu)
		layer3 = Dense(K2, size(target)[1])
		model = Chain(layer1, layer2, layer3) |> gpu
		nlayers = 2
	end

	# record loss/accuracy data for three sets
	trainLoss, trainAcc, valLoss, valAcc = Float32[], Float32[], Float32[], Float32[]
	epochTrainLoss, epochTrainAcc = 0.0, 0.0
	batchData, batchTarget = undef, undef

	# loss function and accuracy measure
	loss(x, y) = Flux.mse(model(x), y)
	accuracy(x, y) = 1 - abs.(mean(y - model(x)))

	opt = ADAM(lr)  # ADAM optimizer
	# opt = Momentum(lr)  # SGD with momentum

	elapsedEpochs = 0
	training_time = 0.0  # excludes early stop condition & checkpointing overhead

	# train with mini batches; if batch_size = 0, train with full batch
	randIdx = collect(1:1:size(trainData)[2])
	if batch_size > 0
		numBatches = round(Int, floor(size(trainData)[2] / batch_size - 1))
	else
		numBatches = 1
		batch_size = size(trainData)[2] - 1
    end
	for epoch = 1:epochs
		time = Base.time()
		println("epoch: $epoch")
		Random.shuffle!(randIdx) # to shuffle training set
		i = 1
		for j = 1:numBatches
			batchData = trainData[:, randIdx[i:i+batch_size]]
			batchTarget = trainTarget[:, randIdx[i:i+batch_size]]
			Flux.train!(loss, Flux.params(model), [(batchData, batchTarget)], opt)
			epochTrainLoss += Tracker.data(loss(batchData, batchTarget))
			epochTrainAcc += Tracker.data(accuracy(batchData, batchTarget))
			i += batch_size
		end
		push!(trainLoss, epochTrainLoss / numBatches)
		push!(trainAcc, epochTrainAcc / numBatches)
		push!(valLoss, Tracker.data(loss(valData, valTarget)))
		push!(valAcc, Tracker.data(accuracy(valData, valTarget)))
		epochTrainLoss, epochTrainAcc = 0.0, 0.0   # reset values
		training_time += Base.time() - time
		elapsedEpochs = epoch

		# checkpoint if val acc increased; to load, change @save to @load
		if epoch > 1 && valAcc[end] > valAcc[end-1]
			model_checkpt = cpu(model)
			@save string(case, "_ep", epoch, "_model.bson") model_checkpt  # to get weights, use Tracker.data()
		end
		# early exit condition
		if trainAcc[end] > 0.99 && valAcc[end] > 0.99
			break
		end
	end
	testAcc = Tracker.data(accuracy(testData, testTarget))  # test set accuracy
	# write result to file
	log = open("$(case)_train_output.log", "a")
	println(log, "Finished training after $elapsedEpochs epochs and $(round(
					training_time, digits=3)) seconds")
	println(log, "Hyperparameters used: learning rate = $lr,
				batch_size = $batch_size, number of hidden layers = $nlayers")
	println(log, "Test set accuracy: $(round(testAcc*100, digits=3))%")
	close(log)
	# save final model
	model = cpu(model)
	@save string(case, "_model.bson") model  # to get weights, use Tracker.data()
	# save loss and accuracy data
	save(string(case, "loss_acc_data.jld2"),
		"trainLoss", trainLoss, "trainAcc", trainAcc, "valLoss", valLoss, "valAcc", valAcc)
	plot_results(trainLoss, trainAcc, valLoss, valAcc, case)
end

"""
Plots the loss and accuracy curves of training and validation sets, and saves
the figures as PNGs in current directory.
"""
function plot_results(trainLoss::Array{Float32, 1}, trainAcc::Array{Float32, 1},
					valLoss::Array{Float32, 1}, valAcc::Array{Float32, 1}, filename::String)
	minAcc = min(minimum(trainAcc), minimum(valAcc))*.9  # for y axis limit
	n = collect(1:1:length(trainLoss))  # horizontal axis
	labels = ["Training", "Validation"]

	plot(n, trainLoss, title="Loss", label=labels[1], xlabel="epoch", ylabel="loss")
	plot!(n, valLoss, label=labels[2], xlabel="epoch", ylabel="loss")
	png(string(filename, "_loss_plot"))
	plot(n, trainAcc, title="Accuracy", label=labels[1], xlabel="epoch", ylabel="accuracy")
	plot!(n, valAcc, label=labels[2], xlabel="epoch", ylabel="accuracy", legend=:right, ylims=(minAcc, 1))
	png(string(filename, "_accuracy_plot"))
end
