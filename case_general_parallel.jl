### Deprecated ###
using Distributed  # parallelize pf computations
using Distributions
using Distributions: Random
using Flux # ML and GPU packages
using Flux.Optimise: ADAM
using CuArrays
using Plots
using JLD2, FileIO, BSON  # saving and loading files
using BSON: @save

# only add processors if haven't added any; needs to be before any @everywhere
# for the workers to have access
# the number of procs in addprocs() will be the ones doing parallel pf computation
# proc 1 is lazy af and won't do shit
if nprocs() == 1
	addprocs(5)
end
numProcs = nprocs()-1
println("Number of workers: ", numProcs)

# bus type
const PQ_BUS = 1
const PV_BUS = 2
const REF_BUS = 3

@everywhere using ParallelDataTransfer
@everywhere using PowerModels, JuMP, Ipopt # powerl flow packages

Random.seed!(521) # random seed for reproducible outputs

# I think this works
@everywhere function compute_pf(i::Int64)
	# println(i)  # uncomment this if you want to see which worker is executing what
	network = deepcopy(ndata)  # we need to write to network dict so make copy
	for j = 1:numPQ
		network["load"][string(j)]["pd"] = P[j, i]
		network["load"][string(j)]["qd"] = Q[j, i]
	end
	PowerModels.silence()
	opt = Ipopt.Optimizer
	dc_result = run_dc_pf(network, JuMP.with_optimizer(opt, print_level=0))
	ac_result = run_ac_pf(network, JuMP.with_optimizer(opt, print_level=0))

	ret = Array{Any}(undef, 4)
	ret[1] = dc_result["solution"]
	ret[2] = ac_result["solution"]
	ret[3] = dc_result["solve_time"]
	ret[4] = ac_result["solve_time"]
	return ret
end

# load variations based on Zhen Dai's paper; NY parameters
function get_PQ_variation(PD::Array{Float64}, baseMVA::Int64, N::Int64)
	numPQ = length(PD)
	P̄ = mean(PD) * baseMVA
	α0 = 5.44130
	α1 = 0.17459
	α2 = 0.001673
	σP = α0 + α1*sqrt(P̄) + α2*P̄

	# Generate N (ΔPD, power factor(0.7~1.0 lagging)) independently Gaussian RVs
	# for load bus PQ variations; Q = tan(acos(power factor)) * P
	PD = rand(Normal(P̄, σP), N*numPQ) ./ baseMVA
	pf = rand(TruncatedNormal(0.85, 0.1, 0.7, 1.0), N*numPQ)
	QD = PD .* tan.(acos.(pf))

	# reshape variations above from 1D vec to numPQ x N, and change all
	# negative values to 0
 	PD = max.(reshape(PD, numPQ, :), 0)
	QD = max.(reshape(QD, numPQ, :), 0)
	return PD, QD
end

# do dc and ac power flow computations or load existing data
function load_data(N::Int64, case::String)
	if isfile(string(case, "_pf_results_parallel.jld2")) == true
		f = string(case, "_pf_results_parallel.jld2")
		data = FileIO.load(f)["data"]
		target = FileIO.load(f)["target"]
		println("Total load data performance:")
		return data, target
	end

	# read matpower case file
	network_data = parse_file(string(case, ".m"))
	# bus = network_data["bus"]  # a dictionary containing bus info (NOT USED)
	load = network_data["load"]
	baseMVA = network_data["baseMVA"]
	numPQ = length(load) # since load dictionary ignores 0 entries, its size <= length(bus)

	# first col is loadIdx, second is its corresponding bus idx
	loadToBusIdx = zeros(Int32, numPQ, 2)
	PD = zeros(numPQ)
	for i = 1:numPQ
		PD[i] = load[string(i)]["pd"]
		loadToBusIdx[i, 1] = 1  # ordered by load index
		loadToBusIdx[i, 2] = load[string(i)]["load_bus"] # record corresponding bus number
	end
	PD, QD = get_PQ_variation(PD, baseMVA, N)
	save(string(case, "_pq_values_parallel.jld2"), "PD", PD, "QD", QD)

	# generate input and label for NN
	# make PD, QD, network_data and numPQ accessable by all workers
	# each worker will only read from them, or first make deep copies then write
	# to copies to avoid race condition
	PowerModels.silence()
	sendto(workers(), ndata = network_data, P = PD, Q = QD, numPQ = numPQ)  # so that every worker can access this

	# run dc and ac pf in parallel
	time = Base.time()
	ret = pmap(compute_pf, 1:N)
	pf_time = Base.time() - time

	numFeature = Int32(4)  # pd, qd, vm_dc, va_dc, 1 PQ bus
	numTarget = Int32(2)  # vm_ac, va_ac
	xIncrement = numFeature - 1  # for indexing
	yIncrement = numTarget - 1
	data = zeros(Float32, numPQ*numFeature, N)  # pd, qd, vm_dc, va_dc ∀ PQ buses; samples are col vectors
	target = zeros(Float32, numPQ*numTarget, N)
	# reduce
	time = Base.time()
	dc_time, ac_time = .0, .0
	for i = 1:N
		x, y = Int32(1), Int32(1)
		for k = 1:numPQ
			# PD, QD access should be optimized here!
			# ret[i][1], ret[i][2] = i-th solution ∀ i = 1, ..., N;
			# 1 = dc solution, 2 = ac solution, 3 = dc solve time
			data[x:x+xIncrement, i] = [
							1.,
							ret[i][1]["bus"][string(loadToBusIdx[k, 2])]["va"],
							PD[k, i],
							QD[k, i]]
			target[y:y+yIncrement, i] = [
							ret[i][2]["bus"][string(loadToBusIdx[k, 2])]["vm"],
							ret[i][2]["bus"][string(loadToBusIdx[k, 2])]["va"]]
			x += numFeature
			y += numTarget
		end
		dc_time += ret[i][3]
		ac_time += ret[i][4]
	end
	reduce_time = Base.time() - time

	# currently dc_time and ac_time are both *actual* times, i.e. there are
	# overlaps between workers due to parallel execution; the ratio however is
	# true, so normalize it down with total parallel pf time
	dc = (dc_time / (dc_time+ac_time)) * 100.0
	ac = (ac_time / (dc_time+ac_time)) * 100.0

	# DC and AC computation time; will be less than the @time macro in main()
	# since that also has other overhead like network data dict accessing
	println("total power flow computation time: $(round(pf_time, digits=3)) seconds")
	println("  => dcpf: $(round(dc, digits=3)) %")
	println("  => acpf: $(round(ac, digits=3)) %")
	println("extracting results time: $(round(reduce_time, digits=3)) seconds")

	# save data and target to disk
	save(string(case, "_pf_results_parallel.jld2"), "data", data, "target", target)
	println("Total load data performance:")
	return data, target
end

# train_net model
function train_net(case::String, data::Array{Float32}, target::Array{Float32},
				lr::Float64, epochs::Int64, batch_size::Int64, K1::Int64; K2::Int64=0)
	f = string(case, "_model.bson")

	# separate out train (70%), validation (15%) and test (15%) data
	# don't move train set to gpu yet; may not have enough memory
	N = size(data)[2]
	trainSplit = round(Int32, N*0.7)
	valSplit = round(Int32, N*0.85)
	trainData = data[:, 1:trainSplit]
	trainTarget = target[:, 1:trainSplit]
	valData = data[:, trainSplit:valSplit] |> gpu
	valTarget = target[:, trainSplit:valSplit] |> gpu
	testData = data[:, valSplit:end] |> gpu
	testTarget = target[:, valSplit:end] |> gpu

	# network model: if K2 is nonzero, there are two hidden layers, otherwise 1
	layer1 = Dense(size(data)[1], K1, relu)  # weight matrix 1
	layer2 = Dense(K1, size(target)[1]) # weight matrix 2
	model = Chain(layer1, layer2) |> gpu  # foward pass: ŷ = model(x)

	if K2 != 0  # second hidden layer, optional
		layer2 = Dense(K1, K2, relu)
		layer3 = Dense(K2, size(target)[1])
		model = Chain(layer1, layer2, layer3) |> gpu
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

	# train with mini batches; only send mini batches to gpu instead of the entire set
	randIdx = collect(1:1:size(trainData)[2])
	numBatches = round(Int, floor(size(trainData)[2] / batch_size - 1))
	for epoch = 1:epochs
		time = Base.time()
		println("epoch: $epoch")
		Random.shuffle!(randIdx) # to shuffle training set
		i = 1  # Julia is 1 indexed
		for j = 1:numBatches
			# println(j)
			batchData = trainData[:, randIdx[i:i+batch_size]] |> gpu
			batchTarget = trainTarget[:, randIdx[i:i+batch_size]] |> gpu
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
			@save string(case, "_ep", epoch, "_model_parallel.bson") model_checkpt  # to get weights, use Tracker.data()
		end
		# early exit condition
		if trainAcc[end] > 0.99 && valAcc[end] > 0.99
			break
		end
	end
	testAcc = Tracker.data(accuracy(testData, testTarget))  # test set accuracy
	println(string("Finished training after $elapsedEpochs epochs and $(round(
					training_time, digits=3)) seconds"))
	println(string("Test set accuracy: $(round(testAcc*100, digits=3))%"))
	model = cpu(model)
	@save string(case, "_model_parallel.bson") model  # to get weights, use Tracker.data()

	return trainLoss, trainAcc, valLoss, valAcc
end

function plot_results(trainLoss::Array{Float64}, trainAcc::Array{Float64},
					valLoss::Array{Float64}, valAcc::Array{Float64}, filename::String)
	minAcc = min(minimum(trainAcc), minimum(valAcc))*.9  # for y axis limit
	n = collect(1:1:length(trainLoss))  # horizontal axis
	labels = ["Training", "Validation"]

	plot(n, trainLoss, title="Loss", label=labels[1], xlabel="epoch", ylabel="loss")
	plot!(n, valLoss, label=labels[2], xlabel="epoch", ylabel="loss")
	png(string(filename, "_loss_plot_parallel"))
	plot(n, trainAcc, title="Accuracy", label=labels[1], xlabel="epoch", ylabel="accuracy")
	plot!(n, valAcc, label=labels[2], xlabel="epoch", ylabel="accuracy", legend=:right, ylims=(minAcc, 1))
	png(string(filename, "_accuracy_plot_parallel"))
end

function main(case::String, N::Int64)
	if isfile(string(case, ".m")) == false
		println("Couldn't find $case.m in current directory! Exiting...")
		return
	end
	filename = case*".m"
	# get data
	@time data, target = load_data(N, case)  # to see time elapsed, add "@time " in front
	# training
	epochs = 30
	lr = 1e-4  # learning rate
	bs = 256
	K1 = round(Int, (size(data)[2] + size(target)[2]) / 2) # hidden layer size
	K2 = 0 #K1  # second hidden layer; by default not used

	retrain = false
	# Float32 should decrease memory allocation demand and run much faster on
	# non professional GPUs
	if typeof(data) != Array{Float32, 2}
		data = convert(Array{Float32}, data)
	end
	if typeof(target) != Array{Float32, 2}
		target = convert(Array{Float32}, target)
	end
	# check if model is trained and does not need to be retrained
	if isfile(string(filename, "_model_parallel.bson")) == true && retrain == false
		println("Model already trained! Loading model for forward pass...")
			start = Base.time()
			BSON.@load string(filename, "_model_parallel.bson") model
			model |> gpu
			data |> gpu
			predict = model(data)  # do stuff with this
			elapsed = Base.time() - start
		println("Loading model and forward pass took $(round(elapsed, digits=3)) seconds.")
	else
		trainLoss, trainAcc, valLoss, valAcc = train_net(filename,
														data, target,
														lr,
														epochs,
														bs,
														K1, K2=K2)
		plot_results(trainLoss, trainAcc, valLoss, valAcc, filename)
	end
end
