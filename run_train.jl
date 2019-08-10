"""
Running on command line (assuming train.jl is in current directory):
default hyperparameters:
`/path/to/julia run_train.jl <case name> -d
custom hyperparameters (currently must be complete):
`/path/to/julia run_train.jl <case name> <second hidden layer Y/N> <retrain Y/N>
 	<learning rate> <epochs> <batch size>`
TODO: accommodate vararg hyperparameter set
"""

using Dates, FileIO

error = false
case = ARGS[1]  # case name, is String
if length(ARGS) != 6 && length(ARGS) != 2
	println("Incorrect number of arguments provided. Expected 6, received $(length(ARGS))")
	error = true
elseif length(ARGS) == 2 && ARGS[2] != "-d"
	println("Unknown option. Expected -d or custom hyperparameters, received \"$(ARGS[2])\"")
	error = true
elseif isfile("train.jl") == false
	println("train.jl not found in current directory. Exiting...")
	error = true
elseif isfile("$(case)_pf_results.jld2") == false
	println("Dataset $(case)_pf_results.jld2 not found in current directory. Exiting...")
	error = true
end

if error == false
	# load dataset from local directory
	data = FileIO.load("$(case)_pf_results.jld2")["data"]
	target = FileIO.load("$(case)_pf_results.jld2")["target"]
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
	if length(ARGS) == 6
		K2 = (ARGS[2]=="Y" || ARGS[2]=="y") ? K1 : 0  # if yes, set to same size as K1
		retrain = (ARGS[3]=="Y" || ARGS[3]=="y") ? true : false
		lr = parse(Float64, ARGS[4])
		epochs = parse(Int64, ARGS[5])
		bs = parse(Int64, ARGS[6])
		default_param = false
	end
	include("train.jl")

	log = open("$(case)_train_output.log", "a")
	println(log, now())
	close(log)
	# train
	println("Training a model for $case...")
	if default_param == true
		@time train_net(case, data, target, K1, K2)
	else
		@time train_net(case, data, target, K1, K2, lr, epochs, bs, retrain)
	end
	println("Program finished. Exiting...")
end
