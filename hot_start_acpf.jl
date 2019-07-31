using FileIO

# bus type
const PQ_BUS = 1
const PV_BUS = 2
const REF_BUS = 3

@everywhere using ParallelDataTransfer
@everywhere using PowerModels, JuMP, Ipopt # powerl flow packages

"""
	compute_acpf(i::Int64)
AC power flow computation in parallel; returns `solution` as dict and `solve_time`
as float
"""
# rerun ac power flow with hot start values from training
@everwhere function compute_acpf(i::Int64)
	network = deepcopy(ndata)  # we need to write to network dict so make copy
	idx = 1
	for j = 1:numPQ
		network["bus"][j]["va"] = start[idx, i]
		network["bus"][j]["vm"] = start[idx+1, i]
		idx += 2  # numFeature = 2: va, vm
	end
	PowerModels.silence()
	opt = Ipopt.Optimizer
	ac_result = run_ac_pf(network, JuMP.with_optimizer(opt, print_level=0))

	ret = Array{Any}(undef, 2)
	ret[1] = ac_result["solution"]
	ret[2] = ac_result["solve_time"]
	return ret
end

"""
	hot_start_acpf(case::String)
Rerun parallel ac power flow with predicted values from NN forward pass.
Not providing options such as save_data and log because this will be a final run
"""
function hot_start_acpf(case::String)
	# check if forward pass results exists
	if isfile("$(case)_predict.jld2") == false
		log = open("$(case)_output_hot_start_acpf.log")
		println(log, "Predicted results not found in current directory")
		close(log)
		return nothing

	predict = FileIO.load("$(case)_predict.jld2")["predict"]
	N = size(predict)[2]
	PowerModels.silence()
	network_data = parse_file("$case.m")
	numPQ = length(network_data["load"])
	# bus = network_data["bus"]
	# numBus = length(bus)

	# record the PQ bus indices
	loadToBusIdx = loadToBusIdx = hcat(collect(Int64, 1:1:numPQ), zeros(Int64, numPQ))
	for i = 1:numPQ
		loadToBusIdx[i, 2] = load[string(i)]["load_bus"]
	end

	sendto(workers(), ndata = network_data, start = predict, numPQ = numPQ)
	ret = pmap(compute_acpf, 1:N)  # parallel ac pf

	# extract results: vm, va at  all PQ buses
	numTarget = Int32(2)  # vm_ac, va_ac
	yIncrement = numTarget - 1
	hs_result = zeros(Float32, numPQ*numTarget, N)
	time = Base.time()
	ac_time = .0
	for i = 1:N
		y = Int32(1)
		for k = 1:numPQ
			# ret[i][1], ret[i][2] = i-th solution ∀ i = 1, ..., N;
			# 1 = ac solution, 2 = ac solve time
			hs_result[y:y+yIncrement, i] = [
							ret[i][1]["bus"][string(loadToBusIdx[k, 2])]["vm"],
							ret[i][1]["bus"][string(loadToBusIdx[k, 2])]["va"]]
			y += numTarget
		end
		ac_time += ret[i][2]
	end
	reduce_time = Base.time() - time
	log = open("$(case)_output_pf.log", "a")
	println(log, "AC power flow computation with hot start time: $(round(
			pf_time, digits=3)) seconds")
	close(log)
	# save final values
	save("$(case)_pf_hot_start_results.jld2", "hs_result", hs_result)
end
