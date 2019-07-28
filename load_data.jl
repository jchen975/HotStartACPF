using Distributions
using Distributions: Random
using JLD2, FileIO

# bus type
const PQ_BUS = 1
const PV_BUS = 2
const REF_BUS = 3

@everywhere using ParallelDataTransfer
@everywhere using PowerModels, JuMP, Ipopt # powerl flow packages

"""
	compute_pf(i::Int64)
Parallel power flow computation
After creating the P, Q matrices and parsing a matpower case file, compute
DC and AC power flow in parallel as follows:
	1. Pass P, Q, network_data to all worker processes
	2. Using `pmap()` and its argument `i`, each worker process will first make a
		deep copy of `network_data` dictionary, and then use the i-th column of
		P, Q to update the `pd`, `qd` fields for all loads in `network_data`. Each
		worker process will then use the `run_dc_pf` and `run_ac_pf` functions in
		PowerModels to compute the DC and AC power flow results. Finally, each
		worker returns the DC, AC power flow solutions and solution times as an
		array of type Any, with the first two element being the solutions
		dictionary and the last two being the solution times as Floats.
	3. This will not cause race conditions as only the i-th column of P, Q are
		accessed by each worker, and no worker process will access it twice.
		No worker process writes to the shared P, Q matrices, and only writes
		to the deep copied network_data, which is an overhead.
Several notes:
	1. `@everywhere` macro must be placed before all function definitions and
		`using` statements if they are shared by workers
	2. `addprocs()`` must be called before all @everywhere instances
"""
# not sure why but if there's no empty line between """ and @everwhere
# docstring breaks
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

"""
Load demand variations based on Zhen Dai's paper; NY parameters

TODO: add variations for PV buses: P and Vm
"""
function get_PQ_variation(PD::Array{Float64}, baseMVA::Int64, N::Int64)
	Random.seed!(521) # random seed for reproducible outputs
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

"""
	load_data(case::String, N::Int64, save_data::Bool=false, log::Bool=false)
Perform dc and ac power flow computations or load existing data
`PowerModels` parses matpower casefiles into a dictionary of dictionaries, where
loads are viewed as explicit components that are separate from buses (they are
two distinct dictionaries). So a bus can have none, 1, or many loads. The load
objects have unique ids, which are independent from the bus ids. These load
object ids are the keys in the load dictionary and the index value. The
`load_bus` value specifies which bus id the load is connected to. Because buses
can have no loads in `PowerModels`, when parsing Matpower data, loads with
`pd`=0.0 and `qd`=0.0 are filter out. `loadToBusIdx` is therefore a 2D array to
help connecting load and bus indices: first column is the load indices and the
second the corresponding bus indices.

ARGUMENTS:

	N = number of samples for the dataset
	case = "caseXXX", a string representing the name of the matpower casefile to
		parse from.

OPTIONAL ARGUMENTS:

	save_data: whether we save the computed pf results, default is false
	log: whether to print outputs to file, default is false

Note that for precompilation, it is best to run `load_data` first with a small
	N once or twice, then the intended N with both save_data and log as true.
	Also, `@time` macro should be placed before calling load_data if we want to
	benchmark the performance, both time and memory usage. This will include the
	pure pf time and other overhead.
"""
function load_data(case::String, N::Int64, save_data::Bool=false,
					log::Bool=false, reload::Bool=false)
	if isfile(string(case, "_pf_results.jld2")) == true && reload == false
		## Uncomment if running in REPL and calling train_net next
		# f = string(case, "_pf_results.jld2")
		# data = FileIO.load(f)["data"]
		# target = FileIO.load(f)["target"]
		if log == true
			log = open("$(case)_pf_output.log", "w")
			println(log, "Dataset already exists in current directory.")
			println("Total load data performance:")
			close(log)
		end
		# return data, target
	end
	# read matpower case file
	network_data = parse_file(string(case, ".m"))
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
	if save_data == true
		save(string(case, "_pq_values_parallel.jld2"), "PD", PD, "QD", QD)
	end

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
	# only write to output.log if we're saving data, i.e. not test runs
	if save_data == true && log == true
		log = open("$(case)_pf_output.log", "w")
		println(log, "Number of workers: $(nprocs()-1)")
		println(log, "Total power flow computation time: $(round(pf_time, digits=3)) seconds")
		println(log, "  => dcpf: $(round(dc, digits=3)) %")
		println(log, "  => acpf: $(round(ac, digits=3)) %")
		println(log, "Extracting results time: $(round(reduce_time, digits=3)) seconds")
		println("Total load data performance:")
		close(log)
	end
	# save data and target to disk
	if save_data == true
		save(string(case, "_pf_results.jld2"), "data", data, "target", target)
	end
	## Uncomment if running in REPL and calling train_net next
	# return data, target
end
