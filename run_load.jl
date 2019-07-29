"""
Only add processors if none has been added. The number of procs in addprocs()
will be the ones doing parallel pf computation, so should be ~40 on clusters and
5~7 on computers with 8 logical processors.

Running on command line (assuming load_data.jl is in current directory):
`/path/to/julia run_load.jl <case name> <number of samples> <number of workers>`
"""

using Dates
using Distributed

if length(ARGS) == 3
    case = ARGS[1]  # case name, is String
    N = parse(Int64, ARGS[2])  # number of samples, is Int
	nworker = parse(Int64, ARGS[3])  # number of worker processes, is Int
    println("Running load_data for $case, $N samples, with $nworker workers")
	if isfile("load_data.jl") == false
		println("load_data.jl not found in current directory. Exiting...")
		exit()
	end
	if isfile("$case.m") == false
		println("$case.m not found in current directory. Exiting...")
		exit()
	end
	# only add worker processes and include load_data.jl if passes error checking
	if nprocs() == 1
		addprocs(nworker)
	end
	include("load_data.jl")
	log = open("$(case)_train_output.log", "a")  # record current time stamp
	println(log, now())
	close(log)

	load_data(case, 10)  # precompilation run 1
	println("Warm up 1 successeful")
	load_data(case, 20)  # precompilation run 2
	println("Warm up 2 successeful")
	println("Finished warming up, starting data generation of $N samples with $nworker workers")
	@time load_data(case, N, true, true, true)  # full set and save outputs to file
	println("Program finished. Exiting...")
else
	println("Incorrect number of arguments provided. Expected 3, received $(length(ARGS))")
end
