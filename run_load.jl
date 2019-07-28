"""
Only add processors if none has been added. The number of procs in addprocs()
will be the ones doing parallel pf computation, so should be ~40 on clusters and
5~7 on computers with 8 logical processors.
When running on clusters, the third argument in bash file should be the number
of worker processes to add.
"""

using Distributed

if length(ARGS) == 3
    case = ARGS[1]  # case name, is String
    N = parse(Int64, ARGS[2])  # number of samples, is Int
	nworker = parse(Int64, ARGS[3])  # number of worker processes, is Int
	if nprocs() == 1
		addprocs(nworker)
	end
    println("$case, $N samples, $nworker workers")
	include("load_data.jl")
	println("Include successful")
	load_data(case, 10)  # precompilation run 1
	println("Warm up 1 successeful")
	load_data(case, 20)  # precompilation run 2
	println("Warm up 2 successeful")
	println("Finished warming up, starting data generation of $N samples with  $nworker workers")
	log = open("$(case)_pf_output.log", "w")
	@time load_data(case, N, true, true)  # full set and save outputs to file
	close(log)
	println("Program finished. Exiting...")
else
	println("Incorrect number of arguments provided. Expected 3, received $(length(ARGS))")
end
