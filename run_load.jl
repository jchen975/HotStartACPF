"""
Only add processors if none has been added. The number of procs in addprocs()
will be the ones doing parallel pf computation, so should be ~40 on clusters and
5~7 on computers with 8 logical processors.

Running on command line (assuming load_data.jl is in current directory):
`/path/to/julia run_load.jl <case name> <number of samples> <number of workers>
		<force generating new dataset Y/N>`
"""

using Dates
using Distributed

error = false
case = ARGS[1]
if length(ARGS) != 4
	println("Incorrect number of arguments provided. Expected 4, received $(length(ARGS))")
	error = true
elseif isfile("load_data.jl") == false
	println("load_data.jl not found in current directory. Exiting...")
	error = true
elseif isfile("$case.m") == false
	println("$case.m not found in current directory. Exiting...")
	error = true
end

if error == false
	N = parse(Int64, ARGS[2])  # number of samples, is Int
	nworker = parse(Int64, ARGS[3])  # number of worker processes, is Int
	reload = (ARGS[4]=="Y" || ARGS[4]=="y") ? true : false

	# only add worker processes and include load_data.jl if passes error checking
	if nprocs() == 1
		addprocs(nworker)
	end
	include("load_data.jl")
	# record current time stamp at the start of log
	log = open("$(case)_output_pf.log", "a")
	println(log, now())
	println(log, "Running load_data for $case, $N samples, with $(length(workers())) workers")
	if reload == true
		println(log, "Force generating new dataset even if one exists.")
	end
	close(log)

	load_data(case, 10)  # precompilation run 1
	println("Warm up 1 successeful")
	load_data(case, 20)  # precompilation run 2
	println("Warm up 2 successeful")
	println("Finished warming up. Starting data generation of $N samples with $nworker workers")
	@time load_data(case, N, true, true, reload)  # full set and save outputs to file
	println("Program finished. Exiting...")
	rmprocs(workers())
end
