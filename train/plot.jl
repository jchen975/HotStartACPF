# using PowerModels
using Plots, GR, MAT

# function plot_pf_distribution(case::String)
#     ndata = parse_file(case*".m")
#     load = ndata["load"]
#     len = length(load)
#     pd, qd = zeros(len), zeros(len)
#     for i = 1:len
#         pd[i] = load[string(i)]["pd"]
#         qd[i] = load[string(i)]["qd"]
#     end
#     pf = cos.(atan.(qd ./ pd))
#     histogram(pf, bin=100, xlims=(0, 1), xticks=0:0.05:1, legend=:topleft)
#     N = length(pf)
#     println("Power factor ∈ (0.9, 1.0]: $(round(sum(0.9 .< pf .<= 1.0) / N *100, digits=3))%")
#     println("Power factor ∈ (0.8, 0.9]: $(round(sum(0.8 .< pf .<= 0.9) / N *100, digits=3))%")
#     println("Power factor ∈ (0.7, 0.8]: $(round(sum(0.7 .< pf .<= 0.8) / N *100, digits=3))%")
#     println("Power factor below 0.7: $(round(sum(0 .< pf .<= 0.7) / N *100, digits=3))%")
# end

"""
Plots the loss and accuracy curves of training and validation sets, and saves
the figures as PNGs in current directory.
"""
function plot_loss(case::String, trainLoss::Array{Float32}, valLoss::Array{Float32})
	n = collect(1:1:length(trainLoss))
	nval = length(valLoss)
	vLoss = zeros(size(trainLoss))
	j = 1
	for i = 1:nval
		vLoss[j:j+32] .= valLoss[i]
		j += 32+1
	end
	println(vLoss[end-5:end])
	labels = ["Training", "Validation"]
	plot(n, trainLoss, title=case, label=labels[1], xlabel="iterations", ylabel="loss")
	plot!(n, vLoss, label=labels[2])
	savefig("$(case)_loss_plot.pdf")
end

function plot_loss(case::String, T::Float64)
	trainLoss = matread("$(case)_loss_T=$T.mat")["trainLoss"]
	valLoss = matread("$(case)_loss_T=$T.mat")["valLoss"]
	fn = split(case, "_")
	if length(fn) == 1  # no "_va"/"_vm", trained together
		case = case*", T=$T Loss"
		plot_loss(case, trainLoss, valLoss)
	else
		fn[2] == "va" ? fn[1] *= " Voltage Angle Loss" : fn[1] *= " Voltage Magnitude Loss"
		fn[1] *= ", T=$T"
		plot_loss(String(fn[1]), trainLoss, valLoss)
	end
end

plot_loss("case118_va", 0.2)
