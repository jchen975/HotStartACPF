using PowerModels, Plots

function plot_pf_distribution(case::String)
    ndata = parse_file(case*".m")
    load = ndata["load"]
    len = length(load)
    pd, qd = zeros(len), zeros(len)
    for i = 1:len
        pd[i] = load[string(i)]["pd"]
        qd[i] = load[string(i)]["qd"]
    end
    pf = cos.(atan.(qd ./ pd))
    histogram(pf, bin=100, xlims=(0, 1), xticks=0:0.05:1, legend=:topleft)
    N = length(pf)
    println("Power factor ∈ (0.9, 1.0]: $(round(sum(0.9 .< pf .<= 1.0) / N *100, digits=3))%")
    println("Power factor ∈ (0.8, 0.9]: $(round(sum(0.8 .< pf .<= 0.9) / N *100, digits=3))%")
    println("Power factor ∈ (0.7, 0.8]: $(round(sum(0.7 .< pf .<= 0.8) / N *100, digits=3))%")
    println("Power factor below 0.7: $(round(sum(0 .< pf .<= 0.7) / N *100, digits=3))%")
end

"""
Plots the loss and accuracy curves of training and validation sets, and saves
the figures as PNGs in current directory.
"""
function plot_results(trainLoss::Array{Float32, 1}, trainAcc::Array{Float32, 1},
					valLoss::Array{Float32, 1}, valAcc::Array{Float32, 1}, case::String)
	minAcc = min(minimum(trainAcc), minimum(valAcc))*.9  # for y axis limit
	n = collect(1:1:length(trainLoss))  # horizontal axis
	labels = ["Training", "Validation"]

	plot(n, trainLoss, title="Loss", label=labels[1], xlabel="epoch", ylabel="loss")
	plot!(n, valLoss, label=labels[2], xlabel="epoch", ylabel="loss")
	png("$(case)_loss_plot")
	plot(n, trainAcc, title="Accuracy", label=labels[1], xlabel="epoch", ylabel="accuracy")
	plot!(n, valAcc, label=labels[2], xlabel="epoch", ylabel="accuracy", legend=:right, ylims=(minAcc, 1))
	png("$(case)_accuracy_plot")
end
