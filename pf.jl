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
