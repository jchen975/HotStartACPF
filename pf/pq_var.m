function [P, Q] = pq_var(str, numSample)
    define_constants;
    rng(0);  % set rand seed
    mpc = loadcase(str);
    numBus = size(mpc.bus, 1);
    non_pq = [];  % indices of PV, slack and isolated buses
    non_pq_qd = [];  % default QD values at the non PQ buses
    
    %% generate PD variations on all PQ buses, i.e. BUS_TYPE == 1
    P = zeros([numBus, numSample], 'single');  % equivalent of Julia Float32
    for i = 1:numBus
        avg_p = mpc.bus(i, PD);  % bus PD
        if mpc.bus(i, BUS_TYPE) == 1
            % avg_p could be negative, but std should be the same as if
            % it's positive
            std_p = 5.44130 + 0.17459*sqrt(abs(avg_p)) + 0.001673*abs(avg_p);
            dist = makedist('Normal', 'mu', avg_p, 'sigma', std_p);
            P(i, :) = random(dist, 1, numSample);
        else
            P(i, :) = mpc.bus(i, PD);  % keep default PD
            non_pq = [non_pq, i];
            non_pq_qd = [non_pq_qd; mpc.bus(i, QD)];
        end
    end
    
    %% generate QD variations
    % power factor distribution
    % truncated normal with mean = 1, std = 0.05 between [0.7, 1.0]
    pf_dist = makedist('Normal', 'mu', 1.0, 'sigma', 0.05);
    pf_dist = truncate(pf_dist, 0.7, 1.0);
    pf_vec = random(pf_dist, 1, numSample);
    pf = zeros(size(P), 'single');
    for i = 1:numBus
        pf(i, :) = pf_vec(randperm(numSample));  % randomly shuffle
    end
    
    % generate QD variations based on P and pf matrices
    Q = P .* tan(acos(pf));
    Q(non_pq, :) = repmat(non_pq_qd, 1, numSample);  % change back to default qd values
    
    %%
    % save PD, QD to file in the case that one or more columns of P,Q 
    % result in failed pf, we know the values that cause the failures
    save(['./results/', str, '_pqvar.mat'], 'P', 'Q'); 
end