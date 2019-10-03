function [P, Q, Qp] = pq_var(str, numSample)
    define_constants;
    rng(99);  % set rand seed
    s = rng;
    mpc = loadcase(str);
    numBus = size(mpc.bus, 1);
    
    %% generate PD variations on all PQ buses, i.e. BUS_TYPE == 1
    P = zeros([numBus, numSample], 'single');  % equivalent of Julia Float32
    for i = 1:numBus
        avg_p = mpc.bus(i, PD);  % bus PD
        % avg_p could be negative, but std should be the same as if
        % it's positive
        std_p = 5.44130 + 0.17459*sqrt(abs(avg_p)) + 0.001673*abs(avg_p);
        dist = makedist('Normal', 'mu', avg_p, 'sigma', std_p);
        P(i, :) = random(dist, 1, numSample);
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
    
    % Keep PV buses PD, QD values as is
    gen_idx = find(mpc.bus(:, BUS_TYPE) == PV);
    P(gen_idx, :) = repmat(mpc.bus(gen_idx, PD), 1, numSample);
    Q(gen_idx, :) = repmat(mpc.bus(gen_idx, QD), 1, numSample);
    
    %% Calculate \deltaQ as percentage
    % if Q_og(i) is 0, percentage is undefined, so set as 0
    Q_og = mpc.bus(:, QD);
    Qp = (Q - Q_og) ./ Q_og;
    Qp(Q_og == 0.0, :) = 0.0;
    
    %%
    % save PD, QD to file in the case that one or more columns of P,Q 
    % result in failed pf, we know the values that cause the failures
    % also save the rng setting
%     save(['./results/', str, '_pqvar.mat'], 'P', 'Q', 's'); 
end