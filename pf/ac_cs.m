function [V_M, V_A, itr_et, itr_n, fail] = ac_cs(str, P, Q, DCVM, DCVA)
    define_constants;
    mpc = loadcase(str);
    
    V_M = zeros(size(P), 'single');  % equivalent of Julia Float32
    V_A = zeros(size(P), 'single');
    
    numSample = size(P, 2);
    itr_n = zeros(numSample, 1);  % number of iteration each sample
    itr_et = zeros(numSample, 1);  % iteration elapsed time
    
    % arr for i-th failed NR
    fail = [];  % should preallocate for perf, but ret.et is the NR time so I don't care
    
    % run acpf numSample times
    mpopt = mpoption('out.all', 0, 'verbose', 0);  % verbose = 2 if want to print max P,Q mismatch
    for i = 1:numSample
        if mod(i, 1000) == 0
            fprintf(' >> acpf cold start iteration %d\n', i);
        end
        mpc.bus(:, PD) = P(:, i);
        mpc.bus(:, QD) = Q(:, i);
        ret = runpf(mpc, mpopt);
        if ret.success == 1
            V_M(:, i) = ret.bus(:, VM);
            V_A(:, i) = ret.bus(:, VA);
            itr_n(i) = ret.iterations;
            itr_et(i) = ret.et;
        else
            fail = [fail, i];
        end
    end
    if length(fail) > 0
        fprintf(' Number of failed samples: %i\n', length(fail));
    end
    assert(size(P, 2) == numSample && size(Q, 2) == numSample && size(V_M, 2) == numSample && size(V_A, 2) == numSample);
    assert(length(itr_et) == numSample && length(itr_n) == numSample);
end