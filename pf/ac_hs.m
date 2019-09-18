function ac_hs(c, T, lambda)
    define_constants;

    c = char(c);
    mpc = loadcase(c);
    if isfile([c, '_predict_', T, 'T_', lambda, 'lambda.mat'])
        load([c, '_predict_', T, 'T_', lambda, 'lambda.mat']);

        numBus = size(pq, 1) / 2;
        P = pq(1:numBus, :) * mpc.baseMVA;
        Q = pq(numBus+1:end, :) * mpc.baseMVA;
        V_A = rad2deg(vpredict(1:numBus, :));
        V_M = vpredict(numBus+1:end, :);
        numSample = size(P, 2);  % N \ T

        itr_ac = zeros(numSample, 1);  % number of iteration each sample
        et_ac = zeros(numSample, 1);  % iteration elapsed time

        % arr for i-th failed NR
        fail = [];  % should preallocate for perf, but ret.et is the NR time so I don't care

        % run acpf numSample times
        warm = 0;  % DC result warm start counter
        flat = 0;  % flat start counter
        mpopt = mpoption('out.all', 0, 'verbose', 0, 'pf.nr.max_it', 30);
        for i = 1:numSample
            mpc.bus(:, PD) = P(:, i);
            mpc.bus(:, QD) = Q(:, i);
            mpc.bus(:, VM) = V_M(:, i);
            mpc.bus(:, VA) = V_A(:, i);
            ret = runpf(mpc, mpopt);
            if ret.success == 1
                itr_ac(i) = ret.iterations;
                et_ac(i) = ret.et;
            else  % resistance is futile
                    fail = [fail, i];
            end
        end
        if length(fail) > 0
            fprintf(' Number of successful warm-starts: %i\n', warm);
            fprintf(' Number of successful flat-starts: %i\n', flat);
        end
        assert(size(P, 2) == numSample && size(Q, 2) == numSample && size(V_M, 2) == numSample && size(V_A, 2) == numSample);
        assert(length(et_ac) == numSample && length(itr_ac) == numSample);
        
        T_str = num2str(T);
        lambda_str = char(sprintf("%.1f", lambda));
        fn = ['../results/', c, '_perf_hs_', T_str, 'T_', lambda_str, 'lambda.mat'];
        save(fn, 'itr_ac', 'et_ac');
        cd ..
        perf(c, 'hs', T_str, lambda_str);
        cd ./pf
    else
        fprintf(' File "%s" does not exist.\n', [c, '_predict_', T, 'T_', lambda, 'lambda.mat']);
        fprintf(' Current directory: %s\n', pwd);
    end
end