% c, T, lambda are all strings, ex. ac_hs('case30', '0.2', '2.0')
function ac_hs(c, T, lambda)  
    define_constants;

    c = char(c);
    mpc = loadcase(c);
    if isfile([c, '_predict_', T, 'T_', lambda, 'lambda.mat']) && isfile([c, '_pqvar.mat'])
        % load predicted vm, va and the original P, Q
        load([c, '_predict_', T, 'T_', lambda, 'lambda.mat']);
        load([c, '_pqvar.mat']);

        numBus = size(mpc.bus, 1);
        N = size(P, 2);
        numSample = int32((1-T)*N);  % N \ T

        P = P(:, (N-numSample+1):end);
        Q = Q(:, (N-numSample+1):end);
        V_A = rad2deg(vpredict(1:numBus, :));
        V_M = vpredict(numBus+1:end, :);
        
        
        % flat start vm = 1 except PV bus, all va = 0
        gen_idx = find(mpc.bus(:, BUS_TYPE) == PV);
        flat_vm = ones(size(V_M, 1), 1, 'single');  
        flat_vm(gen_idx, :) = mpc.bus(gen_idx, VM);
        flat_va = zeros(size(flat_vm), 'single');

        itr_ac = zeros(numSample, 1);  % number of iteration each sample
        et_ac = zeros(numSample, 1);  % iteration elapsed time

        % arr for i-th failed NR
        fail = [];  % should preallocate for perf, but ret.et is the NR time so I don't care

        % run acpf numSample times
        mpopt = mpoption('out.all', 0, 'verbose', 2, 'pf.nr.max_it', 30);
        for i = 1:5%numSample
            mpc.bus(:, PD) = P(:, i);
            mpc.bus(:, QD) = Q(:, i);
            mpc.bus(:, VM) = flat_vm;
            mpc.bus(:, VA) = flat_va;
            fprintf('================== cold %i ==================', i)
            cold = runpf(mpc, mpopt);
            mpc.bus(:, VM) = V_M(:, i);
            mpc.bus(:, VA) = V_A(:, i);
            fprintf('================== hot %i ==================', i)
            ret = runpf(mpc, mpopt);
            if ret.success == 1
                itr_ac(i) = ret.iterations;
                et_ac(i) = ret.et;
            else 
                fail = [fail, i];
            end
        end
        if length(fail) > 0
            fprintf(' Number of failed ACPF: %i\n', length(fail));
        end
        assert(size(P, 2) == numSample && size(Q, 2) == numSample && size(V_M, 2) == numSample && size(V_A, 2) == numSample);
        assert(length(et_ac) == numSample && length(itr_ac) == numSample);
        
        T_str = num2str(T);
        lambda_str = char(sprintf("%.1f", lambda));
        fn = ['./results/', c, '_perf_hs_', T_str, 'T_', lambda_str, 'lambda.mat'];
%         save(fn, 'itr_ac', 'et_ac')
%         cd ..
%         perf(c, 'hs', T_str, lambda_str);  % print performance
%         cd ./pf
    else
        fprintf(' File "%s" does not exist.\n', [c, '_predict_', T, 'T_', lambda, 'lambda.mat']);
        fprintf(' Current directory: %s\n', pwd);
    end
end