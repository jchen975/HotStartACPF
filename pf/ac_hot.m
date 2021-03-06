% c and T are both strings, ex. ac_hot('case30', '2000')
function ac_hot(c, T)  
    define_constants;
    c = char(c);
    mpc = loadcase(c);
    dir_predict = 'D:/Work/Research/Cluster Results/Jan 2020/';
    dir_data = 'D:/Work/Research/DC-Power-Flow-Correction/pf/results/';
    fpredict = [dir_predict, c, '/', c, '_predict_T=', T, '.mat'];
    fdata = [dir_data, c, '_dataset.mat'];
    
    if isfile(fpredict) && isfile(fdata)
        % load predicted vm, va and the original P, Q
        load(fpredict);
        load(fdata);

        N = size(P, 2);
%         numSample = int32((1-T)*N);  % N \ T

        P = P(:, (T+1):end);
        Q = Q(:, (T+1):end);
      
        max_itr = 10;
        itr_ac = zeros(N-T, 1);  % number of iteration each sample
        et_ac = zeros(N-T, 1);  % iteration elapsed time
        mismatch_hot = zeros([max_itr, N-T], 'single');  % PQ mismatch info

        % arr for i-th failed NR
        fail = [];  % should preallocate for perf, but ret.et is the NR time so I don't care

        % run acpf numSample times
        mpopt = mpoption('out.all', 0, 'verbose', 0, 'pf.tol', 1e-3);
        for i = 1:(N-T)
            mpc.bus(:, PD) = P(:, i);
            mpc.bus(:, QD) = Q(:, i);
            mpc.bus(:, VM) = V_M(:, i); 
            mpc.bus(:, VA) = V_A(:, i); 
 
            ret = runpf(mpc, mpopt);
            if ret.success == 1
                itr_ac(i) = ret.iterations;
                et_ac(i) = ret.et;
                mismatch_hot(:, i) = ret.mismatch;
            else 
                itr_ac(i) = max_itr;
                fail = [fail, i];
            end
        end
        if length(fail) > 0
            fprintf(' Number of failed ACPF: %i\n', length(fail));
        end
%         assert(size(P, 2) == numSample && size(Q, 2) == numSample && size(V_M, 2) == numSample && size(V_A, 2) == numSample);
%         assert(length(et_ac) == numSample && length(itr_ac) == numSample);
        
        T_str = num2str(T);
        fn = [dir_predict, c, '/', c, '_perf_hot_T=', T_str, '.mat'];
        save(fn, 'itr_ac', 'et_ac', 'mismatch_hot')
        perf(c, 'hot', T_str);  % print performance

    else
        fprintf(' File "%s" or the corresponding PQ data does not exist.\n', [c, '_predict_T=', T, '.mat']);
        fprintf(' Current directory: %s\n', pwd);
    end
end