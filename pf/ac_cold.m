function [V_M, V_A, itr_et, itr_n, fail, mismatch] = ac_cold(str, P, Q)
    define_constants;
    mpc = loadcase(str);
    max_itr = 10;

    V_M = zeros(size(P), 'single');  % equivalent of Julia Float32
    V_A = zeros(size(P), 'single');
    
    % flat start vm = 1 except PV bus, all va = 0
    gen_idx = find(mpc.bus(:, BUS_TYPE) == PV);
    flat_vm = ones(size(V_M, 1), 1, 'single');   
    flat_vm(gen_idx, :) = mpc.bus(gen_idx, VM);
    flat_va = zeros(size(flat_vm), 'single');
    
    numSample = size(P, 2);
    itr_n = zeros(numSample, 1);  % number of iteration each sample
    itr_et = zeros(numSample, 1);  % iteration elapsed time
    mismatch = zeros([max_itr, numSample], 'single');  % N by itr matrix containing the PQ mismatch info
    
    % arr for i-th failed NR
    fail = [];  % should preallocate for perf, but ret.et is the NR time so I don't care
    
    % run acpf numSample times
    tol = 1e-3;
    mpopt = mpoption('out.all', 0, 'verbose', 0, 'pf.tol', tol, 'pf.nr.max_it', max_itr);
    for i = 1:numSample
        mpc.bus(:, PD) = P(:, i);
        mpc.bus(:, QD) = Q(:, i);
        mpc.bus(:, VM) = flat_vm;
        mpc.bus(:, VA) = flat_va;
        ret = runpf(mpc, mpopt);
        if ret.success == 1
            V_M(:, i) = ret.bus(:, VM);
            V_A(:, i) = ret.bus(:, VA);
            itr_n(i) = ret.iterations;
            itr_et(i) = ret.et;            
        else
            fail = [fail, i];
        end
        mismatch(:, i) = ret.mismatch;
    end
end